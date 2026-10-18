// Debug aid: APS5_ALLOC_PROFILE=1 attributes the driver's live heap to its allocation sites. The
// driver's operator new and delete are replaced (memory still comes from malloc, so blocks freed by
// another module stay valid); allocations of at least APS5_ALLOC_PROFILE_MIN_KIB (default 16) are
// recorded with a short call stack, and every 10 s the sites holding the most live bytes are
// printed as module offsets ([alloc]). Off, the operators are plain malloc and free.
#ifdef _WIN32
#include <windows.h>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <new>
#include <thread>
#include <unordered_map>
#include <vector>
#include <algorithm>

extern "C" IMAGE_DOS_HEADER __ImageBase;

namespace {

bool enabled() {
    static const bool on = std::getenv("APS5_ALLOC_PROFILE") != nullptr;
    return on;
}

std::size_t minimumBytes() {
    static const std::size_t bytes = [] {
        const char* text = std::getenv("APS5_ALLOC_PROFILE_MIN_KIB");
        return static_cast<std::size_t>(text != nullptr ? std::strtoull(text, nullptr, 10) : 16u) << 10u;
    }();
    return bytes;
}

constexpr int Frames = 6;
using Stack = std::array<std::uintptr_t, Frames>;

struct StackHash {
    std::size_t operator()(const Stack& stack) const {
        std::uint64_t hash = 1469598103934665603ull;
        for (const auto frame : stack) hash = (hash ^ frame) * 1099511628211ull;
        return static_cast<std::size_t>(hash);
    }
};

struct Site {
    std::uint64_t liveBytes = 0;
    std::uint64_t liveCount = 0;
    std::uint64_t totalBytes = 0;
};

// The tables allocate with malloc (MallocAllocator) so recording never recurses into operator new.
template<class T>
struct MallocAllocator {
    using value_type = T;
    MallocAllocator() = default;
    template<class U> MallocAllocator(const MallocAllocator<U>&) {}
    T* allocate(std::size_t count) {
        if (auto* p = static_cast<T*>(std::malloc(count * sizeof(T)))) return p;
        std::abort();
    }
    void deallocate(T* p, std::size_t) { std::free(p); }
    template<class U> bool operator==(const MallocAllocator<U>&) const { return true; }
    template<class U> bool operator!=(const MallocAllocator<U>&) const { return false; }
};

struct Record {
    std::size_t bytes;
    const Stack* stack;
};

struct State {
    std::mutex mutex;
    std::unordered_map<Stack, Site, StackHash, std::equal_to<Stack>, MallocAllocator<std::pair<const Stack, Site>>> sites;
    std::unordered_map<void*, Record, std::hash<void*>, std::equal_to<void*>, MallocAllocator<std::pair<void* const, Record>>> live;
    std::uint64_t liveBytes = 0;
    std::chrono::steady_clock::time_point lastReport = std::chrono::steady_clock::now();
};

State& state() {
    static State* created = [] {
        void* memory = std::malloc(sizeof(State));
        if (memory == nullptr) std::abort();
        return new (memory) State();
    }();
    return *created;
}

thread_local bool inside = false;

std::uintptr_t moduleOffset(std::uintptr_t address, char* name, std::size_t nameBytes) {
    HMODULE module = nullptr;
    if (!GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, reinterpret_cast<LPCSTR>(address), &module) || module == nullptr) {
        std::snprintf(name, nameBytes, "?");
        return address;
    }
    char path[MAX_PATH] = {};
    GetModuleFileNameA(module, path, MAX_PATH);
    const char* base = path;
    for (const char* c = path; *c != '\0'; ++c) {
        if (*c == '\\' || *c == '/') base = c + 1;
    }
    std::snprintf(name, nameBytes, "%s", base);
    return address - reinterpret_cast<std::uintptr_t>(module);
}

void report(State& s) {
    std::vector<std::pair<const Stack*, Site>, MallocAllocator<std::pair<const Stack*, Site>>> ranked;
    for (const auto& [stack, site] : s.sites) {
        if (site.liveBytes != 0) ranked.emplace_back(&stack, site);
    }
    std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) { return a.second.liveBytes > b.second.liveBytes; });
    std::fprintf(stderr, "[alloc] driver heap: %.1f MiB live in %zu allocations of >= %zu KiB, %zu sites\n", s.liveBytes / 1048576.0, s.live.size(), minimumBytes() >> 10u, ranked.size());
    for (std::size_t i = 0; i < ranked.size() && i < 25; ++i) {
        const auto& [stack, site] = ranked[i];
        std::fprintf(stderr, "[alloc]   %8.1f MiB in %6llu (%.0f MiB ever):", site.liveBytes / 1048576.0, static_cast<unsigned long long>(site.liveCount), site.totalBytes / 1048576.0);
        for (const auto frame : *stack) {
            if (frame == 0) break;
            char name[64];
            const auto offset = moduleOffset(frame, name, sizeof(name));
            std::fprintf(stderr, " %s+0x%llx", name, static_cast<unsigned long long>(offset));
        }
        std::fputc('\n', stderr);
    }
    std::fflush(stderr);
}

void noteAllocation(void* pointer, std::size_t bytes, void* caller) {
    if (inside || bytes < minimumBytes()) return;
    inside = true;
    Stack stack{};
    void* frames[Frames + 2] = {};
    // The caller of operator new first (the stack walk may stop at frames without unwind data).
    stack[0] = reinterpret_cast<std::uintptr_t>(caller);
    const auto captured = RtlCaptureStackBackTrace(0, Frames + 2, frames, nullptr);
    for (USHORT i = 3; i < captured && i - 2 < Frames; ++i) stack[i - 2] = reinterpret_cast<std::uintptr_t>(frames[i]);
    auto& s = state();
    {
        std::lock_guard lock(s.mutex);
        auto [it, added] = s.sites.try_emplace(stack);
        it->second.liveBytes += bytes;
        it->second.liveCount += 1;
        it->second.totalBytes += bytes;
        s.live[pointer] = Record{bytes, &it->first};
        s.liveBytes += bytes;
        const auto now = std::chrono::steady_clock::now();
        if (now - s.lastReport > std::chrono::seconds(10)) {
            s.lastReport = now;
            report(s);
        }
    }
    inside = false;
}

void noteFree(void* pointer) {
    if (inside || pointer == nullptr) return;
    inside = true;
    auto& s = state();
    {
        std::lock_guard lock(s.mutex);
        if (const auto found = s.live.find(pointer); found != s.live.end()) {
            auto& site = s.sites.at(*found->second.stack);
            site.liveBytes -= found->second.bytes;
            site.liveCount -= 1;
            s.liveBytes -= found->second.bytes;
            s.live.erase(found);
        }
    }
    inside = false;
}

// APS5_ALLOC_RATE=1: allocation counts by calling site (1 in 8 sampled, every size), every 10 s
// ([allocrate]), for finding the call sites that churn the heap.
bool rateEnabled() {
    static const bool on = std::getenv("APS5_ALLOC_RATE") != nullptr;
    return on;
}

constexpr std::size_t RateSlots = 8192;
constexpr std::uint64_t RateSample = 8;

struct RateSite {
    std::atomic<std::uintptr_t> caller{0};
    std::atomic<std::uintptr_t> parent{0};
    std::atomic<std::uint64_t> count{0};
    std::atomic<std::uint64_t> bytes{0};
};

RateSite rateSites[RateSlots];
std::atomic<std::uint64_t> rateCalls{0};
std::array<std::atomic<std::uint64_t>, 24> rateSizes{};
std::atomic<long long> rateReported{0};

void reportRate() {
    std::vector<std::pair<std::uint64_t, std::size_t>, MallocAllocator<std::pair<std::uint64_t, std::size_t>>> ranked;
    for (std::size_t slot = 0; slot < RateSlots; ++slot) {
        const auto count = rateSites[slot].count.exchange(0, std::memory_order_relaxed);
        if (count != 0) ranked.emplace_back(count, slot);
    }
    std::sort(ranked.rbegin(), ranked.rend());
    std::fprintf(stderr, "[allocrate] %llu operator new calls (10 s); sampled sizes by log2:", static_cast<unsigned long long>(rateCalls.exchange(0, std::memory_order_relaxed)));
    for (std::size_t bucket = 0; bucket < rateSizes.size(); ++bucket) {
        const auto count = rateSizes[bucket].exchange(0, std::memory_order_relaxed);
        if (count != 0) std::fprintf(stderr, " %zu:%llu", bucket, static_cast<unsigned long long>(count * RateSample));
    }
    std::fputc('\n', stderr);
    const auto module = reinterpret_cast<std::uintptr_t>(&__ImageBase);
    for (std::size_t i = 0; i < ranked.size() && i < 40; ++i) {
        auto& site = rateSites[ranked[i].second];
        std::fprintf(stderr, "[allocrate]   x%llu %llu KiB: +0x%llx <- +0x%llx\n", static_cast<unsigned long long>(ranked[i].first * RateSample), static_cast<unsigned long long>(site.bytes.exchange(0, std::memory_order_relaxed) * RateSample >> 10u), static_cast<unsigned long long>(site.caller.load(std::memory_order_relaxed) - 1 - module), static_cast<unsigned long long>(site.parent.load(std::memory_order_relaxed) - module));
    }
    std::fflush(stderr);
}

void noteRate(std::size_t bytes, void* caller) {
    const auto serial = rateCalls.fetch_add(1, std::memory_order_relaxed);
    if (serial % RateSample != 0) return;
    void* frames[5] = {};
    const auto captured = RtlCaptureStackBackTrace(0, 5, frames, nullptr);
    const auto first = reinterpret_cast<std::uintptr_t>(caller);
    std::uintptr_t parent = 0;
    for (USHORT i = 0; i + 1 < captured; ++i) {
        if (frames[i] == caller) {
            parent = reinterpret_cast<std::uintptr_t>(frames[i + 1]);
            break;
        }
    }
    rateSizes[std::min<std::size_t>(rateSizes.size() - 1, bytes == 0 ? 0 : 64 - __builtin_clzll(bytes))].fetch_add(1, std::memory_order_relaxed);
    const auto key = (first * 0x9e3779b97f4a7c15ull) ^ (parent * 0xc2b2ae3d27d4eb4full);
    for (std::size_t probe = 0; probe < 32; ++probe) {
        auto& site = rateSites[(key + probe) % RateSlots];
        auto expected = std::uintptr_t{0};
        if (site.caller.load(std::memory_order_relaxed) == 0 && site.caller.compare_exchange_strong(expected, first + 1, std::memory_order_relaxed)) site.parent.store(parent, std::memory_order_relaxed);
        if (site.caller.load(std::memory_order_relaxed) == first + 1 && site.parent.load(std::memory_order_relaxed) == parent) {
            site.count.fetch_add(1, std::memory_order_relaxed);
            site.bytes.fetch_add(bytes, std::memory_order_relaxed);
            break;
        }
    }
    if (serial % (RateSample * 16384) == 0) {
        const auto now = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
        auto last = rateReported.load(std::memory_order_relaxed);
        if (last == 0) rateReported.compare_exchange_strong(last, now);
        else if (now - last >= 10 && rateReported.compare_exchange_strong(last, now)) reportRate();
    }
}

void* allocate(std::size_t bytes, void* caller) {
    void* pointer = std::malloc(bytes == 0 ? 1 : bytes);
    if (pointer == nullptr) throw std::bad_alloc();
    if (enabled()) noteAllocation(pointer, bytes, caller);
    if (rateEnabled()) noteRate(bytes, caller);
    return pointer;
}

void release(void* pointer) {
    if (pointer == nullptr) return;
    if (enabled()) noteFree(pointer);
    std::free(pointer);
}

// Small-block cache: a block of CacheMinBytes..CacheMaxBytes returned by a sized delete is kept on a
// per-thread free list for its exact size and handed to the next allocation of that size, instead
// of going back to the process heap (whose allocate and free were ~15% of queue 0's samples in the
// PPSA03671 cabin). Every cached block is a whole malloc block of at least its list's size, whoever
// allocated it, so it serves the next allocation and stays valid for another module's free().
// APS5_NO_ALLOC_CACHE=1 turns it off (as does APS5_ALLOC_PROFILE, which tracks every block).
constexpr std::size_t CacheMinBytes = sizeof(void*);
constexpr std::size_t CacheMaxBytes = 1024;
constexpr std::size_t CacheClasses = CacheMaxBytes + 1;
constexpr std::size_t CacheBytesPerClass = 16 * 1024;

struct SmallCache {
    void* heads[CacheClasses];
    std::uint32_t counts[CacheClasses];
};

void NTAPI releaseSmallCache(void* data) {
    auto* cache = static_cast<SmallCache*>(data);
    if (cache == nullptr) return;
    for (std::size_t index = 0; index < CacheClasses; ++index) {
        for (void* block = cache->heads[index]; block != nullptr;) {
            void* next = *static_cast<void**>(block);
            std::free(block);
            block = next;
        }
    }
    std::free(cache);
}

bool cacheEnabled() {
    static const bool on = std::getenv("APS5_NO_ALLOC_CACHE") == nullptr && !enabled();
    return on;
}

DWORD cacheSlot() {
    static const DWORD slot = FlsAlloc(&releaseSmallCache);
    return slot;
}

SmallCache* smallCache() {
    const auto slot = cacheSlot();
    if (slot == FLS_OUT_OF_INDEXES) return nullptr;
    const auto error = GetLastError();
    auto* cache = static_cast<SmallCache*>(FlsGetValue(slot));
    if (cache == nullptr) {
        cache = static_cast<SmallCache*>(std::calloc(1, sizeof(SmallCache)));
        if (cache != nullptr && !FlsSetValue(slot, cache)) {
            std::free(cache);
            cache = nullptr;
        }
    }
    SetLastError(error);
    return cache;
}

void* allocateSmall(std::size_t bytes) {
    if (auto* cache = smallCache(); cache != nullptr && cache->heads[bytes] != nullptr) {
        void* block = cache->heads[bytes];
        cache->heads[bytes] = *static_cast<void**>(block);
        --cache->counts[bytes];
        return block;
    }
    void* pointer = std::malloc(bytes);
    if (pointer == nullptr) throw std::bad_alloc();
    return pointer;
}

void releaseSized(void* pointer, std::size_t bytes) {
    if (pointer == nullptr) return;
    if (bytes >= CacheMinBytes && bytes <= CacheMaxBytes && cacheEnabled()) {
        auto* cache = smallCache();
        if (cache != nullptr && cache->counts[bytes] < std::max<std::size_t>(4, CacheBytesPerClass / bytes)) {
            *static_cast<void**>(pointer) = cache->heads[bytes];
            cache->heads[bytes] = pointer;
            ++cache->counts[bytes];
            return;
        }
    }
    release(pointer);
}

void* allocateAny(std::size_t bytes, void* caller) {
    if (bytes >= CacheMinBytes && bytes <= CacheMaxBytes && cacheEnabled()) {
        if (rateEnabled()) noteRate(bytes, caller);
        return allocateSmall(bytes);
    }
    return allocate(bytes, caller);
}

}

void* operator new(std::size_t bytes) { return allocateAny(bytes, __builtin_return_address(0)); }
void* operator new[](std::size_t bytes) { return allocateAny(bytes, __builtin_return_address(0)); }
void* operator new(std::size_t bytes, const std::nothrow_t&) noexcept {
    try {
        return allocate(bytes, __builtin_return_address(0));
    } catch (...) {
        return nullptr;
    }
}
void* operator new[](std::size_t bytes, const std::nothrow_t&) noexcept {
    try {
        return allocate(bytes, __builtin_return_address(0));
    } catch (...) {
        return nullptr;
    }
}
void operator delete(void* pointer) noexcept { release(pointer); }
void operator delete[](void* pointer) noexcept { release(pointer); }
void operator delete(void* pointer, std::size_t bytes) noexcept { releaseSized(pointer, bytes); }
void operator delete[](void* pointer, std::size_t bytes) noexcept { releaseSized(pointer, bytes); }
void operator delete(void* pointer, const std::nothrow_t&) noexcept { release(pointer); }
void operator delete[](void* pointer, const std::nothrow_t&) noexcept { release(pointer); }

#endif
