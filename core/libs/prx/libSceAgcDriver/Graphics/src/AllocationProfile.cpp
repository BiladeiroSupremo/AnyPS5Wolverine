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

void* allocate(std::size_t bytes, void* caller) {
    void* pointer = std::malloc(bytes == 0 ? 1 : bytes);
    if (pointer == nullptr) throw std::bad_alloc();
    if (enabled()) noteAllocation(pointer, bytes, caller);
    return pointer;
}

void release(void* pointer) {
    if (pointer == nullptr) return;
    if (enabled()) noteFree(pointer);
    std::free(pointer);
}

}

void* operator new(std::size_t bytes) { return allocate(bytes, __builtin_return_address(0)); }
void* operator new[](std::size_t bytes) { return allocate(bytes, __builtin_return_address(0)); }
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
void operator delete(void* pointer, std::size_t) noexcept { release(pointer); }
void operator delete[](void* pointer, std::size_t) noexcept { release(pointer); }
void operator delete(void* pointer, const std::nothrow_t&) noexcept { release(pointer); }
void operator delete[](void* pointer, const std::nothrow_t&) noexcept { release(pointer); }

#endif
