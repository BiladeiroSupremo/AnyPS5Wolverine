#include "prx/libSceAgcDriver/Graphics/include/GuestMirror.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestBufferMemory.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "prx/libc/include/GuestArena.hpp"
#ifdef _WIN32
#include <windows.h>
#endif
#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace AgcDriver::Graphics {

namespace {

// One owner of a mirror page: an import's memory and the guest address of its first byte.
struct Owner {
    VkDeviceMemory memory;
    std::uint64_t importBase;
};

struct MirrorState {
    std::mutex mutex;
    std::unique_ptr<GuestMirror> mirror;
    bool failed = false;
    // Owners per page (guest address of the page), newest last; the newest is bound.
    std::map<std::uint64_t, std::vector<Owner>> pages;
    // Binds waiting for the GPU mutex (GuestMirror::FlushPending): page runs of one owner.
    struct Run {
        std::uint64_t first;
        std::uint64_t end;
        Owner owner;
    };
    std::vector<Run> pending;
    // The mirror's own imports (GuestMirror::Bind) per import memory they complete.
    std::map<VkDeviceMemory, std::vector<VkDeviceMemory>> extras;
    // The imports bound and not yet unbound, by guest base (for the [mirror-coverage] line).
    struct Bound {
        std::uint64_t bytes;
        VkDeviceMemory memory;
        bool alias;
        std::uint32_t runs;
    };
    std::multimap<std::uint64_t, Bound> bound;
    std::atomic<std::uint64_t> binds{0}, unbinds{0}, deferred{0}, unaligned{0}, bindUs{0}, boundBytes{0}, extraImports{0}, extraRefused{0}, edgesSkipped{0}, edgesAliased{0};
    std::atomic<std::int64_t> lastReport{0};
};

MirrorState& Mirrors() {
    static auto* state = new MirrorState;
    return *state;
}

bool profiled() {
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    return profile;
}

// Whether every byte of [address, address + size) is committed memory the driver can import
// (writable: it refuses read-only memory).
bool importableHost(std::uint64_t address, std::uint64_t size) {
#ifdef _WIN32
    // APS5_TRACE_MIRROR_EDGES=1: the first 40 refused pages with the region that refused them.
    static const bool trace = std::getenv("APS5_TRACE_MIRROR_EDGES") != nullptr;
    static std::atomic<int> traced{0};
    for (auto cursor = address; cursor < address + size;) {
        MEMORY_BASIC_INFORMATION info{};
        const bool queried = VirtualQuery(reinterpret_cast<const void*>(cursor), &info, sizeof(info)) != 0;
        if (!queried || info.State != MEM_COMMIT || (info.Protect & (PAGE_GUARD | PAGE_NOACCESS)) != 0 || (info.Protect & (PAGE_READWRITE | PAGE_EXECUTE_READWRITE | PAGE_WRITECOPY | PAGE_EXECUTE_WRITECOPY)) == 0) {
            if (trace && traced.fetch_add(1, std::memory_order_relaxed) < 40) std::fprintf(stderr, "[mirror-edge] page 0x%llx refused at 0x%llx: region 0x%llx+0x%llx state 0x%lx protect 0x%lx\n", static_cast<unsigned long long>(address), static_cast<unsigned long long>(cursor), queried ? static_cast<unsigned long long>(reinterpret_cast<std::uint64_t>(info.BaseAddress)) : 0ull, queried ? static_cast<unsigned long long>(info.RegionSize) : 0ull, queried ? info.State : 0ul, queried ? info.Protect : 0ul);
            return false;
        }
        cursor = reinterpret_cast<std::uint64_t>(info.BaseAddress) + info.RegionSize;
    }
    return true;
#else
    static_cast<void>(address);
    static_cast<void>(size);
    return false;
#endif
}

}

GuestMirror::GuestMirror(const Context& context) : context(context) {}

GuestMirror* GuestMirror::Instance() {
    auto& state = Mirrors();
    std::lock_guard lock(state.mutex);
    return state.mirror.get();
}

void GuestMirror::FlushAll() {
    if (auto* mirror = Instance()) {
        mirror->FlushPending();
        mirror->Report();
    }
}

GuestMirror* GuestMirror::Get(const Context& context) {
    if (!context.guestMirror) return nullptr;
    auto& state = Mirrors();
    std::lock_guard lock(state.mutex);
    if (state.mirror != nullptr || state.failed) return state.mirror.get();
    std::uintptr_t arena = 0;
    std::size_t arenaBytes = 0;
    GuestArena::GuestArenaRange_nid_postfix(&arena, &arenaBytes);
    auto mirror = std::unique_ptr<GuestMirror>(new GuestMirror(context));
    const VkExternalMemoryBufferCreateInfo external{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO, nullptr, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT};
    VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, &external};
    info.flags = VK_BUFFER_CREATE_SPARSE_BINDING_BIT | VK_BUFFER_CREATE_SPARSE_RESIDENCY_BIT;
    info.size = arenaBytes;
    info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    const auto fail = [&](const char* step, VkResult result) {
        std::fprintf(stderr, "[mirror] guest mirror of 0x%llx+0x%llx failed at %s (%d)\n", static_cast<unsigned long long>(arena), static_cast<unsigned long long>(arenaBytes), step, static_cast<int>(result));
        if (mirror->buffer != VK_NULL_HANDLE) context.Function<PFN_vkDestroyBuffer>("vkDestroyBuffer")(context.device, mirror->buffer, nullptr);
        state.failed = true;
        return nullptr;
    };
    if (const auto result = context.Function<PFN_vkCreateBuffer>("vkCreateBuffer")(context.device, &info, nullptr, &mirror->buffer); result != VK_SUCCESS) return fail("vkCreateBuffer", result);
    VkMemoryRequirements requirements{};
    context.Function<PFN_vkGetBufferMemoryRequirements>("vkGetBufferMemoryRequirements")(context.device, mirror->buffer, &requirements);
    const VkBufferDeviceAddressInfo addressInfo{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, nullptr, mirror->buffer};
    mirror->address = context.Function<PFN_vkGetBufferDeviceAddressKHR>("vkGetBufferDeviceAddressKHR")(context.device, &addressInfo);
    const VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    if (const auto result = context.Function<PFN_vkCreateFence>("vkCreateFence")(context.device, &fenceInfo, nullptr, &mirror->fence); result != VK_SUCCESS) return fail("vkCreateFence", result);
    mirror->base = arena;
    mirror->bytes = arenaBytes;
    mirror->page = requirements.alignment;
    mirror->memoryTypes = requirements.memoryTypeBits;
    std::fprintf(stderr, "[mirror] guest mirror of 0x%llx+0x%llx at device address 0x%llx, page %llu KiB, memory types 0x%x\n", static_cast<unsigned long long>(arena), static_cast<unsigned long long>(arenaBytes), static_cast<unsigned long long>(mirror->address), static_cast<unsigned long long>(mirror->page >> 10u), requirements.memoryTypeBits);
    state.mirror = std::move(mirror);
    return state.mirror.get();
}

bool GuestMirror::submit(std::uint64_t offset, std::uint64_t size, VkDeviceMemory memory, std::uint64_t memoryOffset) {
    const auto started = std::chrono::steady_clock::now();
    const VkSparseMemoryBind bind{offset, size, memory, memoryOffset, 0};
    const VkSparseBufferMemoryBindInfo bufferBind{buffer, 1, &bind};
    VkBindSparseInfo info{VK_STRUCTURE_TYPE_BIND_SPARSE_INFO};
    info.bufferBindCount = 1;
    info.pBufferBinds = &bufferBind;
    // Sparse binds are not ordered with the queue's command batches: the bind completes here before
    // any later batch may use the pages, and an unbind before the memory is freed.
    context.Function<PFN_vkResetFences>("vkResetFences")(context.device, 1, &fence);
    auto result = context.Function<PFN_vkQueueBindSparse>("vkQueueBindSparse")(context.queue, 1, &info, fence);
    if (result == VK_SUCCESS) result = context.Function<PFN_vkWaitForFences>("vkWaitForFences")(context.device, 1, &fence, VK_TRUE, UINT64_MAX);
    Mirrors().bindUs.fetch_add(static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - started).count()), std::memory_order_relaxed);
    if (result != VK_SUCCESS) std::fprintf(stderr, "[mirror] vkQueueBindSparse of +0x%llx+0x%llx failed (%d)\n", static_cast<unsigned long long>(offset), static_cast<unsigned long long>(size), static_cast<int>(result));
    return result == VK_SUCCESS;
}

VkDeviceMemory GuestMirror::importHost(void* host, std::uint64_t size) {
    VkMemoryHostPointerPropertiesEXT pointer{VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT};
    if (context.Function<PFN_vkGetMemoryHostPointerPropertiesEXT>("vkGetMemoryHostPointerPropertiesEXT")(context.device, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, host, &pointer) != VK_SUCCESS) return VK_NULL_HANDLE;
    const auto types = pointer.memoryTypeBits & memoryTypes;
    if (types == 0) return VK_NULL_HANDLE;
    const VkImportMemoryHostPointerInfoEXT import{VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT, nullptr, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, host};
    const VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO, &import, VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT, 0};
    VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &flags};
    allocation.allocationSize = size;
    allocation.memoryTypeIndex = static_cast<std::uint32_t>(std::countr_zero(types));
    VkDeviceMemory memory = VK_NULL_HANDLE;
    if (context.Function<PFN_vkAllocateMemory>("vkAllocateMemory")(context.device, &allocation, nullptr, &memory) != VK_SUCCESS) return VK_NULL_HANDLE;
    return memory;
}

void GuestMirror::Bind(const HostImport& import) {
    auto& state = Mirrors();
    if (import.base < base || import.base + import.bytes > base + bytes) return;
    const auto start = import.base;
    const auto end = import.base + import.bytes;
    const auto down = [&](std::uint64_t value) { return value / page * page; };
    const auto up = [&](std::uint64_t value) { return (value + page - 1) / page * page; };
    // The guest page's host address (an aliased import maps another host view of the same memory).
    const auto host = [&](std::uint64_t guest) { return import.alias != nullptr ? static_cast<void*>(static_cast<std::byte*>(import.alias) + (guest - start)) : reinterpret_cast<void*>(guest); };
    std::vector<MirrorState::Run> runs;
    std::vector<VkDeviceMemory> extras;
    const auto own = [&](std::uint64_t first, std::uint64_t last) {
        const auto memory = importHost(host(first), last - first);
        if (memory == VK_NULL_HANDLE) {
            state.extraRefused.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        state.extraImports.fetch_add(1, std::memory_order_relaxed);
        extras.push_back(memory);
        runs.push_back({first, last, {memory, first}});
    };
    if (start % page == 0) {
        if (down(end) > start) runs.push_back({start, down(end), {import.memory, start}});
    } else {
        state.unaligned.fetch_add(1, std::memory_order_relaxed);
        if (down(end) > up(start)) own(up(start), down(end));
    }
    // Partial pages at either end: the whole page imported when the host memory around the import
    // allows it (not for an aliased import, whose view ends with it).
    const auto edge = [&](std::uint64_t first) {
        if (import.alias != nullptr || first < base || first + page > base + bytes || !importableHost(first, page)) {
            state.edgesSkipped.fetch_add(1, std::memory_order_relaxed);
            if (import.alias != nullptr) state.edgesAliased.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        own(first, first + page);
    };
    if (start % page != 0) edge(down(start));
    if (end % page != 0 && (start % page == 0 || down(end) != down(start))) edge(down(end));
    {
        std::lock_guard lock(state.mutex);
        state.bound.emplace(start, MirrorState::Bound{import.bytes, import.memory, import.alias != nullptr, static_cast<std::uint32_t>(runs.size())});
    }
    if (runs.empty()) return;
    {
        std::lock_guard lock(state.mutex);
        for (const auto& run : runs) {
            for (auto at = run.first; at < run.end; at += page) state.pages[at].push_back(run.owner);
            state.pending.push_back(run);
        }
        if (!extras.empty()) {
            auto& owned = state.extras[import.memory];
            owned.insert(owned.end(), extras.begin(), extras.end());
        }
    }
    state.deferred.fetch_add(1, std::memory_order_relaxed);
    // Bound now when the GPU mutex is free; otherwise at the next present (FlushPending).
    std::unique_lock gpu(GuestMemory::GpuMutex(), std::try_to_lock);
    if (gpu.owns_lock()) FlushPending();
}

void GuestMirror::FlushPending() {
    auto& state = Mirrors();
    std::vector<MirrorState::Run> runs;
    {
        std::lock_guard lock(state.mutex);
        std::vector<MirrorState::Run> queued;
        queued.swap(state.pending);
        // Only the pages that still show the run's owner (a newer import of some of them, or the
        // owner's unbind, took the others): bound as maximal stretches.
        for (const auto& run : queued) {
            auto stretch = run.first;
            for (auto at = run.first; at <= run.end; at += page) {
                bool shown = false;
                if (at < run.end) {
                    const auto found = state.pages.find(at);
                    shown = found != state.pages.end() && !found->second.empty() && found->second.back().memory == run.owner.memory;
                }
                if (shown) continue;
                if (at > stretch) runs.push_back({stretch, at, run.owner});
                stretch = at + page;
            }
        }
    }
    for (const auto& run : runs) {
        if (submit(run.first - base, run.end - run.first, run.owner.memory, run.first - run.owner.importBase)) {
            state.binds.fetch_add(1, std::memory_order_relaxed);
            state.boundBytes.fetch_add(run.end - run.first, std::memory_order_relaxed);
        }
    }
}

void GuestMirror::Unbind(const HostImport& import) {
    auto& state = Mirrors();
    if (import.base < base || import.base + import.bytes > base + bytes) return;
    // Every page the import or its own extras may show: the end pages included.
    const auto first = import.base / page * page;
    const auto end = (import.base + import.bytes + page - 1) / page * page;
    std::lock_guard gpu(GuestMemory::GpuMutex());
    std::vector<std::pair<std::uint64_t, Owner>> rebinds;
    std::vector<VkDeviceMemory> memories{import.memory};
    {
        std::lock_guard lock(state.mutex);
        if (const auto owned = state.extras.find(import.memory); owned != state.extras.end()) {
            memories.insert(memories.end(), owned->second.begin(), owned->second.end());
            state.extras.erase(owned);
        }
        for (auto [at, last] = state.bound.equal_range(import.base); at != last; ++at) {
            if (at->second.memory != import.memory) continue;
            state.bound.erase(at);
            break;
        }
        const auto gone = [&](VkDeviceMemory memory) { return std::find(memories.begin(), memories.end(), memory) != memories.end(); };
        std::erase_if(state.pending, [&](const MirrorState::Run& run) { return gone(run.owner.memory); });
        for (auto at = first; at < end; at += page) {
            const auto found = state.pages.find(at);
            if (found == state.pages.end()) continue;
            auto& owners = found->second;
            const bool shown = !owners.empty() && gone(owners.back().memory);
            std::erase_if(owners, [&](const Owner& owner) { return gone(owner.memory); });
            if (shown) rebinds.emplace_back(at, owners.empty() ? Owner{VK_NULL_HANDLE, 0} : owners.back());
            if (owners.empty()) state.pages.erase(found);
        }
    }
    for (const auto& [at, owner] : rebinds) {
        submit(at - base, page, owner.memory, owner.memory == VK_NULL_HANDLE ? 0 : at - owner.importBase);
        if (owner.memory == VK_NULL_HANDLE) state.boundBytes.fetch_sub(page, std::memory_order_relaxed);
    }
    // The mirror's own imports are freed once no page shows them (the submits above waited).
    for (std::size_t index = 1; index < memories.size(); ++index) context.Function<PFN_vkFreeMemory>("vkFreeMemory")(context.device, memories[index], nullptr);
    state.unbinds.fetch_add(1, std::memory_order_relaxed);
}

void GuestMirror::Report() {
    if (!profiled()) return;
    auto& state = Mirrors();
    const auto now = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    auto last = state.lastReport.load(std::memory_order_relaxed);
    if (now - last < 10000 || !state.lastReport.compare_exchange_strong(last, now)) return;
    const auto take = [](std::atomic<std::uint64_t>& value) { return static_cast<unsigned long long>(value.exchange(0, std::memory_order_relaxed)); };
    const auto binds = take(state.binds);
    const auto unbinds = take(state.unbinds);
    const auto deferred = take(state.deferred);
    const auto unaligned = take(state.unaligned);
    const auto bindUs = take(state.bindUs);
    const auto extraImports = take(state.extraImports);
    const auto extraRefused = take(state.extraRefused);
    const auto edgesSkipped = take(state.edgesSkipped);
    const auto edgesAliased = take(state.edgesAliased);
    std::size_t pages = 0;
    std::size_t extrasHeld = 0;
    {
        std::lock_guard lock(state.mutex);
        pages = state.pages.size();
        for (const auto& [memory, owned] : state.extras) extrasHeld += owned.size();
    }
    AgcDriver::ProfilePrint_nid_no_patch("[mirror] (10 s) %llu imports bound (%llu requested), %llu unbound, %llu not page-aligned; own imports %llu made, %llu refused, %llu end pages skipped (%llu aliased), %zu held; %.1f ms in sparse binds; %zu pages shown (%.0f MiB bound)\n", binds, deferred, unbinds, unaligned, extraImports, extraRefused, edgesSkipped, edgesAliased, extrasHeld, bindUs / 1000.0, pages, state.boundBytes.load() / 1048576.0);
    // Coverage: the registered GPU ranges' pages the mirror does not show (their reads return zero).
    struct Gap {
        std::uint64_t address;
        std::uint64_t bytes;
        std::uint64_t missing;
        std::uint64_t pages;
    };
    std::vector<Gap> gaps;
    std::uint64_t registered = 0;
    std::uint64_t missing = 0;
    std::uint64_t missingReadOnly = 0;
    {
        const auto lease = GuestAllocations::GuestAllocationsAcquire_nid_postfix();
        std::lock_guard lock(state.mutex);
        for (const auto& range : lease) {
            if (!range->readable || !range->gpu || range->address < base || range->address + range->bytes > base + bytes) continue;
            registered += range->bytes;
            Gap gap{range->address, range->bytes, 0, 0};
            for (auto at = range->address / page * page; at < range->address + range->bytes; at += page) {
                ++gap.pages;
                if (!state.pages.contains(at)) ++gap.missing;
            }
            missing += gap.missing;
            if (!range->writable) missingReadOnly += gap.missing;
            if (gap.missing != 0) gaps.push_back(gap);
        }
    }
    std::sort(gaps.begin(), gaps.end(), [](const Gap& left, const Gap& right) { return left.missing > right.missing; });
    std::string largest;
    {
        std::lock_guard lock(state.mutex);
        for (std::size_t index = 0; index < gaps.size() && index < 8; ++index) {
            // The bound imports covering the gap's first byte (none: no import was ever bound there).
            std::uint32_t covering = 0;
            const MirrorState::Bound* first = nullptr;
            std::uint64_t firstBase = 0;
            for (const auto& [at, entry] : state.bound) {
                if (at > gaps[index].address) break;
                if (at + entry.bytes <= gaps[index].address) continue;
                ++covering;
                first = &entry;
                firstBase = at;
            }
            char part[192];
            if (first == nullptr) std::snprintf(part, sizeof(part), " 0x%llx+0x%llx (%llu/%llu, no bound import)", static_cast<unsigned long long>(gaps[index].address), static_cast<unsigned long long>(gaps[index].bytes), static_cast<unsigned long long>(gaps[index].missing), static_cast<unsigned long long>(gaps[index].pages));
            else std::snprintf(part, sizeof(part), " 0x%llx+0x%llx (%llu/%llu, %u bound, last 0x%llx+0x%llx alias %d runs %u)", static_cast<unsigned long long>(gaps[index].address), static_cast<unsigned long long>(gaps[index].bytes), static_cast<unsigned long long>(gaps[index].missing), static_cast<unsigned long long>(gaps[index].pages), covering, static_cast<unsigned long long>(firstBase), static_cast<unsigned long long>(first->bytes), first->alias ? 1 : 0, first->runs);
            largest += part;
        }
    }
    AgcDriver::ProfilePrint_nid_no_patch("[mirror-coverage] registered GPU ranges %.0f MiB, pages not shown %llu (%.0f MiB, %llu of them read-only) in %zu ranges; largest:%s\n", registered / 1048576.0, static_cast<unsigned long long>(missing), static_cast<double>(missing * page) / 1048576.0, static_cast<unsigned long long>(missingReadOnly), gaps.size(), largest.c_str());
}

}
