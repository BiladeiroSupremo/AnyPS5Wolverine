#include "prx/libSceAgcDriver/Graphics/include/GuestMirror.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestBufferMemory.hpp"
#include "prx/libc/include/GuestArena.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
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
    std::atomic<std::uint64_t> binds{0}, unbinds{0}, deferred{0}, unaligned{0}, bindUs{0}, boundBytes{0};
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

void GuestMirror::Bind(const HostImport& import) {
    auto& state = Mirrors();
    if (import.base < base || import.base + import.bytes > base + bytes) return;
    if (import.base % page != 0) {
        state.unaligned.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    const auto end = (import.base + import.bytes) / page * page;
    if (end <= import.base) return;
    {
        std::lock_guard lock(state.mutex);
        for (auto at = import.base; at < end; at += page) state.pages[at].push_back({import.memory, import.base});
        state.pending.push_back({import.base, end, {import.memory, import.base}});
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
        runs.swap(state.pending);
        // A run whose owner was unbound in the meantime has no pages showing it any more.
        std::erase_if(runs, [&](const MirrorState::Run& run) {
            const auto found = state.pages.find(run.first);
            return found == state.pages.end() || found->second.empty() || found->second.back().memory != run.owner.memory;
        });
    }
    for (const auto& run : runs) {
        if (submit(run.first - base, run.end - run.first, run.owner.memory, 0)) {
            state.binds.fetch_add(1, std::memory_order_relaxed);
            state.boundBytes.fetch_add(run.end - run.first, std::memory_order_relaxed);
        }
    }
}

void GuestMirror::Unbind(const HostImport& import) {
    auto& state = Mirrors();
    if (import.base < base || import.base + import.bytes > base + bytes || import.base % page != 0) return;
    const auto end = (import.base + import.bytes) / page * page;
    if (end <= import.base) return;
    std::lock_guard gpu(GuestMemory::GpuMutex());
    std::vector<std::pair<std::uint64_t, Owner>> rebinds;
    {
        std::lock_guard lock(state.mutex);
        std::erase_if(state.pending, [&](const MirrorState::Run& run) { return run.owner.memory == import.memory; });
        for (auto at = import.base; at < end; at += page) {
            const auto found = state.pages.find(at);
            if (found == state.pages.end()) continue;
            auto& owners = found->second;
            const bool shown = !owners.empty() && owners.back().memory == import.memory;
            std::erase_if(owners, [&](const Owner& owner) { return owner.memory == import.memory; });
            if (shown) rebinds.emplace_back(at, owners.empty() ? Owner{VK_NULL_HANDLE, 0} : owners.back());
            if (owners.empty()) state.pages.erase(found);
        }
    }
    for (const auto& [at, owner] : rebinds) {
        submit(at - base, page, owner.memory, owner.memory == VK_NULL_HANDLE ? 0 : at - owner.importBase);
        if (owner.memory == VK_NULL_HANDLE) state.boundBytes.fetch_sub(page, std::memory_order_relaxed);
    }
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
    std::size_t pages = 0;
    {
        std::lock_guard lock(state.mutex);
        pages = state.pages.size();
    }
    AgcDriver::ProfilePrint_nid_no_patch("[mirror] (10 s) %llu imports bound (%llu requested), %llu unbound, %llu not page-aligned; %.1f ms in sparse binds; %zu pages shown (%.0f MiB bound)\n", binds, deferred, unbinds, unaligned, bindUs / 1000.0, pages, state.boundBytes.load() / 1048576.0);
}

}
