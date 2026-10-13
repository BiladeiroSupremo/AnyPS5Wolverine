#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/BufferPool.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Breadcrumbs.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Execution/include/PerformanceTimer.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <mutex>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
extern "C" __declspec(dllimport) int __stdcall GetModuleHandleExW(unsigned long flags, const wchar_t* name, void** module);
extern "C" __declspec(dllimport) unsigned short __stdcall RtlCaptureStackBackTrace(unsigned long skip, unsigned long count, void** frames, unsigned long* hash);
#endif

namespace AgcDriver::Graphics {

namespace {

VkDeviceSize ClassAllocationBytes(const VkMemoryRequirements& requirements, std::size_t capacity) {
    const auto alignment = std::max<VkDeviceSize>(requirements.alignment, 1);
    return std::max<VkDeviceSize>(requirements.size, (static_cast<VkDeviceSize>(capacity) + alignment - 1) / alignment * alignment);
}

VkBuffer BindExactBuffer(const Context& context, std::size_t size, VkBufferUsageFlags usage, VkDeviceMemory memory, VkDeviceSize offset, VkDeviceSize availableBytes) {
    VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    info.size = size;
    info.usage = usage;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkBuffer buffer = VK_NULL_HANDLE;
    Check(context.Function<PFN_vkCreateBuffer>("vkCreateBuffer")(context.device, &info, nullptr, &buffer), "vkCreateBuffer pooled");
    try {
        VkMemoryRequirements requirements{};
        context.Function<PFN_vkGetBufferMemoryRequirements>("vkGetBufferMemoryRequirements")(context.device, buffer, &requirements);
        Require(requirements.size <= availableBytes, "pooled buffer memory is smaller than the requested buffer");
        Require(requirements.alignment == 0 || offset % requirements.alignment == 0, "pooled buffer offset breaks the requested buffer's alignment");
        Check(context.Function<PFN_vkBindBufferMemory>("vkBindBufferMemory")(context.device, buffer, memory, offset), "vkBindBufferMemory pooled");
    } catch (...) {
        context.Function<PFN_vkDestroyBuffer>("vkDestroyBuffer")(context.device, buffer, nullptr);
        throw;
    }
    return buffer;
}

struct MemoryTrace {
    struct Allocation {
        VkDeviceSize bytes;
        std::uint32_t type;
        std::uintptr_t site;
    };
    struct Site {
        VkDeviceSize bytes = 0;
        std::uint64_t count = 0;
        std::uint64_t made = 0;
    };
    std::mutex mutex;
    PFN_vkAllocateMemory allocate = nullptr;
    PFN_vkFreeMemory free = nullptr;
    std::unordered_map<VkDeviceMemory, Allocation> live;
    std::unordered_map<std::uintptr_t, Site> sites;
    std::array<VkDeviceSize, VK_MAX_MEMORY_TYPES> typeBytes{};
    std::chrono::steady_clock::time_point lastReport = std::chrono::steady_clock::now();
};

MemoryTrace& Trace() {
    static MemoryTrace trace;
    return trace;
}

// The caller's offset in its module, for nm/addr2line on the module.
std::uintptr_t SiteOf(void* caller) {
#ifdef _WIN32
    void* module = nullptr;
    constexpr unsigned long FromAddress = 0x4, UnchangedRefcount = 0x2;
    if (GetModuleHandleExW(FromAddress | UnchangedRefcount, static_cast<const wchar_t*>(caller), &module) && module) return reinterpret_cast<std::uintptr_t>(caller) - reinterpret_cast<std::uintptr_t>(module);
#endif
    return reinterpret_cast<std::uintptr_t>(caller);
}

void Report(MemoryTrace& trace, const char* why) {
    VkDeviceSize total = 0;
    for (const auto bytes : trace.typeBytes) total += bytes;
    std::fprintf(stderr, "[vkmem] %s: %zu live allocations, %.1f MiB;", why, trace.live.size(), total / 1048576.0);
    for (std::size_t type = 0; type < trace.typeBytes.size(); ++type) {
        if (trace.typeBytes[type] != 0) std::fprintf(stderr, " type %zu %.1f MiB", type, trace.typeBytes[type] / 1048576.0);
    }
    std::fputc('\n', stderr);
    std::vector<std::pair<std::uintptr_t, MemoryTrace::Site>> ranked(trace.sites.begin(), trace.sites.end());
    std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) { return a.second.bytes > b.second.bytes; });
    for (std::size_t index = 0; index < ranked.size() && index < 10; ++index) {
        const auto& [site, entry] = ranked[index];
        std::fprintf(stderr, "[vkmem]   site +0x%llx: %.1f MiB in %llu live (%llu made)\n", static_cast<unsigned long long>(site), entry.bytes / 1048576.0, static_cast<unsigned long long>(entry.count), static_cast<unsigned long long>(entry.made));
    }
}

VKAPI_ATTR VkResult VKAPI_CALL TracedAllocate(VkDevice device, const VkMemoryAllocateInfo* info, const VkAllocationCallbacks* callbacks, VkDeviceMemory* memory) {
    auto& trace = Trace();
    const auto site = SiteOf(__builtin_return_address(0));
#ifdef _WIN32
    // Allocations of a GiB and more: the call stack, as module offsets.
    if (info->allocationSize >= (VkDeviceSize{1} << 30u)) {
        void* frames[16] = {};
        const auto count = RtlCaptureStackBackTrace(0, 16, frames, nullptr);
        std::fprintf(stderr, "[vkmem] %.1f MiB allocation (type %u) from", info->allocationSize / 1048576.0, info->memoryTypeIndex);
        for (unsigned index = 0; index < count; ++index) std::fprintf(stderr, " +0x%llx", static_cast<unsigned long long>(SiteOf(frames[index])));
        std::fputc('\n', stderr);
    }
#endif
    const auto result = trace.allocate(device, info, callbacks, memory);
    std::lock_guard lock(trace.mutex);
    if (result != VK_SUCCESS) {
        char why[96];
        std::snprintf(why, sizeof(why), "allocation of %.1f MiB (type %u) at +0x%llx failed with %d", info->allocationSize / 1048576.0, info->memoryTypeIndex, static_cast<unsigned long long>(site), static_cast<int>(result));
        Report(trace, why);
        return result;
    }
    trace.live[*memory] = {info->allocationSize, info->memoryTypeIndex, site};
    auto& entry = trace.sites[site];
    entry.bytes += info->allocationSize;
    ++entry.count;
    ++entry.made;
    if (info->memoryTypeIndex < trace.typeBytes.size()) trace.typeBytes[info->memoryTypeIndex] += info->allocationSize;
    const auto now = std::chrono::steady_clock::now();
    if (now - trace.lastReport > std::chrono::seconds(10)) {
        trace.lastReport = now;
        Report(trace, "10 s");
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL TracedFree(VkDevice device, VkDeviceMemory memory, const VkAllocationCallbacks* callbacks) {
    auto& trace = Trace();
    {
        std::lock_guard lock(trace.mutex);
        if (const auto found = trace.live.find(memory); found != trace.live.end()) {
            auto& entry = trace.sites[found->second.site];
            entry.bytes -= found->second.bytes;
            --entry.count;
            if (found->second.type < trace.typeBytes.size()) trace.typeBytes[found->second.type] -= found->second.bytes;
            trace.live.erase(found);
        }
    }
    trace.free(device, memory, callbacks);
}

}

namespace {

struct DeviceLostReporter {
    std::mutex mutex;
    VkDevice device = VK_NULL_HANDLE;
    PFN_vkGetDeviceFaultInfoEXT getFaultInfo = nullptr;
    bool reported = false;
};

DeviceLostReporter& LostReporter() {
    static DeviceLostReporter reporter;
    return reporter;
}

const char* FaultAddressTypeName(VkDeviceFaultAddressTypeEXT type) {
    switch (type) {
        case VK_DEVICE_FAULT_ADDRESS_TYPE_NONE_EXT: return "none";
        case VK_DEVICE_FAULT_ADDRESS_TYPE_READ_INVALID_EXT: return "invalid read";
        case VK_DEVICE_FAULT_ADDRESS_TYPE_WRITE_INVALID_EXT: return "invalid write";
        case VK_DEVICE_FAULT_ADDRESS_TYPE_EXECUTE_INVALID_EXT: return "invalid execute";
        case VK_DEVICE_FAULT_ADDRESS_TYPE_INSTRUCTION_POINTER_UNKNOWN_EXT: return "instruction pointer (unknown)";
        case VK_DEVICE_FAULT_ADDRESS_TYPE_INSTRUCTION_POINTER_INVALID_EXT: return "instruction pointer (invalid)";
        case VK_DEVICE_FAULT_ADDRESS_TYPE_INSTRUCTION_POINTER_FAULT_EXT: return "instruction pointer (fault)";
        default: return "unknown";
    }
}

}

void SetDeviceLostReporter(VkDevice device, PFN_vkGetDeviceFaultInfoEXT getFaultInfo) {
    auto& reporter = LostReporter();
    std::lock_guard lock(reporter.mutex);
    reporter.device = device;
    reporter.getFaultInfo = getFaultInfo;
    reporter.reported = false;
}

void ReportDeviceLost() {
    auto& reporter = LostReporter();
    std::lock_guard lock(reporter.mutex);
    if (reporter.reported) return;
    reporter.reported = true;
    Breadcrumbs::Report();
    if (reporter.getFaultInfo == nullptr) {
        std::fprintf(stderr, "[gpu] device lost (no VK_EXT_device_fault report)\n");
        return;
    }
    VkDeviceFaultCountsEXT counts{VK_STRUCTURE_TYPE_DEVICE_FAULT_COUNTS_EXT};
    if (const auto result = reporter.getFaultInfo(reporter.device, &counts, nullptr); result != VK_SUCCESS && result != VK_INCOMPLETE) {
        std::fprintf(stderr, "[gpu] device lost; vkGetDeviceFaultInfoEXT counts failed with %d\n", static_cast<int>(result));
        return;
    }
    std::vector<VkDeviceFaultAddressInfoEXT> addresses(counts.addressInfoCount);
    std::vector<VkDeviceFaultVendorInfoEXT> vendors(counts.vendorInfoCount);
    VkDeviceFaultInfoEXT info{VK_STRUCTURE_TYPE_DEVICE_FAULT_INFO_EXT};
    info.pAddressInfos = addresses.empty() ? nullptr : addresses.data();
    info.pVendorInfos = vendors.empty() ? nullptr : vendors.data();
    counts.vendorBinarySize = 0;
    if (const auto result = reporter.getFaultInfo(reporter.device, &counts, &info); result != VK_SUCCESS && result != VK_INCOMPLETE) {
        std::fprintf(stderr, "[gpu] device lost; vkGetDeviceFaultInfoEXT failed with %d\n", static_cast<int>(result));
        return;
    }
    std::fprintf(stderr, "[gpu] device lost: %s (%u addresses, %u vendor records)\n", info.description, counts.addressInfoCount, counts.vendorInfoCount);
    for (std::uint32_t index = 0; index < counts.addressInfoCount; ++index) {
        const auto& address = addresses[index];
        std::fprintf(stderr, "[gpu]   %s at 0x%llx (precision 0x%llx)\n", FaultAddressTypeName(address.addressType), static_cast<unsigned long long>(address.reportedAddress), static_cast<unsigned long long>(address.addressPrecision));
    }
    for (std::uint32_t index = 0; index < counts.vendorInfoCount; ++index) {
        const auto& vendor = vendors[index];
        std::fprintf(stderr, "[gpu]   vendor: %s code 0x%llx data 0x%llx\n", vendor.description, static_cast<unsigned long long>(vendor.vendorFaultCode), static_cast<unsigned long long>(vendor.vendorFaultData));
    }
}

PFN_vkVoidFunction TracedMemoryFunction(const char* name, PFN_vkVoidFunction real) {
    static const bool enabled = std::getenv("APS5_TRACE_VKMEM") != nullptr;
    if (!enabled) return nullptr;
    auto& trace = Trace();
    if (std::strcmp(name, "vkAllocateMemory") == 0) {
        std::lock_guard lock(trace.mutex);
        trace.allocate = reinterpret_cast<PFN_vkAllocateMemory>(real);
        return reinterpret_cast<PFN_vkVoidFunction>(&TracedAllocate);
    }
    if (std::strcmp(name, "vkFreeMemory") == 0) {
        std::lock_guard lock(trace.mutex);
        trace.free = reinterpret_cast<PFN_vkFreeMemory>(real);
        return reinterpret_cast<PFN_vkVoidFunction>(&TracedFree);
    }
    return nullptr;
}

Buffer::Buffer(const Context& context, std::size_t size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties) : context(context), size(size), capacity(BufferPool::Capacity(size, properties)), usage(BufferPool::Usage(usage, properties)), properties(properties) {
    Require(size != 0, "zero-sized GPU buffer");
    const bool addressable = (usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) != 0;
    Require(!addressable || context.bufferDeviceAddress, "buffer device address is not enabled");
    static const bool traceMemory = std::getenv("APS5_TRACE_VKMEM") != nullptr;
    if (traceMemory && size >= (std::size_t{1} << 30u)) std::fprintf(stderr, "[vkmem] Buffer of %.1f MiB made from +0x%llx\n", size / 1048576.0, static_cast<unsigned long long>(SiteOf(__builtin_return_address(0))));
    cache = GetBufferPool(context);
    if (const auto allocation = cache->Take(size, this->usage, properties)) {
        buffer = allocation->buffer;
        memory = allocation->memory;
        mapping = allocation->mapping;
        deviceAddress = allocation->address;
        allocationBytes = allocation->allocationBytes;
        capacity = allocation->bytes;
        offset = allocation->offset;
        slab = allocation->slab;
        if (allocation->bufferBytes != size) {
            context.Function<PFN_vkDestroyBuffer>("vkDestroyBuffer")(context.device, buffer, nullptr);
            buffer = VK_NULL_HANDLE;
            deviceAddress = 0;
            try {
                buffer = BindExactBuffer(context, size, this->usage, memory, offset, allocationBytes);
                initializeAddress(usage);
            } catch (...) {
                release();
                throw;
            }
        }
        ready = true;
        return;
    }
    try {
        VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        info.size = size;
        info.usage = this->usage;
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        Check(context.Function<PFN_vkCreateBuffer>("vkCreateBuffer")(context.device, &info, nullptr, &buffer), "vkCreateBuffer");
        VkMemoryRequirements requirements{};
        context.Function<PFN_vkGetBufferMemoryRequirements>("vkGetBufferMemoryRequirements")(context.device, buffer, &requirements);
        VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        const VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO, nullptr, VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT, 0};
        if (addressable) allocation.pNext = &flags;
        allocation.allocationSize = ClassAllocationBytes(requirements, capacity);
        allocationBytes = allocation.allocationSize;
        // The CPU reads most of these buffers back (write-back, diffs), which is very slow from
        // write-combined memory, so the default host properties prefer cached host memory.
        constexpr VkMemoryPropertyFlags hostDefault = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        if (properties == hostDefault) {
            try {
                allocation.memoryTypeIndex = context.MemoryType(requirements.memoryTypeBits, hostDefault | VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
            } catch (const std::runtime_error&) {
                allocation.memoryTypeIndex = context.MemoryType(requirements.memoryTypeBits, hostDefault);
            }
        } else {
            allocation.memoryTypeIndex = context.MemoryType(requirements.memoryTypeBits, properties);
        }
        if ((properties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0 && BufferPool::SlabEligible(capacity, requirements.alignment, requirements.size, context.limits.nonCoherentAtomSize)) {
            const auto slot = cache->TakeSlot(context, allocation.memoryTypeIndex, capacity, addressable);
            memory = slot.memory;
            offset = slot.offset;
            slab = true;
            allocationBytes = capacity;
            Check(context.Function<PFN_vkBindBufferMemory>("vkBindBufferMemory")(context.device, buffer, memory, offset), "vkBindBufferMemory slab");
            initializeAddress(usage);
            mapping = slot.mapping;
            ready = true;
            return;
        }
        memory = cache->AllocateMemory(allocation, "vkAllocateMemory buffer");
        Check(context.Function<PFN_vkBindBufferMemory>("vkBindBufferMemory")(context.device, buffer, memory, 0), "vkBindBufferMemory");
        initializeAddress(usage);
        if ((properties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0) Check(context.Function<PFN_vkMapMemory>("vkMapMemory")(context.device, memory, 0, VK_WHOLE_SIZE, 0, &mapping), "vkMapMemory");
        ready = true;
    } catch (...) {
        release();
        throw;
    }
}

Buffer::~Buffer() {
    release();
}

void Buffer::release() noexcept {
    if (ready && cache) {
        cache->Put({buffer, memory, mapping, deviceAddress, allocationBytes, capacity, usage, properties, offset, slab, size});
        return;
    }
    if (slab) {
        if (buffer) context.Function<PFN_vkDestroyBuffer>("vkDestroyBuffer")(context.device, buffer, nullptr);
        cache->PutSlot(memory, offset);
        return;
    }
    if (mapping) context.Function<PFN_vkUnmapMemory>("vkUnmapMemory")(context.device, memory);
    if (buffer) context.Function<PFN_vkDestroyBuffer>("vkDestroyBuffer")(context.device, buffer, nullptr);
    if (memory) context.Function<PFN_vkFreeMemory>("vkFreeMemory")(context.device, memory, nullptr);
}

VkBuffer Buffer::Handle() const {
    return buffer;
}

std::span<std::byte> Buffer::Bytes() {
    Require(mapping != nullptr, "device-local buffer has no host mapping");
    return {static_cast<std::byte*>(mapping), size};
}

void Buffer::Invalidate() {
    Require(mapping != nullptr, "cannot invalidate an unmapped GPU buffer");
    VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
    range.memory = memory;
    range.offset = offset;
    range.size = slab ? static_cast<VkDeviceSize>(capacity) : VK_WHOLE_SIZE;
    Check(context.Function<PFN_vkInvalidateMappedMemoryRanges>("vkInvalidateMappedMemoryRanges")(context.device, 1, &range), "vkInvalidateMappedMemoryRanges");
}

namespace {

// APS5_TRACE_VKMEM: device buffers made per caller (module offset) every 10 s, the pool's churn.
void NoteDeviceBuffer(void* caller, std::size_t size) {
    static const bool enabled = std::getenv("APS5_TRACE_VKMEM") != nullptr;
    if (!enabled) return;
    struct Site {
        std::uint64_t made = 0;
        std::uint64_t bytes = 0;
        std::size_t largest = 0;
    };
    static std::mutex mutex;
    static std::unordered_map<std::uintptr_t, Site> sites;
    static auto lastReport = std::chrono::steady_clock::now();
    // The caller of the constructor (inside make_shared/make_unique when one wraps it).
    const auto site = SiteOf(caller);
    std::lock_guard lock(mutex);
    auto& entry = sites[site];
    ++entry.made;
    entry.bytes += size;
    entry.largest = std::max(entry.largest, size);
    const auto now = std::chrono::steady_clock::now();
    if (now - lastReport < std::chrono::seconds(10)) return;
    lastReport = now;
    std::vector<std::pair<std::uintptr_t, Site>> ranked(sites.begin(), sites.end());
    std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) { return a.second.bytes > b.second.bytes; });
    for (std::size_t index = 0; index < ranked.size() && index < 8; ++index) {
        const auto& [where, made] = ranked[index];
        std::fprintf(stderr, "[devbuf] 10 s: site +0x%llx made %llu device buffers, %.1f MiB, largest %.1f MiB\n", static_cast<unsigned long long>(where), static_cast<unsigned long long>(made.made), made.bytes / 1048576.0, made.largest / 1048576.0);
    }
    sites.clear();
}

}

DeviceBuffer::DeviceBuffer(const Context& context, std::size_t size, VkBufferUsageFlags usage) : context(context), size(size), capacity(BufferPool::Capacity(size, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)), usage(BufferPool::Usage(usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
    Require(size != 0, "zero-sized device buffer");
    NoteDeviceBuffer(__builtin_return_address(0), size);
    cache = GetBufferPool(context);
    if (const auto allocation = cache->Take(size, this->usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
        buffer = allocation->buffer;
        memory = allocation->memory;
        allocationBytes = allocation->allocationBytes;
        capacity = allocation->bytes;
        if (allocation->bufferBytes == size) return;
        context.Function<PFN_vkDestroyBuffer>("vkDestroyBuffer")(context.device, buffer, nullptr);
        buffer = VK_NULL_HANDLE;
        try {
            buffer = BindExactBuffer(context, size, this->usage, memory, 0, allocationBytes);
        } catch (...) {
            release();
            throw;
        }
        return;
    }
    try {
        VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        info.size = size;
        info.usage = this->usage;
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        Check(context.Function<PFN_vkCreateBuffer>("vkCreateBuffer")(context.device, &info, nullptr, &buffer), "vkCreateBuffer device");
        VkMemoryRequirements requirements{};
        context.Function<PFN_vkGetBufferMemoryRequirements>("vkGetBufferMemoryRequirements")(context.device, buffer, &requirements);
        VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocation.allocationSize = ClassAllocationBytes(requirements, capacity);
        allocationBytes = allocation.allocationSize;
        allocation.memoryTypeIndex = context.DeviceMemoryType(requirements.memoryTypeBits, allocation.allocationSize, "device buffer");
        hostPlaced = allocation.memoryTypeIndex != context.MemoryType(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        memory = cache->AllocateMemory(allocation, "vkAllocateMemory device buffer");
        Check(context.Function<PFN_vkBindBufferMemory>("vkBindBufferMemory")(context.device, buffer, memory, 0), "vkBindBufferMemory device");
    } catch (...) {
        release();
        throw;
    }
}

DeviceBuffer::~DeviceBuffer() {
    release();
}

void DeviceBuffer::release() noexcept {
    if (buffer && memory && cache && !hostPlaced) {
        cache->Put({buffer, memory, nullptr, 0, allocationBytes, capacity, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, false, size});
        return;
    }
    if (buffer) context.Function<PFN_vkDestroyBuffer>("vkDestroyBuffer")(context.device, buffer, nullptr);
    if (memory) context.Function<PFN_vkFreeMemory>("vkFreeMemory")(context.device, memory, nullptr);
}

VkBuffer DeviceBuffer::Handle() const {
    return buffer;
}

std::size_t DeviceBuffer::Size() const {
    return size;
}

void CopyBuffer(const Context& context, VkCommandBuffer commands, VkBuffer source, VkDeviceSize sourceOffset, VkBuffer destination, VkDeviceSize destinationOffset, VkDeviceSize bytes) {
    const VkBufferCopy region{sourceOffset, destinationOffset, bytes};
    context.Resolved(&DeviceFunctions::cmdCopyBuffer, "vkCmdCopyBuffer")(commands, source, destination, 1, &region);
}

void RecordMemoryBarrier(const Context& context, VkCommandBuffer commands, VkPipelineStageFlags sourceStage, VkPipelineStageFlags destinationStage, VkAccessFlags sourceAccess, VkAccessFlags destinationAccess) {
    const VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, sourceAccess, destinationAccess};
    context.Resolved(&DeviceFunctions::cmdPipelineBarrier, "vkCmdPipelineBarrier")(commands, sourceStage, destinationStage, 0, 1, &barrier, 0, nullptr, 0, nullptr);
}

void FillDeviceFunctions(const Context& context, DeviceFunctions& functions) {
    functions.cmdPipelineBarrier = context.Function<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier");
    functions.cmdCopyBuffer = context.Function<PFN_vkCmdCopyBuffer>("vkCmdCopyBuffer");
    functions.cmdUpdateBuffer = context.Function<PFN_vkCmdUpdateBuffer>("vkCmdUpdateBuffer");
    functions.cmdFillBuffer = context.Function<PFN_vkCmdFillBuffer>("vkCmdFillBuffer");
    functions.cmdBindPipeline = context.Function<PFN_vkCmdBindPipeline>("vkCmdBindPipeline");
    functions.cmdBindDescriptorSets = context.Function<PFN_vkCmdBindDescriptorSets>("vkCmdBindDescriptorSets");
    functions.cmdPushConstants = context.Function<PFN_vkCmdPushConstants>("vkCmdPushConstants");
    functions.cmdDispatch = context.Function<PFN_vkCmdDispatch>("vkCmdDispatch");
    functions.cmdDispatchIndirect = context.Function<PFN_vkCmdDispatchIndirect>("vkCmdDispatchIndirect");
    functions.cmdBeginRenderPass = context.Function<PFN_vkCmdBeginRenderPass>("vkCmdBeginRenderPass");
    functions.cmdEndRenderPass = context.Function<PFN_vkCmdEndRenderPass>("vkCmdEndRenderPass");
    functions.cmdSetViewport = context.Function<PFN_vkCmdSetViewport>("vkCmdSetViewport");
    functions.cmdSetScissor = context.Function<PFN_vkCmdSetScissor>("vkCmdSetScissor");
    functions.cmdSetDepthBounds = context.Function<PFN_vkCmdSetDepthBounds>("vkCmdSetDepthBounds");
    functions.cmdSetDepthBias = context.Function<PFN_vkCmdSetDepthBias>("vkCmdSetDepthBias");
    if (context.graphicsPipelineLibrary) {
        functions.cmdBeginRendering = context.Function<PFN_vkCmdBeginRenderingKHR>("vkCmdBeginRenderingKHR");
        functions.cmdEndRendering = context.Function<PFN_vkCmdEndRenderingKHR>("vkCmdEndRenderingKHR");
        functions.cmdSetCullMode = context.Function<PFN_vkCmdSetCullModeEXT>("vkCmdSetCullModeEXT");
        functions.cmdSetFrontFace = context.Function<PFN_vkCmdSetFrontFaceEXT>("vkCmdSetFrontFaceEXT");
        functions.cmdSetDepthTestEnable = context.Function<PFN_vkCmdSetDepthTestEnableEXT>("vkCmdSetDepthTestEnableEXT");
        functions.cmdSetDepthWriteEnable = context.Function<PFN_vkCmdSetDepthWriteEnableEXT>("vkCmdSetDepthWriteEnableEXT");
        functions.cmdSetDepthCompareOp = context.Function<PFN_vkCmdSetDepthCompareOpEXT>("vkCmdSetDepthCompareOpEXT");
        functions.cmdSetDepthBoundsTestEnable = context.Function<PFN_vkCmdSetDepthBoundsTestEnableEXT>("vkCmdSetDepthBoundsTestEnableEXT");
        functions.cmdSetStencilTestEnable = context.Function<PFN_vkCmdSetStencilTestEnableEXT>("vkCmdSetStencilTestEnableEXT");
        functions.cmdSetStencilOp = context.Function<PFN_vkCmdSetStencilOpEXT>("vkCmdSetStencilOpEXT");
        functions.cmdSetStencilCompareMask = context.Function<PFN_vkCmdSetStencilCompareMask>("vkCmdSetStencilCompareMask");
        functions.cmdSetStencilWriteMask = context.Function<PFN_vkCmdSetStencilWriteMask>("vkCmdSetStencilWriteMask");
        functions.cmdSetStencilReference = context.Function<PFN_vkCmdSetStencilReference>("vkCmdSetStencilReference");
    }
    functions.cmdBindVertexBuffers = context.Function<PFN_vkCmdBindVertexBuffers>("vkCmdBindVertexBuffers");
    functions.cmdBindIndexBuffer = context.Function<PFN_vkCmdBindIndexBuffer>("vkCmdBindIndexBuffer");
    functions.cmdDraw = context.Function<PFN_vkCmdDraw>("vkCmdDraw");
    functions.cmdDrawIndexed = context.Function<PFN_vkCmdDrawIndexed>("vkCmdDrawIndexed");
    functions.cmdDrawIndirect = context.Function<PFN_vkCmdDrawIndirect>("vkCmdDrawIndirect");
    functions.cmdDrawIndexedIndirect = context.Function<PFN_vkCmdDrawIndexedIndirect>("vkCmdDrawIndexedIndirect");
    functions.cmdCopyBufferToImage = context.Function<PFN_vkCmdCopyBufferToImage>("vkCmdCopyBufferToImage");
    functions.cmdCopyImageToBuffer = context.Function<PFN_vkCmdCopyImageToBuffer>("vkCmdCopyImageToBuffer");
    functions.cmdClearColorImage = context.Function<PFN_vkCmdClearColorImage>("vkCmdClearColorImage");
    functions.updateDescriptorSets = context.Function<PFN_vkUpdateDescriptorSets>("vkUpdateDescriptorSets");
    functions.allocateDescriptorSets = context.Function<PFN_vkAllocateDescriptorSets>("vkAllocateDescriptorSets");
    functions.getFenceStatus = context.Function<PFN_vkGetFenceStatus>("vkGetFenceStatus");
}

RenderTarget::RenderTarget(const Context& context, const ColorTarget& target, bool blending) : context(context) {
    VkFormatProperties properties{};
    context.formatProperties(context.physical, target.format, &properties);
    const VkFormatFeatureFlags required = VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT | VK_FORMAT_FEATURE_TRANSFER_SRC_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT | (blending ? VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT : 0u);
    Require((properties.optimalTilingFeatures & required) == required, "render-target format does not support required operations");
    constexpr auto usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    VkImageFormatProperties supported{};
    Check(context.imageFormatProperties(context.physical, target.format, VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_OPTIMAL, usage, 0, &supported), "vkGetPhysicalDeviceImageFormatProperties");
    Require(target.extent.width <= supported.maxExtent.width && target.extent.height <= supported.maxExtent.height && (supported.sampleCounts & VK_SAMPLE_COUNT_1_BIT) != 0 && target.bytes <= supported.maxResourceSize, "render target exceeds device image limits");
    Require(target.extent.width <= context.limits.maxFramebufferWidth && target.extent.height <= context.limits.maxFramebufferHeight, "render target exceeds framebuffer limits");
    try {
        VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        info.imageType = VK_IMAGE_TYPE_2D;
        info.format = target.format;
        info.extent = {target.extent.width, target.extent.height, 1};
        info.mipLevels = 1;
        info.arrayLayers = 1;
        info.samples = VK_SAMPLE_COUNT_1_BIT;
        info.tiling = VK_IMAGE_TILING_OPTIMAL;
        info.usage = usage;
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        Check(context.Function<PFN_vkCreateImage>("vkCreateImage")(context.device, &info, nullptr, &image), "vkCreateImage");
        VkMemoryRequirements requirements{};
        context.Function<PFN_vkGetImageMemoryRequirements>("vkGetImageMemoryRequirements")(context.device, image, &requirements);
        VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocation.allocationSize = requirements.size;
        allocation.memoryTypeIndex = context.MemoryType(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        Check(context.Function<PFN_vkAllocateMemory>("vkAllocateMemory")(context.device, &allocation, nullptr, &memory), "vkAllocateMemory render target");
        Check(context.Function<PFN_vkBindImageMemory>("vkBindImageMemory")(context.device, image, memory, 0), "vkBindImageMemory");
        VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        viewInfo.image = image;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = target.format;
        viewInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        Check(context.Function<PFN_vkCreateImageView>("vkCreateImageView")(context.device, &viewInfo, nullptr, &view), "vkCreateImageView");
    } catch (...) {
        release();
        throw;
    }
}

RenderTarget::~RenderTarget() {
    release();
}

void RenderTarget::release() noexcept {
    if (view) context.Function<PFN_vkDestroyImageView>("vkDestroyImageView")(context.device, view, nullptr);
    if (image) context.Function<PFN_vkDestroyImage>("vkDestroyImage")(context.device, image, nullptr);
    if (memory) context.Function<PFN_vkFreeMemory>("vkFreeMemory")(context.device, memory, nullptr);
}

VkImage RenderTarget::Image() const {
    return image;
}

VkImageView RenderTarget::View() const {
    return view;
}

CommandBatch::CommandBatch(const Context& context) : context(context) {
    // Records into the device's one command pool and submits to its queue: device-lock work only.
    GuestMemory::AssertGpuLockHeld("CommandBatch");
    try {
        VkCommandBufferAllocateInfo allocation{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        allocation.commandPool = context.pool;
        allocation.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocation.commandBufferCount = 1;
        Check(context.Function<PFN_vkAllocateCommandBuffers>("vkAllocateCommandBuffers")(context.device, &allocation, &commands), "vkAllocateCommandBuffers");
        VkFenceCreateInfo info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        Check(context.Function<PFN_vkCreateFence>("vkCreateFence")(context.device, &info, nullptr, &fence), "vkCreateFence");
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        Check(context.Function<PFN_vkBeginCommandBuffer>("vkBeginCommandBuffer")(commands, &begin), "vkBeginCommandBuffer");
        // Work recorded so far goes first in queue order, so this batch sees its results.
        if (auto* recorder = Recorder::Active(); recorder != nullptr && recorder->Recording()) recorder->Submit();
    } catch (...) {
        release();
        throw;
    }
}

CommandBatch::~CommandBatch() {
    release();
}

void CommandBatch::release() noexcept {
    if (pending) {
        auto result = context.Function<PFN_vkGetFenceStatus>("vkGetFenceStatus")(context.device, fence);
        if (result == VK_NOT_READY) result = context.Function<PFN_vkQueueWaitIdle>("vkQueueWaitIdle")(context.queue);
        if (result != VK_SUCCESS && result != VK_ERROR_DEVICE_LOST) std::terminate();
    }
    if (commands) context.Function<PFN_vkFreeCommandBuffers>("vkFreeCommandBuffers")(context.device, context.pool, 1, &commands);
    if (fence) context.Function<PFN_vkDestroyFence>("vkDestroyFence")(context.device, fence, nullptr);
}

VkCommandBuffer CommandBatch::Handle() const {
    return commands;
}

void CommandBatch::SubmitAndWait() {
    Submit();
    Wait();
}

void CommandBatch::Submit() {
    PerformanceTimer timing("Graphics.Submit");
    Require(!submitted, "command batch has already been submitted");
    Check(context.Function<PFN_vkEndCommandBuffer>("vkEndCommandBuffer")(commands), "vkEndCommandBuffer");
    VkSubmitInfo submission{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submission.commandBufferCount = 1;
    submission.pCommandBuffers = &commands;
    timing.Mark("command_end");
    Check(context.Function<PFN_vkQueueSubmit>("vkQueueSubmit")(context.queue, 1, &submission, fence), "vkQueueSubmit graphics");
    timing.Mark("queue_submit");
    pending = true;
    submitted = true;
}

void CommandBatch::Wait() {
    Require(submitted, "command batch has not been submitted");
    if (!pending) return;
    PerformanceTimer timing("Graphics.Wait");
    const auto result = context.Function<PFN_vkWaitForFences>("vkWaitForFences")(context.device, 1, &fence, VK_TRUE, 5'000'000'000ULL);
    timing.Mark("fence_wait");
    if (result == VK_SUCCESS || result == VK_ERROR_DEVICE_LOST) pending = false;
    Check(result, "vkWaitForFences graphics");
    // Recorded batches preceded this one, so their completions (write-backs) can run now.
    if (auto* recorder = Recorder::Active()) recorder->Reap();
}

}
