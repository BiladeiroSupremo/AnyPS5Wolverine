#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_CONTEXT_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_CONTEXT_HPP

#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include <vulkan/vulkan.h>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace AgcDriver::Graphics {

class TextureDetiler;
class GpuColorTransfer;
class BufferPool;
class TextureCache;
class RenderCache;
class DrawQueue;
class GraphicsPipelineCache;
class Recorder;
class DescriptorCache;
class SamplerCache;
class ShaderResources;
class PlaceholderImages;
class BindlessHeapTable;

inline void Require(bool condition, const std::string& reason) {
    if (!condition) throw std::runtime_error("AGC graphics: " + reason);
}

// With VK_EXT_device_fault enabled, the first VK_ERROR_DEVICE_LOST prints the device's fault report
// (addresses, their kind, the vendor's description) to stderr (Resources.cpp).
void SetDeviceLostReporter(VkDevice device, PFN_vkGetDeviceFaultInfoEXT getFaultInfo);
void ReportDeviceLost();

inline void Check(VkResult result, const char* operation) {
    if (result == VK_ERROR_DEVICE_LOST) ReportDeviceLost();
    if (result != VK_SUCCESS) throw std::runtime_error(std::string("AGC graphics: ") + operation + ": Vulkan result " + std::to_string(result));
}

inline void Require(bool condition, const char* reason) {
    if (!condition) throw std::runtime_error(std::string("AGC graphics: ") + reason);
}

// Device entry points resolved once per device (VulkanDevice's State fills it after setup): the
// loader's vkGetDeviceProcAddr is a name lookup under its global mutex per call, paid at every
// record site otherwise. Null members (tests, APS5_NO_PROC_TABLE=1) resolve per call (Resolved).
struct DeviceFunctions {
    PFN_vkCmdPipelineBarrier cmdPipelineBarrier = nullptr;
    PFN_vkCmdCopyBuffer cmdCopyBuffer = nullptr;
    PFN_vkCmdUpdateBuffer cmdUpdateBuffer = nullptr;
    PFN_vkCmdFillBuffer cmdFillBuffer = nullptr;
    PFN_vkCmdBindPipeline cmdBindPipeline = nullptr;
    PFN_vkCmdBindDescriptorSets cmdBindDescriptorSets = nullptr;
    PFN_vkCmdPushConstants cmdPushConstants = nullptr;
    PFN_vkCmdDispatch cmdDispatch = nullptr;
    PFN_vkCmdDispatchIndirect cmdDispatchIndirect = nullptr;
    PFN_vkCmdBeginRenderPass cmdBeginRenderPass = nullptr;
    PFN_vkCmdEndRenderPass cmdEndRenderPass = nullptr;
    PFN_vkCmdBeginRenderingKHR cmdBeginRendering = nullptr;
    PFN_vkCmdEndRenderingKHR cmdEndRendering = nullptr;
    PFN_vkCmdSetViewport cmdSetViewport = nullptr;
    PFN_vkCmdSetScissor cmdSetScissor = nullptr;
    PFN_vkCmdSetDepthBounds cmdSetDepthBounds = nullptr;
    PFN_vkCmdSetDepthBias cmdSetDepthBias = nullptr;
    PFN_vkCmdSetCullModeEXT cmdSetCullMode = nullptr;
    PFN_vkCmdSetFrontFaceEXT cmdSetFrontFace = nullptr;
    PFN_vkCmdSetDepthTestEnableEXT cmdSetDepthTestEnable = nullptr;
    PFN_vkCmdSetDepthWriteEnableEXT cmdSetDepthWriteEnable = nullptr;
    PFN_vkCmdSetDepthCompareOpEXT cmdSetDepthCompareOp = nullptr;
    PFN_vkCmdSetDepthBoundsTestEnableEXT cmdSetDepthBoundsTestEnable = nullptr;
    PFN_vkCmdSetStencilTestEnableEXT cmdSetStencilTestEnable = nullptr;
    PFN_vkCmdSetStencilOpEXT cmdSetStencilOp = nullptr;
    PFN_vkCmdSetStencilCompareMask cmdSetStencilCompareMask = nullptr;
    PFN_vkCmdSetStencilWriteMask cmdSetStencilWriteMask = nullptr;
    PFN_vkCmdSetStencilReference cmdSetStencilReference = nullptr;
    PFN_vkCmdBindVertexBuffers cmdBindVertexBuffers = nullptr;
    PFN_vkCmdBindIndexBuffer cmdBindIndexBuffer = nullptr;
    PFN_vkCmdDraw cmdDraw = nullptr;
    PFN_vkCmdDrawIndexed cmdDrawIndexed = nullptr;
    PFN_vkCmdDrawIndirect cmdDrawIndirect = nullptr;
    PFN_vkCmdDrawIndexedIndirect cmdDrawIndexedIndirect = nullptr;
    PFN_vkCmdCopyBufferToImage cmdCopyBufferToImage = nullptr;
    PFN_vkCmdCopyImageToBuffer cmdCopyImageToBuffer = nullptr;
    PFN_vkCmdClearColorImage cmdClearColorImage = nullptr;
    PFN_vkUpdateDescriptorSets updateDescriptorSets = nullptr;
    PFN_vkAllocateDescriptorSets allocateDescriptorSets = nullptr;
    PFN_vkGetFenceStatus getFenceStatus = nullptr;
};

// Per-thread count of vkGetDeviceProcAddr lookups made through Context::Function (the [vk] line).
inline std::uint64_t& DeviceProcLookups() {
    thread_local std::uint64_t count = 0;
    return count;
}

// APS5_TRACE_VKMEM=1: vkAllocateMemory and vkFreeMemory resolved through Context::Function are
// wrapped to account live device memory by memory type and by call site (the module offset of the
// caller), reported every 10 s and when an allocation fails (Resources.cpp), with live object counts
// (pipelines, descriptor pools, command buffers, ...) and the host heaps' committed bytes. Null when
// disabled.
PFN_vkVoidFunction TracedMemoryFunction(const char* name, PFN_vkVoidFunction real);

inline constexpr std::size_t EmptyBufferBytes = 16;

struct Context {
    VkDevice device;
    VkPhysicalDevice physical;
    VkQueue queue;
    VkCommandPool pool;
    PFN_vkGetDeviceProcAddr deviceProc;
    PFN_vkGetPhysicalDeviceFormatProperties formatProperties;
    PFN_vkGetPhysicalDeviceImageFormatProperties imageFormatProperties;
    VkPhysicalDeviceMemoryProperties memory;
    VkPhysicalDeviceLimits limits;
    bool tessellationShader = false;
    bool meshShader = false;
    VkPhysicalDeviceMeshShaderPropertiesEXT meshLimits{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_PROPERTIES_EXT};
    bool depthClipControl = false;
    bool depthRangeUnrestricted = false;
    bool bufferDeviceAddress = false;
    VkPhysicalDeviceSubgroupProperties subgroup{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
    bool fragmentShaderBarycentric = false;
    bool samplerAnisotropy = false;
    bool textureCompressionBC = false;
    TextureDetiler* detiler = nullptr;
    GpuColorTransfer* colorTransfer = nullptr;
    mutable std::shared_ptr<BufferPool> bufferPool;
    TextureCache* textureCache = nullptr;
    VkPipelineCache pipelineCache = VK_NULL_HANDLE;
    bool depthClamp = false;
    // Nonzero when VK_EXT_external_memory_host is enabled: the required host pointer alignment.
    VkDeviceSize hostImportAlignment = 0;
    bool dmaBufImport = false;
    PFN_vkGetPhysicalDeviceMemoryProperties2 memoryProperties2 = nullptr;
    RenderCache* renderCache = nullptr;
    DrawQueue* drawQueue = nullptr;
    GraphicsPipelineCache* graphicsPipelines = nullptr;
    // Batches GPU work across guest commands (see Recorder); null before the device finished setup.
    Recorder* recorder = nullptr;
    // Per-device caches of the immutable descriptor objects a ShaderResources build needs (set
    // layouts, descriptor pools, samplers); null (tests) means every build makes and destroys its own.
    DescriptorCache* descriptorCache = nullptr;
    SamplerCache* samplerCache = nullptr;
    // Indirect draw features of the device: records with a non-zero first instance, several records
    // per call, and a GPU-side draw count (VK_KHR_draw_indirect_count).
    bool drawIndirectFirstInstance = false;
    bool multiDrawIndirect = false;
    bool drawIndirectCount = false;
    bool occlusionQueryPrecise = false;
    bool depthBounds = false;
    bool depthBiasClamp = false;
    bool dualSrcBlend = false;
    bool samplerFilterMinmax = false;
    bool nonSeamlessCubeMap = false;
    bool conservativeRasterization = false;
    bool meshWave32 = false;
    VkBuffer emptyBuffer = VK_NULL_HANDLE;
    // The bindless heap set (BindlessHeapTable), null without descriptor indexing: every compute
    // pipeline layout takes bindlessLayout as set ShaderRecompiler::BindlessHeaps::DescriptorSet
    // (bindlessEmptyLayout standing in for an unused set below it), and every compute dispatch
    // binds bindlessSet there.
    BindlessHeapTable* bindlessHeaps = nullptr;
    VkDescriptorSetLayout bindlessLayout = VK_NULL_HANDLE;
    VkDescriptorSet bindlessSet = VK_NULL_HANDLE;
    VkDescriptorSetLayout bindlessEmptyLayout = VK_NULL_HANDLE;
    // The device's list of recorded dispatches whose copied written buffers await a CPU write-back
    // (VulkanDevice's State::copiedWriters; the draw counterpart is DrawCopiedWriters): an indirect
    // draw whose records one of them produces reads them on the CPU. Null in tests.
    const std::vector<std::shared_ptr<ShaderResources>>* copiedWriters = nullptr;
    // The device's resolved entry points (see DeviceFunctions); null until the device set up.
    const DeviceFunctions* functions = nullptr;
    // VK_EXT_descriptor_indexing with non-uniform sampled/storage image array indexing enabled
    // (bindless image tables in graphics stages).
    bool descriptorIndexing = false;
    VkPhysicalDeviceDescriptorIndexingPropertiesEXT descriptorIndexingLimits{};
    bool imageInt64Atomics = false;
    bool geometryShader = false;
    bool sampleRateShading = false;
    bool shaderClipDistance = false;
    bool shaderCullDistance = false;
    bool nullDescriptors = false;
    bool bufferInt64Atomics = false;
    const PlaceholderImages* placeholderImages = nullptr;
    bool viewportIndexLayer = false;
    bool primitiveListRestart = false;
    bool imageViewMinLod = false;
    bool pipelineExecutableInfo = false;
    bool graphicsPipelineLibrary = false;
    bool pipelineLinkTimeOptimization = true;
    std::uint32_t srgbDecodeFormats = 0;
    bool provokingVertexLast = false;
    bool provokingVertexModePerPipeline = false;
    bool singlePassStorage = false;
    VkSampleCountFlags sampleLocationSampleCounts = 0;

    template<typename TFunction>
    TFunction Function(const char* name) const {
        Require(deviceProc != nullptr, "missing Vulkan device function resolver");
        // A per-thread cache by device and name literal: every lookup otherwise walks the loader and
        // each implicit layer (RivaTuner's, Steam's) before the driver, thousands of times a second.
        struct Entry {
            VkDevice device;
            const char* name;
            PFN_vkVoidFunction function;
        };
        thread_local std::array<Entry, 256> cache{};
        auto& entry = cache[((reinterpret_cast<std::uintptr_t>(name) >> 3u) ^ reinterpret_cast<std::uintptr_t>(device)) & 255u];
        if (entry.device == device && entry.name == name && entry.function != nullptr) return reinterpret_cast<TFunction>(entry.function);
        ++DeviceProcLookups();
        const auto real = deviceProc(device, name);
        if (real == nullptr) throw std::runtime_error(std::string("AGC graphics: missing Vulkan function: ") + name);
        const auto traced = TracedMemoryFunction(name, real);
        entry = {device, name, traced != nullptr ? traced : real};
        return reinterpret_cast<TFunction>(entry.function);
    }

    // The table's entry point, or a per-call lookup while the table is absent or lacks it.
    template<typename TFunction>
    TFunction Resolved(TFunction DeviceFunctions::*member, const char* name) const {
        if (functions != nullptr && functions->*member != nullptr) return functions->*member;
        return Function<TFunction>(name);
    }

    std::uint32_t MemoryType(std::uint32_t mask, VkMemoryPropertyFlags flags) const {
        for (std::uint32_t i = 0; i < memory.memoryTypeCount; ++i) {
            if ((mask & (1u << i)) != 0 && (memory.memoryTypes[i].propertyFlags & flags) == flags) return i;
        }
        throw std::runtime_error("AGC graphics: required Vulkan memory type is unavailable");
    }

    std::uint32_t DeviceMemoryType(std::uint32_t mask, VkDeviceSize bytes, const char* what) const {
        const auto type = MemoryType(mask, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (memoryProperties2 == nullptr || physical == VK_NULL_HANDLE) return type;
        VkPhysicalDeviceMemoryBudgetPropertiesEXT reported{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
        VkPhysicalDeviceMemoryProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};
        properties.pNext = &reported;
        memoryProperties2(physical, &properties);
        const auto heap = memory.memoryTypes[type].heapIndex;
        const auto budget = reported.heapBudget[heap];
        const auto usage = reported.heapUsage[heap];
        if (budget == 0 || (usage < budget && bytes <= budget - usage)) return type;
        for (std::uint32_t i = 0; i < memory.memoryTypeCount; ++i) {
            if ((mask & (1u << i)) == 0 || (memory.memoryHeaps[memory.memoryTypes[i].heapIndex].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0) continue;
            static std::atomic<std::uint64_t> placements{0}, placedBytes{0};
            const auto count = placements.fetch_add(1) + 1;
            const auto total = placedBytes.fetch_add(bytes) + bytes;
            if (count <= 5 || (count & (count - 1)) == 0) std::fprintf(stderr, "[gpu] %s: device-local heap %u uses %llu of its %llu MiB budget, %llu bytes placed in host memory type %u; %llu such allocations, %llu MiB\n", what, heap, static_cast<unsigned long long>(usage >> 20u), static_cast<unsigned long long>(budget >> 20u), static_cast<unsigned long long>(bytes), i, static_cast<unsigned long long>(count), static_cast<unsigned long long>(total >> 20u));
            return i;
        }
        return type;
    }
};

inline void EndRenderPass(const Context& context, VkCommandBuffer commands) {
    if (context.graphicsPipelineLibrary) context.Resolved(&DeviceFunctions::cmdEndRendering, "vkCmdEndRenderingKHR")(commands);
    else context.Resolved(&DeviceFunctions::cmdEndRenderPass, "vkCmdEndRenderPass")(commands);
}

}

#endif
