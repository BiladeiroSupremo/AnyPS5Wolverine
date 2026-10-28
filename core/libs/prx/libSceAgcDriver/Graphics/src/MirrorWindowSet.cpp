#include "prx/libSceAgcDriver/Graphics/include/MirrorWindowSet.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestMirror.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"
#include "BdaAbi.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace AgcDriver::Graphics {

namespace Abi = ShaderRecompiler::BdaAbi;

bool MirrorWindowSet::Enabled() {
    static const bool enabled = std::getenv("APS5_MIRROR_WINDOWS") != nullptr && std::getenv("APS5_BDA_MIRROR") != nullptr && std::getenv("APS5_GUEST_MIRROR") != nullptr;
    return enabled;
}

MirrorWindowSet::MirrorWindowSet(const Context& context, const GuestMirror& mirror) : context(context) {
    try {
        Require(context.limits.maxStorageBufferRange >= Abi::MirrorWindowBytes, "mirror windows need a 4 GiB storage buffer range");
        Require(context.limits.maxPerStageDescriptorStorageBuffers >= Abi::MirrorWindowCount && context.limits.maxDescriptorSetStorageBuffers >= Abi::MirrorWindowCount, "mirror windows exceed the device's storage buffer descriptor limits");
        Require(context.limits.maxBoundDescriptorSets > Abi::MirrorWindowSet, "mirror windows need a fourth bound descriptor set");
        const std::uint64_t stride = std::uint64_t{1} << Abi::MirrorWindowShift;
        Require(mirror.Bytes() >= stride * 2u && (mirror.Bytes() & 0xffffu) == 0u, "the guest mirror is too small or unaligned for windows");
        const VkDescriptorSetLayoutBinding binding{Abi::MirrorWindowBinding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, Abi::MirrorWindowCount, VK_SHADER_STAGE_ALL, nullptr};
        VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        layoutInfo.bindingCount = 1;
        layoutInfo.pBindings = &binding;
        Check(context.Function<PFN_vkCreateDescriptorSetLayout>("vkCreateDescriptorSetLayout")(context.device, &layoutInfo, nullptr, &layout), "vkCreateDescriptorSetLayout mirror windows");
        const VkDescriptorSetLayoutCreateInfo emptyInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        Check(context.Function<PFN_vkCreateDescriptorSetLayout>("vkCreateDescriptorSetLayout")(context.device, &emptyInfo, nullptr, &emptyLayout), "vkCreateDescriptorSetLayout mirror windows empty set");
        const VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, Abi::MirrorWindowCount};
        VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        poolInfo.maxSets = 1;
        poolInfo.poolSizeCount = 1;
        poolInfo.pPoolSizes = &size;
        Check(context.Function<PFN_vkCreateDescriptorPool>("vkCreateDescriptorPool")(context.device, &poolInfo, nullptr, &pool), "vkCreateDescriptorPool mirror windows");
        VkDescriptorSetAllocateInfo allocation{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        allocation.descriptorPool = pool;
        allocation.descriptorSetCount = 1;
        allocation.pSetLayouts = &layout;
        Check(context.Function<PFN_vkAllocateDescriptorSets>("vkAllocateDescriptorSets")(context.device, &allocation, &set), "vkAllocateDescriptorSets mirror windows");
        // Window w starts at w * stride; one past the mirror's end binds its last page (the shader
        // clamps its window index to the array, and a read there is out of the window's range).
        std::vector<VkDescriptorBufferInfo> infos(Abi::MirrorWindowCount);
        const std::uint64_t lastPage = mirror.Bytes() - 0x10000u;
        for (std::uint32_t window = 0; window < Abi::MirrorWindowCount; ++window) {
            const auto offset = std::min<std::uint64_t>(static_cast<std::uint64_t>(window) * stride, lastPage);
            infos[window] = {mirror.Handle(), offset, std::min<std::uint64_t>(Abi::MirrorWindowBytes, mirror.Bytes() - offset)};
        }
        VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        write.dstSet = set;
        write.dstBinding = Abi::MirrorWindowBinding;
        write.descriptorCount = Abi::MirrorWindowCount;
        write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        write.pBufferInfo = infos.data();
        context.Function<PFN_vkUpdateDescriptorSets>("vkUpdateDescriptorSets")(context.device, 1, &write, 0, nullptr);
        std::fprintf(stderr, "[mirror] %u windows of 0x%llx bytes every 0x%llx over the guest mirror, descriptor set %u\n", Abi::MirrorWindowCount, static_cast<unsigned long long>(Abi::MirrorWindowBytes), static_cast<unsigned long long>(stride), Abi::MirrorWindowSet);
    } catch (...) {
        release();
        throw;
    }
}

MirrorWindowSet::~MirrorWindowSet() {
    release();
}

void MirrorWindowSet::release() noexcept {
    if (pool != VK_NULL_HANDLE) context.Function<PFN_vkDestroyDescriptorPool>("vkDestroyDescriptorPool")(context.device, pool, nullptr);
    if (layout != VK_NULL_HANDLE) context.Function<PFN_vkDestroyDescriptorSetLayout>("vkDestroyDescriptorSetLayout")(context.device, layout, nullptr);
    if (emptyLayout != VK_NULL_HANDLE) context.Function<PFN_vkDestroyDescriptorSetLayout>("vkDestroyDescriptorSetLayout")(context.device, emptyLayout, nullptr);
    pool = VK_NULL_HANDLE;
    layout = VK_NULL_HANDLE;
    emptyLayout = VK_NULL_HANDLE;
    set = VK_NULL_HANDLE;
}

}
