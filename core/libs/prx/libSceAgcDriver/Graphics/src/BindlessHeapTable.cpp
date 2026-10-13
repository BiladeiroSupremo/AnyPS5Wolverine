#include "prx/libSceAgcDriver/Graphics/include/BindlessHeapTable.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestTextureResource.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderResources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <set>
#include <stdexcept>

namespace AgcDriver::Graphics {

namespace Heaps = ShaderRecompiler::BindlessHeaps;

namespace {

// Feedback is scanned at most this often: requests from a frame are resolved while the next frames
// are recorded.
constexpr auto ScanInterval = std::chrono::milliseconds(2);
// Textures resolved per scan: each may record an upload, so a burst of requests spreads over frames.
constexpr std::uint32_t ScanBudget = 256u;
// Published entries are looked up again at most this often, this many per pass (round robin).
constexpr auto RevalidateInterval = std::chrono::milliseconds(30);
constexpr std::size_t RevalidateBudget = 128u;

}

BindlessHeapTable::BindlessHeapTable(const Context& context) : context(context) {
    try {
        std::array<VkDescriptorSetLayoutBinding, Heaps::KindCount + 2u> bindings{};
        std::array<VkDescriptorBindingFlags, Heaps::KindCount + 2u> flags{};
        for (std::uint32_t kind = 0; kind < Heaps::KindCount; kind++) {
            bindings[kind] = {kind, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, Heaps::SlotsPerKind, VK_SHADER_STAGE_ALL, nullptr};
            flags[kind] = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT;
        }
        bindings[Heaps::KindCount] = {Heaps::TranslationBinding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL, nullptr};
        bindings[Heaps::KindCount + 1u] = {Heaps::FeedbackBinding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL, nullptr};
        VkDescriptorSetLayoutBindingFlagsCreateInfo bindingFlags{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO};
        bindingFlags.bindingCount = static_cast<std::uint32_t>(flags.size());
        bindingFlags.pBindingFlags = flags.data();
        VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, &bindingFlags};
        layoutInfo.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
        layoutInfo.bindingCount = static_cast<std::uint32_t>(bindings.size());
        layoutInfo.pBindings = bindings.data();
        Check(context.Function<PFN_vkCreateDescriptorSetLayout>("vkCreateDescriptorSetLayout")(context.device, &layoutInfo, nullptr, &layout), "vkCreateDescriptorSetLayout bindless heaps");
        const VkDescriptorSetLayoutCreateInfo emptyInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        Check(context.Function<PFN_vkCreateDescriptorSetLayout>("vkCreateDescriptorSetLayout")(context.device, &emptyInfo, nullptr, &emptyLayout), "vkCreateDescriptorSetLayout bindless empty set");

        const std::array<VkDescriptorPoolSize, 2> sizes{{{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, Heaps::SlotsPerKind * Heaps::KindCount}, {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2}}};
        VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
        poolInfo.maxSets = 1;
        poolInfo.poolSizeCount = static_cast<std::uint32_t>(sizes.size());
        poolInfo.pPoolSizes = sizes.data();
        Check(context.Function<PFN_vkCreateDescriptorPool>("vkCreateDescriptorPool")(context.device, &poolInfo, nullptr, &pool), "vkCreateDescriptorPool bindless heaps");
        VkDescriptorSetAllocateInfo allocation{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        allocation.descriptorPool = pool;
        allocation.descriptorSetCount = 1;
        allocation.pSetLayouts = &layout;
        Check(context.Function<PFN_vkAllocateDescriptorSets>("vkAllocateDescriptorSets")(context.device, &allocation, &set), "vkAllocateDescriptorSets bindless heaps");

        const auto bytes = static_cast<std::size_t>(Heaps::TranslationEntries) * sizeof(std::uint32_t);
        translation = std::make_unique<Buffer>(context, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        feedback = std::make_unique<Buffer>(context, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        std::fill(Translation().begin(), Translation().end(), Heaps::Pending);
        std::fill(Feedback().begin(), Feedback().end(), 0u);
        const std::array<VkDescriptorBufferInfo, 2> infos{{{translation->Handle(), 0, VK_WHOLE_SIZE}, {feedback->Handle(), 0, VK_WHOLE_SIZE}}};
        std::array<VkWriteDescriptorSet, 2> writes{};
        for (std::uint32_t i = 0; i < writes.size(); i++) {
            writes[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            writes[i].dstSet = set;
            writes[i].dstBinding = i == 0 ? Heaps::TranslationBinding : Heaps::FeedbackBinding;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &infos[i];
        }
        context.Function<PFN_vkUpdateDescriptorSets>("vkUpdateDescriptorSets")(context.device, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
        createPlaceholders();
    } catch (...) {
        release();
        throw;
    }
}

BindlessHeapTable::~BindlessHeapTable() {
    release();
}

void BindlessHeapTable::release() noexcept {
    pinned.clear();
    translation.reset();
    feedback.reset();
    if (pool != VK_NULL_HANDLE) context.Function<PFN_vkDestroyDescriptorPool>("vkDestroyDescriptorPool")(context.device, pool, nullptr);
    if (layout != VK_NULL_HANDLE) context.Function<PFN_vkDestroyDescriptorSetLayout>("vkDestroyDescriptorSetLayout")(context.device, layout, nullptr);
    if (emptyLayout != VK_NULL_HANDLE) context.Function<PFN_vkDestroyDescriptorSetLayout>("vkDestroyDescriptorSetLayout")(context.device, emptyLayout, nullptr);
    pool = VK_NULL_HANDLE;
    layout = VK_NULL_HANDLE;
    emptyLayout = VK_NULL_HANDLE;
    set = VK_NULL_HANDLE;
    for (auto& placeholder : placeholders) {
        if (placeholder.view != VK_NULL_HANDLE) context.Function<PFN_vkDestroyImageView>("vkDestroyImageView")(context.device, placeholder.view, nullptr);
        if (placeholder.image != VK_NULL_HANDLE) context.Function<PFN_vkDestroyImage>("vkDestroyImage")(context.device, placeholder.image, nullptr);
        if (placeholder.memory != VK_NULL_HANDLE) context.Function<PFN_vkFreeMemory>("vkFreeMemory")(context.device, placeholder.memory, nullptr);
        placeholder = {};
    }
}

// A 1x1 transparent-black image of every kind in slot 0, written once and never replaced.
void BindlessHeapTable::createPlaceholders() {
    CommandBatch batch(context);
    const auto commands = batch.Handle();
    std::array<VkDescriptorImageInfo, Heaps::KindCount> infos{};
    std::array<VkWriteDescriptorSet, Heaps::KindCount> writes{};
    for (std::uint32_t kind = 0; kind < Heaps::KindCount; kind++) {
        auto& placeholder = placeholders[kind];
        VkImageCreateInfo imageInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        imageInfo.imageType = kind == Heaps::ImageBinding3D ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D;
        imageInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
        imageInfo.extent = {1, 1, 1};
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        Check(context.Function<PFN_vkCreateImage>("vkCreateImage")(context.device, &imageInfo, nullptr, &placeholder.image), "vkCreateImage bindless placeholder");
        VkMemoryRequirements requirements{};
        context.Function<PFN_vkGetImageMemoryRequirements>("vkGetImageMemoryRequirements")(context.device, placeholder.image, &requirements);
        VkMemoryAllocateInfo memoryInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        memoryInfo.allocationSize = requirements.size;
        memoryInfo.memoryTypeIndex = context.MemoryType(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        Check(context.Function<PFN_vkAllocateMemory>("vkAllocateMemory")(context.device, &memoryInfo, nullptr, &placeholder.memory), "vkAllocateMemory bindless placeholder");
        Check(context.Function<PFN_vkBindImageMemory>("vkBindImageMemory")(context.device, placeholder.image, placeholder.memory, 0), "vkBindImageMemory bindless placeholder");

        const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        VkImageMemoryBarrier toTransfer{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        toTransfer.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        toTransfer.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        toTransfer.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        toTransfer.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toTransfer.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toTransfer.image = placeholder.image;
        toTransfer.subresourceRange = range;
        context.Function<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")(commands, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &toTransfer);
        const VkClearColorValue black{};
        context.Function<PFN_vkCmdClearColorImage>("vkCmdClearColorImage")(commands, placeholder.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &black, 1, &range);
        VkImageMemoryBarrier toShader = toTransfer;
        toShader.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        toShader.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        toShader.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        toShader.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        context.Function<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")(commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &toShader);

        VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        viewInfo.image = placeholder.image;
        viewInfo.viewType = kind == Heaps::ImageBinding2D ? VK_IMAGE_VIEW_TYPE_2D : kind == Heaps::ImageBinding2DArray ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_3D;
        viewInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
        viewInfo.subresourceRange = range;
        Check(context.Function<PFN_vkCreateImageView>("vkCreateImageView")(context.device, &viewInfo, nullptr, &placeholder.view), "vkCreateImageView bindless placeholder");
        infos[kind] = {VK_NULL_HANDLE, placeholder.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        writes[kind] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        writes[kind].dstSet = set;
        writes[kind].dstBinding = kind;
        writes[kind].dstArrayElement = 0;
        writes[kind].descriptorCount = 1;
        writes[kind].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        writes[kind].pImageInfo = &infos[kind];
        nextSlot[kind] = 1;
    }
    batch.SubmitAndWait();
    context.Function<PFN_vkUpdateDescriptorSets>("vkUpdateDescriptorSets")(context.device, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

VkBuffer BindlessHeapTable::TranslationBuffer() const {
    return translation->Handle();
}

std::span<std::uint32_t> BindlessHeapTable::Translation() {
    const auto bytes = translation->Bytes();
    return {reinterpret_cast<std::uint32_t*>(bytes.data()), Heaps::TranslationEntries};
}

std::span<std::uint32_t> BindlessHeapTable::Feedback() {
    const auto bytes = feedback->Bytes();
    return {reinterpret_cast<std::uint32_t*>(bytes.data()), Heaps::TranslationEntries};
}

std::uint32_t BindlessHeapTable::Publish(std::uint32_t kind, const std::shared_ptr<Texture>& texture, VkImageView view, VkImageLayout imageLayout) {
    if (kind >= Heaps::KindCount || nextSlot[kind] >= Heaps::SlotsPerKind) return Heaps::Unusable;
    const auto slot = nextSlot[kind]++;
    const VkDescriptorImageInfo info{VK_NULL_HANDLE, view, imageLayout};
    VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstSet = set;
    write.dstBinding = kind;
    write.dstArrayElement = slot;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    write.pImageInfo = &info;
    context.Function<PFN_vkUpdateDescriptorSets>("vkUpdateDescriptorSets")(context.device, 1, &write, 0, nullptr);
    pinned.push_back(texture);
    return (kind << Heaps::KindShift) | slot;
}

bool BindlessHeapTable::Claim(std::uint32_t index) {
    if (claimed.empty()) claimed.assign(Heaps::TranslationEntries, 0u);
    if (index >= claimed.size() || claimed[index] != 0u) return false;
    claimed[index] = 1u;
    return true;
}

bool BindlessHeapTable::RevalidateDue() {
    const auto now = std::chrono::steady_clock::now();
    if (now - lastRevalidate < RevalidateInterval) return false;
    lastRevalidate = now;
    return true;
}

bool BindlessHeapTable::ScanDue() {
    const auto now = std::chrono::steady_clock::now();
    if (now - lastScan < ScanInterval) return false;
    lastScan = now;
    return true;
}

namespace {

bool integerViewFormat(VkFormat format) {
    switch (format) {
    case VK_FORMAT_R8_UINT: case VK_FORMAT_R8_SINT: case VK_FORMAT_R8G8_UINT: case VK_FORMAT_R8G8_SINT:
    case VK_FORMAT_R8G8B8A8_UINT: case VK_FORMAT_R8G8B8A8_SINT: case VK_FORMAT_R16_UINT: case VK_FORMAT_R16_SINT:
    case VK_FORMAT_R16G16_UINT: case VK_FORMAT_R16G16_SINT: case VK_FORMAT_R16G16B16A16_UINT: case VK_FORMAT_R16G16B16A16_SINT:
    case VK_FORMAT_R32_UINT: case VK_FORMAT_R32_SINT: case VK_FORMAT_R32G32_UINT: case VK_FORMAT_R32G32_SINT:
    case VK_FORMAT_R32G32B32A32_UINT: case VK_FORMAT_R32G32B32A32_SINT: case VK_FORMAT_A2B10G10R10_UINT_PACK32:
        return true;
    default:
        return false;
    }
}

bool readHeapWords(std::uint64_t address, std::array<std::uint32_t, 8>& words) {
    if (!GuestMemory::Accessible(reinterpret_cast<const void*>(address), sizeof(words))) return false;
    GuestMemory::Read(address, std::as_writable_bytes(std::span(words)));
    return !(words[0] == 0u && (words[1] & 0xffu) == 0u);
}

// The translation word of one heap entry: the entry's texture in a slot of the kind its view has,
// or Unusable (null, unreadable, integer or of a kind the heap set lacks). A usable entry is
// recorded for revalidation (`existing`: the entry being revalidated, updated in place).
std::uint32_t resolveHeapEntry(const Context& context, BindlessHeapTable& table, std::uint32_t index, std::uint64_t address, BindlessHeapTable::Entry* existing = nullptr) {
    std::array<std::uint32_t, 8> words{};
    if (!readHeapWords(address, words)) return Heaps::Unusable;
    const auto texture = CachedSampledTexture(context, words);
    // Revalidation: the same T# served by the same texture needs nothing (the lookup refreshed an
    // image-backed view in place).
    if (existing != nullptr && existing->words == words && existing->texture == texture) return Heaps::Pending;
    if (texture == nullptr || integerViewFormat(texture->ViewFormat())) return Heaps::Unusable;
    const auto range = texture->SampledViewRange(false);
    std::uint32_t kind = Heaps::KindCount;
    if (range.type == VK_IMAGE_VIEW_TYPE_2D) kind = Heaps::ImageBinding2D;
    else if (range.type == VK_IMAGE_VIEW_TYPE_2D_ARRAY) kind = Heaps::ImageBinding2DArray;
    else if (range.type == VK_IMAGE_VIEW_TYPE_3D) kind = Heaps::ImageBinding3D;
    // Diagnostic: APS5_TRACE_HEAP_ENTRIES=1 prints every published entry.
    static const bool traceEntries = std::getenv("APS5_TRACE_HEAP_ENTRIES") != nullptr;
    if (kind == Heaps::KindCount) {
        if (traceEntries) std::fprintf(stderr, "[bindless] entry 0x%llx: %ux%u guest format %u, view type %d unusable\n", static_cast<unsigned long long>(address), DecodeTextureResource(words).width, DecodeTextureResource(words).height, DecodeTextureResource(words).format, static_cast<int>(range.type));
        return Heaps::Unusable;
    }
    const auto word = table.Publish(kind, texture, texture->View(), texture->Layout());
    if (word != Heaps::Unusable) {
        if (existing != nullptr) {
            existing->words = words;
            existing->texture = texture;
        } else {
            table.Entries().push_back({index, address, words, texture});
        }
    }
    if (traceEntries) {
        const auto resource = DecodeTextureResource(words);
        std::fprintf(stderr, "[bindless] entry 0x%llx -> T# 0x%llx %ux%u guest format %u tile %d sel %u%u%u%u, view %d format %d layers %u, word 0x%x\n", static_cast<unsigned long long>(address), static_cast<unsigned long long>(resource.baseAddress), resource.width, resource.height, resource.format, static_cast<int>(resource.tileMode), resource.dstSelX, resource.dstSelY, resource.dstSelZ, resource.dstSelW, static_cast<int>(range.type), static_cast<int>(texture->ViewFormat()), range.layers, word);
    }
    return word;
}

struct HeapCounters {
    std::atomic<std::uint64_t> requests{0};
    std::atomic<std::uint64_t> usable{0};
    std::atomic<std::uint64_t> failed{0};
    std::atomic<std::uint64_t> republished{0};
    std::atomic<long long> lastReport{0};
};

HeapCounters& heapCounters() {
    static HeapCounters counters;
    return counters;
}

// One line per 10 s while the heaps resolve anything: requests read, entries published, failures.
void reportHeaps() {
    auto& counters = heapCounters();
    const auto now = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    auto last = counters.lastReport.load(std::memory_order_relaxed);
    if (now - last < 10 || !counters.lastReport.compare_exchange_strong(last, now, std::memory_order_relaxed)) return;
    const auto requests = counters.requests.exchange(0, std::memory_order_relaxed);
    const auto republished = counters.republished.exchange(0, std::memory_order_relaxed);
    if (requests == 0 && republished == 0) return;
    std::fprintf(stderr, "[bindless] heap (10 s): %llu requests, %llu published, %llu unusable, %llu republished\n", static_cast<unsigned long long>(requests), static_cast<unsigned long long>(counters.usable.exchange(0, std::memory_order_relaxed)), static_cast<unsigned long long>(counters.failed.exchange(0, std::memory_order_relaxed)), static_cast<unsigned long long>(republished));
}

}

void ResolveBindlessHeapRequests(const Context& context, Recorder& recorder) {
    auto* table = context.bindlessHeaps;
    if (table == nullptr || Heaps::RegionsGeneration() == 0u) return;
    std::lock_guard lock(table->Mutex());
    if (!table->ScanDue()) return;
    const auto regions = Heaps::Regions();
    auto feedback = table->Feedback();
    auto& counters = heapCounters();
    // Diagnostic: every 10 s, the words of each region as the CPU sees them.
    static auto lastProbe = std::chrono::steady_clock::now();
    if (std::chrono::steady_clock::now() - lastProbe > std::chrono::seconds(10)) {
        lastProbe = std::chrono::steady_clock::now();
        auto translationWords = table->Translation();
        for (const auto& region : regions) {
            std::uint32_t requested = 0, pending = 0, mapped = 0;
            for (std::uint32_t key = 0; key < std::min(region.entries, region.capacity); key++) {
                requested += feedback[region.base + key] != 0u ? 1u : 0u;
                const auto word = translationWords[region.base + key];
                pending += word == Heaps::Pending ? 1u : 0u;
                mapped += word != Heaps::Pending && word != Heaps::Unusable ? 1u : 0u;
            }
            std::fprintf(stderr, "[bindless] probe: region %u (%u entries): %u requested, %u pending, %u mapped\n", region.base, region.entries, requested, pending, mapped);
        }
        std::fprintf(stderr, "[bindless] probe: last out-of-table key 0x%x\n", feedback[Heaps::TranslationEntries - 1u]);
    }
    std::vector<std::pair<std::uint32_t, std::uint32_t>> published;
    for (const auto& region : regions) {
        const auto count = std::min(region.entries, region.capacity);
        for (std::uint32_t key = 0; key < count && published.size() < ScanBudget; key++) {
            const auto index = region.base + key;
            if (feedback[index] == 0u) continue;
            feedback[index] = 0u;
            if (!table->Claim(index)) continue;
            std::uint32_t word = Heaps::Unusable;
            try {
                word = resolveHeapEntry(context, *table, index, region.heapBase + region.entryOffset + static_cast<std::uint64_t>(key) * 32u);
            } catch (const std::exception& error) {
                static std::atomic<int> reported{0};
                if (reported.fetch_add(1, std::memory_order_relaxed) < 16) std::fprintf(stderr, "[bindless] heap 0x%llx key %u is unusable: %s\n", static_cast<unsigned long long>(region.heapBase), key, error.what());
            }
            published.emplace_back(index, word);
        }
    }
    counters.requests.fetch_add(published.size(), std::memory_order_relaxed);
    for (const auto& entry : published) (entry.second != Heaps::Unusable ? counters.usable : counters.failed).fetch_add(1, std::memory_order_relaxed);
    // Revalidation: published entries whose T# or texture changed are published again.
    // APS5_NO_HEAP_REVALIDATE=1 keeps every entry as first resolved.
    static const bool revalidate = std::getenv("APS5_NO_HEAP_REVALIDATE") == nullptr;
    auto& entries = table->Entries();
    if (revalidate && !entries.empty() && table->RevalidateDue()) {
        auto& cursor = table->RevalidateCursor();
        for (std::size_t visited = 0; visited < std::min(RevalidateBudget, entries.size()); ++visited) {
            if (cursor >= entries.size()) cursor = 0;
            auto& entry = entries[cursor++];
            try {
                const auto word = resolveHeapEntry(context, *table, entry.index, entry.address, &entry);
                if (word == Heaps::Pending) continue;
                published.emplace_back(entry.index, word);
                counters.republished.fetch_add(1, std::memory_order_relaxed);
            } catch (const std::exception&) {
            }
        }
    }
    reportHeaps();
    if (published.empty()) return;
    // The words reach the shaders through the GPU, after the uploads the lookups recorded (the
    // batch is taken after them: a lookup may have submitted the previous one).
    const auto commands = recorder.Commands();
    const auto updateBuffer = context.Function<PFN_vkCmdUpdateBuffer>("vkCmdUpdateBuffer");
    for (const auto& [index, word] : published) updateBuffer(commands, table->TranslationBuffer(), static_cast<VkDeviceSize>(index) * sizeof(std::uint32_t), sizeof(std::uint32_t), &word);
    // Diagnostic: APS5_TRACE_HEAP_ENTRIES=2 also reads back each newly published image-backed entry.
    static const bool dumpEntries = [] { const char* text = std::getenv("APS5_TRACE_HEAP_ENTRIES"); return text != nullptr && text[0] == '2'; }();
    if (dumpEntries) {
        static std::set<const StorageTexture*> dumped;
        std::vector<std::shared_ptr<StorageTexture>> images;
        for (const auto& [index, word] : published) {
            for (const auto& entry : table->Entries()) {
                if (entry.index != index || entry.texture == nullptr || !entry.texture->ViewsStorageImage()) continue;
                if (dumped.insert(entry.texture->StorageSource()).second) images.push_back(entry.texture->SharedStorageSource());
            }
        }
        if (!images.empty()) {
            recorder.Sync();
            for (const auto& image : images) {
                const bool done = image->DumpImage("heap");
                std::fprintf(stderr, "[bindless] dumped heap image 0x%llx %ux%u format %u%s\n", static_cast<unsigned long long>(image->Descriptor().baseAddress), image->Descriptor().width, image->Descriptor().height, image->Descriptor().format, done ? "" : " (format not dumped)");
            }
            return;
        }
    }
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    context.Function<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")(commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
}

}
