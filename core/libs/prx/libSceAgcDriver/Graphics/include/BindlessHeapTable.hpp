#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_BINDLESSHEAPTABLE_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_BINDLESSHEAPTABLE_HPP

#include "prx/libSceAgcDriver/Graphics/include/Context.hpp"
#include "Optimization/BindlessHeaps.hpp"
#include <array>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <vector>

namespace AgcDriver::Graphics {

class Buffer;
class Recorder;
class Texture;

// The device's bindless heap set (see Optimization/BindlessHeaps.hpp): its layout, pool and set,
// the host-visible translation and feedback buffers, and the textures its slots hold. Slot 0 of
// every kind holds a 1x1 placeholder, so a word of a full kind never reads an unbound descriptor.
// Every compute pipeline layout takes the set as set BindlessHeaps::DescriptorSet.
class BindlessHeapTable {
public:
    explicit BindlessHeapTable(const Context& context);
    ~BindlessHeapTable();
    BindlessHeapTable(const BindlessHeapTable&) = delete;
    BindlessHeapTable& operator=(const BindlessHeapTable&) = delete;

    VkDescriptorSetLayout Layout() const { return layout; }
    VkDescriptorSetLayout EmptyLayout() const { return emptyLayout; }
    VkDescriptorSet Set() const { return set; }
    VkBuffer TranslationBuffer() const;
    std::span<std::uint32_t> Translation();
    std::span<std::uint32_t> Feedback();
    // Puts the texture's view in the next free slot of `kind` (a BindlessHeaps::ImageBinding*) and
    // keeps the texture alive; the translation word for it, or Unusable when the kind is full.
    std::uint32_t Publish(std::uint32_t kind, const std::shared_ptr<Texture>& texture, VkImageView view, VkImageLayout imageLayout);
    // Whether a feedback scan is due (at most one per interval).
    bool ScanDue();
    // Marks a translation entry handled (its word published or about to be); false when it already
    // was. The GPU writes the word later, so the CPU copy can still read Pending meanwhile.
    bool Claim(std::uint32_t index);
    std::mutex& Mutex() { return mutex; }

    // A published entry, revalidated over time (ResolveBindlessHeapRequests): a guest texture the
    // CPU or GPU keeps writing (a glyph atlas gaining glyphs) is looked up again, which refreshes an
    // image-backed view in place or, when the cache hands back another texture or the T# changed,
    // publishes it again under a new slot.
    struct Entry {
        std::uint32_t index = 0;
        std::uint64_t address = 0;
        std::array<std::uint32_t, 8> words{};
        std::shared_ptr<Texture> texture;
    };
    std::vector<Entry>& Entries() { return entries; }
    std::size_t& RevalidateCursor() { return revalidateCursor; }
    // Whether a revalidation pass is due (at most one per interval).
    bool RevalidateDue();

private:
    void createPlaceholders();
    void release() noexcept;

    Context context;
    VkDescriptorSetLayout layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout emptyLayout = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE;
    std::unique_ptr<Buffer> translation;
    std::unique_ptr<Buffer> feedback;
    std::array<std::uint32_t, ShaderRecompiler::BindlessHeaps::KindCount> nextSlot{};
    struct Placeholder {
        VkImage image = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
    };
    std::array<Placeholder, ShaderRecompiler::BindlessHeaps::KindCount> placeholders{};
    std::vector<std::shared_ptr<Texture>> pinned;
    std::vector<std::uint8_t> claimed;
    std::chrono::steady_clock::time_point lastScan{};
    std::vector<Entry> entries;
    std::size_t revalidateCursor = 0;
    std::chrono::steady_clock::time_point lastRevalidate{};
    std::mutex mutex;
};

// Resolves the keys shaders requested through the feedback buffer since the last scan: reads each
// heap T#, looks its texture up (recording its upload), publishes a slot and stores the slot words
// into the translation buffer with commands ordered after the uploads. Called before a compute
// dispatch is recorded (outside any render pass), under GuestMemory::GpuMutex.
void ResolveBindlessHeapRequests(const Context& context, Recorder& recorder);

}

#endif
