#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_GUESTMIRROR_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_GUESTMIRROR_HPP

#include "prx/libSceAgcDriver/Graphics/include/Context.hpp"
#include <cstdint>
#include <functional>
#include <span>
#include <vector>

namespace AgcDriver::Graphics {

struct HostImport;

// The guest mirror (M4 native draws, APS5_GUEST_MIRROR=1): one sparse buffer spanning the guest
// arena whose 64 KiB pages are bound to the host imports of guest memory at their guest offsets, so
// the GPU reaches guest byte A at Address() + (A - Base()) without a lookup table. Pages no import
// covers stay unbound (reads return zero, writes are dropped: residencyNonResidentStrict). Imports
// overlap (joined and span imports cover others), so each page keeps its owners and shows the
// newest live one.
class GuestMirror {
public:
    // Null unless the device enabled sparse binding for it (Context::guestMirror).
    static GuestMirror* Get(const Context& context);
    // The mirror made so far, or null (for callers without a context: retired imports).
    static GuestMirror* Instance();
    // Binds the imports waiting for the GPU mutex; the caller holds it (the present path).
    static void FlushAll();
    void FlushPending();
    VkDeviceAddress Address() const { return address; }
    std::uint64_t Base() const { return base; }
    std::uint64_t Bytes() const { return bytes; }
    // Binds the import's pages. Its memory serves the pages at page-aligned offsets from its base;
    // the rest (an import whose base is not page-aligned, and partial pages at either end) is served
    // by host imports of the mirror's own: the aligned interior, and each partial end page whose
    // whole 64 KiB is committed writable memory. They live as long as the import's binding.
    void Bind(const HostImport& import);
    // Before the import's memory is freed: its pages show the next owner, or none; then `release`
    // (frees the import) runs. Inside an UnbindBatch on this thread both wait for the batch's end.
    void Unbind(const HostImport& import, std::function<void()> release);
    // The unbinds of a release of many imports (a finished batch's kept objects, the import reaper)
    // in one vkQueueBindSparse, one fence wait and one GPU mutex hold, instead of one each: the
    // release thread took the GPU mutex 1-2k times per 10 s for them (trab3). Nests. Opt-in
    // (APS5_MIRROR_UNBIND_BATCH=1): it measured slower (GuestMirror.cpp).
    class UnbindBatch {
    public:
        UnbindBatch();
        ~UnbindBatch();
        UnbindBatch(const UnbindBatch&) = delete;
        UnbindBatch& operator=(const UnbindBatch&) = delete;
    };
    // An unbind waiting for its batch's end.
    struct Unbinding {
        std::uint64_t base;
        std::uint64_t bytes;
        VkDeviceMemory memory;
        std::function<void()> release;
    };
    // [mirror] line every 10 s (APS5_PROFILE_DRAW).
    void Report();

private:
    explicit GuestMirror(const Context& context);
    bool submit(std::uint64_t offset, std::uint64_t size, VkDeviceMemory memory, std::uint64_t memoryOffset);
    // All the binds in one vkQueueBindSparse and one fence wait.
    bool submit(std::span<const VkSparseMemoryBind> binds);
    // A host import of [host, host + size) for the mirror's own pages, or null.
    VkDeviceMemory importHost(void* host, std::uint64_t size);
    // The unbinds of `unbinding` in one submit, then their releases.
    void unbindAll(std::vector<Unbinding>& unbinding);

    // A copy: the context Get was called with need not outlive the mirror.
    const Context context;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceAddress address = 0;
    VkFence fence = VK_NULL_HANDLE;
    std::uint64_t base = 0;
    std::uint64_t bytes = 0;
    std::uint64_t page = 0;
    std::uint32_t memoryTypes = 0;
};

}

#endif
