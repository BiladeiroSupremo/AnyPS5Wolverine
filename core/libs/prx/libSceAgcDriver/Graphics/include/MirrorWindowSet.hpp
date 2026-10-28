#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_MIRRORWINDOWSET_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_MIRRORWINDOWSET_HPP

#include "prx/libSceAgcDriver/Graphics/include/Context.hpp"

namespace AgcDriver::Graphics {

class GuestMirror;

// The guest mirror as a static descriptor set of storage-buffer windows (BdaAbi::MirrorWindow*,
// APS5_MIRROR_WINDOWS=1): window w is [w * 2^MirrorWindowShift, + MirrorWindowBytes) of the mirror
// buffer (windows past its end bind its last page, so they read zero). Written once; every
// pipeline layout takes Layout() as set BdaAbi::MirrorWindowSet (EmptyLayout() standing in for an
// unused set below it) and every draw and dispatch binds Set() there.
class MirrorWindowSet {
public:
    MirrorWindowSet(const Context& context, const GuestMirror& mirror);
    ~MirrorWindowSet();
    MirrorWindowSet(const MirrorWindowSet&) = delete;
    MirrorWindowSet& operator=(const MirrorWindowSet&) = delete;
    // APS5_MIRROR_WINDOWS=1 (with the guest mirror and APS5_BDA_MIRROR).
    static bool Enabled();
    // Writes the windows into `set` at BdaAbi::MirrorWindowBinding (the bindless heap set, whose
    // layout has the binding when Enabled()). The layout of the class's own set (set 3) is unused
    // since 2026-10-28: a fourth set in graphics shaders broke the game's rendering.
    static void WriteInto(const Context& context, const GuestMirror& mirror, VkDescriptorSet set);

    VkDescriptorSetLayout Layout() const { return layout; }
    VkDescriptorSetLayout EmptyLayout() const { return emptyLayout; }
    VkDescriptorSet Set() const { return set; }

private:
    void release() noexcept;

    Context context;
    VkDescriptorSetLayout layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout emptyLayout = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE;
};

}

#endif
