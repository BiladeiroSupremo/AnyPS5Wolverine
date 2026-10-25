#ifndef CORE_SHADER_RECOMPILIER_OPTIMIZATION_INCLUDE_OPTIMIZATION_BINDLESSHEAPS_HPP
#define CORE_SHADER_RECOMPILIER_OPTIMIZATION_INCLUDE_OPTIMIZATION_BINDLESSHEAPS_HPP

#include "IntermediateRepresentation/IrMetadata/ShaderStage.hpp"
#include <cstdint>
#include <vector>

// Bindless image heaps (after KytyPS5's bindless table): a bindless image table too wide to bind
// slot by slot (BindlessSlots) is sampled through one descriptor set the device shares between
// compute pipelines, instead of binding its entries per dispatch.
//
// Set `DescriptorSet` holds a runtime array of sampled float images per kind (2D, 2D array, 3D),
// a translation buffer and a feedback buffer. Each heap (a table V# base and entry offset) owns a
// region of the translation buffer: the shader reads `translation[region + key]`, which is either a
// slot (`kind << KindShift | index` into that kind's array), Pending (not resolved yet: the shader
// writes 1 to `feedback[region + key]` and samples zeros) or Unusable (resolution failed: zeros,
// no request). The driver reads the feedback, looks the requested textures up, writes their
// descriptors and publishes the slots, ordered after the uploads on the GPU.
//
// Every compute table root of an eligible shape (HeapEligible) is emitted with both paths: its
// mapping block (ImageResource::indirectMappingOffset) starts with HeapMarker when the dispatch's
// table went through the heap, followed by the region base and the entry count; otherwise it is
// the usual sorted key/slot list.
namespace ShaderRecompiler::BindlessHeaps {

// After the resource set (0) and the workgroup memory set (1).
inline constexpr std::uint32_t DescriptorSet = 2u;
inline constexpr std::uint32_t KindCount = 4u;
// Binding of each kind's image array; the translation and feedback buffers follow.
inline constexpr std::uint32_t ImageBinding2D = 0u;
inline constexpr std::uint32_t ImageBinding2DArray = 1u;
inline constexpr std::uint32_t ImageBinding3D = 2u;
inline constexpr std::uint32_t ImageBinding1D = 3u;
inline constexpr std::uint32_t TranslationBinding = 4u;
inline constexpr std::uint32_t FeedbackBinding = 5u;
inline constexpr std::uint32_t SlotsPerKind = 16384u;
inline constexpr std::uint32_t TranslationEntries = 1u << 20u;
inline constexpr std::uint32_t KindShift = 24u;
inline constexpr std::uint32_t IndexMask = (1u << 16u) - 1u;
static_assert(SlotsPerKind <= IndexMask + 1u, "a slot index fits the word's index bits");
// Bits 16-19 of a slot word: levels the slot's texture leaves out at the top of its chain (a
// streamed texture held from its first loaded level, after KytyPS5's image heap). The shader
// subtracts them from explicit levels and adds them back to level and size queries; an implicit
// level follows from the smaller image by itself.
inline constexpr std::uint32_t LevelShift = 16u;
inline constexpr std::uint32_t LevelMask = 0xfu;
inline constexpr std::uint32_t Pending = 0xffffffffu;
inline constexpr std::uint32_t Unusable = 0xfffffffeu;
// First word of a table root's mapping block when its table went through the heap.
inline constexpr std::uint32_t HeapMarker = 0xffffffffu;

// GPU-resolved T#s (APS5_GPU_IMAGES=1): a sampled image whose eight dwords are constant-offset
// reads of one uniform buffer is not evaluated on the CPU at each draw. The shader loads the words,
// hashes them (HashWords) and probes HashProbes buckets of the key buffer (binding KeyBinding,
// eight words per bucket, written only by the driver). A matching bucket's word sits in the
// translation buffer at HashBase + bucket. No match, or a Pending word, appends the words to the
// request ring in the feedback buffer (feedback[RingBase] counts, then RingStride words per request:
// the T#, then a nonzero completion word); the driver resolves it like a heap entry and stores the
// key and the word. The ring reuses the feedback words of the hash range, which regions never take.
inline constexpr std::uint32_t KeyBinding = 6u;
inline constexpr std::uint32_t HashEntries = 1u << 16u;
inline constexpr std::uint32_t HashProbes = 4u;
inline constexpr std::uint32_t HashBase = TranslationEntries - 1u - HashEntries;
inline constexpr std::uint32_t RingBase = HashBase;
inline constexpr std::uint32_t RingStride = 9u;
inline constexpr std::uint32_t RingCapacity = (HashEntries - 1u) / RingStride;
// Bits 20-22 of a hashed entry's word: the view's numeric class (0 float, 1 uint, 2 sint) and
// whether it is a cube view (stored in the 2D array binding). Table entries never set them.
inline constexpr std::uint32_t ClassShift = 20u;
inline constexpr std::uint32_t ClassMask = 0x3u;
inline constexpr std::uint32_t CubeBit = 1u << 22u;

// GPU-resolved S#s share the key buffer and ring: a sampler's key is its four words, SamplerMarker,
// then three zeros (never a T#'s eight words, never an empty bucket). Its word is SamplerKind <<
// KindShift | pair index; pair k sits at elements 2k (as the S# says) and 2k + 1 (point filtered) of
// the sampler array binding SamplerBinding. Pair 0 is the placeholder.
inline constexpr std::uint32_t SamplerBinding = 7u;
inline constexpr std::uint32_t SamplerPairs = 2048u;
inline constexpr std::uint32_t SamplerKind = 0x40u;
inline constexpr std::uint32_t SamplerMarker = 0x53414d50u;

// The bucket hash of a T#; the SPIR-V emits the same arithmetic.
[[nodiscard]] constexpr std::uint32_t HashWords(const std::uint32_t* words) {
    std::uint32_t hash = 0x9e3779b9u;
    for (std::uint32_t i = 0; i < 8u; ++i) hash = (hash ^ words[i]) * 0x01000193u;
    return hash ^ (hash >> 16u);
}

// Whether shaders take GPU-resolved T#s (APS5_GPU_IMAGES=1 and the heap set exists).
[[nodiscard]] bool HashedImagesEnabled();

// The class and cube bits (ClassShift, CubeBit) of a GPU-resolved T#'s word: the runtime mode the
// recompiler picks for the T# (ResourceMaterializer::RuntimeImageMode, as a direct image of its own
// dimension would) when that mode is one a hashed access emits (float, uint or sint, a cube); else
// Unusable, with `reason` naming the mode.
[[nodiscard]] std::uint32_t HashedWordBits(const std::uint32_t* words, std::uint32_t srgbDecodeFormats, const char** reason);

struct Region {
    std::uint64_t heapBase = 0;
    std::uint32_t entryOffset = 0;
    // First translation entry of the region, and how many it holds.
    std::uint32_t base = 0;
    std::uint32_t capacity = 0;
    // The table's entry count when it was last materialized.
    std::uint32_t entries = 0;
};

// Whether the device provides the heap set; set by the driver once it created the set, before
// any shader is prepared. Off, a table wider than BindlessSlots is rejected as before.
[[nodiscard]] bool Enabled();
void SetEnabled(bool enabled);

// Whether a table root of this stage, dimension and access is emitted with the heap path (the
// SPIR-V and the materializer must agree). `dimension` is an RdnaImageDimension value.
[[nodiscard]] bool HeapEligible(IrShaderStage stage, std::uint32_t dimension, bool sampled, bool depthCompare, bool r128);

// The region of the heap, allocated (with headroom) on first use or when the table outgrew it.
// Throws BindlessTableRejected when the translation buffer is full.
[[nodiscard]] std::uint32_t RegionBase(std::uint64_t heapBase, std::uint32_t entryOffset, std::uint32_t entries);

// Every region, and a counter that changes whenever a region is added or grows.
[[nodiscard]] std::vector<Region> Regions();
[[nodiscard]] std::uint64_t RegionsGeneration();

}

#endif
