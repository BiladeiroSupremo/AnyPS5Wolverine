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
inline constexpr std::uint32_t IndexMask = (1u << KindShift) - 1u;
inline constexpr std::uint32_t Pending = 0xffffffffu;
inline constexpr std::uint32_t Unusable = 0xfffffffeu;
// First word of a table root's mapping block when its table went through the heap.
inline constexpr std::uint32_t HeapMarker = 0xffffffffu;

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
