#ifndef CORE_SHADER_RECOMPILIER_OPTIMIZATION_INCLUDE_OPTIMIZATION_RESOURCEMATERIALIZER_HPP
#define CORE_SHADER_RECOMPILIER_OPTIMIZATION_INCLUDE_OPTIMIZATION_RESOURCEMATERIALIZER_HPP

#include "IntermediateRepresentation/IrProgram.hpp"
#include "Optimization/SrtWalker.hpp"
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace ShaderRecompiler {

// Why a bindless image table (a T# loaded from a table buffer at a runtime key) was not bound;
// counted on the [bindless] line (APS5_PROFILE_DRAW).
enum class BindlessRejection { Capacity, MaterialScan, NoEntry, Storage, NonUniform, ImageSlots, HeapOnly, Count };

// A bindless image table the materializer cannot bind (BindlessRejection): the work using it cannot
// run, which a driver may report and skip instead of failing the process.
struct BindlessTableRejected : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// An image descriptor the compiled interface cannot read (a dimension, class or format the shader
// was not compiled for): that work cannot run, which a driver may report and skip as above.
struct RuntimeImageMismatch : std::runtime_error {
    using std::runtime_error::runtime_error;
};

class ResourceMaterializer {
public:
    static std::uint32_t EmulatedCompareState(const ShaderInfo& info, const ResourceSnapshot& snapshot, std::uint32_t index);
    void ApplyStaticInterface(IrProgram& program, bool nativeSampleOffsets = true) const;
    static std::vector<ImageResource> RuntimeImageModes(const ImageResource& image);
    static std::uint32_t RuntimeImageMode(const ImageResource& image, const DescriptorValue& descriptor, std::span<const ImageResource> modes);
    // Whether `mode` (one of RuntimeImageModes(image)) can serve a heap-sampled table: the heap path
    // takes the mode of the table's representative entry, which is a float entry of the access's own
    // dimension, so the integer, conversion and packed modes and the other dimensions never apply
    // there (SpirvImageEmitter emits only these in the heap branch).
    static bool HeapImageMode(const ImageResource& image, const ImageResource& mode);
    // APS5_HEAP_ONLY=1: graphics stages emit only the heap branch of a heap-branched table (no slot
    // search); a table the driver would bind slot by slot is rejected instead (BindlessRejection::HeapOnly).
    static bool HeapOnlyGraphics();
    // The slot search of a heap-branched table in a graphics stage also holds only the heap modes (a
    // compile-size cut that keeps the branch, unlike APS5_HEAP_ONLY); a table bound slot by slot
    // with an entry of another mode is rejected (BindlessRejection::HeapOnly). RGA on PPSA03671's
    // shader_11d28c9e00 prepared module: 1.42 M words 56.9 s -> 1.20 M words 37.7 s; gameplay
    // renders correctly with 0 rejections (trsh1). APS5_SLOT_ALL_MODES=1 emits every mode as before.
    static bool SlotHeapModes();
    static void PrepareImageModes(ShaderInfo& info);
    static const char* UnnormalizedSampleMismatch(const ImageResource& image);
    [[nodiscard]] IrResourcePlan ExtractPlan(const IrProgram& program) const;
    static void CompactPlan(IrResourcePlan& plan);
    void Materialize(const IrResourcePlan& program, const SrtRuntime& runtime, ResourceSnapshot& snapshot) const;
    static std::uint64_t SpecializationNanoseconds();
    static std::uint32_t BindlessSlots();
    static void CountBindlessRejection(BindlessRejection reason);
    static std::uint64_t EvaluateNanoseconds();
    static std::uint64_t MaterializeNanoseconds();
};

}

#endif
