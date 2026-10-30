#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DRAWCACHE_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DRAWCACHE_HPP

#include "prx/libSceAgcDriver/Execution/include/Driver/Dispatch/DispatchCache.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include <array>
#include <atomic>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace AgcDriver::DriverDetail {

struct DrawProgram {
    ShaderRecompiler::ShaderBinary binary;
    std::uint32_t userDataBase;
    std::uint32_t firstUserSgpr = 8;
    std::vector<std::uint32_t> userData;
    std::array<ShaderRecompiler::MemoryRegion, 2> memory;

    std::shared_ptr<const ShaderSnapshot> snapshot;
    std::size_t codeOffset = 0;
};

struct DrawDecode {
    Graphics::State state;
    ShaderRecompiler::ShaderPixelStageInfo pixel;
    std::vector<DrawProgram> programs;
    std::vector<ShaderRecompiler::ProgramRole> roles;
};

void DecodeGraphicsPrograms(DrawDecode& decoded, const QueueState& queue, const ShaderRegistry& registry, bool staticAbi, bool includeFragment);

struct PreparedGraphicsStage {
    std::shared_ptr<const ShaderSnapshot> snapshot;
    PreparedShaders::Entry entry;
    ShaderRecompiler::ShaderStage stage = ShaderRecompiler::ShaderStage::Vertex;
    std::uint32_t pushOffset = 0;
    std::shared_ptr<const ShaderRecompiler::RecompileResult> registration;
};

std::vector<PreparedGraphicsStage> PrepareGraphicsStages(const DrawDecode& decoded, const ShaderRecompiler::SpirvTarget& target);

struct DrawRecipeRecord {
    std::vector<std::weak_ptr<const DispatchVariant>> stages;
    std::shared_ptr<const DrawRecipe> recipe;
    bool Matches(const std::vector<std::shared_ptr<DispatchVariant>>& variants) const;
    bool Expired() const;
};

struct DrawEntry {

    std::shared_ptr<const DrawDecode> decode;

    std::vector<std::vector<std::shared_ptr<DispatchVariant>>> stages;

    std::atomic<std::shared_ptr<const std::vector<DrawRecipeRecord>>> recipes;
    std::uint64_t touched = 0;
    std::list<std::uint64_t>::iterator order;
};

enum class DrawMiss : std::size_t { FrontDiffering, FragmentDiffering, OtherDiffering, Layout, Gate, Stages, Count };

struct DrawEntryCounters {
    std::uint64_t lookups = 0, absent = 0, hits = 0, stageValidations = 0, stageEqual = 0, variantsCompared = 0, inserts = 0, variantsInserted = 0, variantsEvicted = 0, present = 0, unstable = 0, touches = 0, verifyHits = 0, verifyMismatches = 0;
    std::array<std::uint64_t, static_cast<std::size_t>(DrawMiss::Count)> misses{};
    std::array<std::uint64_t, MaxDispatchVariants> variantHitsByRank{};
    double validateUs = 0;

    std::uint64_t registerKeyLookups = 0, registerKeyHits = 0, decodeSkipped = 0, decodePartial = 0, facadeMismatches = 0, verifyDecodes = 0, verifyDecodeMismatches = 0;
    double keyUs = 0;
    std::chrono::steady_clock::time_point lastReport = std::chrono::steady_clock::now();
};

class DrawKeySightings {
public:
    static constexpr std::size_t Ways = 4;
    explicit DrawKeySightings(std::size_t keys);
    bool SeenBefore(std::uint64_t key);

private:
    std::vector<std::uint64_t> keys;
    std::vector<std::uint8_t> filled;
    std::vector<std::uint8_t> next;
    std::size_t bucketMask = 0;
};

enum class DrawVerdict { Drawn, Nothing, Rejected };

struct StageCapture {
    std::shared_ptr<const ShaderRecompiler::RecompileResult> compiled;
    std::vector<ShaderRecompiler::MemoryRegion> regions;
    std::uint64_t forgetSerial = 0;
    std::uint32_t pushOffset = 0;
    // The user words the capture's walk read through (asBase) and specialized on (asValue), for
    // APS5_RELOCATED_DRAW_KEYS; captured is set when this draw ran the walk.
    std::uint64_t userWordsAsBase = 0;
    std::uint64_t userWordsAsValue = 0;
    // APS5_NATIVE_DRAWS: the walk's reads, each with the user word its base pointer came from (-1:
    // a pointer read from memory), to relocate the captured runs exactly.
    std::vector<std::pair<std::uint64_t, std::int32_t>> walkReads;
    // ... and the capture's read trace: flattened-SRT slots read verbatim from one address (leaves)
    // and the addresses read for anything else (ResourceCapture::readTrace), for its data words.
    std::vector<std::pair<std::uint32_t, std::uint64_t>> dataLeaves;
    std::vector<std::uint64_t> otherReads;
    bool captured = false;
};

bool DecodeProgramsCurrent(const DrawDecode& decode, const ShaderRegistry& registry);

// Copies of a native stage's result that hits patch in place: one nothing else holds any more is
// reused, so a hit allocates nothing (a copy per hit cost ~12 us per draw, trno1 2026-10-30).
struct NativeResultPool {
    std::mutex mutex;
    std::vector<std::shared_ptr<ShaderRecompiler::RecompileResult>> results;
};

// APS5_NATIVE_DRAWS (native path, Draw.cpp and Dispatch.cpp): a stage as materialized, the patches
// that carry a later draw's user words into it, its program and vertex stage info (a hit needs the
// same), the user words its capture used as values and as base pointers, the capturing draw's user
// words, and for a stage that read guest memory its captured words as runs relocated at their
// pointers (NativeRunsFor), which a hit compares at its own pointers.
struct NativeDrawStage {
    std::shared_ptr<const ShaderRecompiler::RecompileResult> result;
    std::vector<DispatchVariant::UserWordPatch> patches;
    std::uint64_t program = 0;
    std::optional<ShaderRecompiler::ShaderVertexStageInfo> vertexInfo;
    std::uint64_t asValue = 0;
    std::vector<std::uint32_t> userData;
    std::uint64_t asBase = 0;
    std::shared_ptr<DispatchVariant> runs;
    // Runs read through a pointer the walk loaded from `runs` (absolute as captured): per run, the
    // position of that pointer's low word in runs->words (chainParents) and its value then
    // (chained->runPointerValues); a hit moves each run by the pointer's change (Driver::nativeGather).
    std::shared_ptr<DispatchVariant> chained;
    std::vector<std::uint32_t> chainParents;
    // Where those pointers' low words appear in the result (binding or NoBinding for push data, word,
    // and in `userWord` the pointer's position in runs->words): a hit writes their values now there.
    std::vector<DispatchVariant::UserWordPatch> chainPatches;
    // Dispatches: value-used user words not used as a read base whose every use the patches carry
    // (patchableValues). A dispatch where only such words changed is built once as before and
    // compared with this stage patched (learning): equal adds them to `confirmed`, unequal sets
    // `refused`; later dispatches with only confirmed words changed reuse the stage.
    std::uint64_t patchableValues = 0;
    std::shared_ptr<std::atomic<std::uint64_t>> confirmed = std::make_shared<std::atomic<std::uint64_t>>(0);
    std::shared_ptr<std::atomic<bool>> refused = std::make_shared<std::atomic<bool>>(false);
    std::shared_ptr<NativeResultPool> pool = std::make_shared<NativeResultPool>();
};

// The value-used user words that differ from the stage's (bit per word below 64), or ~0 when the
// words cannot be compared that way (sizes, untracked words, base-pointer nullness).
std::uint64_t NativeValueChanges(const NativeDrawStage& stage, std::span<const std::uint32_t> userData);

// A stage's captured words (`regions`, the words its walk read from pages) as runs, each relocated
// at the user word its walk read it through (`walkReads`: address and origin, SrtUserDataUse::reads);
// words the walk did not read itself stay absolute. Null when a word was read through two bases or
// a base is not a non-null pointer pair. With `stage`, words read through a pointer loaded from
// those runs go to stage->chained (one level; the pointer must not appear in the result) and the
// pointers' positions to `parents` (to join the runs' data positions, NativeAddDataPositions).
std::shared_ptr<DispatchVariant> NativeRunsFor(std::span<const ShaderRecompiler::MemoryRegion> regions, std::span<const std::pair<std::uint64_t, std::int32_t>> walkReads, const std::vector<std::uint32_t>& userData, const std::shared_ptr<const ShaderRecompiler::RecompileResult>& result, NativeDrawStage* stage = nullptr, std::vector<std::uint32_t>* parents = nullptr);

// Positions of runs.words a hit may find changed with nothing to patch (chain pointers).
void NativeAddDataPositions(DispatchVariant& runs, std::span<const std::uint32_t> positions);

// The data words of a stage's runs (as the dispatch cache's data hits): captured words its walk read
// only to copy them into the flattened SRT (`leaves`, not among `otherReads`), which a hit may find
// changed and patches in (NativeWithData). Sets runs.flatBinding, dataPositions and dataSlots.
void NativeDataWords(DispatchVariant& runs, const ShaderRecompiler::RecompileResult& result, std::span<const std::pair<std::uint32_t, std::uint64_t>> leaves, std::span<const std::uint64_t> otherReads);

// APS5_NATIVE_DRAWS=1: every 64th reuse (NativeSampled) is also built as before, rendered from that
// and compared; a mismatch turns native reuse off for the rest of the run (TripNative, logged), so a
// state the verify runs did not meet costs a few wrong draws at most. NativeTripped: turned off.
bool NativeSampled();
bool NativeTripped();
void TripNative(const char* what, std::uint64_t program);

// Whether a stage's stored user words admit these: the value-used words equal, the base pointers
// null where they were null (a walk through a null pointer records no read).
bool NativeSameValues(const NativeDrawStage& stage, std::span<const std::uint32_t> userData);

// The stage with this draw's values: its user words patched in, its vertex attributes' V#s from
// `vertex`, and the words the gather found (`live`: the stage's runs->words with the changed ones;
// empty: as stored) at its data slots and chain pointers' places. A pooled copy (NativeResultPool)
// with every such place rewritten; the stage itself when nothing varies; null when a patch lies
// outside it.
std::shared_ptr<const ShaderRecompiler::RecompileResult> NativePatchedResult(const NativeDrawStage& stage, std::span<const std::uint32_t> userData, const std::optional<ShaderRecompiler::ShaderVertexStageInfo>& vertex, std::span<const std::uint32_t> live = {});

// The shader register a draw program's user word comes from, or 0 for a merged stage's driver
// words (DrawRelocation.cpp).
std::uint32_t DrawUserWordRegister(const DrawProgram& program, std::uint32_t word);

}

#endif
