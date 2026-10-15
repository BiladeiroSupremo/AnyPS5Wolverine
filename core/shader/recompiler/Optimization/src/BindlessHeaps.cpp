#include "Optimization/BindlessHeaps.hpp"
#include "Optimization/ResourceMaterializer.hpp"
#include "RdnaDecoder/RdnaInstruction.hpp"
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>

namespace ShaderRecompiler::BindlessHeaps {

namespace {

// A new region holds at least this many entries, or twice the table, so a heap that streams in more
// entries keeps its region (and the dispatches' region words) for a while.
constexpr std::uint32_t MinimumCapacity = 16384u;

struct Registry {
    std::mutex mutex;
    std::vector<Region> regions;
    std::uint32_t next = 0;
    std::atomic<std::uint64_t> generation{0};
    std::atomic<bool> enabled{false};
};

Registry& registry() {
    static Registry instance;
    return instance;
}

}

bool Enabled() {
    return registry().enabled.load(std::memory_order_acquire);
}

void SetEnabled(bool enabled) {
    registry().enabled.store(enabled, std::memory_order_release);
}

namespace {

// APS5_GRAPHICS_HEAPS: the graphics stages that take the heap, every one by default ("all"; was
// "pixel" until the load freezes it caused were traced to revalidation, 715b27741; 25 min clean
// run hall3 2026-10-15); "0" for none, "1" or "all" for every graphics stage, or a comma list
// (vertex, pixel, mesh, local, hull, domain).
bool graphicsStageHeaps(IrShaderStage stage) {
    static const std::string list = [] { const char* text = std::getenv("APS5_GRAPHICS_HEAPS"); return std::string(text != nullptr ? text : "all"); }();
    if (list.empty() || list == "0") return false;
    if (list == "1" || list == "all") return true;
    const char* name = stage == IrShaderStage::Vertex ? "vertex" : stage == IrShaderStage::Pixel ? "pixel" : stage == IrShaderStage::Mesh ? "mesh" : stage == IrShaderStage::Local ? "local" : stage == IrShaderStage::TessellationControl ? "hull" : stage == IrShaderStage::TessellationEvaluation ? "domain" : "other";
    std::size_t start = 0;
    while (start <= list.size()) {
        const auto end = std::min(list.find(',', start), list.size());
        if (list.compare(start, end - start, name) == 0) return true;
        start = end + 1;
    }
    return false;
}

}

bool HeapEligible(IrShaderStage stage, std::uint32_t dimension, bool sampled, bool depthCompare, bool r128) {
    const bool compute = stage == IrShaderStage::Compute;
    const auto shape = static_cast<RdnaImageDimension>(dimension);
    // Dim1D is eligible: the uncaught exception (0x20474343) PPSA03671 ended with once its 1D heap
    // path was emitted was a name-table overrun, fixed in b1d1b84b1 (run hall3 clean).
    // APS5_NO_HEAP_1D=1 makes it ineligible again.
    static const bool heap1D = std::getenv("APS5_NO_HEAP_1D") == nullptr;
    const bool kind = shape == RdnaImageDimension::Dim2D || shape == RdnaImageDimension::Dim2DArray || shape == RdnaImageDimension::Dim3D || (heap1D && shape == RdnaImageDimension::Dim1D);
    // Graphics stages take the heap per APS5_GRAPHICS_HEAPS (every graphics pipeline layout carries
    // the set either way; tables wider than the static limit are rejected in the others, and their
    // draws skipped). PPSA03671's terrain and rock materials need it in pixel shaders. Those draws
    // hung the GPU until 2026-10-15: their pixel shaders loop on a row scan read with v_readlane
    // (EmitRowReduction); the draws only ran once the heap let their tables bind. The materializer
    // decides at each draw, so a shader prepared with the heap path is safe either way.
    return Enabled() && (compute || graphicsStageHeaps(stage)) && kind && sampled && !depthCompare && !r128;
}

std::uint32_t RegionBase(std::uint64_t heapBase, std::uint32_t entryOffset, std::uint32_t entries) {
    auto& state = registry();
    std::lock_guard lock(state.mutex);
    auto found = std::find_if(state.regions.begin(), state.regions.end(), [&](const Region& region) { return region.heapBase == heapBase && region.entryOffset == entryOffset; });
    if (found != state.regions.end() && entries <= found->capacity) {
        if (entries > found->entries) {
            found->entries = entries;
            state.generation.fetch_add(1, std::memory_order_release);
        }
        return found->base;
    }
    const auto capacity = std::max(MinimumCapacity, entries > TranslationEntries / 2u ? entries : entries * 2u);
    if (capacity > TranslationEntries - state.next) {
        throw BindlessTableRejected("bindless image heap: the translation buffer has no room for " + std::to_string(capacity) + " more entries");
    }
    // A table that outgrew its region moves to a new one; the old region's words stay published
    // for dispatches still recorded with it.
    if (found != state.regions.end()) state.regions.erase(found);
    state.regions.push_back({heapBase, entryOffset, state.next, capacity, entries});
    std::fprintf(stderr, "[bindless] heap 0x%llx+0x%x: %u entries in translation region %u (capacity %u)\n", static_cast<unsigned long long>(heapBase), entryOffset, entries, state.next, capacity);
    state.next += capacity;
    state.generation.fetch_add(1, std::memory_order_release);
    return state.regions.back().base;
}

std::vector<Region> Regions() {
    auto& state = registry();
    std::lock_guard lock(state.mutex);
    return state.regions;
}

std::uint64_t RegionsGeneration() {
    return registry().generation.load(std::memory_order_acquire);
}

}
