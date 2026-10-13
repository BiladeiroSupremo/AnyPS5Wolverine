#include "Optimization/BindlessHeaps.hpp"
#include "Optimization/ResourceMaterializer.hpp"
#include "RdnaDecoder/RdnaInstruction.hpp"
#include <algorithm>
#include <atomic>
#include <cstdio>
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

bool HeapEligible(bool compute, std::uint32_t dimension, bool sampled, bool depthCompare, bool r128) {
    const auto shape = static_cast<RdnaImageDimension>(dimension);
    const bool kind = shape == RdnaImageDimension::Dim2D || shape == RdnaImageDimension::Dim2DArray || shape == RdnaImageDimension::Dim3D;
    return compute && Enabled() && kind && sampled && !depthCompare && !r128;
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
