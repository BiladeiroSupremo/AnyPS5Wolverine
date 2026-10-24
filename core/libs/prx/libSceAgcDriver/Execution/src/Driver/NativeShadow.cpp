#include "prx/libSceAgcDriver/Execution/include/Driver/NativeShadow.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <iterator>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace AgcDriver::DriverDetail {

namespace {

constexpr std::uint32_t ShaderFileHeader = 0x34333231u;
// Per-stage records of the title's Agc command buffer (PPSA03671): 0x70 bytes each from +0x38,
// record +0 = the stage's Shader.
constexpr std::uint64_t StageRecords = 0x38;
constexpr std::uint64_t StageRecordBytes = 0x70;

struct ShadowState {
    std::mutex mutex;
    std::unordered_map<std::uint64_t, NativeDrawRecord> records;
    std::deque<std::uint64_t> order;
    // Decoded draws: every decoded Shader among the record's; some missing (by count of decoded
    // stages without a registered header apart); no record at all.
    std::uint64_t covered = 0, uncovered = 0, unregistered = 0, missing = 0, noted = 0;
    // Record slots holding the decoded shaders (which stage records the title uses).
    std::array<std::uint64_t, 16> slotUse{};
    int examples = 0;
    std::chrono::steady_clock::time_point lastReport = std::chrono::steady_clock::now();
};

ShadowState& Shadow() {
    static auto* state = new ShadowState();
    return *state;
}

}

bool NativeShadowActive() {
    static const bool active = std::getenv("APS5_NATIVE_SHADOW") != nullptr;
    return active;
}

std::optional<NativeDrawRecord> TakeNativeDraw(std::uint64_t packetAddress) {
    auto& state = Shadow();
    std::lock_guard lock(state.mutex);
    const auto found = state.records.find(packetAddress);
    if (found == state.records.end()) return std::nullopt;
    auto record = found->second;
    state.records.erase(found);
    return record;
}

void CompareNativeDraw(const NativeDrawRecord* record, std::span<const std::uint64_t> decodedShaders) {
    auto& state = Shadow();
    std::lock_guard lock(state.mutex);
    if (record == nullptr) {
        ++state.missing;
    } else {
        bool covered = true;
        for (const auto shader : decodedShaders) {
            if (shader == 0) {
                ++state.unregistered;
                continue;
            }
            const auto slot = std::find(record->shaders.begin(), record->shaders.end(), shader);
            if (slot == record->shaders.end()) covered = false;
            else ++state.slotUse[static_cast<std::size_t>(slot - record->shaders.begin())];
        }
        ++(covered ? state.covered : state.uncovered);
        if (!covered && state.examples < 12) {
            ++state.examples;
            std::string text = " record";
            char item[32];
            for (std::size_t slot = 0; slot < record->shaders.size(); ++slot) {
                if (record->shaders[slot] == 0) continue;
                std::snprintf(item, sizeof(item), " %zu:%llx", slot, static_cast<unsigned long long>(record->shaders[slot]));
                text += item;
            }
            text += " | decoded";
            for (const auto shader : decodedShaders) {
                std::snprintf(item, sizeof(item), " %llx", static_cast<unsigned long long>(shader));
                text += item;
            }
            std::fprintf(stderr, "[native-shadow] not covered:%s\n", text.c_str());
        }
    }
    const auto now = std::chrono::steady_clock::now();
    if (now - state.lastReport < std::chrono::seconds(10)) return;
    state.lastReport = now;
    std::string slots;
    for (std::size_t slot = 0; slot < state.slotUse.size(); ++slot) {
        if (state.slotUse[slot] == 0) continue;
        char item[32];
        std::snprintf(item, sizeof(item), " %zu:%llu", slot, static_cast<unsigned long long>(state.slotUse[slot]));
        slots += item;
    }
    std::fprintf(stderr, "[native-shadow] 10 s: %llu draw packets noted; decoded draws: every shader in the record %llu, not %llu, no record %llu; decoded stages without a registered header %llu; record slots used:%s\n", static_cast<unsigned long long>(state.noted), static_cast<unsigned long long>(state.covered), static_cast<unsigned long long>(state.uncovered), static_cast<unsigned long long>(state.missing), static_cast<unsigned long long>(state.unregistered), slots.c_str());
    state.covered = state.uncovered = state.unregistered = state.missing = state.noted = 0;
    state.slotUse = {};
}

}

extern "C" void AgcDriverNoteEngineDraw_nid_postfix(const CommandBuffer* buffer, const std::uint32_t* packet) {
    using namespace AgcDriver::DriverDetail;
    if (!NativeShadowActive() || buffer == nullptr || packet == nullptr) return;
    const auto base = reinterpret_cast<std::uint64_t>(buffer);
    NativeDrawRecord record;
    for (std::uint64_t stage = 0; stage < record.shaders.size(); ++stage) {
        const auto at = base + StageRecords + stage * StageRecordBytes;
        if (!AgcDriver::GuestMemory::Accessible(reinterpret_cast<const void*>(at), 8)) break;
        const auto shader = *reinterpret_cast<const std::uint64_t*>(at);
        if (shader < 0x10000 || !AgcDriver::GuestMemory::Accessible(reinterpret_cast<const void*>(shader), sizeof(Shader))) continue;
        if (reinterpret_cast<const Shader*>(shader)->file_header != ShaderFileHeader) continue;
        record.shaders[stage] = shader;
    }
    auto& state = Shadow();
    std::lock_guard lock(state.mutex);
    const auto address = reinterpret_cast<std::uint64_t>(packet);
    state.records[address] = record;
    state.order.push_back(address);
    ++state.noted;
    while (state.order.size() > (std::size_t{1} << 16)) {
        state.records.erase(state.order.front());
        state.order.pop_front();
    }
}
