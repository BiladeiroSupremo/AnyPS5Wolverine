#include "prx/libSceAgcDriver/Execution/include/Driver/NativeShadow.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
    // Per decoded program with its Shader in the record: user data equal, equal over the shorter
    // length (one side longer), different; the record had no user data for it.
    std::uint64_t dataEqual = 0, dataPrefix = 0, dataDiffer = 0, dataNone = 0;
    std::array<std::uint64_t, 33> dataOffsets{};
    int dataExamples = 0;
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

void DumpNativeDraw(const NativeDrawRecord& record, std::span<const std::uint64_t> decodedShaders, std::span<const std::vector<std::uint32_t>> decodedUserData) {
    static std::atomic<int> dumped{0};
    if (record.raw.empty() || dumped.fetch_add(1, std::memory_order_relaxed) >= 24) return;
    std::string text;
    char item[64];
    for (const auto& [slot, bytes] : record.raw) {
        std::snprintf(item, sizeof(item), "\n  record %u:", static_cast<unsigned>(slot));
        text += item;
        for (const auto word : bytes) {
            std::snprintf(item, sizeof(item), " %016llx", static_cast<unsigned long long>(word));
            text += item;
        }
        // The words the record's +0x10 and +0x18 point at, when they are pointers.
        for (const std::size_t field : {2u, 3u}) {
            const auto pointer = bytes[field];
            if (pointer < 0x10000 || !AgcDriver::GuestMemory::Accessible(reinterpret_cast<const void*>(pointer), 32 * 4)) continue;
            std::snprintf(item, sizeof(item), "\n    [+0x%zx] ->", field * 8);
            text += item;
            for (std::size_t i = 0; i < 32; ++i) {
                std::snprintf(item, sizeof(item), " %08x", reinterpret_cast<const std::uint32_t*>(pointer)[i]);
                text += item;
            }
        }
    }
    for (std::size_t i = 0; i < decodedShaders.size() && i < decodedUserData.size(); ++i) {
        std::snprintf(item, sizeof(item), "\n  decoded shader %llx user data:", static_cast<unsigned long long>(decodedShaders[i]));
        text += item;
        for (const auto word : decodedUserData[i]) {
            std::snprintf(item, sizeof(item), " %08x", word);
            text += item;
        }
    }
    std::fprintf(stderr, "[native-shadow] study:%s\n", text.c_str());
}

void CompareNativeUserData(const NativeDrawRecord& record, std::span<const std::uint64_t> decodedShaders, std::span<const std::vector<std::uint32_t>> decodedUserData) {
    auto& state = Shadow();
    std::lock_guard lock(state.mutex);
    for (std::size_t i = 0; i < decodedShaders.size() && i < decodedUserData.size(); ++i) {
        const auto slot = std::find(record.shaders.begin(), record.shaders.end(), decodedShaders[i]);
        if (decodedShaders[i] == 0 || slot == record.shaders.end()) continue;
        const auto& native = record.userData[static_cast<std::size_t>(slot - record.shaders.begin())];
        const auto& decoded = decodedUserData[i];
        if (native.empty()) {
            ++state.dataNone;
            continue;
        }
        // The title's payload starts at the first user SGPR it sets (a geometry stage's at SGPR 8,
        // where the decoded words start at 0): equal at some word offset of the decoded words.
        std::size_t offset = 0;
        bool same = false;
        for (; offset + native.size() <= decoded.size() && offset <= 32; ++offset) {
            if (std::equal(native.begin(), native.end(), decoded.begin() + static_cast<std::ptrdiff_t>(offset))) {
                same = true;
                break;
            }
        }
        if (same && offset == 0 && native.size() == decoded.size()) ++state.dataEqual;
        else if (same) {
            ++state.dataPrefix;
            if (offset < state.dataOffsets.size()) ++state.dataOffsets[offset];
        } else {
            ++state.dataDiffer;
        }
        if (!same && state.dataExamples < 12) {
            ++state.dataExamples;
            std::string text;
            char item[24];
            text += " native";
            for (const auto word : native) {
                std::snprintf(item, sizeof(item), " %08x", word);
                text += item;
            }
            text += " | decoded";
            for (const auto word : decoded) {
                std::snprintf(item, sizeof(item), " %08x", word);
                text += item;
            }
            std::fprintf(stderr, "[native-shadow] user data of shader %llx (record %zu):%s\n", static_cast<unsigned long long>(decodedShaders[i]), static_cast<std::size_t>(slot - record.shaders.begin()), text.c_str());
        }
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
    std::string offsets;
    for (std::size_t offset = 0; offset < state.dataOffsets.size(); ++offset) {
        if (state.dataOffsets[offset] == 0) continue;
        char item[32];
        std::snprintf(item, sizeof(item), " +%zu:%llu", offset, static_cast<unsigned long long>(state.dataOffsets[offset]));
        offsets += item;
    }
    std::fprintf(stderr, "[native-shadow] 10 s: %llu draw packets noted; decoded draws: every shader in the record %llu, not %llu, no record %llu; decoded stages without a registered header %llu; record slots used:%s; user data per program: equal %llu, equal at a word offset %llu (by offset:%s), different %llu, none in the record %llu\n", static_cast<unsigned long long>(state.noted), static_cast<unsigned long long>(state.covered), static_cast<unsigned long long>(state.uncovered), static_cast<unsigned long long>(state.missing), static_cast<unsigned long long>(state.unregistered), slots.c_str(), static_cast<unsigned long long>(state.dataEqual), static_cast<unsigned long long>(state.dataPrefix), offsets.c_str(), static_cast<unsigned long long>(state.dataDiffer), static_cast<unsigned long long>(state.dataNone));
    state.covered = state.uncovered = state.unregistered = state.missing = state.noted = 0;
    state.dataEqual = state.dataPrefix = state.dataDiffer = state.dataNone = 0;
    state.dataOffsets = {};
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
        if (AgcDriver::GuestMemory::Accessible(reinterpret_cast<const void*>(at), StageRecordBytes)) {
            const auto words = *reinterpret_cast<const std::uint64_t*>(at + 0x10);
            const auto count = *reinterpret_cast<const std::uint32_t*>(at + 0x38);
            if (words >= 0x10000 && count != 0 && count <= 64 && AgcDriver::GuestMemory::Accessible(reinterpret_cast<const void*>(words), count * 4u)) {
                record.userData[stage].assign(reinterpret_cast<const std::uint32_t*>(words), reinterpret_cast<const std::uint32_t*>(words) + count);
            }
        }
        static const bool study = [] { const char* text = std::getenv("APS5_NATIVE_SHADOW"); return text != nullptr && std::strtol(text, nullptr, 10) >= 2; }();
        static std::atomic<int> studied{0};
        if (study && studied.load(std::memory_order_relaxed) < 64 && AgcDriver::GuestMemory::Accessible(reinterpret_cast<const void*>(at), StageRecordBytes)) {
            std::array<std::uint64_t, 14> bytes{};
            std::memcpy(bytes.data(), reinterpret_cast<const void*>(at), StageRecordBytes);
            record.raw.emplace_back(static_cast<std::uint8_t>(stage), bytes);
        }
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
