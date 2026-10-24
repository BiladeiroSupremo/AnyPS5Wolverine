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
    // Bound descriptors by role (buffer, image, sampler): found among the record's placed ones, not.
    std::array<std::uint64_t, 3> descriptorsFound{}, descriptorsMissed{};
    int descriptorExamples = 0;
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
        // The Shader's resource layout (ShaderUserData): direct resources (u16 user-data positions)
        // and the four sharp tables (ShaderSharp {offset_dw:15, size:1}).
        const auto shader = bytes[0];
        if (shader >= 0x10000 && AgcDriver::GuestMemory::Accessible(reinterpret_cast<const void*>(shader), sizeof(Shader))) {
            const auto* userData = reinterpret_cast<const Shader*>(shader)->user_data;
            if (userData != nullptr && AgcDriver::GuestMemory::Accessible(userData, sizeof(ShaderUserData))) {
                std::snprintf(item, sizeof(item), "\n    layout: eud %u srt %u direct %u:", userData->eud_size_dw, userData->srt_size_dw, userData->direct_resource_count);
                text += item;
                if (userData->direct_resource_offset != nullptr && AgcDriver::GuestMemory::Accessible(userData->direct_resource_offset, userData->direct_resource_count * 2u)) {
                    for (std::uint32_t i = 0; i < userData->direct_resource_count && i < 32; ++i) {
                        std::snprintf(item, sizeof(item), " %x", userData->direct_resource_offset[i]);
                        text += item;
                    }
                }
                for (std::uint32_t table = 0; table < 4; ++table) {
                    const auto count = userData->sharp_resource_count[table];
                    std::snprintf(item, sizeof(item), "; sharp[%u] %u:", table, count);
                    text += item;
                    const auto* sharps = userData->sharp_resource_offset[table];
                    if (sharps == nullptr || count == 0 || !AgcDriver::GuestMemory::Accessible(sharps, count * 2u)) continue;
                    for (std::uint32_t i = 0; i < count && i < 24; ++i) {
                        std::snprintf(item, sizeof(item), " %u/%u", sharps[i].offset_dw, sharps[i].size);
                        text += item;
                    }
                }
            }
        }
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

void CompareNativeDescriptors(const NativeDrawRecord& record, std::span<const DecodedDescriptor> decoded) {
    // The descriptors the record's Shaders place: first four words of each sharp entry.
    std::vector<std::array<std::uint32_t, 4>> placed;
    for (std::size_t slot = 0; slot < record.shaders.size(); ++slot) {
        const auto shader = record.shaders[slot];
        if (shader == 0) continue;
        const auto* layout = reinterpret_cast<const Shader*>(shader)->user_data;
        if (layout == nullptr || !AgcDriver::GuestMemory::Accessible(layout, sizeof(ShaderUserData))) continue;
        const auto& words = record.userData[slot];
        const auto& extended = record.eud[slot];
        for (std::uint32_t table = 0; table < 4; ++table) {
            const auto count = layout->sharp_resource_count[table];
            const auto* sharps = layout->sharp_resource_offset[table];
            if (sharps == nullptr || count == 0 || !AgcDriver::GuestMemory::Accessible(sharps, count * 2u)) continue;
            for (std::uint32_t i = 0; i < count; ++i) {
                const std::uint32_t offset = sharps[i].offset_dw;
                if (offset == 0x7fffu) continue;
                const auto& source = offset < 32 ? words : extended;
                const auto at = offset < 32 ? offset : offset - 32;
                if (at + 4 > source.size()) continue;
                placed.push_back({source[at], source[at + 1], source[at + 2], source[at + 3]});
            }
        }
    }
    auto& state = Shadow();
    std::lock_guard lock(state.mutex);
    for (const auto& descriptor : decoded) {
        if (descriptor.role > 2) continue;
        if (std::find(placed.begin(), placed.end(), descriptor.words) != placed.end()) {
            ++state.descriptorsFound[descriptor.role];
            continue;
        }
        ++state.descriptorsMissed[descriptor.role];
        if (state.descriptorExamples < 40) {
            ++state.descriptorExamples;
            // Where the descriptor lives: searched in the memory each user-data word pair points at
            // (as a raw pointer, or as a V#'s base with its size), up to 1 MiB.
            std::string found;
            for (std::size_t slot = 0; slot < record.shaders.size() && found.empty(); ++slot) {
                for (const auto* source : {&record.userData[slot], &record.eud[slot]}) {
                    for (std::size_t k = 0; k + 1 < source->size() && found.empty(); ++k) {
                        const auto low = (*source)[k];
                        const auto high = (*source)[k + 1];
                        std::array<std::pair<std::uint64_t, std::uint64_t>, 2> targets{{
                            {static_cast<std::uint64_t>(low) | (static_cast<std::uint64_t>(high) << 32u), 0x10000},
                            {static_cast<std::uint64_t>(low) | (static_cast<std::uint64_t>(high & 0xffffu) << 32u), k + 2 < source->size() ? std::min<std::uint64_t>(static_cast<std::uint64_t>((*source)[k + 2]) * 16u, 0x100000) : 0}}};
                        for (const auto& [base, bytes] : targets) {
                            if (base < 0x100000 || bytes < 16 || (base & 3u) != 0 || !AgcDriver::GuestMemory::Accessible(reinterpret_cast<const void*>(base), static_cast<std::size_t>(bytes))) continue;
                            const auto* words = reinterpret_cast<const std::uint32_t*>(base);
                            for (std::uint64_t i = 0; i + 4 <= bytes / 4; ++i) {
                                if (words[i] == descriptor.words[0] && words[i + 1] == descriptor.words[1] && words[i + 2] == descriptor.words[2] && words[i + 3] == descriptor.words[3]) {
                                    char text[160];
                                    std::snprintf(text, sizeof(text), "found via record %zu %s word %zu (%s 0x%llx) at +0x%llx", slot, source == &record.userData[slot] ? "user data" : "eud", k, bytes == 0x10000 ? "pointer" : "V# base", static_cast<unsigned long long>(base), static_cast<unsigned long long>(i * 4));
                                    found = text;
                                    break;
                                }
                            }
                            if (!found.empty()) break;
                        }
                    }
                    if (!found.empty()) break;
                }
            }
            std::fprintf(stderr, "[native-shadow] bound %s %08x %08x %08x %08x not placed by the layouts (%zu placed): %s\n", descriptor.role == 0 ? "buffer" : descriptor.role == 1 ? "image" : "sampler", descriptor.words[0], descriptor.words[1], descriptor.words[2], descriptor.words[3], placed.size(), found.empty() ? "not found behind any user-data pointer" : found.c_str());
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
    std::fprintf(stderr, "[native-shadow] bound descriptors placed by the title's Shader layouts: buffers %llu of %llu, images %llu of %llu, samplers %llu of %llu\n", static_cast<unsigned long long>(state.descriptorsFound[0]), static_cast<unsigned long long>(state.descriptorsFound[0] + state.descriptorsMissed[0]), static_cast<unsigned long long>(state.descriptorsFound[1]), static_cast<unsigned long long>(state.descriptorsFound[1] + state.descriptorsMissed[1]), static_cast<unsigned long long>(state.descriptorsFound[2]), static_cast<unsigned long long>(state.descriptorsFound[2] + state.descriptorsMissed[2]));
    state.covered = state.uncovered = state.unregistered = state.missing = state.noted = 0;
    state.dataEqual = state.dataPrefix = state.dataDiffer = state.dataNone = 0;
    state.dataOffsets = {};
    state.descriptorsFound = {};
    state.descriptorsMissed = {};
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
            const auto extended = *reinterpret_cast<const std::uint64_t*>(at + 0x18);
            const auto* layout = reinterpret_cast<const Shader*>(shader)->user_data;
            if (extended >= 0x10000 && layout != nullptr && AgcDriver::GuestMemory::Accessible(layout, sizeof(ShaderUserData))) {
                const auto size = std::min<std::uint32_t>(layout->eud_size_dw, 256);
                if (size != 0 && AgcDriver::GuestMemory::Accessible(reinterpret_cast<const void*>(extended), size * 4u)) record.eud[stage].assign(reinterpret_cast<const std::uint32_t*>(extended), reinterpret_cast<const std::uint32_t*>(extended) + size);
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
