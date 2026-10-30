#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Diagnostics.hpp"
#include "UserDataUse.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>

namespace AgcDriver::DriverDetail {

// Relocated draw keys (APS5_RELOCATED_DRAW_KEYS=1). PPSA03671 writes a fresh copy of the same
// resource tables for every draw, so 4-6 user-data pointers change on every draw while what the
// stages capture through them stays equal (APS5_PROFILE_CAPTURE_DIFF, 2026-10-23: 0.0-1.6% of the
// captured words differ between a program's draws, and every region lies at the same pointer and
// offset). The register key hashes every user word, so ~98% of draws missed and took the full
// capture (~40 us on queue 0). A stage whose every run lies within 4 KiB after one of its pointers
// is kept relocated: the draw's key leaves those pointer words out, and a later draw compares the
// runs at its own pointers plus the same offsets. Stages captured through other addresses keep
// their user words in the key; an entry is not inserted under a key that leaves out the words of
// a stage it keeps absolute.

namespace {

constexpr std::uint64_t RelocationWindow = 0x1000;

std::uint64_t PointerAt(std::span<const std::uint32_t> userData, std::size_t word) {
    return static_cast<std::uint64_t>(userData[word]) | (static_cast<std::uint64_t>(userData[word + 1] & 0xffffu) << 32u);
}

// The nearest user-data pointer (an even word and the next) the address lies within the window
// after, or -1.
int PointerFor(std::span<const std::uint32_t> userData, std::uint64_t address) {
    int found = -1;
    std::uint64_t best = RelocationWindow;
    for (std::size_t word = 0; word + 1 < userData.size(); word += 2) {
        const auto pointer = PointerAt(userData, word);
        if (pointer == 0 || address < pointer || address - pointer >= best) continue;
        best = address - pointer;
        found = static_cast<int>(word);
    }
    return found;
}

std::uint32_t RegisterOf(const DrawProgram& program, std::uint32_t word) {
    return DrawUserWordRegister(program, word);
}

}

// A merged stage's prefix (DecodeGraphicsPrograms): words 0-1 are the merged user-data pointer at
// base - 0x0a (0x82 or 0x102), words 2-7 the driver's, words 8.. base + (word - 8). A pixel program
// also starts at SGPR 0 but has no prefix: word w is base + w (it was mapped as merged before
// 2026-10-30, so relocated keys refreshed and masked the wrong registers for it).
std::uint32_t DrawUserWordRegister(const DrawProgram& program, std::uint32_t word) {
    if (program.firstUserSgpr == 8 || program.binary.stage == ShaderRecompiler::ShaderStage::Fragment) return program.userDataBase + word;
    if (program.firstUserSgpr != 0) return 0;
    if (word >= 8) return program.userDataBase + (word - 8);
    if (word < 2) return program.userDataBase - 0x0a + word;
    return 0;
}

bool Driver::relocatedDrawKeys() {
    static const bool on = std::getenv("APS5_RELOCATED_DRAW_KEYS") != nullptr;
    return on;
}

bool Driver::relocatedRuns(const DispatchVariant& variant, std::span<const std::uint32_t> userData, std::vector<std::pair<std::uint64_t, std::uint64_t>>& runs) {
    runs.clear();
    runs.reserve(variant.runs.size());
    for (std::size_t i = 0; i < variant.runs.size(); ++i) {
        // A run no pointer covers stays where it was (learning checked it does not move).
        if (variant.runPointerWords[i] < 0) {
            runs.push_back(variant.runs[i]);
            continue;
        }
        const auto word = static_cast<std::size_t>(variant.runPointerWords[i]);
        if (word + 1 >= userData.size()) return false;
        const auto pointer = PointerAt(userData, word);
        if (pointer == 0) return false;
        const auto begin = pointer + (variant.runs[i].first - variant.runPointerValues[i]);
        runs.emplace_back(begin, begin + (variant.runs[i].second - variant.runs[i].first));
    }
    return true;
}

void Driver::appendEntryRegionsAt(const DispatchVariant& variant, const std::vector<std::pair<std::uint64_t, std::uint64_t>>& runs, std::vector<ShaderRecompiler::MemoryRegion>& regions) {
    regions.reserve(regions.size() + runs.size());
    std::size_t offset = 0;
    for (const auto& [begin, end] : runs) {
        const auto count = static_cast<std::size_t>((end - begin) / sizeof(std::uint32_t));
        regions.push_back({begin, std::as_bytes(std::span<const std::uint32_t>(variant.words).subspan(offset, count))});
        offset += count;
    }
}

void Driver::refreshRelocatedUserWords(const QueueState& queue, const std::vector<std::uint32_t>& relocated, std::vector<DrawProgram>& programs) {
    for (auto& program : programs) {
        for (std::uint32_t word = 0; word < program.userData.size(); ++word) {
            const auto offset = RegisterOf(program, word);
            if (offset != 0 && std::binary_search(relocated.begin(), relocated.end(), offset)) program.userData[word] = readUserData(queue.shader, offset);
        }
    }
}

namespace {

std::atomic<std::uint64_t> patchConfirmed{0}, patchRefused{0}, patchHits{0}, patchVerifyMismatches{0};
std::atomic<std::uint64_t> planNs{0}, patchNs{0};
std::array<std::atomic<std::uint64_t>, 64> uncoveredDistance{};

std::uint64_t NowNs() {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}

}

std::vector<DispatchVariant::UserWordPatch> Driver::userWordPatchesFor(const ShaderRecompiler::RecompileResult& result, std::span<const std::uint32_t> userData) {
    std::vector<DispatchVariant::UserWordPatch> patches;
    const auto scan = [&](std::span<const std::uint32_t> words, std::uint32_t binding) {
        for (std::size_t word = 0; word + 1 < words.size(); ++word) {
            for (std::size_t user = 0; user + 1 < userData.size(); user += 2) {
                if (PointerAt(userData, user) < 0x10000 || words[word] != userData[user] || words[word + 1] != userData[user + 1]) continue;
                patches.push_back({binding, static_cast<std::uint32_t>(word), static_cast<std::uint32_t>(user)});
                patches.push_back({binding, static_cast<std::uint32_t>(word + 1), static_cast<std::uint32_t>(user + 1)});
                break;
            }
        }
    };
    for (std::size_t binding = 0; binding < result.bindings.size(); ++binding) scan(result.bindings[binding].guestDescriptor, static_cast<std::uint32_t>(binding));
    std::vector<std::uint32_t> push(result.pushConstants.size() / sizeof(std::uint32_t));
    std::memcpy(push.data(), result.pushConstants.data(), push.size() * sizeof(std::uint32_t));
    scan(push, DispatchVariant::UserWordPatch::NoBinding);
    return patches;
}

// The shader data's leading words are the stage's user words (ShaderDataDwordsFor), in push data
// or in a ShaderData binding: a patched copy always carries the draw's own.
std::vector<DispatchVariant::UserWordPatch> Driver::shaderDataPatchesFor(const ShaderRecompiler::RecompileResult& result) {
    std::vector<DispatchVariant::UserWordPatch> patches;
    for (std::size_t i = 0; i < result.shaderDataUserWords.size(); ++i) {
        const auto user = result.shaderDataUserWords[i];
        if ((i + 1) * sizeof(std::uint32_t) <= result.pushConstants.size()) patches.push_back({DispatchVariant::UserWordPatch::NoBinding, static_cast<std::uint32_t>(i), user});
        for (std::size_t binding = 0; binding < result.bindings.size(); ++binding) {
            if (result.bindings[binding].role == ShaderRecompiler::DescriptorRole::ShaderData && i < result.bindings[binding].guestDescriptor.size()) patches.push_back({static_cast<std::uint32_t>(binding), static_cast<std::uint32_t>(i), user});
        }
    }
    return patches;
}

std::shared_ptr<const ShaderRecompiler::RecompileResult> Driver::patchedResult(const DispatchVariant& variant, std::span<const std::uint32_t> userData) {
    const auto started = NowNs();
    struct Timed {
        std::uint64_t started;
        ~Timed() { patchNs += NowNs() - started; }
    } timed{started};
    auto patched = std::make_shared<ShaderRecompiler::RecompileResult>(*variant.compiled);
    for (const auto& patch : variant.userWordPatches) {
        if (patch.userWord >= userData.size()) return nullptr;
        const auto value = userData[patch.userWord];
        if (patch.binding == DispatchVariant::UserWordPatch::NoBinding) {
            if ((patch.word + 1) * sizeof(std::uint32_t) > patched->pushConstants.size()) return nullptr;
            std::memcpy(patched->pushConstants.data() + patch.word * sizeof(std::uint32_t), &value, sizeof(value));
        } else {
            if (patch.binding >= patched->bindings.size() || patch.word >= patched->bindings[patch.binding].guestDescriptor.size()) return nullptr;
            patched->bindings[patch.binding].guestDescriptor[patch.word] = value;
        }
    }
    return patched;
}

bool Driver::sameCompiledResult(const ShaderRecompiler::RecompileResult& a, const ShaderRecompiler::RecompileResult& b) {
    if (a.spirv.KeyHashes() != b.spirv.KeyHashes()) return false;
    if (a.specializationId != b.specializationId || a.variantId != b.variantId || a.pushConstants != b.pushConstants || a.shaderDataUserWords != b.shaderDataUserWords || a.bindings.size() != b.bindings.size()) return false;
    for (std::size_t i = 0; i < a.bindings.size(); ++i) {
        const auto& x = a.bindings[i];
        const auto& y = b.bindings[i];
        if (x.kind != y.kind || x.role != y.role || x.descriptorSet != y.descriptorSet || x.binding != y.binding || x.count != y.count || x.readOnly != y.readOnly || x.guestDescriptor != y.guestDescriptor || x.imageShape != y.imageShape) return false;
    }
    return true;
}

void Driver::notePatchOutcome(int outcome) {
    switch (outcome) {
        case 1: ++patchConfirmed; break;
        case -1: ++patchRefused; break;
        case 2: ++patchHits; break;
        case 3: ++patchVerifyMismatches; break;
        default: break;
    }
}

bool Driver::relocateVariant(DispatchVariant& variant, std::span<const std::uint32_t> userData) {
    variant.runPointerWords.clear();
    variant.runPointerValues.clear();
    // Runs no pointer covers stay absolute (-1): a stage's static tables, reached through pointers
    // its captured words hold. The learning draw checks they did not move.
    for (const auto& run : variant.runs) {
        const auto word = PointerFor(userData, run.first);
        variant.runPointerWords.push_back(static_cast<std::int16_t>(word));
        variant.runPointerValues.push_back(word < 0 ? 0 : PointerAt(userData, static_cast<std::size_t>(word)));
    }
    return !variant.runs.empty();
}

Driver::DrawRelocationPlan Driver::planDrawRelocation(const std::function<std::uint64_t(const std::vector<std::uint32_t>*)>& keyFor, const std::vector<DrawProgram>& programs, const std::vector<StageCapture>& stageCaptures, const std::vector<std::vector<Graphics::DecodeRead>>& decodeReads, std::uint64_t shapeKey, const std::shared_ptr<const std::vector<std::uint32_t>>& relocation, std::uint64_t drawKey) {
    struct Timed {
        std::uint64_t started = NowNs();
        ~Timed() { planNs += NowNs() - started; }
    } timed;
    DrawRelocationPlan plan;
    plan.insertKey = drawKey;
    plan.stages.assign(programs.size(), false);
    plan.patches.assign(programs.size(), {});
    std::vector<std::uint32_t> mask;
    // Per stage, when this draw ran its capture walk: every captured address must lie after one of
    // its pointers (the runs move with them), and the key keeps only the user words the walk
    // specialized on (asValue); words the walk read through or did not use leave it.
    // A stage kept absolute: its user-word registers (they must all stay in the key).
    std::vector<std::uint32_t> absoluteRegisters;
    std::uint64_t relocatedStages = 0, maskedStages = 0;
    std::array<std::uint64_t, 4> absoluteBy{};
    const auto keepAbsolute = [&](const DrawProgram& program, std::size_t reason) {
        for (std::uint32_t word = 0; word < program.userData.size(); ++word) {
            if (const auto offset = RegisterOf(program, word); offset != 0) absoluteRegisters.push_back(offset);
        }
        ++absoluteBy[reason];
    };
    for (std::size_t i = 0; i < programs.size() && i < stageCaptures.size(); ++i) {
        const auto& program = programs[i];
        const auto& capture = stageCaptures[i];
        const auto size = static_cast<std::uint32_t>(program.userData.size());
        if (program.firstUserSgpr != 8 && program.firstUserSgpr != 0) {
            keepAbsolute(program, 0);
            continue;
        }
        if (!capture.captured) {
            keepAbsolute(program, 1);
            continue;
        }
        if (size > ShaderRecompiler::SrtUserDataUse::TrackedWords || capture.userWordsAsValue == ~std::uint64_t{0}) {
            keepAbsolute(program, 2);
            continue;
        }
        std::vector<std::uint64_t> addresses;
        for (const auto& region : capture.regions) addresses.push_back(region.guestAddress);
        if (i < decodeReads.size()) {
            for (const auto& read : decodeReads[i]) addresses.push_back(read.address);
        }
        const bool all = std::all_of(addresses.begin(), addresses.end(), [&](std::uint64_t address) { return PointerFor(program.userData, address) >= 0; });
        static const bool mixed = std::getenv("APS5_NO_MIXED_RELOCATION") == nullptr;
        if (!all && !mixed) {
            // Where the runs no pointer covers lie: log2 of the distance above the nearest pointer
            // below them (bucket 63: no pointer below), printed with the [draw-relocation] line.
            for (const auto address : addresses) {
                if (PointerFor(program.userData, address) >= 0) continue;
                std::uint64_t nearest = ~std::uint64_t{0};
                for (std::size_t word = 0; word + 1 < program.userData.size(); word += 2) {
                    const auto pointer = PointerAt(program.userData, word);
                    if (pointer >= 0x10000 && address >= pointer) nearest = std::min(nearest, address - pointer);
                }
                ++uncoveredDistance[nearest == ~std::uint64_t{0} ? 63 : std::bit_width(nearest)];
            }
            keepAbsolute(program, 3);
            continue;
        }
        plan.stages[i] = !addresses.empty();
        ++(addresses.empty() ? maskedStages : relocatedStages);
        // A value-used pointer the compiled result holds verbatim (a buffer descriptor in user
        // SGPRs) leaves the key too: a hit gets a copy with its own pointer there, once confirmed.
        std::vector<DispatchVariant::UserWordPatch> patches;
        if (capture.compiled != nullptr) patches = userWordPatchesFor(*capture.compiled, program.userData);
        std::uint64_t patchedWords = 0;
        for (const auto& patch : patches) patchedWords |= std::uint64_t{1} << patch.userWord;
        for (std::uint32_t word = 0; word < size; ++word) {
            const auto offset = RegisterOf(program, word);
            if (offset != 0 && (((capture.userWordsAsValue >> word) & 1u) == 0 || ((patchedWords >> word) & 1u) != 0)) mask.push_back(offset);
        }
        // Words in the shader data are patched too, but stay keyed when value-used.
        if (capture.compiled != nullptr) {
            const auto data = shaderDataPatchesFor(*capture.compiled);
            patches.insert(patches.end(), data.begin(), data.end());
        }
        plan.patches[i] = std::move(patches);
    }
    {
        static std::atomic<std::uint64_t> plans{0}, relocated{0}, masked{0}, otherSgpr{0}, uncaptured{0}, untracked{0}, unrelated{0};
        static std::atomic<std::int64_t> reportedAt{0};
        ++plans;
        relocated += relocatedStages;
        masked += maskedStages;
        otherSgpr += absoluteBy[0];
        uncaptured += absoluteBy[1];
        untracked += absoluteBy[2];
        unrelated += absoluteBy[3];
        const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
        auto last = reportedAt.load();
        if (last == 0) reportedAt.compare_exchange_strong(last, now);
        else if (now - last >= 10'000'000'000ll && reportedAt.compare_exchange_strong(last, now)) {
            const auto take = [](std::atomic<std::uint64_t>& counter) { return static_cast<unsigned long long>(counter.exchange(0)); };
            std::string distances;
            for (std::size_t bucket = 0; bucket < uncoveredDistance.size(); ++bucket) {
                if (const auto count = uncoveredDistance[bucket].exchange(0); count != 0) distances += " 2^" + std::to_string(bucket) + ":" + std::to_string(count);
            }
            std::fprintf(stderr, "[draw-relocation] uncovered runs by distance above the nearest pointer (2^63 = none):%s\n", distances.c_str());
            std::fprintf(stderr, "[draw-relocation] (10 s) %llu plans: stages relocated %llu, masked without runs %llu; kept absolute: user SGPRs not at 0/8 %llu, not captured this draw %llu, untracked words %llu, runs not after a pointer %llu; %zu shapes; pointer patches: %llu confirmed, %llu refused, %llu patched stage hits, %llu verify mismatches; plans %.1f ms, patched copies %.1f ms\n", take(plans), take(relocated), take(masked), take(otherSgpr), take(uncaptured), take(untracked), take(unrelated), [&] { std::lock_guard cacheLock(drawCacheMutex); return drawRelocations.size(); }(), take(patchConfirmed), take(patchRefused), take(patchHits), take(patchVerifyMismatches), static_cast<double>(planNs.exchange(0)) / 1e6, static_cast<double>(patchNs.exchange(0)) / 1e6);
        }
    }
    std::sort(mask.begin(), mask.end());
    mask.erase(std::unique(mask.begin(), mask.end()), mask.end());
    std::sort(absoluteRegisters.begin(), absoluteRegisters.end());
    const auto effective = relocation != nullptr ? *relocation : mask;
    // Diagnostic (APS5_PROFILE_RELOCATION_MISSES): a draw of a known shape that missed - which of the
    // user words its key kept changed since the shape's previous miss (stage:word, value-used v,
    // in a stage kept absolute a), every 10 s.
    if (static const bool missProfile = std::getenv("APS5_PROFILE_RELOCATION_MISSES") != nullptr; missProfile && relocation != nullptr) {
        static std::mutex profileMutex;
        static std::unordered_map<std::uint64_t, std::vector<std::vector<std::uint32_t>>> last;
        static std::map<std::string, std::uint64_t> changed;
        static std::uint64_t misses = 0, sameWords = 0;
        static auto windowStart = std::chrono::steady_clock::now();
        std::lock_guard profileLock(profileMutex);
        ++misses;
        auto& previous = last[shapeKey];
        std::vector<std::vector<std::uint32_t>> now(programs.size());
        bool any = false;
        for (std::size_t i = 0; i < programs.size(); ++i) {
            const auto& program = programs[i];
            now[i] = program.userData;
            if (previous.size() != programs.size() || previous[i].size() != program.userData.size()) continue;
            for (std::uint32_t word = 0; word < program.userData.size(); ++word) {
                const auto offset = RegisterOf(program, word);
                if (offset == 0 || std::binary_search(relocation->begin(), relocation->end(), offset)) continue;
                if (previous[i][word] == program.userData[word]) continue;
                any = true;
                const bool value = i < stageCaptures.size() && stageCaptures[i].captured && ((stageCaptures[i].userWordsAsValue >> word) & 1u) != 0;
                const bool absolute = std::binary_search(absoluteRegisters.begin(), absoluteRegisters.end(), offset);
                ++changed[std::to_string(i) + ":" + std::to_string(word) + (value ? "v" : "") + (absolute ? "a" : "")];
            }
        }
        if (!any) ++sameWords;
        previous = std::move(now);
        const auto at = std::chrono::steady_clock::now();
        if (at - windowStart >= std::chrono::seconds(10)) {
            windowStart = at;
            std::vector<std::pair<std::uint64_t, std::string>> top;
            for (const auto& [where, count] : changed) top.emplace_back(count, where);
            std::sort(top.rbegin(), top.rend());
            std::string text;
            for (std::size_t k = 0; k < top.size() && k < 16; ++k) text += " " + top[k].second + "=" + std::to_string(top[k].first);
            std::fprintf(stderr, "[relocation-misses] (10 s) %llu misses of known shapes, %llu with no kept word changed; changed kept words:%s\n", static_cast<unsigned long long>(misses), static_cast<unsigned long long>(sameWords), text.c_str());
            changed.clear();
            misses = sameWords = 0;
        }
    }
    // A stage kept absolute must have all its user words in the key.
    for (const auto offset : absoluteRegisters) {
        if (std::binary_search(effective.begin(), effective.end(), offset)) plan.insert = false;
    }
    if (relocation != nullptr) {
        // The shape's key already leaves its words out (drawKey): only the same relocation inserts,
        // and like any key only once seen before (Driver::admitDrawKey): most misses of a known
        // shape are keys that do not come back, and inserting each cost ~12 us per draw (tr226).
        if (*relocation != mask) plan.insert = false;
        return plan;
    }
    if (mask.empty() || !plan.insert) {
        plan.stages.assign(programs.size(), false);
        plan.insert = true;
        return plan;
    }
    auto shared = std::make_shared<const std::vector<std::uint32_t>>(std::move(mask));
    plan.insertKey = keyFor(shared.get());
    plan.admit = true;
    std::lock_guard cacheLock(drawCacheMutex);
    drawRelocations.emplace(shapeKey, std::move(shared));
    return plan;
}

}
