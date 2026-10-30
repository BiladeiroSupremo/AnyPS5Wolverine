#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Diagnostics.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawAhead.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPipeline.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Synchronization/DeferredLabels.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/PerformanceTimer.hpp"
#include "prx/libSceAgcDriver/Execution/include/Pm4.hpp"
#include "Optimization/ResourceProgram.hpp"
#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <exception>
#include <algorithm>
#include <cstdlib>
#include <mutex>
#include <set>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace AgcDriver::DriverDetail {

namespace {

// APS5_NATIVE_DRAWS (native path N2, see Driver::draw): a walk-free shape's stages as first
// materialized, the patches that carry a draw's user words into them, the program and the vertex
// stage info they were built for (a hit needs the same; empty for a GeometryBack role).
struct NativeDrawStage {
    std::shared_ptr<const ShaderRecompiler::RecompileResult> result;
    std::vector<DispatchVariant::UserWordPatch> patches;
    std::uint64_t program = 0;
    std::optional<ShaderRecompiler::ShaderVertexStageInfo> vertexInfo;
};

struct NativeDrawCache {
    std::mutex mutex;
    std::unordered_map<std::uint64_t, std::shared_ptr<const std::vector<NativeDrawStage>>> shapes;
    std::uint64_t hits = 0, stored = 0, refused = 0, walking = 0;
    // APS5_NATIVE_DRAWS=verify: stages whose reused copy equals / differs from the fresh one.
    std::uint64_t verifyEqual = 0, verifyDiffer = 0, verifyReports = 0;
    std::chrono::steady_clock::time_point reported = std::chrono::steady_clock::now();
};

NativeDrawCache& nativeDrawCache() {
    static auto* cache = new NativeDrawCache();
    return *cache;
}

// Whether two vertex stage infos have the same shape: everything but each V#'s base address and
// record count (word 0, word 1 bits 0-15, word 2), which the reused stage takes from this draw
// (nativePatchedResult). The compiled stage depends on the rest (formats and selectors in word 3,
// stride and swizzle in word 1, the fetch layout).
bool sameVertexShape(const ShaderRecompiler::ShaderVertexStageInfo& a, const ShaderRecompiler::ShaderVertexStageInfo& b) {
    if (a.resourcesNum != b.resourcesNum || a.fetchAttribReg != b.fetchAttribReg || a.fetchBufferReg != b.fetchBufferReg || a.fetchEmbedded != b.fetchEmbedded || a.paClVsOutCntl != b.paClVsOutCntl) return false;
    for (std::uint32_t i = 0; i < a.resourcesNum && i < a.resources.size(); ++i) {
        if (a.resources[i].fields[3] != b.resources[i].fields[3] || (a.resources[i].fields[1] >> 16u) != (b.resources[i].fields[1] >> 16u)) return false;
        const auto& x = a.resourcesDst[i];
        const auto& y = b.resourcesDst[i];
        if (x.registerStart != y.registerStart || x.registersNum != y.registersNum || x.attrId != y.attrId || x.fetchIndex != y.fetchIndex) return false;
    }
    return true;
}

// The cached stage with this draw's per-draw values: its user words patched in, and its vertex
// attributes' V#s (the compiled result keeps each attribute's whole V#, base address included:
// Recompiler.cpp materializeResult) taken from this draw's vertex stage info. The stage itself when
// there is nothing to change; null when a patch or an attribute lies outside it.
std::shared_ptr<const ShaderRecompiler::RecompileResult> nativePatchedResult(const NativeDrawStage& stage, std::span<const std::uint32_t> userData, const std::optional<ShaderRecompiler::ShaderVertexStageInfo>& vertex) {
    if (stage.patches.empty() && stage.result->vertexAttributes.empty()) return stage.result;
    auto patched = std::make_shared<ShaderRecompiler::RecompileResult>(*stage.result);
    for (auto& attribute : patched->vertexAttributes) {
        if (!vertex || attribute.location >= vertex->resourcesNum || attribute.location >= vertex->resources.size()) return nullptr;
        attribute.resource = vertex->resources[attribute.location];
    }
    for (const auto& patch : stage.patches) {
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

// APS5_NATIVE_CENSUS (see Driver::draw): shapes seen (capped, then forgotten) and 10 s counters.
void noteNativeCensus(std::uint64_t shapeKey, bool drawHit, const std::vector<StageCapture>& captures, const std::vector<ShaderRecompiler::ProgramRole>& roles) {
    struct Census {
        std::mutex mutex;
        std::unordered_set<std::uint64_t> seen;
        std::uint64_t draws = 0, cacheHits = 0, repeats = 0, walkFree = 0, walkFreeRepeats = 0, walkingRepeats = 0, repeatWords = 0, uncaptured = 0;
        std::chrono::steady_clock::time_point reported = std::chrono::steady_clock::now();
    };
    static auto* census = new Census();
    bool walkFree = true;
    bool allCaptured = true;
    std::uint64_t words = 0;
    for (std::size_t i = 0; i < captures.size(); ++i) {
        if (i < roles.size() && roles[i] == ShaderRecompiler::ProgramRole::GeometryBack) continue;
        if (!captures[i].captured) {
            allCaptured = false;
            continue;
        }
        for (const auto& region : captures[i].regions) words += region.bytes.size() / sizeof(std::uint32_t);
        if (!captures[i].regions.empty()) walkFree = false;
    }
    std::lock_guard lock(census->mutex);
    ++census->draws;
    if (census->seen.size() > (1u << 20)) census->seen.clear();
    const bool repeat = !census->seen.insert(shapeKey).second;
    if (drawHit) ++census->cacheHits;
    else if (!allCaptured) ++census->uncaptured;
    else {
        if (repeat) ++census->repeats;
        if (walkFree) ++census->walkFree;
        if (repeat && walkFree) ++census->walkFreeRepeats;
        if (repeat && !walkFree) {
            ++census->walkingRepeats;
            census->repeatWords += words;
        }
    }
    const auto now = std::chrono::steady_clock::now();
    if (now - census->reported < std::chrono::seconds(10)) return;
    census->reported = now;
    std::fprintf(stderr, "[native-census] (10 s) %llu draws: %llu draw-cache hits, %llu not all captured; of the captured: shape seen before %llu, walk-free %llu, both %llu (reuse with values rebuilt), walking repeats %llu (%.1f words to gather each); %zu shapes known\n", static_cast<unsigned long long>(census->draws), static_cast<unsigned long long>(census->cacheHits), static_cast<unsigned long long>(census->uncaptured), static_cast<unsigned long long>(census->repeats), static_cast<unsigned long long>(census->walkFree), static_cast<unsigned long long>(census->walkFreeRepeats), static_cast<unsigned long long>(census->walkingRepeats), census->walkingRepeats == 0 ? 0.0 : static_cast<double>(census->repeatWords) / static_cast<double>(census->walkingRepeats), census->seen.size());
    census->draws = census->cacheHits = census->repeats = census->walkFree = census->walkFreeRepeats = census->walkingRepeats = census->repeatWords = census->uncaptured = 0;
}

// APS5_NATIVE_VERIFY_PATCH (see Driver::draw): a stage as materialized, the patches that carry its
// user words into it, the user words and the guest regions (address, bytes) it was captured with.
struct NativePatchStage {
    std::shared_ptr<const ShaderRecompiler::RecompileResult> result;
    std::vector<DispatchVariant::UserWordPatch> patches;
    std::vector<std::uint32_t> userData;
    std::uint64_t program = 0;
    std::vector<std::pair<std::uint64_t, std::vector<std::byte>>> regions;
};

// The user-data pointer (even word and the next, 48 bits) an address lies within 4 KiB after, or -1
// (the relocated-key rule, DrawRelocation.cpp).
int NativePointerFor(const std::vector<std::uint32_t>& userData, std::uint64_t address) {
    int found = -1;
    std::uint64_t best = 0x1000;
    for (std::size_t word = 0; word + 1 < userData.size(); word += 2) {
        const auto pointer = static_cast<std::uint64_t>(userData[word]) | (static_cast<std::uint64_t>(userData[word + 1] & 0xffffu) << 32u);
        if (pointer == 0 || address < pointer || address - pointer >= best) continue;
        best = address - pointer;
        found = static_cast<int>(word);
    }
    return found;
}

std::uint64_t NativePointerAt(const std::vector<std::uint32_t>& userData, int word) {
    return static_cast<std::uint64_t>(userData[static_cast<std::size_t>(word)]) | (static_cast<std::uint64_t>(userData[static_cast<std::size_t>(word) + 1] & 0xffffu) << 32u);
}

void noteNativePatchVerify(std::uint64_t shapeKey, std::vector<NativePatchStage> now) {
    // Per kind (0 walk-free stages, 1 walking stages).
    struct Counters {
        std::uint64_t stages = 0, equal = 0, program = 0, layout = 0, push = 0, unpatchable = 0;
        std::array<std::uint64_t, 8> descriptor{};
    };
    struct Verify {
        std::mutex mutex;
        std::unordered_map<std::uint64_t, std::vector<NativePatchStage>> shapes;
        std::uint64_t draws = 0;
        std::array<Counters, 2> kinds{};
        // A walking stage the relocated compare would miss: region count, a region at another
        // place than its pointer + offset (or a static one that moved), bytes differ.
        std::uint64_t missCount = 0, missMoved = 0, missBytes = 0;
        int examples = 0;
        std::chrono::steady_clock::time_point reported = std::chrono::steady_clock::now();
    };
    static auto* verify = new Verify();
    static constexpr std::array<const char*, 8> RoleNames{"buffers", "images", "samplers", "gds", "pagetable", "fault", "flattened-srt", "shader-data"};
    std::lock_guard lock(verify->mutex);
    if (verify->shapes.size() > 8192) verify->shapes.clear();
    auto& before = verify->shapes[shapeKey];
    if (before.size() == now.size()) {
        ++verify->draws;
        for (std::size_t i = 0; i < now.size(); ++i) {
            const auto& earlier = before[i];
            const auto& current = now[i];
            const bool walking = !earlier.regions.empty() || !current.regions.empty();
            auto& counters = verify->kinds[walking ? 1 : 0];
            if (walking) {
                // What the relocated hit would compare: each earlier region at this draw's pointer
                // plus the same offset (or the same address when no pointer covers it), same bytes.
                if (earlier.regions.size() != current.regions.size()) {
                    ++verify->missCount;
                    continue;
                }
                bool moved = false, differ = false;
                for (std::size_t r = 0; r < earlier.regions.size() && !moved; ++r) {
                    const auto& [address, bytes] = earlier.regions[r];
                    const auto word = NativePointerFor(earlier.userData, address);
                    const auto expected = word < 0 ? address : (static_cast<std::size_t>(word) + 1 < current.userData.size() ? NativePointerAt(current.userData, word) + (address - NativePointerAt(earlier.userData, word)) : ~std::uint64_t{0});
                    if (current.regions[r].first != expected) moved = true;
                    else if (current.regions[r].second != bytes) differ = true;
                }
                if (moved) {
                    ++verify->missMoved;
                    continue;
                }
                if (differ) {
                    ++verify->missBytes;
                    continue;
                }
            }
            const auto& old = *earlier.result;
            const auto& fresh = *current.result;
            const auto& userData = current.userData;
            ++counters.stages;
            if (earlier.program != current.program || old.spirv.KeyHashes() != fresh.spirv.KeyHashes() || old.specializationId != fresh.specializationId || old.variantId != fresh.variantId) {
                ++counters.program;
                continue;
            }
            bool sameLayout = old.shaderDataUserWords == fresh.shaderDataUserWords && old.bindings.size() == fresh.bindings.size() && old.pushConstants.size() == fresh.pushConstants.size();
            for (std::size_t b = 0; sameLayout && b < old.bindings.size(); ++b) {
                const auto& x = old.bindings[b];
                const auto& y = fresh.bindings[b];
                sameLayout = x.kind == y.kind && x.role == y.role && x.descriptorSet == y.descriptorSet && x.binding == y.binding && x.count == y.count && x.readOnly == y.readOnly && x.imageShape == y.imageShape && x.guestDescriptor.size() == y.guestDescriptor.size();
            }
            if (!sameLayout) {
                ++counters.layout;
                continue;
            }
            // The earlier stage's per-draw values with this draw's user words patched in.
            auto pushes = old.pushConstants;
            std::vector<std::vector<std::uint32_t>> descriptors;
            for (const auto& binding : old.bindings) descriptors.push_back(binding.guestDescriptor);
            bool patchable = true;
            for (const auto& patch : earlier.patches) {
                if (patch.userWord >= userData.size()) {
                    patchable = false;
                    break;
                }
                const auto value = userData[patch.userWord];
                if (patch.binding == DispatchVariant::UserWordPatch::NoBinding) {
                    if ((patch.word + 1) * sizeof(std::uint32_t) > pushes.size()) patchable = false;
                    else std::memcpy(pushes.data() + patch.word * sizeof(std::uint32_t), &value, sizeof(value));
                } else if (patch.binding >= descriptors.size() || patch.word >= descriptors[patch.binding].size()) {
                    patchable = false;
                } else {
                    descriptors[patch.binding][patch.word] = value;
                }
            }
            if (!patchable) {
                ++counters.unpatchable;
                continue;
            }
            bool same = true;
            if (pushes != fresh.pushConstants) {
                ++counters.push;
                same = false;
                if (verify->examples < 12) {
                    ++verify->examples;
                    std::size_t word = 0;
                    while ((word + 1) * 4 <= pushes.size() && std::memcmp(pushes.data() + word * 4, fresh.pushConstants.data() + word * 4, 4) == 0) ++word;
                    std::uint32_t patched = 0, wanted = 0;
                    if ((word + 1) * 4 <= pushes.size()) {
                        std::memcpy(&patched, pushes.data() + word * 4, 4);
                        std::memcpy(&wanted, fresh.pushConstants.data() + word * 4, 4);
                    }
                    std::fprintf(stderr, "[native-patch] %s program 0x%llx: push word %zu patched 0x%08x, fresh 0x%08x\n", walking ? "walking" : "walk-free", static_cast<unsigned long long>(current.program), word, patched, wanted);
                }
            }
            for (std::size_t b = 0; b < descriptors.size(); ++b) {
                if (descriptors[b] == fresh.bindings[b].guestDescriptor) continue;
                const auto role = std::min<std::size_t>(static_cast<std::size_t>(fresh.bindings[b].role), 7);
                ++counters.descriptor[role];
                same = false;
                if (verify->examples < 12) {
                    ++verify->examples;
                    std::size_t word = 0;
                    while (word < descriptors[b].size() && descriptors[b][word] == fresh.bindings[b].guestDescriptor[word]) ++word;
                    std::fprintf(stderr, "[native-patch] %s program 0x%llx: %s binding %zu word %zu patched 0x%08x, fresh 0x%08x\n", walking ? "walking" : "walk-free", static_cast<unsigned long long>(current.program), RoleNames[role], b, word, word < descriptors[b].size() ? descriptors[b][word] : 0u, word < descriptors[b].size() ? fresh.bindings[b].guestDescriptor[word] : 0u);
                }
            }
            if (same) ++counters.equal;
        }
    }
    before = std::move(now);
    const auto clock = std::chrono::steady_clock::now();
    if (clock - verify->reported < std::chrono::seconds(10)) return;
    verify->reported = clock;
    std::string line;
    for (std::size_t kind = 0; kind < 2; ++kind) {
        const auto& counters = verify->kinds[kind];
        std::string roles;
        for (std::size_t r = 0; r < counters.descriptor.size(); ++r) {
            if (counters.descriptor[r] != 0) roles += " " + std::string(RoleNames[r]) + " " + std::to_string(counters.descriptor[r]);
        }
        char text[320];
        std::snprintf(text, sizeof(text), "; %s stages compared %llu: patched equals fresh %llu, differs: program %llu, layout %llu, push %llu, unpatchable %llu, descriptors by role:%s", kind == 0 ? "walk-free" : "walking (relocated runs equal)", static_cast<unsigned long long>(counters.stages), static_cast<unsigned long long>(counters.equal), static_cast<unsigned long long>(counters.program), static_cast<unsigned long long>(counters.layout), static_cast<unsigned long long>(counters.push), static_cast<unsigned long long>(counters.unpatchable), roles.empty() ? " none" : roles.c_str());
        line += text;
    }
    std::fprintf(stderr, "[native-patch] (10 s) %llu repeated draws; walking stages the relocated compare would miss: region count %llu, moved %llu, bytes differ %llu%s\n", static_cast<unsigned long long>(verify->draws), static_cast<unsigned long long>(verify->missCount), static_cast<unsigned long long>(verify->missMoved), static_cast<unsigned long long>(verify->missBytes), line.c_str());
    verify->draws = verify->missCount = verify->missMoved = verify->missBytes = 0;
    verify->kinds = {};
    verify->examples = 0;
}

// Which user-data words vary from draw to draw, per program (APS5_PROFILE_USER_WORDS): every 10 s,
// for the programs drawn most, the distinct values each word took (capped at 64). The draw keys
// hash every user word, so a word that changes per draw (a per-draw constant pointer, say) makes
// each draw a new key; this shows which words those are.
struct UserWordProfile {
    std::mutex mutex;
    struct Program {
        std::uint64_t draws = 0;
        std::vector<std::set<std::uint32_t>> words;
    };
    std::unordered_map<std::uint64_t, Program> programs;
    std::chrono::steady_clock::time_point windowStart = std::chrono::steady_clock::now();
};

void ProfileUserWords(const std::vector<DrawProgram>& programs) {
    static const bool on = std::getenv("APS5_PROFILE_USER_WORDS") != nullptr;
    if (!on) return;
    static auto* state = new UserWordProfile();
    std::lock_guard lock(state->mutex);
    for (const auto& program : programs) {
        auto& entry = state->programs[program.binary.codeAddress];
        ++entry.draws;
        if (entry.words.size() < program.userData.size()) entry.words.resize(program.userData.size());
        for (std::size_t i = 0; i < program.userData.size(); ++i) {
            if (entry.words[i].size() < 64) entry.words[i].insert(program.userData[i]);
        }
    }
    const auto now = std::chrono::steady_clock::now();
    if (now - state->windowStart < std::chrono::seconds(10)) return;
    state->windowStart = now;
    std::vector<std::pair<std::uint64_t, const UserWordProfile::Program*>> top;
    std::uint64_t draws = 0;
    for (const auto& [address, entry] : state->programs) {
        top.emplace_back(address, &entry);
        draws += entry.draws;
    }
    std::sort(top.begin(), top.end(), [](const auto& a, const auto& b) { return a.second->draws > b.second->draws; });
    std::fprintf(stderr, "[user-words] %zu programs, %llu program draws (10 s); distinct values per word of the top programs:\n", top.size(), static_cast<unsigned long long>(draws));
    for (std::size_t i = 0; i < top.size() && i < 16; ++i) {
        std::string text;
        for (const auto& values : top[i].second->words) text += " " + std::to_string(values.size());
        std::fprintf(stderr, "[user-words]   0x%llx x%llu:%s\n", static_cast<unsigned long long>(top[i].first), static_cast<unsigned long long>(top[i].second->draws), text.c_str());
    }
    state->programs.clear();
}

// How a program's captured memory changes between its draws (APS5_PROFILE_CAPTURE_DIFF): each
// draw's captured regions are compared, region by region and word by word, with the program's
// previous draw. Every 10 s, for the programs drawn most: regions and words per draw, the share of
// words that differed, and where the differing words sit in their region (word index mod 8: a
// buffer descriptor's base address is word 0 and the low bits of word 1 of its 4 words).
struct CaptureDiffProfile {
    std::mutex mutex;
    struct Program {
        std::vector<std::vector<std::uint32_t>> last;
        // Per region: the user-data word pair whose pointer it lies within 4 KiB after, and the
        // offset (or -1, -1), compared with the previous draw.
        std::vector<std::pair<int, std::uint64_t>> lastRelative;
        std::vector<std::uint64_t> lastAddresses;
        std::vector<std::vector<std::uint32_t>> lastDescriptors;
        std::uint64_t bindings = 0, bindingsChanged = 0, bindingsChangedFromUser = 0;
        std::uint64_t draws = 0, compared = 0, words = 0, differing = 0, shapeChanged = 0;
        std::uint64_t regions = 0, relative = 0, relativeStable = 0, absoluteStable = 0, absoluteMoved = 0;
        std::array<std::uint64_t, 8> positions{};
    };
    std::unordered_map<std::uint64_t, Program> programs;
    std::chrono::steady_clock::time_point windowStart = std::chrono::steady_clock::now();
};

void ProfileCaptureDiff(const std::vector<DrawProgram>& programs, const std::vector<StageCapture>& captures) {
    static const bool on = std::getenv("APS5_PROFILE_CAPTURE_DIFF") != nullptr;
    if (!on) return;
    static auto* state = new CaptureDiffProfile();
    std::lock_guard lock(state->mutex);
    for (std::size_t i = 0; i < programs.size() && i < captures.size(); ++i) {
        if (captures[i].regions.empty()) continue;
        auto& entry = state->programs[programs[i].binary.codeAddress];
        ++entry.draws;
        std::vector<std::vector<std::uint32_t>> now;
        std::vector<std::pair<int, std::uint64_t>> relative;
        std::vector<std::uint64_t> addresses;
        const auto& user = programs[i].userData;
        for (const auto& region : captures[i].regions) {
            std::vector<std::uint32_t> words(region.bytes.size() / 4);
            std::memcpy(words.data(), region.bytes.data(), words.size() * 4);
            now.push_back(std::move(words));
            std::pair<int, std::uint64_t> found{-1, 0};
            for (std::size_t w = 0; w + 1 < user.size(); w += 2) {
                const auto pointer = static_cast<std::uint64_t>(user[w]) | (static_cast<std::uint64_t>(user[w + 1] & 0xffffu) << 32u);
                if (pointer != 0 && region.guestAddress >= pointer && region.guestAddress - pointer < 0x1000 && (found.first < 0 || region.guestAddress - pointer < found.second)) found = {static_cast<int>(w), region.guestAddress - pointer};
            }
            relative.push_back(found);
            addresses.push_back(region.guestAddress);
        }
        if (entry.lastRelative.size() == relative.size()) {
            for (std::size_t r = 0; r < relative.size(); ++r) {
                ++entry.regions;
                if (relative[r].first >= 0) {
                    ++entry.relative;
                    if (relative[r] == entry.lastRelative[r]) ++entry.relativeStable;
                } else if (addresses[r] == entry.lastAddresses[r]) {
                    ++entry.absoluteStable;
                } else {
                    ++entry.absoluteMoved;
                }
            }
        }
        entry.lastRelative = std::move(relative);
        entry.lastAddresses = std::move(addresses);
        // The compiled stage's binding descriptors: how many changed since the program's previous
        // draw, and how many of the changed ones are user words of this draw (a V# in user SGPRs).
        if (captures[i].compiled != nullptr) {
            std::vector<std::vector<std::uint32_t>> descriptors;
            for (const auto& binding : captures[i].compiled->bindings) descriptors.push_back(binding.guestDescriptor);
            if (entry.lastDescriptors.size() == descriptors.size()) {
                for (std::size_t b = 0; b < descriptors.size(); ++b) {
                    ++entry.bindings;
                    if (descriptors[b] == entry.lastDescriptors[b]) continue;
                    ++entry.bindingsChanged;
                    const auto& words = descriptors[b];
                    const bool fromUser = !words.empty() && std::search(user.begin(), user.end(), words.begin(), words.begin() + std::min<std::size_t>(words.size(), 2)) != user.end();
                    if (fromUser) ++entry.bindingsChangedFromUser;
                }
            }
            entry.lastDescriptors = std::move(descriptors);
        }
        if (!entry.last.empty()) {
            bool sameShape = entry.last.size() == now.size();
            for (std::size_t r = 0; sameShape && r < now.size(); ++r) sameShape = entry.last[r].size() == now[r].size();
            if (!sameShape) {
                ++entry.shapeChanged;
            } else {
                ++entry.compared;
                for (std::size_t r = 0; r < now.size(); ++r) {
                    for (std::size_t w = 0; w < now[r].size(); ++w) {
                        ++entry.words;
                        if (now[r][w] == entry.last[r][w]) continue;
                        ++entry.differing;
                        ++entry.positions[w % 8];
                    }
                }
            }
        }
        entry.last = std::move(now);
    }
    const auto now = std::chrono::steady_clock::now();
    if (now - state->windowStart < std::chrono::seconds(10)) return;
    state->windowStart = now;
    std::vector<std::pair<std::uint64_t, CaptureDiffProfile::Program*>> top;
    for (auto& [address, entry] : state->programs) top.emplace_back(address, &entry);
    std::sort(top.begin(), top.end(), [](const auto& a, const auto& b) { return a.second->draws > b.second->draws; });
    std::fprintf(stderr, "[capture-diff] %zu programs with captures (10 s):\n", top.size());
    for (std::size_t i = 0; i < top.size() && i < 16; ++i) {
        const auto& entry = *top[i].second;
        std::string positions;
        for (const auto count : entry.positions) positions += " " + std::to_string(count);
        std::fprintf(stderr, "[capture-diff]   0x%llx bindings %llu, changed %llu (from user words %llu)\n", static_cast<unsigned long long>(top[i].first), static_cast<unsigned long long>(entry.bindings), static_cast<unsigned long long>(entry.bindingsChanged), static_cast<unsigned long long>(entry.bindingsChangedFromUser));
        std::fprintf(stderr, "[capture-diff]   0x%llx x%llu: %zu regions %zu words, compared %llu (shape changed %llu), %.1f%% of words differed, by word mod 8:%s; regions %llu: pointer-relative %llu (same pointer and offset %llu), absolute same %llu, absolute moved %llu\n",
                     static_cast<unsigned long long>(top[i].first), static_cast<unsigned long long>(entry.draws), entry.last.size(),
                     [&] { std::size_t n = 0; for (const auto& r : entry.last) n += r.size(); return n; }(),
                     static_cast<unsigned long long>(entry.compared), static_cast<unsigned long long>(entry.shapeChanged),
                     entry.words != 0 ? 100.0 * static_cast<double>(entry.differing) / static_cast<double>(entry.words) : 0.0, positions.c_str(),
                     static_cast<unsigned long long>(entry.regions), static_cast<unsigned long long>(entry.relative), static_cast<unsigned long long>(entry.relativeStable),
                     static_cast<unsigned long long>(entry.absoluteStable), static_cast<unsigned long long>(entry.absoluteMoved));
    }
    for (auto& [address, entry] : state->programs) {
        entry.draws = entry.compared = entry.words = entry.differing = entry.shapeChanged = 0;
        entry.regions = entry.relative = entry.relativeStable = entry.absoluteStable = entry.absoluteMoved = 0;
        entry.bindings = entry.bindingsChanged = entry.bindingsChangedFromUser = 0;
        entry.positions = {};
    }
}

}

DrawVerdict Driver::draw(QueueState& queue, std::span<const std::uint32_t> packet, const Submission& submission, std::string& rejected) {
    PerformanceTimer timing("Driver.Draw");
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    std::array<double, DrawDriverPhaseCount> phaseMs{};
    std::uint64_t captures = 0;
    auto phaseLap = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    DrawPhaseTiming phaseTiming{profile, phaseMs, phaseLap};
    if (profile && packetStartedAt() != std::chrono::steady_clock::time_point{}) phaseMs[DrawRowPrologue] = std::chrono::duration<double, std::milli>(phaseLap - packetStartedAt()).count();
    // A draw that throws leaves its phases for the slow-skip line (PacketExecution).
    struct ThrownPhases {
        DrawPhaseTiming& timing;
        int exceptions = std::uncaught_exceptions();
        ~ThrownPhases() {
            if (!timing.profile || std::uncaught_exceptions() <= exceptions) return;
            timing.Phase(DrawRowSkipped);
            ThrownDrawPhases() = timing.phaseMs;
        }
    } thrownPhases{phaseTiming};
    // Causal-profile probe (APS5_SLOW_DRAW_PACKET_US=<us>): every draw packet spins this long on the
    // packet worker first, so a run shows how much the frame depends on this thread's per-draw cost.
    static const auto slowPacket = std::chrono::microseconds(std::getenv("APS5_SLOW_DRAW_PACKET_US") != nullptr ? std::strtoll(std::getenv("APS5_SLOW_DRAW_PACKET_US"), nullptr, 10) : 0);
    if (slowPacket.count() > 0) {
        const auto until = std::chrono::steady_clock::now() + slowPacket;
        while (std::chrono::steady_clock::now() < until) {
        }
    }

    const auto drawn = [&] {
        phaseTiming.Phase(DrawRowVectors);
        if (!profile) return DrawVerdict::Drawn;
        auto& pending = pendingDrawPhases();
        pending.phases = true;
        pending.captures = captures;
        pending.ms = phaseMs;
        pending.tailAt = phaseLap;
        return DrawVerdict::Drawn;
    };
    auto drawParameters = Pm4::ResolveDraw(packet, queue);
    // APS5_PIPELINED_DRAWS: draws are recorded by the committer behind this worker (see the end of
    // this function); everything recorded inline, or that reads memory queued draws may write (the
    // records of a CPU-path indirect draw), waits for the queued ones first.
    // Prepared ahead on a pool thread (APS5_PREPARE_AHEAD, DrawAhead.hpp): the draw ends in its task
    // instead of the committer, and any path that would wait for queued draws or record here gives
    // it back to the worker (DrawAheadSerial).
    auto* const ahead = CurrentDrawAhead();
    const bool pipelined = DrawPipeline::Active() || ahead != nullptr;
    const auto drainPipeline = [&](DrawPipeline::DrainReason reason) {
        if (ahead != nullptr) throw DrawAheadSerial{};
        if (pipelined && DrawPipeline::Queue0().Busy()) DrawPipeline::Queue0().Drain(reason, (packet[0] >> 8u) & 0xffu);
    };
    if (drawParameters.indirect && drawParameters.indirect->unsizedIndices) {
        drainPipeline(DrawPipeline::DrainReason::Indirect);
        recordQueuedLabelsBeforeRead(submission.queue);
        Pm4::BoundUnsizedIndices(drawParameters);
        static std::atomic<std::uint64_t> unsizedDraws{0};
        if (const auto drawn = unsizedDraws.fetch_add(1, std::memory_order_relaxed); drawn < 4 || (drawn & (drawn + 1)) == 0) std::fprintf(stderr, "[draw] indexed indirect draw without INDEX_BUFFER_SIZE drawn with the records' %u indices (%llu so far)\n", drawParameters.indexCount, static_cast<unsigned long long>(drawn + 1));
    }
    bool traceIndirect = false;
    if (const auto verdict = precheckDraw(queue, submission, packet, drawParameters, rejected, traceIndirect)) return *verdict;
    phaseTiming.Phase(DrawRowPrecheck);
    using Stage = ShaderRecompiler::ShaderStage;
    using Role = ShaderRecompiler::ProgramRole;

    static const std::uint64_t dumpTarget = [] { const char* text = std::getenv("APS5_DUMP_DRAW_SHADERS"); return text ? std::strtoull(text, nullptr, 16) : 0ull; }();

    static const std::uint64_t dumpSlot1 = [] { const char* text = std::getenv("APS5_DUMP_DRAW_SLOT1"); return text ? std::strtoull(text, nullptr, 16) : 0ull; }();

    static const bool lockedPrepare = std::getenv("APS5_LOCKED_DRAW_PREPARE") != nullptr;
    std::unique_lock gpuLock(GuestMemory::GpuMutex(), std::defer_lock);
    std::shared_ptr<VulkanDevice> localDevice;
    if (lockedPrepare) {

        GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Draw);
        gpuLock.lock();
        timing.Mark("gpu_mutex_wait");
        phaseTiming.Phase(DrawRowLockWait);
        if (device == nullptr) device = std::make_shared<VulkanDevice>();
        localDevice = device;

        recordLabelsForPacket(localDevice.get(), submission.queue);
        phaseTiming.Phase(DrawRowLabels);
    } else if ((localDevice = device.Load()) == nullptr) {
        GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Draw);
        std::lock_guard createLock(GuestMemory::GpuMutex());
        if (device == nullptr) device = std::make_shared<VulkanDevice>();
        localDevice = device;
    }
    timing.Mark("device_setup");
    phaseTiming.Phase(DrawRowVectors);

    const bool useDrawEntries = drawEntries() && !ShaderRecompiler::DebugProbeActive() && dumpTarget == 0 && dumpSlot1 == 0;
    const bool registerKey = useDrawEntries && registerKeyEnabled();
    std::uint64_t drawKey = 0;
    std::shared_ptr<DrawEntry> entry;
    std::shared_ptr<const DrawDecode> decode;
    // APS5_RELOCATED_DRAW_KEYS (DrawRelocation.cpp): the shape's relocated pointer words, left out of the key.
    std::uint64_t shapeKey = 0;
    std::shared_ptr<const std::vector<std::uint32_t>> relocation;
    if (registerKey) {
        const auto keyStart = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        if (relocatedDrawKeys()) {
            shapeKey = drawRegisterKey(queue, *submission.shaders, localDevice->Serial(), nullptr, true);
            std::lock_guard cacheLock(drawCacheMutex);
            if (const auto found = drawRelocations.find(shapeKey); found != drawRelocations.end()) relocation = found->second;
        }
        drawKey = drawRegisterKey(queue, *submission.shaders, localDevice->Serial(), relocation.get());
        std::lock_guard cacheLock(drawCacheMutex);
        ++drawEntryCounters.lookups;
        ++drawEntryCounters.registerKeyLookups;
        if (profile) drawEntryCounters.keyUs += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - keyStart).count();
        const auto found = drawCache.find(drawKey);
        if (found != drawCache.end()) {
            entry = found->second;
            decode = entry->decode;
        } else {
            ++drawEntryCounters.absent;
        }
    }
    if (decode != nullptr && !DecodeProgramsCurrent(*decode, *submission.shaders)) {
        std::lock_guard cacheLock(drawCacheMutex);
        eraseDrawEntry(drawKey, entry);
        ++drawEntryCounters.absent;
        entry = nullptr;
        decode = nullptr;
    }
    phaseTiming.Phase(DrawRowKeyLookupValidate);

    resolveDrawDecode(queue, submission, decode, registerKey, drawKey, profile);
    const auto& graphics = decode->state;
    auto pixel = decode->pixel;
    std::vector<DrawProgram> programs = decode->programs;
    // A cached decode holds the user words of the draw that made it; the key left these out.
    if (relocation != nullptr) refreshRelocatedUserWords(queue, *relocation, programs);
    // APS5_NATIVE_SHADOW: the title's own view of this draw's shaders against the decoded programs.
    const NativeDrawRecord* nativeRecord = nullptr;
    if (NativeShadowActive() && packet.data() >= submission.commands.data() && packet.data() < submission.commands.data() + submission.commands.size()) {
        std::vector<std::uint64_t> decodedShaders;
        for (const auto& program : programs) decodedShaders.push_back(program.snapshot != nullptr ? program.snapshot->headerAddress : 0);
        const auto found = submission.nativeDraws.find(static_cast<std::size_t>(packet.data() - submission.commands.data()));
        CompareNativeDraw(found != submission.nativeDraws.end() ? &found->second : nullptr, decodedShaders);
        if (found != submission.nativeDraws.end()) {
            nativeRecord = &found->second;
            std::vector<std::vector<std::uint32_t>> userData;
            for (const auto& program : programs) userData.push_back(program.userData);
            CompareNativeUserData(found->second, decodedShaders, userData);
            if (!found->second.raw.empty()) DumpNativeDraw(found->second, decodedShaders, userData);
        }
    }
    // Debug aid: draws of APS5_TRACE_DRAW_PROGRAMS print their resources ([draw-io]) and skip recipes.
    const bool tracedPrograms = std::any_of(programs.begin(), programs.end(), [](const DrawProgram& program) { return Graphics::TraceDrawProgram(program.binary.codeAddress); });
    if (tracedPrograms) {
        // Each distinct program combination of a traced draw once: every stage's code address.
        static std::mutex seenMutex;
        static std::set<std::vector<std::uint64_t>> seen;
        std::vector<std::uint64_t> addresses;
        for (const auto& program : programs) addresses.push_back(program.binary.codeAddress);
        std::lock_guard seenLock(seenMutex);
        if (seen.insert(addresses).second && seen.size() <= 64) {
            std::string text;
            for (const auto& program : programs) {
                char item[64];
                std::snprintf(item, sizeof(item), " stage %u 0x%llx", static_cast<unsigned>(program.binary.stage), static_cast<unsigned long long>(program.binary.codeAddress));
                text += item;
            }
            std::fprintf(stderr, "[draw-programs]%s\n", text.c_str());
        }
    }
    struct TracedDrawMark {
        bool on;
        const std::string& rejected;
        std::uint64_t program;
        TracedDrawMark(bool traced, const std::string& reason, std::uint64_t address) : on(traced), rejected(reason), program(address) { if (on) Graphics::MarkTracedDraw(true); }
        ~TracedDrawMark() {
            if (!on) return;
            Graphics::MarkTracedDraw(false);
            static std::atomic<int> printed{0};
            if (!rejected.empty() && printed.fetch_add(1, std::memory_order_relaxed) < 200) std::fprintf(stderr, "[draw-io] traced draw (program 0x%llx) rejected: %s\n", static_cast<unsigned long long>(program), rejected.c_str());
        }
    } tracedMark(tracedPrograms, rejected, programs.empty() ? 0 : programs.front().binary.codeAddress);
    Graphics::SetDrawTimingProgram(programs.empty() ? 0 : programs.back().binary.codeAddress);
    if (const auto skipped = SkippedShader(); skipped != 0 && std::any_of(programs.begin(), programs.end(), [skipped](const DrawProgram& program) { return program.binary.codeAddress == skipped; })) return drawn();
    const auto setMeshIndexBuffer = [&](const Pm4::DrawParameters& parameters) {
        if (!graphics.stages.mesh) return;
        auto& words = programs.front().userData;
        require(programs.front().firstUserSgpr == 0 && words.size() >= ShaderRecompiler::MeshIndexBufferUserWord + 4, "mesh program lacks the hidden user words");
        const auto descriptor = Graphics::MeshIndexBufferDescriptor(parameters);
        std::copy(descriptor.begin(), descriptor.end(), words.begin() + ShaderRecompiler::MeshIndexBufferUserWord);
    };
    if (!drawParameters.indirect) setMeshIndexBuffer(drawParameters);
    else if (graphics.stages.mesh) setMeshIndexBuffer(Pm4::DrawParameters{drawParameters.indexAddress, std::max(drawParameters.indexCount, 1u), drawParameters.indexSize, 1, 0, drawParameters.indexed});
    const std::vector<Role>& roles = decode->roles;
    phaseTiming.Phase(DrawRowDecode);

    const auto locate = [&](std::uint32_t location) -> std::optional<std::pair<std::size_t, std::size_t>> {
        if (location == 0x280u) return std::nullopt;
        for (std::size_t i = 0; i < programs.size(); ++i) {
            if (roles[i] == Role::Fragment || roles[i] == Role::GeometryBack || location < programs[i].userDataBase) continue;
            const auto word = location - programs[i].userDataBase + (8u - programs[i].firstUserSgpr);
            if (word < programs[i].userData.size()) return std::make_pair(i, static_cast<std::size_t>(word));
        }
        return std::nullopt;
    };
    if (drawParameters.indirect) {
        auto& indirect = *drawParameters.indirect;
        const auto sgprOf = [&](std::uint32_t location) -> std::int32_t {
            const auto word = locate(location);
            if (!word || word->first != 0) return -1;
            return static_cast<std::int32_t>(programs.front().firstUserSgpr + word->second);
        };
        indirect.baseVertexSgpr = sgprOf(indirect.baseVertexLocation);
        indirect.startInstanceSgpr = sgprOf(indirect.startInstanceLocation);
        indirect.drawIndexSgpr = sgprOf(indirect.drawIndexLocation);
    }
    std::vector<ShaderRecompiler::MemoryRegion> memory;
    std::vector<ShaderRecompiler::LinkedProgram> linked;
    for (std::size_t i = 0; i < programs.size(); ++i) {
        const auto& program = programs[i];
        for (const auto& region : program.memory) {
            if (!region.bytes.empty()) memory.push_back(region);
        }
        linked.push_back({roles[i], program.binary, program.userDataBase, program.firstUserSgpr, program.userData});
    }
    timing.Mark("prepare");
    phaseTiming.Phase(DrawRowProgramPrepare);

    std::vector<std::optional<ShaderRecompiler::ShaderVertexStageInfo>> vertexInfos(programs.size());
    std::vector<std::vector<Graphics::DecodeRead>> decodeReads(programs.size());
    const auto decodeVertexInfo = [&](std::size_t i) {
        const auto& program = programs[i];
        if (program.binary.stage == Stage::Fragment || roles[i] == Role::GeometryBack) return;
        decodeReads[i].clear();
        vertexInfos[i] = Graphics::DecodeVertexStageInfo(program.binary.header, program.binary.headerAddress, program.userData, &decodeReads[i], false, 8u - program.firstUserSgpr);
        if (vertexInfos[i]) vertexInfos[i]->paClVsOutCntl = PositionExportLayout(program.snapshot.get(), program.codeOffset, program.binary.code, readRegister(queue.context, 0x207)) | graphics.layerExports;
    };
    if (!registerKey) {
        for (std::size_t i = 0; i < programs.size(); ++i) decodeVertexInfo(i);
        phaseTiming.Phase(DrawRowDecode);
    }
    // Heap-owned, so a pipelined draw's commit owns the capture's pages (memory's regions point there).
    auto shaderMemory = std::make_unique<ShaderMemory>(memory, &queryPendingWrite, &observePendingWrite, hookWaitCounter());
    std::vector<ShaderRecompiler::RecompileResult> results;
    // Materialized stages, shared with the capture (not copied); `results` holds the derived and
    // patched ones. `stages` points into both, so neither grows past its reservation.
    std::vector<std::shared_ptr<const ShaderRecompiler::RecompileResult>> heldResults;
    heldResults.reserve(programs.size());
    std::vector<Graphics::CompiledShader> stages;
    results.reserve(2u * programs.size() + 3u);
    stages.reserve(programs.size());
    std::uint32_t pushCursorBytes = 0;

    std::vector<const ShaderRecompiler::RecompileResult*> programResults(programs.size(), nullptr);

    std::vector<StageCapture> stageCaptures(programs.size());
    std::vector<std::shared_ptr<DispatchVariant>> matched(programs.size());
    std::vector<std::vector<ShaderRecompiler::MemoryRegion>> matchedRegions(programs.size());

    std::vector<std::shared_ptr<DispatchVariant>> fresh(programs.size());

    std::vector<bool> recompiled(programs.size(), false);
    bool drawHit = false;
    bool verifyHit = false;
    ProfileUserWords(programs);
    // The key's inputs as looked up: the miss path below may change pixel.
    const auto lookupPixel = pixel;
    bool learnHit = false;
    lookupDraw(submission, localDevice, graphics, pixel, programs, roles, vertexInfos, useDrawEntries, registerKey, profile, drawKey, entry, matched, matchedRegions, drawHit, verifyHit, phaseTiming, phaseMs, shapeKey, relocation, learnHit);

    if (registerKey) {

        for (std::size_t i = 0; i < programs.size(); ++i) {
            if (matched[i] != nullptr && drawHit && !verifyDrawRecipe()) {
                if (matched[i]->vertexInfo != nullptr) vertexInfos[i] = *matched[i]->vertexInfo;
                continue;
            }
            decodeVertexInfo(i);
            if (matched[i] != nullptr && verifyDrawRecipe() && (vertexInfos[i].has_value() != (matched[i]->vertexInfo != nullptr) || (vertexInfos[i] && !sameVertexInfo(*vertexInfos[i], *matched[i]->vertexInfo)))) {
                static std::atomic<std::uint64_t> reports{0};
                if (reports.fetch_add(1) < 20) std::fprintf(stderr, "[draw-cache] verify: stage %zu (program 0x%llx) of a hit has a vertex stage info unlike its variant's\n", i, static_cast<unsigned long long>(programs[i].binary.codeAddress));
                std::lock_guard cacheLock(drawCacheMutex);
                ++drawEntryCounters.verifyDecodeMismatches;
            }
        }
        phaseTiming.Phase(DrawRowDecode);
    }

    const auto fold = [&](const ShaderRecompiler::RecompileResult& main, Pm4::DrawParameters& parameters) { FoldDrawOffsets(main, programs.front(), parameters); };

    std::optional<Graphics::IndirectDrawPath> indirectCpu;
    std::vector<std::uint32_t> pushOffsets(programs.size(), 0);
    std::vector<std::size_t> resultIndex(programs.size(), 0);
    // APS5_NATIVE_DRAWS=1 (native path N2; part of the native switch the user wants built whole and
    // turned on at once): a walk-free draw (no stage reads guest memory for its resources) whose
    // shape (the register key without user words: programs and render state) was drawn before reuses
    // that draw's compiled stages with this draw's user words patched in, skipping capture and
    // materialize. APS5_NATIVE_VERIFY_PATCH found that exact on every walk-free stage compared
    // (trnv1, 2026-10-30: ~27k, 0 mismatches). Walking shapes (85% of draws) still capture: their
    // flattened SRTs carry bindless-table state a patch cannot rebuild (N3, not built). The vertex
    // stage info (memory-derived) must also match the cached stage's.
    // APS5_NATIVE_DRAWS=verify: every hit also materializes the stages as before, renders with those
    // and compares them with the reused ones (rendering unchanged).
    static const bool nativeDraws = std::getenv("APS5_NATIVE_DRAWS") != nullptr;
    static const bool nativeVerify = nativeDraws && std::string_view(std::getenv("APS5_NATIVE_DRAWS")) == "verify";
    const auto nativeKey = nativeDraws && !drawHit ? drawRegisterKey(queue, *submission.shaders, localDevice->Serial(), nullptr, true) : 0;
    std::shared_ptr<const std::vector<NativeDrawStage>> nativeHit;
    if (nativeDraws && !drawHit) {
        auto& cache = nativeDrawCache();
        {
            std::lock_guard lock(cache.mutex);
            if (const auto found = cache.shapes.find(nativeKey); found != cache.shapes.end()) nativeHit = found->second;
        }
        bool same = nativeHit != nullptr && nativeHit->size() == programs.size();
        for (std::size_t i = 0; same && i < programs.size(); ++i) {
            if (roles[i] == Role::GeometryBack) continue;
            const auto& stage = (*nativeHit)[i];
            same = stage.result != nullptr && stage.program == programs[i].binary.codeAddress && stage.vertexInfo.has_value() == vertexInfos[i].has_value() && (!vertexInfos[i] || sameVertexShape(*vertexInfos[i], *stage.vertexInfo));
        }
        if (!same) nativeHit = nullptr;
    }
    for (std::size_t i = 0; i < programs.size(); ++i) {
        if (roles[i] == Role::GeometryBack) continue;
        const auto& program = programs[i];
        pushCursorBytes = Graphics::StagePushOffset(pushCursorBytes, program.binary.stage, localDevice->GraphicsPipelineLibraries());
        pushOffsets[i] = pushCursorBytes;
        if (drawHit) {

            programResults[i] = matched[i]->compiled.get();
            // A confirmed pointer-patched stage (DrawRelocation.cpp): a copy with this draw's pointers.
            if (!matched[i]->userWordPatches.empty()) {
                auto patched = patchedResult(*matched[i], programs[i].userData);
                require(patched != nullptr, "pointer patch outside the cached stage result");
                heldResults.push_back(std::move(patched));
                programResults[i] = heldResults.back().get();
                notePatchOutcome(2);
            }
            memory.insert(memory.end(), matchedRegions[i].begin(), matchedRegions[i].end());
        } else {
            if (program.binary.stage == ShaderRecompiler::ShaderStage::Fragment) {
                for (auto previous = i; previous-- > 0;) {
                    if (programResults[previous] == nullptr) continue;
                    pixel = Graphics::DefaultUnexportedInputs(decode->pixel, programResults[previous]->parameterExports);
                    break;
                }
            }
            if (nativeHit != nullptr && !nativeVerify) {
                auto patched = nativePatchedResult((*nativeHit)[i], programs[i].userData, vertexInfos[i]);
                require(patched != nullptr, "native draw: a user-word patch lies outside the cached stage result");
                heldResults.push_back(std::move(patched));
                programResults[i] = heldResults.back().get();
            } else {
                heldResults.push_back(materializeDrawStage(i, pushCursorBytes, queue, submission, programs, graphics, pixel, vertexInfos, memory, linked, drawParameters, localDevice, *shaderMemory, stageCaptures, recompiled, drawHit, matched, matchedRegions, profile, dumpTarget, dumpSlot1, captures, phaseTiming, phaseMs, rejected));
                if (!rejected.empty()) return DrawVerdict::Rejected;
                programResults[i] = heldResults.back().get();
                if (nativeHit != nullptr) {
                    const auto patched = nativePatchedResult((*nativeHit)[i], programs[i].userData, vertexInfos[i]);
                    // sameCompiledResult leaves out the vertex attributes and poisoned SRT reads,
                    // which are per draw too.
                    const auto& fresh = *heldResults.back();
                    const auto sameAttributes = [&] {
                        if (patched->vertexAttributes.size() != fresh.vertexAttributes.size()) return false;
                        for (std::size_t a = 0; a < fresh.vertexAttributes.size(); ++a) {
                            const auto& x = patched->vertexAttributes[a];
                            const auto& y = fresh.vertexAttributes[a];
                            if (x.location != y.location || x.components != y.components || x.fetchIndex != y.fetchIndex || x.formatComponents != y.formatComponents || x.resource.fields != y.resource.fields) return false;
                        }
                        return true;
                    };
                    const bool equal = patched != nullptr && sameCompiledResult(*patched, fresh) && patched->poisonedSrtReads == fresh.poisonedSrtReads && sameAttributes();
                    auto& cache = nativeDrawCache();
                    std::lock_guard lock(cache.mutex);
                    ++(equal ? cache.verifyEqual : cache.verifyDiffer);
                    if (!equal && cache.verifyReports < 12) {
                        ++cache.verifyReports;
                        std::fprintf(stderr, "[native-draws] verify: stage %zu of program 0x%llx differs from its reused copy%s\n", i, static_cast<unsigned long long>(program.binary.codeAddress), patched == nullptr ? " (patch outside the result)" : "");
                    }
                }
            }
        }
        const auto& result = *programResults[i];
        if (i == 0 && drawParameters.indirect) {
            indirectCpu = ClassifyIndirectDraw(result, graphics, programs.front(), localDevice, drawParameters, traceIndirect);
        } else if (i == 0) {
            fold(result, drawParameters);
        }
        const auto slotEnd = (pushCursorBytes / Graphics::PipelinePushSlotBytes + 1u) * Graphics::PipelinePushSlotBytes;
        require(result.pushConstants.size() <= slotEnd - pushCursorBytes, "stage push constants exceed the pipeline push constant block");
        stages.push_back({program.binary.stage, &result, result.pushConstants.empty() ? 0u : pushCursorBytes});
        pushCursorBytes += static_cast<std::uint32_t>(result.pushConstants.size());
    }

    // APS5_NATIVE_DRAWS: a walk-free miss keeps its stages for its shape ([native-draws] line).
    if (nativeDraws && !drawHit) {
        auto& cache = nativeDrawCache();
        std::shared_ptr<std::vector<NativeDrawStage>> store;
        bool walking = false;
        if (nativeHit == nullptr) {
            store = std::make_shared<std::vector<NativeDrawStage>>(programs.size());
            for (std::size_t i = 0; store != nullptr && i < programs.size(); ++i) {
                if (roles[i] == Role::GeometryBack) continue;
                const auto held = std::find_if(heldResults.begin(), heldResults.end(), [&](const auto& result) { return result.get() == programResults[i]; });
                // Vertex stage info read from memory is fine: every draw decodes its own and a hit
                // requires it equal to the stored one (requiring none cut hits from ~14k to ~600
                // per 10 s, trn2v).
                if (held == heldResults.end() || !stageCaptures[i].captured || !stageCaptures[i].regions.empty()) {
                    walking = true;
                    store = nullptr;
                    break;
                }
                auto patches = userWordPatchesFor(**held, programs[i].userData);
                const auto data = shaderDataPatchesFor(**held);
                patches.insert(patches.end(), data.begin(), data.end());
                (*store)[i] = {*held, std::move(patches), programs[i].binary.codeAddress, vertexInfos[i]};
            }
        }
        std::lock_guard lock(cache.mutex);
        if (nativeHit != nullptr) ++cache.hits;
        else if (store != nullptr) {
            if (cache.shapes.size() > (1u << 14)) cache.shapes.clear();
            cache.shapes[nativeKey] = std::move(store);
            ++cache.stored;
        } else if (walking) ++cache.walking;
        else ++cache.refused;
        if (const auto clock = std::chrono::steady_clock::now(); clock - cache.reported >= std::chrono::seconds(10)) {
            cache.reported = clock;
            std::fprintf(stderr, "[native-draws] (10 s) walk-free reuse: %llu hits, %llu stored, %llu walking draws (capture as before), %llu refused; %zu shapes kept; verify: %llu stages equal, %llu differ\n", static_cast<unsigned long long>(cache.hits), static_cast<unsigned long long>(cache.stored), static_cast<unsigned long long>(cache.walking), static_cast<unsigned long long>(cache.refused), cache.shapes.size(), static_cast<unsigned long long>(cache.verifyEqual), static_cast<unsigned long long>(cache.verifyDiffer));
            cache.hits = cache.stored = cache.walking = cache.refused = cache.verifyEqual = cache.verifyDiffer = 0;
        }
    }
    // APS5_NATIVE_CENSUS=1 (native path groundwork, count only): whether the draw's shape (the
    // register key without user words: programs and render state) was seen before, and whether its
    // stages captured nothing (walk-free); a [native-census] line every 10 s. A repeated walk-free
    // shape can reuse its compiled stages with only the per-draw values rebuilt; a repeated walking
    // shape would gather its runs at the draw's own pointers.
    static const bool nativeCensus = std::getenv("APS5_NATIVE_CENSUS") != nullptr;
    // APS5_NATIVE_VERIFY_PATCH=1 (native path groundwork, compare only, rendering unchanged): for a
    // draw whose shape was drawn before, the earlier draw's compiled stages with this draw's user
    // words patched in (the pointer pairs they hold verbatim and the shader-data words, as
    // relocated keys patch them) against this draw's freshly materialized stages: whether patching
    // rebuilds every per-draw value, or which part it misses ([native-patch] line). Walking stages
    // are compared when their regions match at this draw's pointers (what a relocated hit checks).
    static const bool verifyPatch = std::getenv("APS5_NATIVE_VERIFY_PATCH") != nullptr;
    const auto nativeShape = nativeCensus || verifyPatch ? drawRegisterKey(queue, *submission.shaders, localDevice->Serial(), nullptr, true) : 0;
    if (nativeCensus) noteNativeCensus(nativeShape, drawHit, stageCaptures, roles);
    if (verifyPatch && !drawHit) {
        std::vector<NativePatchStage> patchStages;
        bool eligible = true;
        for (std::size_t i = 0; eligible && i < programs.size(); ++i) {
            if (roles[i] == Role::GeometryBack) continue;
            const auto held = std::find_if(heldResults.begin(), heldResults.end(), [&](const auto& result) { return result.get() == programResults[i]; });
            eligible = held != heldResults.end() && stageCaptures[i].captured;
            if (!eligible) break;
            auto patches = userWordPatchesFor(**held, programs[i].userData);
            const auto data = shaderDataPatchesFor(**held);
            patches.insert(patches.end(), data.begin(), data.end());
            std::vector<std::pair<std::uint64_t, std::vector<std::byte>>> regions;
            for (const auto& region : stageCaptures[i].regions) regions.emplace_back(region.guestAddress, std::vector<std::byte>(region.bytes.begin(), region.bytes.end()));
            patchStages.push_back({*held, std::move(patches), programs[i].userData, programs[i].binary.codeAddress, std::move(regions)});
        }
        if (eligible && !patchStages.empty()) noteNativePatchVerify(nativeShape, std::move(patchStages));
    }
    ProfileCaptureDiff(programs, stageCaptures);
    std::optional<DrawRelocationPlan> relocationPlan;
    if (useDrawEntries && relocatedDrawKeys() && !drawHit) {
        // A known shape's miss is planned (~8 us) only when it may be inserted: its key seen before
        // (Driver::admitDrawKey, as cacheDrawStages asks) and not a learning draw.
        if (relocation == nullptr || (!learnHit && admitDrawKey(drawKey))) {
            const auto keyFor = [&](const std::vector<std::uint32_t>* mask) {
                return registerKey ? drawRegisterKey(queue, *submission.shaders, localDevice->Serial(), mask) : drawProgramKey(localDevice->Serial(), graphics, lookupPixel, programs, roles, vertexInfos, mask);
            };
            relocationPlan = planDrawRelocation(keyFor, programs, stageCaptures, decodeReads, shapeKey, relocation, drawKey);
            relocationPlan->learning = learnHit;
            if (relocation != nullptr) relocationPlan->admit = true;
        } else {
            relocationPlan.emplace();
            relocationPlan->insert = false;
            relocationPlan->learning = learnHit;
        }
    }
    const bool relocatedDraw = relocation != nullptr || (relocationPlan && std::any_of(relocationPlan->stages.begin(), relocationPlan->stages.end(), [](bool stage) { return stage; }));
    // A native hit captured nothing: the draw cache gets no entry from it.
    cacheDrawStages(useDrawEntries && nativeHit == nullptr, drawHit, drawParameters, indirectCpu, programs, stageCaptures, vertexInfos, decodeReads, verifyHit, matched, fresh, relocationPlan ? relocationPlan->insertKey : drawKey, registerKey, decode, phaseTiming, relocationPlan ? &*relocationPlan : nullptr);
    timing.Mark("shader_compile_and_link");

    for (const auto& reads : decodeReads) {
        for (const auto& read : reads) memory.push_back({read.address, std::as_bytes(std::span(read.bytes))});
    }

    if (recordQueuedLabelsAfterCapture(submission.queue, memory)) return draw(queue, packet, submission, rejected);

    bool rectListBuilt = false;

    std::size_t rectIndex = 0;
    const auto buildRectList = [&] {
        phaseTiming.Phase(DrawRowVectors);
        require(programs.size() == 2 && programResults[0] != nullptr && programResults[1] != nullptr, "rect-list requires vertex and fragment programs");
        auto rectangle = DrawRectangle(*programs[0].snapshot, programs[1].snapshot, programResults[0]->variantId, programResults[1]->variantId, localDevice->Target());
        if (rectListBuilt) {
            results[rectIndex] = std::move(rectangle.control);
            results[rectIndex + 1] = std::move(rectangle.evaluation);
            phaseTiming.Phase(DrawRowRectList);
            return;
        }
        require(stages.size() == 2, "rect-list requires vertex and fragment programs");
        rectIndex = results.size();
        results.push_back(std::move(rectangle.control));
        results.push_back(std::move(rectangle.evaluation));
        stages.insert(stages.begin() + 1, {{Stage::TessellationControl, &results[rectIndex], 0}, {Stage::TessellationEvaluation, &results[rectIndex + 1], 0}});
        rectListBuilt = true;
        phaseTiming.Phase(DrawRowRectList);
    };
    if (graphics.rectList) buildRectList();
    if (!programs.empty() && programResults.back() != nullptr && programResults.back()->barycentricEmulation.active) {
        require(!graphics.rectList && graphics.stages.path == Graphics::ShaderPath::Vertex && programs.size() == 2 && programResults[0] != nullptr && stages.size() == 2, "a pixel shader that reads barycentrics without VK_KHR_fragment_shader_barycentric needs a vertex shader before it");
        results.push_back(ShaderRecompiler::BuildBarycentricGeometryShader(*programResults[0], *programResults[1], localDevice->Target(), localDevice->GeometryLimits()));
        stages.insert(stages.begin() + 1, Graphics::CompiledShader{Stage::Geometry, &results.back(), 0});
    }
    std::vector<Graphics::GuestMemorySnapshot> snapshots;
    const auto snapshot = [&] {
        snapshots.clear();
        for (const auto& region : memory) snapshots.push_back({region.guestAddress, region.bytes});
    };
    snapshot();
    timing.Mark("post_compile_prepare");
    const auto lockForDraw = [&] {
        if (gpuLock.owns_lock()) return;
        drainPipeline(DrawPipeline::DrainReason::Packet);
        if (DrawWindowWrites() != nullptr) DrawWindowInvalidated() = true;
        phaseTiming.Phase(DrawRowVectors);
        GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Draw);
        gpuLock.lock();
        timing.Mark("gpu_mutex_wait");
        phaseTiming.Phase(DrawRowLockWait);

        if (auto current = device.Load(); current != nullptr && current != localDevice) {
            static std::atomic<std::uint64_t> replaced{0};
            std::fprintf(stderr, "[draw] device replaced during unlocked preparation (%llu)\n", static_cast<unsigned long long>(++replaced));
            localDevice = std::move(current);
        }

        recordLabelsForPacket(localDevice.get(), submission.queue);
        phaseTiming.Phase(DrawRowLabels);
    };
    if (drawParameters.indirect && indirectCpu) {
        // The records are read here, after what queued work writes: the queue is waited for when its
        // write ranges cover the records or the count (captures order themselves the same way,
        // queryPendingWrite). The previous CPU-read indirect draw's records go to the committer as one
        // item, and waiting for it unconditionally put back most of what that saved (indirect drains
        // 980 -> 4310 per 10 s, 0.1 -> 0.77 s). APS5_INLINE_INDIRECT_RECORDS=1 always waits.
        const auto& args = *drawParameters.indirect;
        static const bool alwaysDrain = std::getenv("APS5_INLINE_INDIRECT_RECORDS") != nullptr;
        const bool overlapping = alwaysDrain || !pipelined || DrawPipeline::Queue0().Overlaps(args.arguments, static_cast<std::size_t>(args.RangeBytes())) || (args.countIndirect && DrawPipeline::Queue0().Overlaps(args.countAddress, 4));
        if (overlapping) drainPipeline(DrawPipeline::DrainReason::Indirect);
        const auto indirect = *drawParameters.indirect;
        // The records patch the results below: each stage's (a cached variant's or a shared
        // materialized one) is copied into `results` first.
        {
            for (std::size_t i = 0; i < programs.size(); ++i) {
                if (programResults[i] == nullptr) continue;
                resultIndex[i] = results.size();
                results.push_back(ShaderRecompiler::RecompileResult(*programResults[i]));
                for (auto& stage : stages) {
                    if (stage.program == programResults[i]) stage.program = &results[resultIndex[i]];
                }
                programResults[i] = &results[resultIndex[i]];
            }
        }
        recordQueuedLabelsBeforeRead(submission.queue);
        const auto readStart = std::chrono::steady_clock::now();
        const auto count = std::min(indirect.countIndirect ? Pm4::ReadDrawCount(indirect) : indirect.count, indirect.count);
        std::vector<Pm4::DrawArguments> records;
        for (std::uint32_t record = 0; record < count; ++record) records.push_back(Pm4::ReadDrawArguments(indirect, record));
        Graphics::CountIndirectDraw(*indirectCpu, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - readStart).count());
        std::vector<std::vector<Graphics::DecodeRead>> retiredDecodeReads;
        // With the draw pipeline the records go to the committer as one item after the loop: each
        // record keeps its own copies of the results it patched (the others stay in `results`, which
        // the loop no longer changes) and its snapshots, and the item takes over what they point
        // into. Recording each record here under the GPU mutex took ~25 us per record on the queue-0
        // worker (~1.3 s per 10 s in PPSA03671) while the committer idled. APS5_INLINE_INDIRECT_RECORDS=1
        // records them here as before.
        struct BatchedRecord {
            Pm4::DrawParameters direct;
            std::vector<ShaderRecompiler::RecompileResult> results;
            std::vector<Graphics::CompiledShader> stages;
            std::vector<Graphics::GuestMemorySnapshot> snapshots;
        };
        static const bool inlineRecords = std::getenv("APS5_INLINE_INDIRECT_RECORDS") != nullptr;
        const bool batchRecords = !inlineRecords && pipelined && ahead == nullptr && deferredLabels().labels.empty() && DrawWindowWrites() == nullptr;
        std::vector<BatchedRecord> batched;
        const auto flushBatch = [&] {
            if (batched.empty()) return;
            auto writes = drawWriteRanges(graphics, stages);
            const auto commitQueue = submission.queue;
            const auto commitPacket = (packet[0] >> 8u) & 0xffu;
            const auto epoch = DrawPipeline::EpochToken().load(std::memory_order_relaxed);
            auto commit = [this, localDevice, decode, entry, commitQueue, commitPacket, epoch, batched = std::move(batched), programs = std::move(programs), results = std::move(results), heldResults = std::move(heldResults), memory = std::move(memory), linked = std::move(linked), stageCaptures = std::move(stageCaptures), matched = std::move(matched), matchedRegions = std::move(matchedRegions), fresh = std::move(fresh), decodeReads = std::move(decodeReads), retiredDecodeReads = std::move(retiredDecodeReads), shaderMemory = std::move(shaderMemory)] {
                DrawPipeline::FollowEpoch(epoch);
                Graphics::SetDrawTimingProgram(programs.empty() ? 0 : programs.back().binary.codeAddress);
                for (const auto& record : batched) commitDraw(localDevice, commitQueue, commitPacket, decode->state, record.direct, record.stages, record.snapshots, nullptr, {}, 0);
            };
            auto held = std::make_shared<decltype(commit)>(std::move(commit));
            DrawPipeline::Queue0().Enqueue([held] { (*held)(); }, std::move(writes));
            timing.Mark("draw_enqueued");
        };
        const auto baseVertexWord = locate(indirect.baseVertexLocation);
        const auto startInstanceWord = locate(indirect.startInstanceLocation);
        const auto drawIndexWord = indirect.drawIndexEnabled ? locate(indirect.drawIndexLocation) : std::nullopt;
        for (std::uint32_t record = 0; record < records.size(); ++record) {
            const auto& arguments = records[record];
            if (traceIndirect) std::fprintf(stderr, "[draw]   record %u: count %u instances %u first %u vertexOffset %u startInstance %u\n", record, arguments.count, arguments.instances, arguments.firstVertexOrIndex, arguments.vertexOffset, arguments.firstInstance);
            if (arguments.count == 0 || arguments.instances == 0) continue;
            std::set<std::size_t> patched;
            const auto patch = [&](const std::optional<std::pair<std::size_t, std::size_t>>& word, std::uint32_t value) {
                if (!word) return;
                programs[word->first].userData[word->second] = value;
                patched.insert(word->first);
            };
            patch(baseVertexWord, indirect.recordBytes == 20 ? arguments.vertexOffset : arguments.firstVertexOrIndex);
            patch(startInstanceWord, arguments.firstInstance);
            patch(drawIndexWord, record);
            Pm4::DrawParameters direct{0, arguments.count, 0, arguments.instances, drawParameters.flags, drawParameters.indexed, 0, 0};
            if (drawParameters.indexed) {

                if (arguments.firstVertexOrIndex >= drawParameters.indexCount) continue;
                direct.indexAddress = drawParameters.indexAddress + static_cast<std::uint64_t>(arguments.firstVertexOrIndex) * drawParameters.indexSize;
                direct.indexCount = std::min(arguments.count, drawParameters.indexCount - arguments.firstVertexOrIndex);
                direct.indexSize = drawParameters.indexSize;
            } else {
                direct.firstVertex = indirect.indxOffset;
            }
            if (graphics.stages.mesh) {
                setMeshIndexBuffer(direct);
                patched.insert(0);
            }
            for (const auto programIndex : patched) {
                auto& result = results[resultIndex[programIndex]];
                const auto pushBytes = result.pushConstants.size();
                // `memory` (and every record's snapshots) points into the decode reads' bytes: the
                // re-decode below would free them (clear), leaving the snapshots reading freed memory.
                // Moved aside, they stay where they are until the draw returns.
                retiredDecodeReads.push_back(std::move(decodeReads[programIndex]));
                decodeVertexInfo(programIndex);
                result = *materializeDrawStage(programIndex, pushOffsets[programIndex], queue, submission, programs, graphics, pixel, vertexInfos, memory, linked, drawParameters, localDevice, *shaderMemory, stageCaptures, recompiled, drawHit, matched, matchedRegions, profile, dumpTarget, dumpSlot1, captures, phaseTiming, phaseMs, rejected);
                if (!rejected.empty()) {
                    flushBatch();
                    return DrawVerdict::Rejected;
                }
                require(result.pushConstants.size() == pushBytes, "patched program changed its push constant layout");
            }
            fold(*programResults[0], direct);
            if (graphics.rectList && patched.contains(0)) buildRectList();
            snapshot();
            if (auto known = localDevice->KnownDrawRejection(graphics, stages)) {
                flushBatch();
                rejected = std::move(*known);
                return DrawVerdict::Rejected;
            }
            if (batchRecords) {
                // The results this record changed: its patched programs' and, with them, a rebuilt
                // rect list's. The stages point at this record's copies of those.
                std::vector<std::size_t> changed;
                for (const auto programIndex : patched) changed.push_back(resultIndex[programIndex]);
                if (graphics.rectList && patched.contains(0)) {
                    changed.push_back(rectIndex);
                    changed.push_back(rectIndex + 1);
                }
                auto& record = batched.emplace_back();
                record.direct = direct;
                record.snapshots = snapshots;
                record.stages = stages;
                record.results.reserve(changed.size());
                for (const auto index : changed) {
                    record.results.push_back(results[index]);
                    for (auto& stage : record.stages) {
                        if (stage.program == &results[index]) stage.program = &record.results.back();
                    }
                }
                phaseTiming.Phase(DrawRowVectors);
                continue;
            }
            lockForDraw();
            noteDrawWriters(stages, submission.queue);
            phaseTiming.Phase(DrawRowVectors);
            localDevice->Draw(graphics, direct, stages, snapshots);
            phaseTiming.Phase(DrawRowGraphics);
            // The next record's materialization runs without the GPU mutex: held across the loop, it
            // kept the other queues and the committer waiting (~200 us per hold, 1.7 s per 10 s in
            // PPSA03671). Each record is recorded under the mutex, as separate draws are.
            // APS5_LOCKED_INDIRECT_RECORDS=1 keeps the mutex across the records as before.
            static const bool lockedRecords = std::getenv("APS5_LOCKED_INDIRECT_RECORDS") != nullptr;
            if (!lockedRecords && !lockedPrepare && gpuLock.owns_lock()) gpuLock.unlock();
        }
        flushBatch();
        timing.Mark("draw_and_resource_release");
        return drawn();
    }

    std::vector<std::shared_ptr<DispatchVariant>> recipeStages;
    // Recipes hold resources found at the recording draw's addresses: not for relocated draws.
    if (registerKey && !drawParameters.indirect && Graphics::DrawRecipes() && !relocatedDraw) {
        recipeStages.reserve(programs.size());
        for (std::size_t i = 0; i < programs.size(); ++i) recipeStages.push_back(drawHit ? matched[i] : fresh[i]);
        if (std::all_of(recipeStages.begin(), recipeStages.end(), [](const std::shared_ptr<DispatchVariant>& variant) { return variant == nullptr; })) recipeStages.clear();
    }
    std::shared_ptr<const DrawRecipe> recipe;
    if (drawHit && !recipeStages.empty() && !tracedPrograms) {
        recipe = findDrawRecipe(drawKey, recipeStages);
        if (recipe == nullptr) VulkanDevice::NoteDrawRecipeMiss(VulkanDevice::DrawRecipePrecheck::NoRecipe);
    }
    if (recipe == nullptr) {
        if (auto known = localDevice->KnownDrawRejection(graphics, stages)) {
            rejected = std::move(*known);
            return DrawVerdict::Rejected;
        }
    }
    // APS5_NATIVE_SHADOW: the descriptors the recompiler bound, against the ones the title's Shader
    // layouts place in this draw's user data.
    if (nativeRecord != nullptr) {
        std::vector<DecodedDescriptor> decoded;
        std::uint64_t structure = 0xcbf29ce484222325ull;
        std::uint64_t pushes = 0xcbf29ce484222325ull;
        const auto mix = [](std::uint64_t& hash, std::uint64_t value) { hash = (hash ^ value) * 0x100000001b3ull; };
        for (const auto& stage : stages) {
            if (stage.program == nullptr) continue;
            const auto [first, second] = stage.program->spirv.KeyHashes();
            mix(structure, first);
            mix(structure, second);
            mix(structure, stage.program->PipelineVariantId());
            for (const auto byte : stage.program->pushConstants) mix(pushes, static_cast<std::uint64_t>(byte));
            for (const auto word : stage.program->shaderDataUserWords) mix(structure, word);
            for (const auto& binding : stage.program->bindings) {
                mix(structure, static_cast<std::uint64_t>(binding.kind) << 32u | static_cast<std::uint64_t>(binding.role));
                mix(structure, static_cast<std::uint64_t>(binding.binding) << 32u | binding.count);
                const auto role = binding.role == ShaderRecompiler::DescriptorRole::GuestBuffers ? 0 : binding.role == ShaderRecompiler::DescriptorRole::GuestImages ? 1 : binding.role == ShaderRecompiler::DescriptorRole::GuestSamplers ? 2 : 3;
                if (role == 3 || binding.count == 0) continue;
                const auto elementWords = binding.guestDescriptor.size() / binding.count;
                if (elementWords < 4) continue;
                for (std::uint32_t element = 0; element < binding.count; ++element) {
                    const auto* words = binding.guestDescriptor.data() + element * elementWords;
                    if (words[0] == 0 && words[1] == 0 && words[2] == 0 && words[3] == 0) continue;
                    decoded.push_back({static_cast<std::uint8_t>(role), {words[0], words[1], words[2], words[3]}});
                }
            }
        }
        CompareNativeDescriptors(*nativeRecord, decoded, structure, pushes);
    }
    // A pipelined direct draw: the committer records it (commitDraw) after the draws queued before it.
    // The commit owns everything the recording reads: the stages' results, the snapshots and the
    // regions and pages they point into. With labels deferred on this worker the draw stays inline
    // (they would be recorded after it).
    // GPU-side indirect draws too: their records are read by the GPU, or by Graphics::Draw on the
    // committer, after everything before them.
    if (pipelined && deferredLabels().labels.empty()) {
        phaseTiming.Phase(DrawRowVectors);
        auto writes = drawWriteRanges(graphics, stages);
        // The guest memory this draw's preparation read (its captures, the decode's reads).
        std::vector<std::pair<std::uint64_t, std::uint64_t>> reads;
        if (ahead != nullptr) {
            reads.reserve(memory.size());
            for (const auto& region : memory) reads.emplace_back(region.guestAddress, region.guestAddress + region.bytes.size());
        }
        const auto commitQueue = submission.queue;
        const auto commitPacket = (packet[0] >> 8u) & 0xffu;
        const auto epoch = DrawPipeline::EpochToken().load(std::memory_order_relaxed);
        // The draw's resource-cache key, made here beside the committer's work (see commitDraw).
        auto probe = recipe == nullptr ? localDevice->ProbeDraw(graphics, drawParameters, stages, false) : Graphics::DrawResourceProbe{};
        auto commit = [this, localDevice = std::move(localDevice), decode = std::move(decode), probe = std::move(probe), entry = std::move(entry), drawParameters, stages = std::move(stages), snapshots = std::move(snapshots), recipe = std::move(recipe), recipeStages = std::move(recipeStages), drawKey, commitQueue, commitPacket, epoch, programs = std::move(programs), results = std::move(results), heldResults = std::move(heldResults), memory = std::move(memory), linked = std::move(linked), stageCaptures = std::move(stageCaptures), matched = std::move(matched), matchedRegions = std::move(matchedRegions), fresh = std::move(fresh), decodeReads = std::move(decodeReads), shaderMemory = std::move(shaderMemory)]() mutable {
            DrawPipeline::FollowEpoch(epoch);
            // The committer records the draw: APS5_GPU_TIME_DRAW_PROGRAMS names its pass by this program.
            Graphics::SetDrawTimingProgram(programs.empty() ? 0 : programs.back().binary.codeAddress);
            commitDraw(localDevice, commitQueue, commitPacket, decode->state, drawParameters, stages, snapshots, recipe, recipeStages, drawKey, std::move(probe));
        };
        auto held = std::make_shared<decltype(commit)>(std::move(commit));
        if (ahead != nullptr) {
            ahead->commit = [held] { (*held)(); };
            ahead->writes = std::move(writes);
            ahead->reads = std::move(reads);
            return drawn();
        }
        if (auto* windowWrites = DrawWindowWrites()) windowWrites->insert(windowWrites->end(), writes.begin(), writes.end());
        DrawPipeline::Queue0().Enqueue([held] { (*held)(); }, std::move(writes));
        phaseTiming.Phase(DrawRowLockWait);
        timing.Mark("draw_enqueued");
        return drawn();
    }
    lockForDraw();
    noteDrawWriters(stages, submission.queue);
    phaseTiming.Phase(DrawRowVectors);
    if (recipe != nullptr) {
        if (localDevice->DrawFromRecipe(graphics, drawParameters, stages, snapshots, recipe) == RecipeOutcome::Recorded) {
            phaseTiming.Phase(DrawRowGraphics);
            timing.Mark("draw_and_resource_release");
            return drawn();
        }

        VulkanDevice::NoteRecipe(VulkanDevice::RecipeEvent::Restart, VulkanDevice::RecipeKind::Draw);
    }
    std::shared_ptr<const DrawRecipe> built;
    localDevice->Draw(graphics, drawParameters, stages, snapshots, recipeStages.empty() ? nullptr : &built);
    phaseTiming.Phase(DrawRowGraphics);
    if (built != nullptr) attachDrawRecipe(drawKey, recipeStages, std::move(built));
    timing.Mark("draw_and_resource_release");
    return drawn();
}

// A pipelined draw's recording, on the committer. A draw that cannot be recorded is reported and
// skipped as an inline one is (PacketExecution); APS5_STRICT_DRAWS=1 makes it fatal, which stops
// the pipeline.
void Driver::commitDraw(std::shared_ptr<VulkanDevice> localDevice, std::uint32_t queue, std::uint32_t packetOpcode, const Graphics::State& graphics, const Pm4::DrawParameters& drawParameters, std::span<const Graphics::CompiledShader> stages, std::span<const Graphics::GuestMemorySnapshot> snapshots, const std::shared_ptr<const DrawRecipe>& recipe, const std::vector<std::shared_ptr<DispatchVariant>>& recipeStages, std::uint64_t drawKey, Graphics::DrawResourceProbe probe) {
    GuestMemory::SetCurrentPacket(packetOpcode, queue);
    try {
        // The resource-cache probe before the lock, of the key the worker made when it queued the
        // draw (Graphics::DrawResourceProbe). A device swapped meanwhile makes Draw ignore it.
        Graphics::FindDrawResources(probe);
        GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Draw);
        std::lock_guard gpuLock(GuestMemory::GpuMutex());
        if (auto current = device.Load(); current != nullptr && current != localDevice) localDevice = std::move(current);
        recordLabelsForPacket(localDevice.get(), queue);
        noteDrawWriters(stages, queue);
        if (recipe != nullptr) {
            if (localDevice->DrawFromRecipe(graphics, drawParameters, stages, snapshots, recipe) == RecipeOutcome::Recorded) return;
            VulkanDevice::NoteRecipe(VulkanDevice::RecipeEvent::Restart, VulkanDevice::RecipeKind::Draw);
        }
        std::shared_ptr<const DrawRecipe> built;
        localDevice->Draw(graphics, drawParameters, stages, snapshots, recipeStages.empty() ? nullptr : &built, &probe);
        if (built != nullptr) attachDrawRecipe(drawKey, recipeStages, std::move(built));
        // The batch cap's submit, which the worker leaves to the committer while draws are queued.
        if (batchCap() != 0 && Graphics::Recorder::RecordedWorkSinceSubmit() >= batchCap()) localDevice->SubmitRecorded(queue == 0);
    } catch (const std::exception& error) {
        static const bool strictDraws = std::getenv("APS5_STRICT_DRAWS") != nullptr;
        if (strictDraws) throw;
        countFrameSkip(error.what());
        reportSkip("draw", error.what());
    }
}

}
