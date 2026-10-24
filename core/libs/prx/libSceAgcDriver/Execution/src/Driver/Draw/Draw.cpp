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
#include <unordered_map>
#include <vector>

namespace AgcDriver::DriverDetail {

namespace {

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
            heldResults.push_back(materializeDrawStage(i, pushCursorBytes, queue, submission, programs, graphics, pixel, vertexInfos, memory, linked, drawParameters, localDevice, *shaderMemory, stageCaptures, recompiled, drawHit, matched, matchedRegions, profile, dumpTarget, dumpSlot1, captures, phaseTiming, phaseMs, rejected));
            if (!rejected.empty()) return DrawVerdict::Rejected;
            programResults[i] = heldResults.back().get();
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
    cacheDrawStages(useDrawEntries, drawHit, drawParameters, indirectCpu, programs, stageCaptures, vertexInfos, decodeReads, verifyHit, matched, fresh, relocationPlan ? relocationPlan->insertKey : drawKey, registerKey, decode, phaseTiming, relocationPlan ? &*relocationPlan : nullptr);
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
        auto commit = [this, localDevice = std::move(localDevice), decode = std::move(decode), entry = std::move(entry), drawParameters, stages = std::move(stages), snapshots = std::move(snapshots), recipe = std::move(recipe), recipeStages = std::move(recipeStages), drawKey, commitQueue, commitPacket, epoch, programs = std::move(programs), results = std::move(results), heldResults = std::move(heldResults), memory = std::move(memory), linked = std::move(linked), stageCaptures = std::move(stageCaptures), matched = std::move(matched), matchedRegions = std::move(matchedRegions), fresh = std::move(fresh), decodeReads = std::move(decodeReads), shaderMemory = std::move(shaderMemory)] {
            DrawPipeline::FollowEpoch(epoch);
            // The committer records the draw: APS5_GPU_TIME_DRAW_PROGRAMS names its pass by this program.
            Graphics::SetDrawTimingProgram(programs.empty() ? 0 : programs.back().binary.codeAddress);
            commitDraw(localDevice, commitQueue, commitPacket, decode->state, drawParameters, stages, snapshots, recipe, recipeStages, drawKey);
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
void Driver::commitDraw(std::shared_ptr<VulkanDevice> localDevice, std::uint32_t queue, std::uint32_t packetOpcode, const Graphics::State& graphics, const Pm4::DrawParameters& drawParameters, std::span<const Graphics::CompiledShader> stages, std::span<const Graphics::GuestMemorySnapshot> snapshots, const std::shared_ptr<const DrawRecipe>& recipe, const std::vector<std::shared_ptr<DispatchVariant>>& recipeStages, std::uint64_t drawKey) {
    GuestMemory::SetCurrentPacket(packetOpcode, queue);
    try {
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
        localDevice->Draw(graphics, drawParameters, stages, snapshots, recipeStages.empty() ? nullptr : &built);
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
