#include "prx/libSceAgcDriver/Execution/include/PerformanceTimer.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Diagnostics.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Shaders/ShaderRegistry.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/Pm4.hpp"
#include "Optimization/ResourceProgram.hpp"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace AgcDriver::DriverDetail {

std::shared_ptr<const ShaderRecompiler::RecompileResult> Driver::materializeDrawStage(std::size_t i, std::uint32_t pushOffset, const QueueState& queue, const Submission& submission, const std::vector<DrawProgram>& programs, const Graphics::State& graphics, const ShaderRecompiler::ShaderPixelStageInfo& pixel, const std::vector<std::optional<ShaderRecompiler::ShaderVertexStageInfo>>& vertexInfos, std::vector<ShaderRecompiler::MemoryRegion>& memory, const std::vector<ShaderRecompiler::LinkedProgram>& linked, const Pm4::DrawParameters& drawParameters, const std::shared_ptr<VulkanDevice>& localDevice, ShaderMemory& shaderMemory, std::vector<StageCapture>& stageCaptures, std::vector<bool>& recompiled, bool drawHit, const std::vector<std::shared_ptr<DispatchVariant>>& matched, const std::vector<std::vector<ShaderRecompiler::MemoryRegion>>& matchedRegions, bool profile, std::uint64_t dumpTarget, std::uint64_t dumpSlot1, std::uint64_t& captures, DrawPhaseTiming& phaseTiming, std::array<double, DrawDriverPhaseCount>& phaseMs, std::string& rejected) {
    PerformanceTimer timing("Shader.DrawStage");
    using Stage = ShaderRecompiler::ShaderStage;
    phaseTiming.Phase(DrawRowVectors);
    const auto& program = programs[i];
    const auto waveSize = program.binary.stage == Stage::Fragment ? graphics.stages.fragmentWaveSize : graphics.stages.vertexWaveSize;
    ShaderRecompiler::RecompileRequest request{
        program.binary,
        {waveSize, program.firstUserSgpr, program.userData, std::nullopt, program.binary.stage == Stage::Fragment ? std::optional(pixel) : std::nullopt, vertexInfos[i], memory, RegisteredFloatMode(*program.snapshot)},
        localDevice->Target(),
        {0, 0, pushOffset, (graphics.stages.mesh ? ShaderRecompiler::MeshDrawPushOffsetBytes : graphics.stages.tessellation && pushOffset < Graphics::PipelinePushSlotBytes ? ShaderRecompiler::TessellationLevelPushOffsetBytes : Graphics::PipelinePushSlotBytes) - pushOffset % Graphics::PipelinePushSlotBytes},
        ShaderRecompiler::GraphicsCompileContext{program.firstUserSgpr, linked, graphics.stages.mesh, graphics.stages.tessellation, {drawParameters.indexAddress, drawParameters.indexCount, drawParameters.indexSize, drawParameters.instanceCount}}
    };
    const auto waitedBefore = traceCapSync() || profile ? Graphics::Recorder::ThreadWaitedMs() : 0.0;
    // Diagnostic: APS5_DUMP_SHADERS also writes the requests of graphics stages prepared at a draw
    // (once per address, after the capture so the request holds its memory; agc_shader_replay
    // reads them).
    static const bool dumpShaders = std::getenv("APS5_DUMP_SHADERS") != nullptr;
    const auto invocation = InvocationFor(*program.snapshot, program.codeOffset, request, true);
    timing.Mark("prepared_invocation");
    auto& stageCapture = stageCaptures[i];
    stageCapture.forgetSerial = GuestMemory::ForgetSerial();
    stageCapture.pushOffset = pushOffset;
    const auto capture = [&] {
        const SampledReadScope sampling(evidenceReads);
        ShaderMemory::NoteCaptureProgram(program.binary.codeAddress);
        return shaderMemory.Capture(invocation);
    }();

    stageCapture.regions = shaderMemory.TakeRecentRegions();
    if (capture != nullptr) {
        stageCapture.userWordsAsBase = capture->userDataUse.asBase;
        stageCapture.userWordsAsValue = capture->userDataUse.asValue;
        stageCapture.captured = true;
        // APS5_PROFILE_DRAW: native coverage, the stages whose resources need no CPU read of guest
        // memory (user data alone; everything else resolved on the GPU), on a [walk-free] line.
        if (profile) {
            static std::atomic<std::uint64_t> stages{0}, walkFree{0}, words{0};
            static std::atomic<long long> lastReport{0};
            ++stages;
            words += capture->userDataUse.reads.size();
            if (capture->userDataUse.reads.empty()) ++walkFree;
            const auto now = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
            auto last = lastReport.load(std::memory_order_relaxed);
            if (last == 0) lastReport.compare_exchange_strong(last, now);
            else if (now - last >= 10 && lastReport.compare_exchange_strong(last, now)) {
                const auto total = stages.exchange(0);
                const auto free = walkFree.exchange(0);
                std::fprintf(stderr, "[walk-free] (10 s): %llu of %llu captured draw stages read no guest memory (%.1f%%), %.1f guest words read per stage\n", static_cast<unsigned long long>(free), static_cast<unsigned long long>(total), total != 0 ? 100.0 * static_cast<double>(free) / static_cast<double>(total) : 0.0, total != 0 ? static_cast<double>(words.exchange(0)) / static_cast<double>(total) : 0.0);
            }
        }
    }
    recompiled[i] = true;
    memory = shaderMemory.Regions();

    if (drawHit) {
        for (std::size_t j = 0; j < programs.size(); ++j) {
            if (matched[j] != nullptr && !recompiled[j]) memory.insert(memory.end(), matchedRegions[j].begin(), matchedRegions[j].end());
        }
    }
    request.context.memory = memory;
    if (dumpShaders) static_cast<void>(dumpRequest(program.binary.codeAddress, request));
    if (traceCapSync()) traceCapture("draw-capture", program.binary.codeAddress, submission.queue, memory, Graphics::Recorder::ThreadWaitedMs() - waitedBefore);
    if (profile) {
        ++captures;
        phaseTiming.Phase(DrawRowCapture);

        const auto waited = std::min(Graphics::Recorder::ThreadWaitedMs() - waitedBefore, phaseMs[DrawRowCapture]);
        phaseMs[DrawRowCapture] -= waited;
        phaseMs[DrawRowCaptureHookWaits] += waited;
    }
    if (dumpTarget != 0) {

        const auto slot0 = (static_cast<std::uint64_t>(readRegister(queue.context, 0x390)) << 40u) | (static_cast<std::uint64_t>(readRegister(queue.context, 0x318)) << 8u);
        if ((graphics.hasColorTarget && graphics.color.address == dumpTarget) || slot0 == dumpTarget) static_cast<void>(dumpRequest(program.binary.codeAddress, request));
    }
    if (dumpSlot1 != 0) {
        const auto value = [&](std::uint32_t offset) -> std::uint64_t { const auto it = queue.context.find(offset); return it == queue.context.end() ? 0u : it->second; };
        const auto slot1 = (value(0x391) << 40u) | (value(0x327) << 8u);
        if (slot1 == dumpSlot1) {
            static_cast<void>(dumpRequest(program.binary.codeAddress, request));
            if (std::FILE* file = std::fopen("draw_slot1.regs", "w")) {
                for (const auto& [offset, value] : queue.context) std::fprintf(file, "context %x %08x\n", offset, value);
                for (const auto& [offset, value] : queue.userConfig) std::fprintf(file, "uconfig %x %08x\n", offset, value);
                for (const auto& [offset, value] : queue.shader) std::fprintf(file, "shader %x %08x\n", offset, value);
                std::fclose(file);
            }
        }
    }

    phaseTiming.Phase(DrawRowCapture);

    timing.Mark("capture_resources");
    // A specialized module the draw needs that is not built yet is built in the background and the
    // draw skipped (ShaderRecompiler::SpecializationPending), as for its pipeline parts.
    stageCapture.compiled = [&] {
        const ShaderRecompiler::DeferSpecializationScope defer;
        return invocation.Materialize(*capture);
    }();
    timing.Mark("materialize");
    // The shared result itself: a copy per stage and draw (its binding, push and export vectors)
    // was most of the recompile phase. A caller that patches it (CPU-side indirect records)
    // copies it there.
    phaseTiming.Phase(DrawRowRecompile);
    return stageCapture.compiled;
}

void Driver::cacheDrawStages(bool useDrawEntries, bool drawHit, const Pm4::DrawParameters& drawParameters, const std::optional<Graphics::IndirectDrawPath>& indirectCpu, const std::vector<DrawProgram>& programs, const std::vector<StageCapture>& stageCaptures, const std::vector<std::optional<ShaderRecompiler::ShaderVertexStageInfo>>& vertexInfos, const std::vector<std::vector<Graphics::DecodeRead>>& decodeReads, bool verifyHit, const std::vector<std::shared_ptr<DispatchVariant>>& matched, std::vector<std::shared_ptr<DispatchVariant>>& fresh, std::uint64_t drawKey, bool registerKey, const std::shared_ptr<const DrawDecode>& decode, DrawPhaseTiming& phaseTiming, const DrawRelocationPlan* relocation) {
    // A relocated insert (DrawRelocation.cpp) of a new shape goes under its own key at once: that
    // key recurs. A plan that may not insert still lets a learning or verified hit compare below.
    const bool insertable = relocation == nullptr || (relocation->insert && !relocation->learning);
    if (!insertable && !verifyHit) return;
    const bool admitted = relocation != nullptr && relocation->admit;
    if (useDrawEntries && !drawHit && !(drawParameters.indirect && indirectCpu) && (verifyHit || admitted || admitDrawKey(drawKey))) {
        phaseTiming.Phase(DrawRowVectors);
        std::uint64_t unstable = 0, mismatches = 0;
        for (std::size_t i = 0; i < programs.size(); ++i) {
            const auto& stageCapture = stageCaptures[i];
            if (stageCapture.compiled == nullptr || !CacheableResult(*stageCapture.compiled)) continue;
            auto variant = std::make_shared<DispatchVariant>();
            variant->compiled = stageCapture.compiled;
            if (programs[i].binary.stage == ShaderRecompiler::ShaderStage::Mesh) {
                std::copy_n(programs[i].userData.begin() + ShaderRecompiler::MeshIndexBufferUserWord, variant->meshIndexDescriptor.size(), variant->meshIndexDescriptor.begin());
            }
            variant->shader = programs[i].snapshot;
            variant->forgetSerial = stageCapture.forgetSerial;
            variant->pushOffset = stageCapture.pushOffset;
            if (vertexInfos[i]) variant->vertexInfo = std::make_shared<const ShaderRecompiler::ShaderVertexStageInfo>(*vertexInfos[i]);

            std::vector<ShaderRecompiler::MemoryRegion> regions(stageCapture.regions.begin(), stageCapture.regions.end());
            for (const auto& read : decodeReads[i]) regions.push_back({read.address, std::as_bytes(std::span(read.bytes))});
            std::stable_sort(regions.begin(), regions.end(), [](const ShaderRecompiler::MemoryRegion& a, const ShaderRecompiler::MemoryRegion& b) { return a.guestAddress < b.guestAddress; });
            for (const auto& region : regions) {
                variant->runs.emplace_back(region.guestAddress, region.guestAddress + region.bytes.size());
                const auto count = region.bytes.size() / sizeof(std::uint32_t);
                const auto offset = variant->words.size();
                variant->words.resize(offset + count);
                std::memcpy(variant->words.data() + offset, region.bytes.data(), count * sizeof(std::uint32_t));
            }
            if (insertable && relocation != nullptr && i < relocation->stages.size() && relocation->stages[i] && !relocateVariant(*variant, programs[i].userData)) return;
            if (insertable && relocation != nullptr && i < relocation->patches.size()) variant->userWordPatches = relocation->patches[i];
            std::vector<std::pair<std::uint64_t, std::uint64_t>> matchedRuns;
            if (verifyHit && matched[i] != nullptr) {
                if (matched[i]->runPointerWords.empty() || !relocatedRuns(*matched[i], programs[i].userData, matchedRuns)) matchedRuns = matched[i]->runs;
            }
            // A relocated or pointer-patched stage (DrawRelocation.cpp): this draw's full compile
            // confirms or refuses it once - the same runs at this draw's pointers (absolute ones
            // unmoved), the same words, and a patched copy equal to this compile - and checks it
            // again under APS5_VERIFY_DRAW_ENTRIES.
            if (verifyHit && matched[i] != nullptr && (!matched[i]->userWordPatches.empty() || !matched[i]->runPointerWords.empty()) && stageCapture.compiled != nullptr) {
                const auto patched = patchedResult(*matched[i], programs[i].userData);
                const bool same = patched != nullptr && sameCompiledResult(*patched, *stageCapture.compiled) && matchedRuns == variant->runs && matched[i]->words == variant->words;
                auto state = matched[i]->patchState.load();
                if (state == 0 && matched[i]->patchState.compare_exchange_strong(state, same ? 1 : -1)) notePatchOutcome(same ? 1 : -1);
                else if (state == 1 && !same) notePatchOutcome(3);
            }
            if (verifyHit && matched[i] != nullptr && (matchedRuns != variant->runs || matched[i]->words != variant->words)) {
                ++mismatches;
                static std::atomic<std::uint64_t> reports{0};
                if (reports.fetch_add(1) < 20) std::fprintf(stderr, "[draw-cache] verify: stage %zu (program 0x%llx) of a hit captured differently: %zu runs / %zu words matched, %zu / %zu fresh\n", i, static_cast<unsigned long long>(programs[i].binary.codeAddress), matched[i]->runs.size(), matched[i]->words.size(), variant->runs.size(), variant->words.size());
            }
            if (insertCompare()) {
                const GuestMemory::ReadSiteScope site(GuestMemory::ReadSite::DrawCache);
                if (!captureStable(stageCapture.regions)) {
                    ++unstable;
                    continue;
                }
            }
            fresh[i] = std::move(variant);
        }
        // A learning draw keeps the entry it confirmed or refused.
        if (!insertable) return;
        insertDrawEntry(drawKey, fresh, registerKey ? decode : nullptr);
        if (unstable != 0 || mismatches != 0) {
            std::lock_guard cacheLock(drawCacheMutex);
            drawEntryCounters.unstable += unstable;
            drawEntryCounters.verifyMismatches += mismatches;
        }
        phaseTiming.Phase(DrawRowKeyLookupValidate);
    }
}

}
