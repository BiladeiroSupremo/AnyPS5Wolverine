#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "ThreadOwned.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Diagnostics.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawAhead.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPipeline.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Synchronization/SynchronizationStatistics.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Synchronization/DeferredLabels.hpp"
#include "prx/libSceAgcDriver/Execution/include/CaptureTrace.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/PerformanceTimer.hpp"
#include "prx/libSceAgcDriver/Execution/include/Pm4.hpp"
#include "prx/libSceAgcDriver/Execution/include/ShaderMemory.hpp"
#include "prx/libSceAgcDriver/Eq/include/Event.hpp"
#include "Optimization/ResourceMaterializer.hpp"
#include <atomic>
#include <cstdlib>
#include <functional>
#include <map>
#include <string>
#include <shared_mutex>

namespace AgcDriver::DriverDetail {

namespace {

// The packets queue 0 executes with pipelined draws still queued (APS5_PIPELINED_DRAWS): draws (an
// indirect one drains in draw()), the register, constant-RAM and draw-state packets (the indirect
// register loads 0x63/0x64/0x9f take lists read at submission), and the events this driver executes
// as nothing (0x42, 0x46 but the sample dump 0x39, 0x58, 0x59). None reads or writes guest memory or
// records GPU work. Every other packet waits for the queued draws first. Upstream's list also keeps
// fills, DMA, labels and satisfied waits, which need its ordered commit items.
bool keepsDrawPipeline(std::span<const std::uint32_t> packet, std::uint32_t header, std::uint32_t opcode) {
    if (header == FlipPacketHeader || header == RenderingWaitPacketHeader) return false;
    if (Pm4::DrawOpcode(opcode)) return true;
    switch (opcode) {
        case 0x11: case 0x12: case 0x13: case 0x26: case 0x2a: case 0x2f: case 0x3f: case 0x42: case 0x58: case 0x59: case 0x63: case 0x64: case 0x69: case 0x76: case 0x79: case 0x7a: case 0x81: case 0x9f: return true;
        case 0x46: return packet.size() >= 2 && (packet[1] & 0x3fu) != 0x39u;
        case 0x10: {
            const auto operation = (header >> 2u) & 0x3fu;
            return operation == 0 || operation == 0x09 || operation == 0x0b || operation == 0x0c || operation == 0x1a;
        }
        default: return false;
    }
}

}

template <typename TWork>
void Driver::timed(double WorkerProfile::*bucket, TWork&& work) {
    static thread_local WorkerProfile profile;
    const auto begin = std::chrono::steady_clock::now();
    work();
    const auto end = std::chrono::steady_clock::now();
    profile.*bucket += std::chrono::duration<double, std::milli>(end - begin).count();
    static const bool report = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    if (report && end - profile.reported > std::chrono::seconds(10)) {
        profile.reported = end;
        AgcDriver::ProfilePrint_nid_no_patch( "[gpu] worker at %.0f s: dispatch %.1f s, draw %.1f s, wait %.1f s\n", std::chrono::duration<double>(end - profile.start).count(), profile.dispatchMs / 1000, profile.drawMs / 1000, profile.waitMs / 1000);
    }
}

void Driver::execute(const Submission& submission) {
    auto* submissionTiming = includeTimingSubmission(submission, true);
    struct FlipHold {
        Driver& driver;
        bool held;
        void Release() {
            if (!held) return;
            held = false;
            driver.releaseFlipHold();
        }
        ~FlipHold() { Release(); }
    } flipHold{*this, submission.holdsFlip};
    // APS5_PIPELINED_DRAWS: queue 0's worker hands its direct draws to the committer
    // (DrawPipeline); a submission ends with them recorded, so its completion and everything after it
    // see their work (also when a packet throws).
    const bool pipelined = submission.queue == 0 && DrawPipeline::Depth() != 0;
    DrawPipeline::Active() = pipelined;
    struct PipelineDrain {
        bool on;
        ~PipelineDrain() {
            if (!on) return;
            // Draws prepared ahead read this submission's packets: none outlives it.
            Driver::Get().finishDrawWindow();
            if (!DrawPipeline::Queue0().Busy()) return;
            try {
                DrawPipeline::Queue0().Drain(DrawPipeline::DrainReason::Submission);
            } catch (...) {
            }
        }
    } pipelineDrain{pipelined};
    if (submission.suspend) {
        if (pipelined) DrawPipeline::Queue0().Drain(DrawPipeline::DrainReason::Submission);
        PerformanceContext timingContext(submissionTiming);
        PerformanceTimer timing("Driver.Suspend");

        static const bool suspendDrain = std::getenv("APS5_SUSPEND_DRAIN") != nullptr;
        static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
        ++suspendPoints;
        auto& costs = submissionCosts(submission.queue);

        if (suspendDrain || !deferredLabels().labels.empty() || Graphics::Recorder::PendingLabelSince().has_value() || Graphics::Recorder::RecordedWorkSinceSubmit() != 0) {
            const auto start = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Flush);
            std::lock_guard gpuLock(GuestMemory::GpuMutex());
            const auto localDevice = device.Load();
            recordDeferredLabels(localDevice.get(), submission.queue);
            if (localDevice != nullptr) {
                if (suspendDrain) localDevice->WaitIdle();
                else localDevice->SubmitRecorded(submission.queue == 0);
            }
            ++costs.suspends;
            if (profile) costs.suspendNs += static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count());
        } else {
            ++costs.suspendsSkipped;
        }
        resetGraphics = true;
        return;
    }
    QueueState* state = nullptr;
    {
        std::lock_guard lock(mutex);
        if (submission.queue == 0 && resetGraphics) {
            queues.erase(0);
            resetGraphics = false;
        }
        state = &queues[submission.queue];
    }
    auto& queue = *state;
    static const bool traceGpu = std::getenv("APS5_TRACE_GPU") != nullptr;
    if (traceGpu) std::fprintf(stderr, "[gpu] %.1f execute serial=%llu queue=0x%x dwords=%zu\n", TraceMs(), static_cast<unsigned long long>(submission.serial), submission.queue, submission.commands.size());

    static const long dumpQueue = [] { const char* text = std::getenv("APS5_DUMP_QUEUE"); return text ? std::strtol(text, nullptr, 16) : -1L; }();
    if (static_cast<long>(submission.queue) == dumpQueue) {

        static int dumped = 0;
        if (dumped++ < 40) {
            std::string text = "[queue] submission " + std::to_string(submission.serial) + ":\n";
            for (std::size_t cursor = 0; cursor < submission.commands.size();) {
                const auto header = submission.commands[cursor];
                const auto count = Pm4::PacketWords(header);
                char line[200];
                int length = std::snprintf(line, sizeof(line), "[queue]   %s", Pm4::Name(header).c_str());
                for (std::size_t i = 1; i < count && i < 10 && length < 180; ++i) length += std::snprintf(line + length, sizeof(line) - length, " %08x", submission.commands[cursor + i]);
                text += line;
                text += "\n";
                cursor += count;
            }
            std::fputs(text.c_str(), stderr);
        }
    }
    PacketHistory recent{submission.commands};

    static const bool profilePackets = std::getenv("APS5_PROFILE_DRAW") != nullptr;

    thread_local PacketProfile* packetProfileSlot = nullptr;
    auto& packetProfile = ShaderRecompiler::ThreadOwned(packetProfileSlot);
    ++packetProfile.submissions;

    bumpEpoch(&EpochBumps::submissions);
    clearResolvedAhead();
    gateOpened() = true;
    struct ClearAhead {
        ~ClearAhead() { clearResolvedAhead(); }
    } clearAhead;
    for (std::size_t cursor = 0; cursor < submission.commands.size();) {
        if (packetEpoch()) bumpEpoch(&EpochBumps::packets);
        CheckFailure();
        const auto header = submission.commands[cursor];
        if (Pm4::FillerPacket(header)) { ++cursor; continue; }
        const auto count = Pm4::PacketWords(header);
        const auto packet = std::span(submission.commands).subspan(cursor, count);
        const auto opcode = (header >> 8u) & 0xffu;
        auto nextCursor = cursor + count;
        // A label behind queued draws, or inside a window of draws prepared ahead (DrawAhead.hpp), is
        // stored by the committer after them (commitLabelPacket); APS5_NO_PIPELINED_LABELS=1 waits for
        // the draws and stores it here instead.
        static const bool labelItems = std::getenv("APS5_NO_PIPELINED_LABELS") == nullptr;
        if (pipelined && labelItems && (DrawPipeline::Queue0().Busy() || DrawWindowWrites() != nullptr) && !Pm4::Predicated(header) && labelCommitItem(packet)) {
            const auto label = Pm4::DecodeLabelWrite(packet);
            std::vector<DrawPipeline::Range> writes{{label->address, label->address + label->Bytes().size()}};
            if (auto* windowWrites = DrawWindowWrites()) windowWrites->insert(windowWrites->end(), writes.begin(), writes.end());
            const auto labelQueue = submission.queue;
            DrawPipeline::Queue0().Enqueue([this, labelQueue, words = std::vector<std::uint32_t>(packet.begin(), packet.end())] { commitLabelPacket(labelQueue, words); }, std::move(writes));
            cursor = nextCursor;
            continue;
        }
        if (pipelined && DrawPipeline::Queue0().Busy()) {
            // A predicated packet reads its predicate first: queued draws that may write it go first.
            const bool predicateQueued = Pm4::Predicated(header) && queue.predication.operation != 0 && DrawPipeline::Queue0().Overlaps(queue.predication.address, queue.predication.operation == 3 ? 8 : 4);
            if (!keepsDrawPipeline(packet, header, opcode) || predicateQueued) DrawPipeline::Queue0().Drain(DrawPipeline::DrainReason::Packet, header == FlipPacketHeader ? 0x100u : opcode);
        }
        if (APS5_ENABLE_TIMING_LOG && submission.queue == 0 && pendingFrameTiming == nullptr) includeTimingSubmission(submission, false);
        PerformanceContext timingContext(submission.queue == 0 ? pendingFrameTiming.get() : FrameTiming::Async());
        PerformanceTimer timing("Driver.Packet");
        std::shared_lock deviceUse(deviceReplacement, std::defer_lock);
        if (opcode != 0x3c && opcode != 0x93 && header != RenderingWaitPacketHeader && header != FlipPacketHeader) deviceUse.lock();

        timing.Mark("device_lock");
        GuestMemory::SetCurrentPacket(header == FlipPacketHeader ? 0xffffu : opcode, submission.queue);
        CaptureTrace::Log("packet submission=%llu queue=%x offset=%zu header=%08x words=%zu", static_cast<unsigned long long>(submission.serial), submission.queue, cursor, header, packet.size());
        if (Pm4::Predicated(header) && queue.predication.operation != 0) {
            landPendingWaits(submission.queue);
            recordQueuedLabelsBeforeRead(submission.queue);
            if (!Pm4::PredicationPasses(queue)) {
                cursor = opcode == 0x3f ? submission.conditionalEnds.at(cursor) : nextCursor;
                continue;
            }
        }
        if (opcode == 0x3f) {
            cursor = nextCursor;
            continue;
        }

        const auto flushStart = profilePackets ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        flushBetweenPackets(submission.queue, header, opcode == 0x49 || opcode == 0x37);
        timing.Mark("flush_between_packets");
        PacketTimer packetTimer{profilePackets, header == FlipPacketHeader ? 0xffffu : opcode, submission.queue, packetProfile, std::chrono::steady_clock::now()};

        if (profilePackets) {
            packetStartedAt() = packetTimer.start;
            pendingDispatchPhases() = {};
            pendingDrawPhases() = {};
        }
        const auto finishDispatchPacket = [&](bool indirect) {
            if (!profilePackets) return;
            const auto now = std::chrono::steady_clock::now();
            auto& pending = pendingDispatchPhases();
            if (pending.phases) {
                pending.ms[PhaseEpilogue] = std::chrono::duration<double, std::milli>(now - pending.tailAt).count();
                addDriverPhases(submission.queue != 0 ? OtherQueues : indirect ? Queue0Indirect : Queue0Direct, pending.ms, pending.hit, pending.validated);
            }
            if (indirect) return;
            auto& row = packetProfile.dispatchOutcomes[static_cast<std::size_t>(pending.outcome)];
            ++row.first;
            row.second += std::chrono::duration<double, std::milli>(now - packetTimer.start).count();
        };
        const auto finishDrawPacket = [&](bool drawn) {
            if (!profilePackets) return;
            const auto now = std::chrono::steady_clock::now();
            auto& pending = pendingDrawPhases();
            std::array<double, DrawDriverPhaseCount> ms{};
            if (drawn && pending.phases) {
                ms = pending.ms;
                ms[DrawRowEpilogue] = std::chrono::duration<double, std::milli>(now - pending.tailAt).count();
            } else {
                ms[DrawRowSkipped] = std::chrono::duration<double, std::milli>(now - packetTimer.start).count();
            }
            addDrawPhases(ms, drawn && pending.phases, pending.captures);
        };
        if (profilePackets) packetProfile.flushMs += std::chrono::duration<double, std::milli>(packetTimer.start - flushStart).count();

        bool wroteOnGpu = false, endOfPipeInterrupt = false, interruptDeferred = false, drawPacket = false, sampleDump = false;
        const bool drains = preparePacketMemory(submission, queue, packet, header, opcode, wroteOnGpu, endOfPipeInterrupt, interruptDeferred, drawPacket, sampleDump);
        traceLabel(packet, submission.queue);
        timing.Mark("prepare_memory");

        const bool waitPacket = opcode == 0x3c || opcode == 0x93 || header == RenderingWaitPacketHeader;
        struct Progress {
            Driver& driver;
            bool counted;
            ~Progress() {
                if (!counted) return;
                --driver.packetsInFlight;
                ++driver.packetsDone;
            }
        } progress{*this, !waitPacket};
        if (!waitPacket) ++packetsInFlight;
        recent.Record(cursor);
        if (header == RenderingWaitPacketHeader) {
            timed(&WorkerProfile::waitMs, [&] { submission.renderingWaits.at(cursor)->Wait(); });
        } else if (header == FlipPacketHeader) {
            CheckFailure();
            std::uint64_t batchesAtFlip = 0, unsignaledAtFlip = 0;
            bool flipInterruptDeferred = false;
            if (!drains) {

                GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Flush);
                std::lock_guard gpuLock(GuestMemory::GpuMutex());
                const auto localDevice = device.Load();

                if (!recordLabelsForPacket(localDevice.get(), submission.queue) && localDevice != nullptr) localDevice->SubmitRecorded(submission.queue == 0);
                if (localDevice != nullptr) {
                    localDevice->FlipBatches(batchesAtFlip, unsignaledAtFlip);
                    const auto queueId = submission.queue;
                    flipInterruptDeferred = localDevice->AfterRecordedWork([queueId] { AgcDriverDeliverEopInterrupt(queueId); }, false);
                }
            }
            if (!flipInterruptDeferred) AgcDriverDeliverEopInterrupt(submission.queue);
            ++flipsCounted;
            if (batchesAtFlip != 0) flipSerial = batchesAtFlip;
            flipBatchesUnsignaled += unsignaledAtFlip;

            timing.Mark("flip_flush");
            timing.FinishPacket(submission.serial, cursor, 0xffffu, count);
            auto frame = APS5_ENABLE_TIMING_LOG ? std::move(pendingFrameTiming) : std::make_shared<FrameTiming>(++frameSerial);
            require(frame != nullptr, "flip has no frame timing");
            const auto now = FrameTiming::Clock::now();
            if (!APS5_ENABLE_TIMING_LOG) frame->IncludeSubmission(submission.serial, now, now, now, true);
            frame->SetFlip(submission.serial, cursor, APS5_ENABLE_TIMING_LOG ? submission.receivedAt : now, now);
            if (APS5_ENABLE_TIMING_LOG) frame->CollectBackground();
            frame->NoteFlipBatches(batchesAtFlip, unsignaledAtFlip);
            CaptureTrace::Log("flip frame=%llu submission=%llu offset=%zu batch=%llu unsignaled=%llu", static_cast<unsigned long long>(frameSerial), static_cast<unsigned long long>(submission.serial), cursor, static_cast<unsigned long long>(batchesAtFlip), static_cast<unsigned long long>(unsignaledAtFlip));
            flipHold.Release();
            submission.flips.at(cursor)->GpuReady(frame);
        } else if (opcode == 0x15 || opcode == 0x16) {
            landPendingWaits(submission.queue);
            currentPacketOffset() = cursor;
            if (gateOpened()) {
                gateOpened() = false;
                timed(&WorkerProfile::dispatchMs, [&] { resolveGroupAhead(submission, queue, cursor); });
            }
            // A dispatch whose bindless image table cannot be bound is reported and skipped, as draws
            // that cannot be translated are; every other failure stays fatal.
            // A dispatch whose resource walk read through a null pointer (an async-compute dispatch
            // left with null user data by the title) is reported and skipped too: on hardware such a
            // dispatch faults unless it runs no group.
            timed(&WorkerProfile::dispatchMs, [&] {
                static_cast<void>(ShaderMemory::TakeNullPageRead());
                try {
                    if (opcode == 0x15) dispatch(queue, packet, submission);
                    else dispatchIndirect(queue, packet, submission);
                } catch (const ShaderRecompiler::BindlessTableRejected& error) {
                    reportSkip("dispatch", error.what());
                } catch (const ShaderRecompiler::RuntimeImageMismatch& error) {
                    // PPSA03671 ended (std::terminate through the failed flip) when its frame
                    // composite 0x11d03ebc00 was handed a 3D T# for a 2D interface once.
                    reportSkip("dispatch", error.what());
                } catch (const DispatchShaderRejected& error) {
                    reportSkip("dispatch", error.what());
                } catch (const std::runtime_error& error) {
                    if (!ShaderMemory::TakeNullPageRead()) throw;
                    reportSkip("dispatch", error.what());
                }
            });
            Graphics::Recorder::CountRecordedWork();
            finishDispatchPacket(opcode == 0x16);
        } else if (opcode == 0x3c || opcode == 0x93) {
            static const bool traceGpu = std::getenv("APS5_TRACE_GPU") != nullptr;
            const auto waitStart = std::chrono::steady_clock::now();
            timed(&WorkerProfile::waitMs, [&] { waitMemory(packet, submission.queue, recent, submission.received, submission.heldAtSubmit.contains(cursor)); });
            gateOpened() = true;
            if (traceGpu && std::chrono::steady_clock::now() - waitStart > std::chrono::milliseconds(200)) {

                for (std::size_t next = cursor + count, shown = 0; next < submission.commands.size() && shown < 48; ++shown) {
                    const auto nextHeader = submission.commands[next];
                    const auto nextCount = Pm4::PacketWords(nextHeader);
                    const auto nextPacket = std::span(submission.commands).subspan(next, nextCount);
                    std::fprintf(stderr, "[gpu]   then %s", Pm4::Name(nextHeader).c_str());
                    for (std::size_t i = 1; i < nextPacket.size() && i < 7; ++i) std::fprintf(stderr, " %08x", nextPacket[i]);
                    std::fprintf(stderr, "\n");
                    next += nextCount;
                }
            }
        } else if (drawPacket) {
            landPendingWaits(submission.queue);
            bool drawn = false;
            timed(&WorkerProfile::drawMs, [&] {
                static const bool traceDraws = std::getenv("APS5_TRACE_DRAWS") != nullptr;
                static const bool profileDraws = std::getenv("APS5_PROFILE_DRAW") != nullptr;
                const auto color = (static_cast<std::uint64_t>(readRegister(queue.context, 0x390)) << 40u) | (static_cast<std::uint64_t>(readRegister(queue.context, 0x318)) << 8u);
                const auto started = profileDraws ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
                const auto countSkip = [&](Graphics::DrawSkip kind) {
                    if (profileDraws) Graphics::CountDrawSkip(kind, std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - started).count());
                };

                try {
                    std::string rejected;
                    const auto verdict = pipelined && DrawAheadThreads() != 0 ? drawAhead(queue, packet, submission, rejected, cursor) : draw(queue, packet, submission, rejected);
                    drawn = verdict == DrawVerdict::Drawn;
                    CaptureTrace::Log("draw submission=%llu queue=%x offset=%zu target=%llx mask=%x verdict=%d reason=%.256s", static_cast<unsigned long long>(submission.serial), submission.queue, cursor, static_cast<unsigned long long>(color), readRegister(queue.context, 0x8e), static_cast<int>(verdict), rejected.c_str());
                    if (verdict == DrawVerdict::Rejected) {
                        countSkip(Graphics::DrawSkip::Prechecked);
                        throw std::runtime_error(rejected);
                    } else if (verdict == DrawVerdict::Nothing) {
                        countSkip(Graphics::DrawSkip::Nothing);
                    } else if (traceDraws) {
                        std::fprintf(stderr, "[draw] target 0x%llx mask 0x%x ok\n",static_cast<unsigned long long>(color), readRegister(queue.context, 0x8e));
                    }
                } catch (const std::exception& error) {
                    CaptureTrace::Log("draw-error submission=%llu offset=%zu reason=%.256s", static_cast<unsigned long long>(submission.serial), cursor, error.what());
                    countSkip(Graphics::DrawSkip::Thrown);
                    // The skips that cost a frame (APS5_PROFILE_DRAW): which reason took >= 5 ms.
                    if (profileDraws) {
                        static std::atomic<int> slowSkips{0};
                        const auto ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
                        if (ms >= 5.0 && slowSkips.fetch_add(1, std::memory_order_relaxed) < 400) std::fprintf(stderr, "[draw] slow skip %.1f ms at present %llu (%s): %.200s\n", ms, static_cast<unsigned long long>(Graphics::Recorder::Presents()), DescribeThrownDrawPhases().c_str(), error.what());
                        ThrownDrawPhases() = {};
                        // Per presented frame, the skipped draws by reason when a frame lost 5 or more
                        // (which frames a missing pass or a black picture came from).
                        static std::uint64_t framePresent = 0;
                        static std::map<std::string, std::uint32_t> frameReasons;
                        static std::atomic<int> framesPrinted{0};
                        const auto present = Graphics::Recorder::Presents();
                        if (present != framePresent) {
                            // Counted: the skips of work still being built in the background (shader
                            // preparation, specialization, pipeline parts); the steady per-frame ones
                            // (unsupported states) would print every frame.
                            std::uint32_t skipped = 0;
                            for (const auto& [reason, count] : frameReasons) {
                                if (reason.find("background") != std::string::npos || reason.find("being prepared") != std::string::npos || reason.find("being built") != std::string::npos) skipped += count;
                            }
                            if (skipped >= 3 && framesPrinted.fetch_add(1, std::memory_order_relaxed) < 2000) {
                                std::string text;
                                for (const auto& [reason, count] : frameReasons) text += " " + std::to_string(count) + "x " + reason + ";";
                                std::fprintf(stderr, "[draw] present %llu skipped %u draws:%s\n", static_cast<unsigned long long>(framePresent), skipped, text.c_str());
                            }
                            frameReasons.clear();
                            framePresent = present;
                        }
                        ++frameReasons[std::string(error.what()).substr(0, 60)];
                    }
                    // A draw that cannot be translated is reported and skipped, so one unsupported
                    // state does not end a title; APS5_STRICT_DRAWS=1 makes it fatal.
                    static const bool strictDraws = std::getenv("APS5_STRICT_DRAWS") != nullptr;
                    if (strictDraws) throw;
                    reportSkip("draw", error.what());
                }
            });
            finishDrawPacket(drawn);
        } else if (opcode == 0x22) {
            landPendingWaits(submission.queue);
            recordQueuedLabelsBeforeRead(submission.queue);
            const auto condition = Pm4::ReadCondition(packet);
            if (condition == 0) nextCursor = submission.conditionalEnds.at(cursor);
            if (traceGpu) std::fprintf(stderr, "[gpu] %.1f queue 0x%x COND_EXEC at DWORD %zu reads 0x%x at 0x%llx: %s %zu dwords\n", TraceMs(), submission.queue, cursor, condition, static_cast<unsigned long long>(packet[1] | (static_cast<std::uint64_t>(packet[2]) << 32u)), condition == 0 ? "skips" : "executes", submission.conditionalEnds.at(cursor) - cursor - count);
        } else if (sampleDump && !wroteOnGpu) {
            dumpSampleCounters(packet[2] | (static_cast<std::uint64_t>(packet[3]) << 32u));
        } else if (opcode != 0x42 && opcode != 0x46 && opcode != 0x58) {
            if (!wroteOnGpu) {
                if (Pm4::IndirectRegisterOpcode(opcode)) Pm4::ExecuteIndirectRegisters(packet, submission.registerLists.at(cursor), queue);
                else Pm4::Execute(packet, queue);
                if (opcode == 0x49 || opcode == 0x37) {
                    if (const auto label = Pm4::DecodeLabelWrite(packet)) noteLabelStore(label->address, label->Bytes(), ++eventSerial, submission.queue);
                }
            }
            if (endOfPipeInterrupt && !interruptDeferred) AgcDriverDeliverEopInterrupt(submission.queue);
        }
        if (drawPacket || (sampleDump && wroteOnGpu)) Graphics::Recorder::CountRecordedWork();
        timing.Mark(header == RenderingWaitPacketHeader ? "rendering_wait" : opcode == 0x3c || opcode == 0x93 ? "memory_wait" : opcode == 0x15 || opcode == 0x16 ? "dispatch" : drawPacket ? "draw" : "execute");
        if (submission.queue == 0) timing.FinishPacket(submission.serial, cursor, opcode, count);
        else timing.Finish();
        cursor = nextCursor;
    }
    if (pipelined) {
        finishDrawWindow();
        DrawPipeline::Queue0().Drain(DrawPipeline::DrainReason::Submission);
    }

    PerformanceContext endContext(submission.queue == 0 ? frameTiming() : FrameTiming::Async());
    PerformanceTimer endTiming("Driver.EndSubmission");
    static const bool submitAtEnd = std::getenv("APS5_SUBMIT_AT_END") != nullptr;
    if (!deferredLabels().labels.empty() || Graphics::Recorder::PendingLabelSince().has_value() || Graphics::Recorder::RecordedWorkSinceSubmit() != 0) {
        auto& costs = submissionCosts(submission.queue);
        if (!submitAtEnd && submission.rewindTail == nullptr && submission.queue == 0 && workerQueued() != nullptr && workerQueued()->load(std::memory_order_acquire) != 0) {
            ++costs.endSkipped;
            return;
        }
        ++costs.endSubmits;
        GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::End);
        std::lock_guard gpuLock(GuestMemory::GpuMutex());
        const auto localDevice = device.Load();

        recordDeferredLabels(localDevice.get(), submission.queue);
        if (localDevice != nullptr) localDevice->SubmitRecorded(submission.queue == 0);
    }
    endTiming.Finish();
    if (submission.rewindTail != nullptr) executeRewindTail(submission);
}

namespace {

// The queue-0 worker's window of draws prepared ahead (APS5_PREPARE_AHEAD, DrawAhead.hpp): the tasks
// by packet offset, and the ranges the draws queued before or enqueued since their captures may write.
struct DrawWindow {
    const Submission* submission = nullptr;
    std::map<std::size_t, std::shared_ptr<DrawAheadTask>> tasks;
    std::vector<DrawPipeline::Range> writes;
    std::uint64_t windows = 0, prepared = 0, used = 0, serial = 0, overlapped = 0, invalidated = 0;
    std::chrono::steady_clock::time_point reported = std::chrono::steady_clock::now();
};

DrawWindow& drawWindow() {
    static thread_local DrawWindow window;
    return window;
}

bool overlapsAny(const std::vector<std::pair<std::uint64_t, std::uint64_t>>& reads, const std::vector<DrawPipeline::Range>& writes) {
    for (const auto& [begin, end] : reads) {
        for (const auto& [first, limit] : writes) {
            if (begin < limit && first < end) return true;
        }
    }
    return false;
}

}

void Driver::finishDrawWindow() {
    auto& window = drawWindow();
    for (auto& [cursor, task] : window.tasks) task->ready.wait();
    window.tasks.clear();
    window.writes.clear();
    window.submission = nullptr;
    DrawWindowWrites() = nullptr;
    DrawWindowInvalidated() = false;
}

void Driver::startDrawWindow(const QueueState& live, const Submission& submission, std::size_t from) {
    static const std::size_t maxDraws = [] {
        const char* text = std::getenv("APS5_PREPARE_AHEAD_DRAWS");
        return text != nullptr ? std::max<std::size_t>(1, std::strtoull(text, nullptr, 10)) : std::max<std::size_t>(1, DrawAheadThreads() * 4);
    }();
    // Captures read memory after the waits this worker deferred and before labels it would record.
    if (DrawAheadThreads() == 0 || !deferredLabels().labels.empty() || !pendingWaits().empty()) return;
    auto& window = drawWindow();
    QueueState scratch = live;
    scratch.markers.clear();
    const auto& commands = submission.commands;
    std::size_t launched = 0;
    for (std::size_t cursor = from; cursor < commands.size() && launched < maxDraws;) {
        const auto header = commands[cursor];
        if (Pm4::FillerPacket(header)) {
            ++cursor;
            continue;
        }
        const auto count = Pm4::PacketWords(header);
        if (cursor + count > commands.size()) break;
        const auto packet = std::span(commands).subspan(cursor, count);
        const auto opcode = (header >> 8u) & 0xffu;
        // A label the worker will hand to the committer (in the window it always does) changes no state.
        static const bool labelItems = std::getenv("APS5_NO_PIPELINED_LABELS") == nullptr;
        if (labelItems && (opcode == 0x49 || opcode == 0x37) && !Pm4::Predicated(header) && labelCommitItem(packet)) {
            cursor += count;
            continue;
        }
        if (!keepsDrawPipeline(packet, header, opcode) || (Pm4::Predicated(header) && scratch.predication.operation != 0)) break;
        if (Pm4::DrawOpcode(opcode)) {
            if (Pm4::IndirectDrawOpcode(opcode)) break;
            if (window.tasks.empty()) {
                window.submission = &submission;
                window.writes = DrawPipeline::Queue0().QueuedWrites();
                DrawWindowWrites() = &window.writes;
                DrawWindowInvalidated() = false;
                ++window.windows;
            }
            auto task = std::make_shared<DrawAheadTask>();
            task->state = scratch;
            task->packet = packet;
            task->ready = task->done.get_future().share();
            window.tasks.emplace(cursor, task);
            ++launched;
            ++window.prepared;
            RunDrawAhead([this, task, prepared = &submission] {
                CurrentDrawAhead() = task.get();
                GuestMemory::SetCurrentPacket((task->packet[0] >> 8u) & 0xffu, prepared->queue);
                try {
                    std::string rejected;
                    task->verdict = draw(task->state, task->packet, *prepared, rejected);
                    task->rejected = std::move(rejected);
                } catch (const DrawAheadSerial&) {
                    task->serial = true;
                } catch (const std::exception& error) {
                    task->error = error.what();
                } catch (...) {
                    task->serial = true;
                }
                CurrentDrawAhead() = nullptr;
                task->done.set_value();
            });
        } else if (opcode != 0x3f && opcode != 0x42 && opcode != 0x46 && opcode != 0x58) {
            // The same state the worker's own execution of the packet leaves.
            try {
                if (Pm4::IndirectRegisterOpcode(opcode)) Pm4::ExecuteIndirectRegisters(packet, submission.registerLists.at(cursor), scratch);
                else Pm4::Execute(packet, scratch);
            } catch (const std::exception&) {
                break;
            }
        }
        cursor += count;
    }
}

DrawVerdict Driver::drawAhead(QueueState& queue, std::span<const std::uint32_t> packet, const Submission& submission, std::string& rejected, std::size_t cursor) {
    auto& window = drawWindow();
    if (window.submission != nullptr && window.submission != &submission) finishDrawWindow();
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    if (profile && std::chrono::steady_clock::now() - window.reported > std::chrono::seconds(10)) {
        window.reported = std::chrono::steady_clock::now();
        std::fprintf(stderr, "[draw-ahead] (10 s) %llu windows, %llu draws prepared ahead: %llu used, %llu given back, %llu read what an earlier draw writes, %llu after an inline draw\n", static_cast<unsigned long long>(window.windows), static_cast<unsigned long long>(window.prepared), static_cast<unsigned long long>(window.used), static_cast<unsigned long long>(window.serial), static_cast<unsigned long long>(window.overlapped), static_cast<unsigned long long>(window.invalidated));
        window.windows = window.prepared = window.used = window.serial = window.overlapped = window.invalidated = 0;
    }
    if (const auto found = window.tasks.find(cursor); found != window.tasks.end()) {
        const auto task = found->second;
        window.tasks.erase(found);
        task->ready.wait();
        bool usable = true;
        if (task->serial) {
            usable = false;
            ++window.serial;
        } else if (DrawWindowInvalidated()) {
            usable = false;
            ++window.invalidated;
        } else if (task->commit && overlapsAny(task->reads, window.writes)) {
            usable = false;
            ++window.overlapped;
        }
        if (usable) {
            ++window.used;
            if (window.tasks.empty()) finishDrawWindow();
            if (task->error) throw std::runtime_error(*task->error);
            if (task->commit) {
                if (DrawWindowWrites() != nullptr) window.writes.insert(window.writes.end(), task->writes.begin(), task->writes.end());
                DrawPipeline::Queue0().Enqueue(std::move(task->commit), std::move(task->writes));
            }
            rejected = std::move(task->rejected);
            if (window.tasks.empty()) startDrawWindow(queue, submission, cursor + packet.size());
            return task->verdict;
        }
    }
    const auto verdict = draw(queue, packet, submission, rejected);
    if (window.tasks.empty()) {
        finishDrawWindow();
        startDrawWindow(queue, submission, cursor + packet.size());
    }
    return verdict;
}

}
