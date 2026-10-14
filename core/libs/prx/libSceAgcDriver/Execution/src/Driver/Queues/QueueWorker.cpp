#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Diagnostics.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Synchronization/SynchronizationStatistics.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Queues/WorkerAffinity.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Synchronization/WaitMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/WorkerSampler.hpp"
#include "prx/libSceAgcDriver/Execution/include/PerformanceTimer.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <string>
#include <cstdlib>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace AgcDriver::DriverDetail {

namespace {

// How long every queue must stay quiet before pending storage image results are written back to
// guest memory (APS5_IDLE_RESULTS_MS, default 1). PPSA03671 waits for such a write-back every frame
// (a CPU read the driver cannot see), so the quiet period was frame latency: 50 ms ran its
// cinematic at 3.2 FPS, 1 ms at 3.8, 2000 ms at 0.4 (2026-10-14).
const std::chrono::milliseconds IdleResultsQuietPeriod = [] {
    const char* value = std::getenv("APS5_IDLE_RESULTS_MS");
    return std::chrono::milliseconds(value != nullptr ? std::strtoll(value, nullptr, 10) : 1);
}();

// Debug aid: APS5_TRACE_IDLE=<hex guest address>[,<hex guest address>] prints queue 0's idle-path
// events (completions cleared, idle store begun and done) with a millisecond clock and the guest
// qwords at those addresses (a title's frame label, a readback it waits for), to place the idle
// store against the title's frame. The lines are buffered and written 64 KiB at a time: a write to
// a redirected stderr took ~4 ms each and slowed the title down to a crawl.
std::uint64_t PeekGuest(std::uint64_t address) {
    if (address == 0) return 0;
    // The guest address is read only once it is committed (it is unmapped early in the boot).
#ifdef _WIN32
    MEMORY_BASIC_INFORMATION memory{};
    if (VirtualQuery(reinterpret_cast<const void*>(address), &memory, sizeof(memory)) == sizeof(memory) && memory.State == MEM_COMMIT && (memory.Protect & (PAGE_READONLY | PAGE_READWRITE | PAGE_WRITECOPY | PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE)) != 0) return *reinterpret_cast<const volatile std::uint64_t*>(address);
#endif
    return 0;
}

void TraceIdle(const char* event) {
    static const std::array<std::uint64_t, 2> peek = [] {
        std::array<std::uint64_t, 2> addresses{};
        const char* text = std::getenv("APS5_TRACE_IDLE");
        if (text == nullptr) return addresses;
        char* end = nullptr;
        addresses[0] = std::strtoull(text, &end, 16);
        if (*end == ',') addresses[1] = std::strtoull(end + 1, nullptr, 16);
        return addresses;
    }();
    if (peek[0] == 0) return;
    // A repeat of the previous event (the loop going round) is counted, not printed.
    static const char* previous = nullptr;
    static std::uint64_t repeats = 0;
    static std::string lines;
    if (event == previous) {
        ++repeats;
        return;
    }
    char line[160];
    if (repeats != 0) {
        std::snprintf(line, sizeof(line), "[idle]   (x%llu)\n", static_cast<unsigned long long>(repeats));
        lines += line;
    }
    previous = event;
    repeats = 0;
    static const auto start = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    std::snprintf(line, sizeof(line), "[idle] %.3f ms %s label 0x%llx peek 0x%llx\n", ms, event, static_cast<unsigned long long>(PeekGuest(peek[0])), static_cast<unsigned long long>(PeekGuest(peek[1])));
    lines += line;
    if (lines.size() >= 65536) {
        std::fwrite(lines.data(), 1, lines.size(), stderr);
        lines.clear();
    }
}

}

FrameTiming* Driver::frameTiming() {
    if (!APS5_ENABLE_TIMING_LOG) return nullptr;
    if (pendingFrameTiming == nullptr) pendingFrameTiming = std::make_shared<FrameTiming>(++frameSerial, true);
    return pendingFrameTiming.get();
}

FrameTiming* Driver::includeTimingSubmission(const Submission& submission, bool firstSegment) {
    if (!APS5_ENABLE_TIMING_LOG) return nullptr;
    auto* frame = submission.queue == 0 ? frameTiming() : FrameTiming::Async();
    require(submission.receivedAt != FrameTiming::Clock::time_point{} && submission.receivedAt <= submission.enqueuedAt && submission.enqueuedAt <= submission.dequeuedAt && submission.dequeuedAt <= submission.orderedAt, "invalid submission timing order");
    if (submission.queue == 0) frame->IncludeSubmission(submission.serial, submission.receivedAt, submission.enqueuedAt, submission.orderedAt, firstSegment);
    else if (firstSegment) {
        frame->Add(frame->Get("Submission", "accept"), submission.enqueuedAt - submission.receivedAt);
        frame->Add(frame->Get("Submission", "queue"), submission.orderedAt - submission.enqueuedAt);
    }
    if (firstSegment) {
        frame->Add(frame->Get("Submission", "dequeue"), submission.dequeuedAt - submission.enqueuedAt);
        frame->Add(frame->Get("Submission", "order_wait"), submission.orderedAt - submission.dequeuedAt);
        if (!submission.suspend) {
            frame->Add(frame->Get("Submission", "copy"), submission.copiedAt - submission.receivedAt, submission.commands.size() * sizeof(std::uint32_t));
            frame->Add(frame->Get("Submission", "validate"), submission.validatedAt - submission.copiedAt);
            frame->Add(frame->Get("Submission", "flip_room_wait"), submission.roomReadyAt - submission.validatedAt);
            frame->Add(frame->Get("Submission", "reserve_enqueue"), submission.enqueuedAt - submission.roomReadyAt);
        }
    }
    return frame;
}

void Driver::markCompleted(std::uint64_t serial) {
    completedOutOfOrder.insert(serial);
    while (!completedOutOfOrder.empty() && *completedOutOfOrder.begin() == completed + 1) {
        completed = *completedOutOfOrder.begin();
        completedOutOfOrder.erase(completedOutOfOrder.begin());
    }
}

const std::atomic<std::uint64_t>*& Driver::workerQueued() {
    static thread_local const std::atomic<std::uint64_t>* queued = nullptr;
    return queued;
}

void Driver::reapCompletionLabels() {
    std::unique_lock gpuLock(GuestMemory::GpuMutex(), std::try_to_lock);
    if (!gpuLock.owns_lock()) {
        ++triesFailed[TryIdle];
        return;
    }

    if (!completionsPending()) return;

    bumpEpoch(&EpochBumps::reaps);
    if (const auto localDevice = device.Load()) localDevice->ReapRecorded();
}

void Driver::run(std::uint32_t id) noexcept {
    onWorkerThread() = true;
    char role[32];
    std::snprintf(role, sizeof role, "queue worker 0x%x", id);
    PinWorkerThread(role);

    GuestMemory::TagGpuLockThread(id);
    if (id == 0) StartWorkerSampler();
    Submission submission;
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    auto& costs = submissionCosts(id);
    try {
        if (PerformanceTrace::Configured()) PerformanceTrace::BindThread(role);
        for (;;) {
            submission = Submission{};
            struct Running {
                std::atomic<std::uint32_t>& count;
                std::atomic<std::uint64_t>& idleEpoch;
                std::condition_variable& changed;
                bool active = false;
                Running(std::atomic<std::uint32_t>& count, std::atomic<std::uint64_t>& idleEpoch, std::condition_variable& changed)
                    : count(count), idleEpoch(idleEpoch), changed(changed) {}
                void Start() {
                    count.fetch_add(1, std::memory_order_acq_rel);
                    active = true;
                }
                ~Running() {
                    if (active && count.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                        idleEpoch.fetch_add(1, std::memory_order_release);
                        changed.notify_all();
                    }
                }
            } running{runningWorkers, idleEpoch, changed};
            static const bool traceGpu = std::getenv("APS5_TRACE_GPU") != nullptr;
            {
                PerformanceContext timingContext(id == 0 ? frameTiming() : FrameTiming::Async());
                PerformanceTimer timing("Driver.WorkerWait");
                std::unique_lock lock(mutex);
                timing.Mark("mutex_wait");
                auto& worker = workers.at(id);
                auto& pending = worker.pending;
                workerQueued() = &worker.queued;
                if (traceGpu && pending.empty()) std::fprintf(stderr, "[gpu] %.1f idle queue=0x%x\n", TraceMs(), id);
                const auto ready = [&] { return failure || stopping || !pending.empty(); };
                const auto allQueuesIdle = [&] {
                    return runningWorkers.load(std::memory_order_acquire) == 0
                        && std::all_of(workers.begin(), workers.end(), [](const auto& item) { return item.second.pending.empty(); });
                };
                const auto idleResultsPending = [&] {
                    return id == 0 && Graphics::StorageTexture::AnyPending() && allQueuesIdle() && !completionsPending();
                };
                const auto poll = [&](const auto& done) {
#ifdef _WIN32
                    lock.unlock();
                    PollSleep();
                    lock.lock();
                    return done();
#else
                    return changed.wait_for(lock, std::chrono::milliseconds(1), done);
#endif
                };

                bool polled = false;
                while (!ready()) {
                    if (!completionsPending() || (id != 0 && Graphics::Recorder::PendingCompletionLabels() == 0)) {
                        if (id == 0 && polled) TraceIdle("completions clear");
                        polled = false;
                        const auto idleEpochAtCheck = idleEpoch.load(std::memory_order_acquire);
                        const bool storeIdleResults = idleResultsPending();
                        if (id == 0) TraceIdle(storeIdleResults ? "idle store due" : Graphics::StorageTexture::AnyPending() ? (allQueuesIdle() ? "no store: completions" : "no store: queues busy") : "no store: nothing pending");
                        if (storeIdleResults) {
                            queue0Dormant.store(true, std::memory_order_relaxed);
                            const auto quietUntil = std::chrono::steady_clock::now() + IdleResultsQuietPeriod;
                            const auto idleEpochAtWait = idleEpoch.load(std::memory_order_acquire);
                            changed.wait_until(lock, quietUntil, [&] {
                                return ready() || idleEpoch.load(std::memory_order_acquire) != idleEpochAtWait;
                            });
                            queue0Dormant.store(false, std::memory_order_relaxed);
                            if (ready()) break;
                            if (idleEpoch.load(std::memory_order_acquire) != idleEpochAtWait) {
                                continue;
                            }
                            if (!idleResultsPending()) continue;
                        }
                        lock.unlock();
                        if (storeIdleResults) {
                            TraceIdle("idle store begin");
                            if (const auto localDevice = device.Load()) localDevice->StoreIdleResults();
                            TraceIdle("idle store submitted");
                        }
                        lock.lock();
                        if (ready()) break;
                        if (id == 0 && allQueuesIdle() && !completionsPending()
                            && ((!storeIdleResults && Graphics::StorageTexture::AnyPending()) || idleEpoch.load(std::memory_order_acquire) != idleEpochAtCheck)) continue;
                        if (id == 0) queue0Dormant.store(true, std::memory_order_relaxed);
                        const auto idleEpochAtWait = idleEpoch.load(std::memory_order_acquire);
                        if (id == 0) TraceIdle("dormant");
                        changed.wait(lock, [&] { return ready() || (id == 0 && idleEpoch.load(std::memory_order_acquire) != idleEpochAtWait); });
                        if (id == 0) queue0Dormant.store(false, std::memory_order_relaxed);
                        if (id == 0) TraceIdle(ready() ? "woken: work" : "woken: idle epoch");
                        if (id == 0 && !ready() && idleEpoch.load(std::memory_order_acquire) != idleEpochAtWait) continue;
                        break;
                    }
                    if (id == 0 && !polled) TraceIdle("polling completions");
                    polled = true;
                    if (poll(ready)) break;
                    lock.unlock();
                    reapCompletionLabels();
                    lock.lock();
                }
                timing.Mark("idle_and_completion_poll");
                rethrowFailure();
                if (stopping || shutdownToken.stop_requested() || pending.empty()) {
                    break;
                }
                submission = std::move(pending.front());
                pending.pop_front();
                if (APS5_ENABLE_TIMING_LOG) submission.dequeuedAt = std::chrono::steady_clock::now();
                if (id == 0) queue0Executing = submission.suspend ? 0 : submission.received;
                if (submission.waitFree) {
                    orderHolders.fetch_add(1, std::memory_order_acq_rel);
                    const auto released = [&] { return failure || stopping || orderReleased(id, submission.received); };
                    while (!released()) poll(released);
                    orderHolders.fetch_sub(1, std::memory_order_acq_rel);
                    rethrowFailure();
                }
                if (APS5_ENABLE_TIMING_LOG) submission.orderedAt = std::chrono::steady_clock::now();
                timing.Mark("dequeue_order");
                worker.queued.fetch_sub(1, std::memory_order_acq_rel);
                running.Start();
                if (profile && submission.enqueuedAt != std::chrono::steady_clock::time_point{}) costs.dequeueNs += static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - submission.enqueuedAt).count());
            }
            execute(submission);
            if (traceGpu) std::fprintf(stderr, "[gpu] %.1f done serial=%llu queue=0x%x\n", TraceMs(), static_cast<unsigned long long>(submission.serial), id);
            const auto completeStart = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            PerformanceContext completionContext(id == 0 ? frameTiming() : FrameTiming::Async());
            PerformanceTimer completionTiming("Driver.Completion");
            bool notify = true;
            {
                std::lock_guard lock(mutex);
                rethrowFailure();
                markCompleted(submission.serial);
                forgetUnfinishedWrites(workers.at(id), submission);
                if (id == 0) queue0Executing = 0;

                notify = idleWaiters != 0 || orderHolders.load(std::memory_order_acquire) != 0;
            }
            if (notify) changed.notify_all();
            else ++costs.notifiesSkipped;
            if (profile) costs.completeNs += static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - completeStart).count());
        }
    } catch (const ProcessShutdown&) {
        submission = Submission{};
    } catch (...) {
        const auto error = std::current_exception();
        for (const auto& [offset, flip] : submission.flips) flip->Fail(error);
        ReportFailure(error);
    }
}

}
