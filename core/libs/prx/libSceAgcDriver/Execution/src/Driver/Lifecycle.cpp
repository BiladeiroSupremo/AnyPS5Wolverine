#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Diagnostics.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawAhead.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPipeline.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Shaders/ShaderRegistry.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/WorkerSampler.hpp"
#include "prx/libSceAgcDriver/Execution/include/PerformanceTrace.hpp"
#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestBufferMemory.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include <atomic>
#include <condition_variable>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <string>
#include <vector>
#include <exception>
#include <mutex>
#include <utility>
#include <system_error>
#include <thread>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <pthread.h>
#endif

namespace AgcDriver::DriverDetail {

namespace {

constexpr std::uintptr_t InlineImportStackBytes = 256 * 1024;

std::atomic<std::uint64_t> gpuMapImportThreads{0};

// Imports of GPU-mapped ranges (the registry's map observer, see Driver::Driver) run on one
// background thread: the mapping call returns at once instead of creating the imports first. In
// PPSA03671 the title's streaming thread maps through sceAmpr while holding a lock its render
// thread then spins on, and those imports were ~16% of that thread during 200+ ms hitches
// (APS5_SAMPLE_HITCHES, 2026-10-23). A range unmapped before its turn is skipped (the ranges are
// held weakly); one the queue needs first is imported at that use, as before the map-time import.
// APS5_SYNC_MAP_IMPORT=1 imports on a thread the mapping call joins, as before.
struct MapImports {
    std::mutex mutex;
    std::condition_variable wake;
    std::deque<std::pair<GuestAllocations::Mapped, std::uint64_t>> pending;
    bool started = false;
};

MapImports& BackgroundMapImports() {
    static auto* state = new MapImports();
    return *state;
}

// Experiments on the title's own code (APS5_CODE_PATCHES="rva:expected:replacement;...", hex, rva
// relative to the main executable): each patch is written only where the expected bytes are found,
// before the title draws anything (the driver starts with its first graphics call). The process'
// image is patched in memory; nothing on disk changes. First use (PPSA03671, 2026-10-23): the render
// thread's frame wait (app.exe 0x143f113f0) waits for frame<<32 instead of frame<<32 | stage<<20 |
// 0x7b, i.e. for the GPU to start the closed frame instead of finishing it: two frames in flight.
void ApplyCodePatches() {
    const char* text = std::getenv("APS5_CODE_PATCHES");
    if (text == nullptr || *text == '\0') return;
#ifdef _WIN32
    const auto base = reinterpret_cast<std::uint8_t*>(GetModuleHandleW(nullptr));
    const auto hexBytes = [](const std::string& hex) {
        std::vector<std::uint8_t> bytes;
        for (std::size_t i = 0; i + 1 < hex.size(); i += 2) bytes.push_back(static_cast<std::uint8_t>(std::stoul(hex.substr(i, 2), nullptr, 16)));
        return bytes;
    };
    std::string list(text);
    std::size_t start = 0;
    while (start < list.size()) {
        const auto end = std::min(list.find(';', start), list.size());
        const auto entry = list.substr(start, end - start);
        start = end + 1;
        const auto first = entry.find(':');
        const auto second = first == std::string::npos ? std::string::npos : entry.find(':', first + 1);
        if (second == std::string::npos) {
            std::fprintf(stderr, "[code-patch] malformed entry '%s' (rva:expected:replacement)\n", entry.c_str());
            continue;
        }
        const auto rva = std::stoull(entry.substr(0, first), nullptr, 16);
        const auto expected = hexBytes(entry.substr(first + 1, second - first - 1));
        const auto replacement = hexBytes(entry.substr(second + 1));
        auto* target = base + rva;
        if (expected.size() != replacement.size() || expected.empty() || std::memcmp(target, expected.data(), expected.size()) != 0) {
            std::fprintf(stderr, "[code-patch] 0x%llx: expected bytes not found (or sizes differ), not patched\n", static_cast<unsigned long long>(rva));
            continue;
        }
        DWORD previous = 0;
        if (!VirtualProtect(target, replacement.size(), PAGE_EXECUTE_READWRITE, &previous)) {
            std::fprintf(stderr, "[code-patch] 0x%llx: VirtualProtect failed (%lu)\n", static_cast<unsigned long long>(rva), GetLastError());
            continue;
        }
        std::memcpy(target, replacement.data(), replacement.size());
        VirtualProtect(target, replacement.size(), previous, &previous);
        FlushInstructionCache(GetCurrentProcess(), target, replacement.size());
        std::fprintf(stderr, "[code-patch] 0x%llx: patched %zu bytes\n", static_cast<unsigned long long>(rva), replacement.size());
    }
#else
    std::fprintf(stderr, "[code-patch] APS5_CODE_PATCHES is Windows only\n");
#endif
}

bool SyncMapImports() {
    static const bool sync = std::getenv("APS5_SYNC_MAP_IMPORT") != nullptr;
    return sync;
}

void PrintFailure(std::exception_ptr error) {
    try {
        std::rethrow_exception(error);
    } catch (const std::exception& cause) {
        std::fprintf(stderr, "[gpu] driver failure: %s\n", cause.what());
    } catch (...) {
        std::fprintf(stderr, "[gpu] driver failure with a non-standard exception\n");
    }
    std::fflush(stderr);
}

bool StackHoldsImport() {
#ifdef _WIN32
    ULONG_PTR low = 0;
    ULONG_PTR high = 0;
    GetCurrentThreadStackLimits(&low, &high);
#else
    struct Bounds {
        std::uintptr_t low = 0;
        std::uintptr_t high = 0;
        Bounds() {
            pthread_attr_t attributes;
            if (pthread_getattr_np(pthread_self(), &attributes) != 0) return;
            void* base = nullptr;
            std::size_t size = 0;
            if (pthread_attr_getstack(&attributes, &base, &size) == 0 && base != nullptr) {
                low = reinterpret_cast<std::uintptr_t>(base);
                high = low + size;
            }
            pthread_attr_destroy(&attributes);
        }
    };
    thread_local const Bounds bounds;
    const auto low = bounds.low;
    const auto high = bounds.high;
#endif
    const volatile unsigned char marker = 0;
    const auto frame = reinterpret_cast<std::uintptr_t>(&marker);
    return frame >= low && frame < high && frame - low >= InlineImportStackBytes;
}

}

std::uint64_t Driver::GpuMapImportThreads() {
    return gpuMapImportThreads.load(std::memory_order_relaxed);
}

Driver& Driver::Get() {
    static Driver driver;
    return driver;
}

Driver::~Driver() {
    GuestAllocations::GuestAllocationsSetGpuMapObserver_nid_postfix(nullptr);
    stop();
}

void Driver::Shutdown() {
    stop();
    std::lock_guard lock(mutex);
    rethrowFailure();
}

void Driver::stop() {
    require(!onWorkerThread(), "worker cannot stop itself");
    std::lock_guard shutdownLock(shutdownMutex);
    if (stopped) return;
    LibcRequestShutdown_nid_postfix();
    {
        std::lock_guard lock(mutex);
        stopping = true;
        for (auto& [queue, worker] : workers) {
            worker.pending.clear();
            worker.unfinishedWrites.clear();
            worker.queued.store(0, std::memory_order_release);
        }
    }
    changed.notify_all();
    for (auto& [queue, worker] : workers) {
        if (worker.thread.joinable()) worker.thread.join();
    }
    // The draw committer runs what queue 0 handed it, then stops (before the device goes); so do the
    // draw prepare-ahead threads.
    DrawPipeline::Queue0().Stop();
    StopDrawAhead();
    StopWorkerSampler();
    CloseRegistrationPreparation();
    std::lock_guard gpuLock(GuestMemory::GpuMutex());
    device.Reset();
    replacedDevices.clear();
    Graphics::ShutdownGuestBufferWorkers();
    PerformanceTrace::StopOutput();
    StopProfileOutput_nid_no_patch();
    stopped = true;
}

Driver::Driver() {
    ApplyCodePatches();
    static_cast<void>(GuestMemory::TrackerGeneration());
    ShaderMemory::SetWaitedMsProvider(&Graphics::Recorder::ThreadWaitedMs);
    try {
        LibcRegisterShutdown_nid_postfix([] { Driver::Get().Shutdown(); });
    } catch (...) {
        stop();
        throw;
    }
    GuestAllocations::GuestAllocationsSetGpuMapObserver_nid_postfix([](const GuestAllocations::Mapped& ranges, std::uint64_t generation) {
        const auto import = [&] {
            try {
                const auto current = Driver::Get().device.Load();
                if (current == nullptr || current->ImportGuestMemory(ranges, generation, false)) return;
                std::lock_guard lock(GuestMemory::GpuMutex());
                if (Driver::Get().device.Load() == current) current->ImportGuestMemory(ranges, generation, true);
            } catch (const std::exception& error) {
                std::fprintf(stderr, "[gpu] import of GPU memory at its mapping failed: %s\n", error.what());
            }
        };
        if (GuestMemory::GpuMutex().HeldByThisThread() || (SyncMapImports() && StackHoldsImport())) {
            import();
            return;
        }
        if (!SyncMapImports()) {
            auto& background = BackgroundMapImports();
            std::lock_guard lock(background.mutex);
            background.pending.emplace_back(ranges, generation);
            if (!background.started) {
                try {
                    std::thread([] {
                        auto& state = BackgroundMapImports();
                        for (;;) {
                            std::unique_lock lock(state.mutex);
                            state.wake.wait(lock, [&] { return !state.pending.empty(); });
                            auto [mapped, mappedGeneration] = std::move(state.pending.front());
                            state.pending.pop_front();
                            lock.unlock();
                            try {
                                const auto current = Driver::Get().device.Load();
                                if (current == nullptr || current->ImportGuestMemory(mapped, mappedGeneration, false)) continue;
                                std::lock_guard gpuLock(GuestMemory::GpuMutex());
                                if (Driver::Get().device.Load() == current) current->ImportGuestMemory(mapped, mappedGeneration, true);
                            } catch (const std::exception& error) {
                                std::fprintf(stderr, "[gpu] import of GPU memory after its mapping failed: %s\n", error.what());
                            }
                        }
                    }).detach();
                    background.started = true;
                } catch (const std::system_error& error) {
                    std::fprintf(stderr, "[gpu] import of GPU memory at its mapping failed: no background thread: %s\n", error.what());
                    background.pending.clear();
                    return;
                }
            }
            background.wake.notify_one();
            return;
        }
        std::thread host;
        try {
            host = std::thread(import);
        } catch (const std::system_error& error) {
            std::fprintf(stderr, "[gpu] import of GPU memory at its mapping failed: no host thread to run it on: %s\n", error.what());
            return;
        }
        gpuMapImportThreads.fetch_add(1, std::memory_order_relaxed);
        host.join();
    });
}

void Driver::WaitIdle() {
    require(!onWorkerThread(), "worker cannot wait for itself");
    std::unique_lock lock(mutex);
    const auto target = accepted;
    ++idleWaiters;
    changed.wait(lock, [&] { return failure != nullptr || stopping || completed >= target; });
    --idleWaiters;
    rethrowFailure();
    checkStopping();
}

void Driver::CheckFailure() {
    if (failed.load(std::memory_order_acquire)) {
        std::lock_guard lock(mutex);
        rethrowFailure();
    }
    checkStopping();
}

void Driver::ReportFailure(std::exception_ptr error) {
    require(error != nullptr, "null asynchronous failure");
    bool first = false;
    {
        std::lock_guard lock(mutex);
        first = !failure;
        if (first) failure = error;
        failed.store(true, std::memory_order_release);
        for (const auto& [handle, output] : outputs) output->Fail(failure);
        for (auto& [queue, worker] : workers) {
            for (const auto& item : worker.pending) {
                for (const auto& [offset, flip] : item.flips) flip->Fail(failure);
            }
            worker.pending.clear();
            worker.unfinishedWrites.clear();
            worker.queued.store(0, std::memory_order_release);
        }
        if (first) PrintFailure(error);
    }
    changed.notify_all();
}

void Driver::rethrowFailure() const {
    if (failure != nullptr) {
        std::rethrow_exception(failure);
    }
}

void Driver::checkStopping() const {
    if (stopping.load(std::memory_order_acquire) || shutdownToken.stop_requested()) throw ProcessShutdown{};
}

bool& Driver::onWorkerThread() {
    static thread_local bool worker = false;
    return worker;
}

}
