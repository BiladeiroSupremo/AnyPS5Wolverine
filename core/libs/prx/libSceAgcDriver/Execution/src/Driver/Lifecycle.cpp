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
#include "prx/libc/include/general/VabiMacros.hpp"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <algorithm>
#include <array>
#include <cstdint>
#include <map>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <string>
#include <unordered_set>
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

// Hooks on the title's own functions, to measure its engine layer before replacing it
// (APS5_GUEST_HOOKS="rva:stolen[:p];...", hex, Windows only). The stolen bytes (whole, position
// independent instructions, at least 5, checked against the image) move into a stub next to the
// image; the function starts with a jump to it. The stub saves the SysV argument registers, calls
// GuestHookCall with the arguments and the hook's index, restores them, runs the moved bytes and
// jumps back. 'p' also reads the draw's shader set: in PPSA03671 the recorder's draws (app.exe
// 0x143f7b9a0 and its siblings, all starting 55 48 89 e5 41 57 41 56) pass this+0x78 as the command
// buffer to every libSceAgc call, and that buffer keeps per-stage records of 0x70 bytes from +0x38
// whose first field points at the stage's shader (stages 2..7 are used); the six pointers stand in for
// the pipeline state. Every 5 s the hooks print calls per second, distinct this pointers and shader
// sets, and how many of the window's shader sets the previous window used too.
// The draw-call census (native port milestone M1, C:\wolvie\work\native_port_plan.md) adds, per hook
// and window: how many distinct values each SysV argument register took (rsi rdx rcx r8 r9) with two
// example calls; and, for 'p' hooks, the libSceAgc command buffer's write cursor at entry
// (CommandBuffer::cursor_up, +0x10 of this+0x78): the PM4 bytes from one hooked call to the next on
// the thread, and the draw packets the queue worker decodes (NoteGuestHookDrawPacket), charged by
// opcode to the hooked call whose cursor is the nearest at or below the packet (within 64 KiB; the
// rest are counted unmatched).
struct GuestHook {
    std::uint64_t rva = 0;
    bool readsPipeline = false;
    std::uint64_t calls = 0;
    std::unordered_set<std::uint64_t> objects;
    std::unordered_set<std::uint64_t> pipelines;
    std::array<std::unordered_set<std::uint64_t>, 5> argValues;
    std::vector<std::array<std::uint64_t, 6>> examples;
    std::uint64_t pm4Calls = 0;
    std::uint64_t pm4Bytes = 0;
    std::map<std::uint32_t, std::uint64_t> drawOpcodes;
};

struct GuestHookCursor {
    std::uint64_t serial;
    std::size_t index;
};

struct GuestHooks {
    std::mutex mutex;
    std::vector<GuestHook> hooks;
    std::unordered_set<std::uint64_t> previousPipelines;
    std::chrono::steady_clock::time_point windowStart = std::chrono::steady_clock::now();
    // Recent entry cursors by address, oldest evicted first (an address reused by a later call
    // keeps that call's entry).
    std::map<std::uint64_t, GuestHookCursor> cursors;
    std::deque<std::pair<std::uint64_t, std::uint64_t>> cursorOrder;
    std::uint64_t serial = 0;
    std::map<std::uint32_t, std::uint64_t> unmatchedDraws;
};

std::atomic<bool> guestHooksActive{false};

GuestHooks& GuestHookState() {
    static auto* state = new GuestHooks();
    return *state;
}

std::string GuestHookOpcodes(const std::map<std::uint32_t, std::uint64_t>& opcodes) {
    std::string text;
    for (const auto& [opcode, count] : opcodes) {
        char item[32];
        std::snprintf(item, sizeof(item), " 0x%02x x%llu", opcode, static_cast<unsigned long long>(count));
        text += item;
    }
    return text.empty() ? " none" : text;
}

void ReportGuestHooks(GuestHooks& state, double seconds) {
    std::unordered_set<std::uint64_t> pipelines;
    for (auto& hook : state.hooks) {
        if (hook.calls == 0) continue;
        std::size_t reused = 0;
        for (const auto pipeline : hook.pipelines) reused += state.previousPipelines.contains(pipeline) ? 1 : 0;
        std::fprintf(stderr, "[guest-hook] 0x%llx: %.0f calls/s, %zu objects, %zu shader sets (%zu also in the last window)\n",
                     static_cast<unsigned long long>(hook.rva), static_cast<double>(hook.calls) / seconds, hook.objects.size(), hook.pipelines.size(), reused);
        std::string examples;
        for (const auto& args : hook.examples) {
            char item[160];
            std::snprintf(item, sizeof(item), " (%llx %llx %llx %llx %llx %llx)", static_cast<unsigned long long>(args[0]), static_cast<unsigned long long>(args[1]), static_cast<unsigned long long>(args[2]), static_cast<unsigned long long>(args[3]), static_cast<unsigned long long>(args[4]), static_cast<unsigned long long>(args[5]));
            examples += item;
        }
        std::fprintf(stderr, "[guest-hook-census] 0x%llx: args distinct rsi %zu rdx %zu rcx %zu r8 %zu r9 %zu; e.g. (rdi rsi rdx rcx r8 r9)%s; PM4 to the next hooked call %.0f bytes avg over %llu; draw packets decoded:%s\n",
                     static_cast<unsigned long long>(hook.rva), hook.argValues[0].size(), hook.argValues[1].size(), hook.argValues[2].size(), hook.argValues[3].size(), hook.argValues[4].size(), examples.c_str(),
                     hook.pm4Calls == 0 ? 0.0 : static_cast<double>(hook.pm4Bytes) / static_cast<double>(hook.pm4Calls), static_cast<unsigned long long>(hook.pm4Calls), GuestHookOpcodes(hook.drawOpcodes).c_str());
        pipelines.insert(hook.pipelines.begin(), hook.pipelines.end());
        hook.calls = 0;
        hook.objects.clear();
        hook.pipelines.clear();
        for (auto& values : hook.argValues) values.clear();
        hook.examples.clear();
        hook.pm4Calls = hook.pm4Bytes = 0;
        hook.drawOpcodes.clear();
    }
    std::size_t reused = 0;
    for (const auto pipeline : pipelines) reused += state.previousPipelines.contains(pipeline) ? 1 : 0;
    std::fprintf(stderr, "[guest-hook] window %.1f s: %zu shader sets, %zu also in the last window; draw packets matched to no hooked call:%s\n", seconds, pipelines.size(), reused, GuestHookOpcodes(state.unmatchedDraws).c_str());
    state.previousPipelines = std::move(pipelines);
    state.unmatchedDraws.clear();
}

APS5_VABI void GuestHookCall(std::uint64_t self, std::uint64_t rsi, std::uint64_t rdx, std::uint64_t rcx, std::uint64_t r8, std::uint64_t r9, std::uint64_t index) {
    auto& state = GuestHookState();
    std::lock_guard lock(state.mutex);
    auto& hook = state.hooks[index];
    ++hook.calls;
    hook.objects.insert(self);
    const std::array<std::uint64_t, 6> args{self, rsi, rdx, rcx, r8, r9};
    for (std::size_t i = 0; i < hook.argValues.size(); ++i) {
        if (hook.argValues[i].size() < 4096) hook.argValues[i].insert(args[i + 1]);
    }
    if (hook.examples.size() < 2 && hook.calls % 97 == 1) hook.examples.push_back(args);
    if (hook.readsPipeline && self >= 0x10000) {
        const auto buffer = *reinterpret_cast<const std::uint64_t*>(self + 0x78);
        std::uint64_t shaders = 0xcbf29ce484222325ull;
        for (std::uint64_t stage = 2; buffer >= 0x10000 && stage < 8; ++stage) {
            shaders = (shaders ^ *reinterpret_cast<const std::uint64_t*>(buffer + 0x38 + stage * 0x70)) * 0x100000001b3ull;
        }
        hook.pipelines.insert(shaders);
        if (buffer >= 0x10000) {
            const auto cursor = *reinterpret_cast<const std::uint64_t*>(buffer + 0x10);
            thread_local std::uint64_t lastCursor = 0;
            thread_local std::size_t lastIndex = 0;
            if (lastCursor != 0 && cursor > lastCursor && cursor - lastCursor < 0x100000) {
                auto& previous = state.hooks[lastIndex];
                ++previous.pm4Calls;
                previous.pm4Bytes += cursor - lastCursor;
            }
            lastCursor = cursor;
            lastIndex = index;
            const auto serial = ++state.serial;
            // Entries ahead of the cursor are from earlier recordings of this buffer (the title has
            // not written there yet): they would take the packets this call and the next ones write.
            state.cursors.erase(state.cursors.upper_bound(cursor), state.cursors.upper_bound(cursor + 0x10000));
            state.cursors[cursor] = {serial, index};
            state.cursorOrder.emplace_back(cursor, serial);
            while (state.cursorOrder.size() > (std::size_t{1} << 18)) {
                const auto [address, oldest] = state.cursorOrder.front();
                state.cursorOrder.pop_front();
                if (const auto found = state.cursors.find(address); found != state.cursors.end() && found->second.serial == oldest) state.cursors.erase(found);
            }
        }
    }
    const auto now = std::chrono::steady_clock::now();
    const auto seconds = std::chrono::duration<double>(now - state.windowStart).count();
    if (seconds >= 5.0) {
        ReportGuestHooks(state, seconds);
        state.windowStart = now;
    }
}

void noteGuestHookDrawPacket(std::uint64_t address, std::uint32_t opcode) {
    if (!guestHooksActive.load(std::memory_order_relaxed) || address == 0) return;
    auto& state = GuestHookState();
    std::lock_guard lock(state.mutex);
    auto found = state.cursors.upper_bound(address);
    if (found == state.cursors.begin() || address - std::prev(found)->first >= 0x10000) {
        ++state.unmatchedDraws[opcode];
        return;
    }
    ++state.hooks[std::prev(found)->second.index].drawOpcodes[opcode];
}

void InstallGuestHooks() {
    const char* text = std::getenv("APS5_GUEST_HOOKS");
    if (text == nullptr || *text == '\0') return;
#ifdef _WIN32
    const auto base = reinterpret_cast<std::uint8_t*>(GetModuleHandleW(nullptr));
    // Stubs must be within a rel32 jump of the image: try 64 KiB steps below it.
    std::uint8_t* stubs = nullptr;
    for (std::uintptr_t step = 1; step < 0x7000 && stubs == nullptr; ++step) {
        const auto address = reinterpret_cast<std::uintptr_t>(base) - step * 0x10000;
        stubs = static_cast<std::uint8_t*>(VirtualAlloc(reinterpret_cast<void*>(address), 0x10000, MEM_RESERVE | MEM_COMMIT, PAGE_EXECUTE_READWRITE));
    }
    if (stubs == nullptr) {
        std::fprintf(stderr, "[guest-hook] no stub memory near the image\n");
        return;
    }
    auto& state = GuestHookState();
    std::size_t used = 0;
    std::string list(text);
    std::size_t start = 0;
    while (start < list.size() && used + 256 <= 0x10000) {
        const auto end = std::min(list.find(';', start), list.size());
        const auto entry = list.substr(start, end - start);
        start = end + 1;
        const auto first = entry.find(':');
        if (first == std::string::npos) continue;
        const auto second = entry.find(':', first + 1);
        const auto rva = std::stoull(entry.substr(0, first), nullptr, 16);
        const auto stolenHex = entry.substr(first + 1, (second == std::string::npos ? entry.size() : second) - first - 1);
        std::vector<std::uint8_t> stolen;
        for (std::size_t i = 0; i + 1 < stolenHex.size(); i += 2) stolen.push_back(static_cast<std::uint8_t>(std::stoul(stolenHex.substr(i, 2), nullptr, 16)));
        auto* target = base + rva;
        if (stolen.size() < 5 || std::memcmp(target, stolen.data(), stolen.size()) != 0) {
            std::fprintf(stderr, "[guest-hook] 0x%llx: stolen bytes not found, not hooked\n", static_cast<unsigned long long>(rva));
            continue;
        }
        const auto index = state.hooks.size();
        state.hooks.push_back({rva, second != std::string::npos && entry.substr(second + 1).find('p') != std::string::npos});
        std::vector<std::uint8_t> code = {
            0x57, 0x56, 0x52, 0x51, 0x41, 0x50, 0x41, 0x51, 0x50, 0x41, 0x52, 0x41, 0x53, // push rdi rsi rdx rcx r8 r9 rax r10 r11
            0x48, 0x81, 0xec, 0x80, 0x00, 0x00, 0x00,                                     // sub rsp, 0x80
        };
        for (std::uint8_t reg = 0; reg < 8; ++reg) code.insert(code.end(), {0xf3, 0x0f, 0x7f, static_cast<std::uint8_t>(0x44 | reg << 3), 0x24, static_cast<std::uint8_t>(reg * 0x10)});
        code.insert(code.end(), {0x48, 0x83, 0xec, 0x08, 0x68}); // sub rsp, 8; push index (7th argument)
        for (int i = 0; i < 4; ++i) code.push_back(static_cast<std::uint8_t>(index >> (i * 8)));
        code.insert(code.end(), {0x48, 0xb8}); // mov rax, GuestHookCall
        const auto call = reinterpret_cast<std::uint64_t>(&GuestHookCall);
        for (int i = 0; i < 8; ++i) code.push_back(static_cast<std::uint8_t>(call >> (i * 8)));
        code.insert(code.end(), {0xff, 0xd0, 0x48, 0x83, 0xc4, 0x10}); // call rax; add rsp, 16
        for (std::uint8_t reg = 0; reg < 8; ++reg) code.insert(code.end(), {0xf3, 0x0f, 0x6f, static_cast<std::uint8_t>(0x44 | reg << 3), 0x24, static_cast<std::uint8_t>(reg * 0x10)});
        code.insert(code.end(), {0x48, 0x81, 0xc4, 0x80, 0x00, 0x00, 0x00,                 // add rsp, 0x80
                                 0x41, 0x5b, 0x41, 0x5a, 0x58, 0x41, 0x59, 0x41, 0x58, 0x59, 0x5a, 0x5e, 0x5f}); // pop in reverse
        code.insert(code.end(), stolen.begin(), stolen.end());
        code.insert(code.end(), {0xff, 0x25, 0x00, 0x00, 0x00, 0x00}); // jmp [rip]: back after the stolen bytes
        const auto back = reinterpret_cast<std::uint64_t>(target + stolen.size());
        for (int i = 0; i < 8; ++i) code.push_back(static_cast<std::uint8_t>(back >> (i * 8)));
        auto* stub = stubs + used;
        std::memcpy(stub, code.data(), code.size());
        used += (code.size() + 15) & ~std::size_t{15};
        std::vector<std::uint8_t> jump(stolen.size(), 0x90);
        jump[0] = 0xe9;
        const auto rel = static_cast<std::int32_t>(reinterpret_cast<std::intptr_t>(stub) - reinterpret_cast<std::intptr_t>(target + 5));
        std::memcpy(jump.data() + 1, &rel, 4);
        DWORD previous = 0;
        VirtualProtect(target, jump.size(), PAGE_EXECUTE_READWRITE, &previous);
        std::memcpy(target, jump.data(), jump.size());
        VirtualProtect(target, jump.size(), previous, &previous);
        FlushInstructionCache(GetCurrentProcess(), target, jump.size());
        std::fprintf(stderr, "[guest-hook] 0x%llx: hooked (%zu bytes moved)\n", static_cast<unsigned long long>(rva), stolen.size());
        guestHooksActive.store(true, std::memory_order_relaxed);
    }
    FlushInstructionCache(GetCurrentProcess(), stubs, used);
#else
    std::fprintf(stderr, "[guest-hook] APS5_GUEST_HOOKS is Windows only\n");
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

bool GuestHooksActive() {
    return guestHooksActive.load(std::memory_order_relaxed);
}

void NoteGuestHookDrawPacket(std::uint64_t guestAddress, std::uint32_t opcode) {
    noteGuestHookDrawPacket(guestAddress, opcode);
}

Driver::Driver() {
    ApplyCodePatches();
    InstallGuestHooks();
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
