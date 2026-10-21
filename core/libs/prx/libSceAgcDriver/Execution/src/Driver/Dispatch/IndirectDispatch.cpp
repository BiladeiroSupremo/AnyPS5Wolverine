#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Dispatch/IndirectDispatch.hpp"
#include "prx/libSceAgcDriver/Execution/include/Pm4.hpp"
#include <cstdlib>

namespace AgcDriver::DriverDetail {

void Driver::countIndirect(int path, double readMs) {
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    if (!profile || path < 0 || path >= IndirectPaths) return;
    static std::mutex countsMutex;
    static std::uint64_t counts[IndirectPaths] = {};
    static double readWaitedMs = 0;
    static auto lastReport = std::chrono::steady_clock::now();
    std::lock_guard lock(countsMutex);
    ++counts[path];
    readWaitedMs += readMs;
    const auto now = std::chrono::steady_clock::now();
    if (now - lastReport < std::chrono::seconds(10)) return;
    lastReport = now;
    std::uint64_t cpu = 0;
    for (int i = 1; i < IndirectPaths; ++i) cpu += counts[i];
    std::fprintf(stderr, "[indirect] gpu-side %llu, cpu-side %llu (thread dimensions %llu, fill kernel %llu, pending image results %llu, pending label or copied write %llu, not imported %llu, misaligned %llu, disabled %llu, device memory LDS %llu); CPU argument reads took %.1f s\n", static_cast<unsigned long long>(counts[IndirectGpu]), static_cast<unsigned long long>(cpu), static_cast<unsigned long long>(counts[IndirectThreadDimensions]), static_cast<unsigned long long>(counts[IndirectFillKernel]), static_cast<unsigned long long>(counts[IndirectPendingImage]), static_cast<unsigned long long>(counts[IndirectCopiedWrite]), static_cast<unsigned long long>(counts[IndirectNotImported]), static_cast<unsigned long long>(counts[IndirectMisaligned]), static_cast<unsigned long long>(counts[IndirectDisabled]), static_cast<unsigned long long>(counts[IndirectWorkgroupMemory]), readWaitedMs / 1000);
}

void Driver::dispatchIndirect(QueueState& queue, std::span<const std::uint32_t> packet, const Submission& submission) {
    // A skip-listed program is dropped before its arguments are read (as KytyPS5 41c11018): the
    // title's ray-tracing dispatches are thousands a frame, and their arguments are garbage.
    {
        const auto shaderRegister = [&](std::uint32_t offset) { const auto it = queue.shader.find(offset); return it == queue.shader.end() ? 0u : it->second; };
        const auto program = (static_cast<std::uint64_t>(shaderRegister(0x20c)) << 8u) | (static_cast<std::uint64_t>(shaderRegister(0x20d) & 0xffu) << 40u);
        if (const auto snapshot = RegisteredProgram(*submission.shaders, program, {0}); snapshot != nullptr && SkipListed(*snapshot)) return;
    }
    static const bool gpuIndirect = std::getenv("APS5_NO_GPU_INDIRECT") == nullptr;
    const auto arguments = Pm4::DispatchArgumentAddress(packet, queue);
    const auto initiator = packet.back();
    int path = IndirectGpu;
    if (!gpuIndirect) path = IndirectDisabled;
    // Thread dimensions (bit 0x20): read on the CPU. APS5_GPU_THREAD_DIMENSIONS=1 (experimental)
    // has the device convert the thread counts on the GPU instead, except with a partial last group
    // (PARTIAL_TG_EN, bit 0x2), whose extra threads only the CPU path's direct dispatch limits
    // (Pm4::PartialGroupThreads). Unsolved: with it the title froze after its first-boot dialog
    // (run tg1) and, with partial groups excluded, lost the device within ~3 minutes (tg2,
    // 2026-10-14); a debug readback comparing the kernel's groups to the CPU's is the next step.
    else if ((initiator & 0x20u) != 0 && ((initiator & 0x2u) != 0 || std::getenv("APS5_GPU_THREAD_DIMENSIONS") == nullptr)) path = IndirectThreadDimensions;
    else if (arguments % 4 != 0) path = IndirectMisaligned;
    if (path != IndirectGpu) {
        if (!resolvingAhead()) {
            const auto& resolved = resolvedAhead();
            if (const auto found = resolved.find(currentPacketOffset()); found != resolved.end() && found->second.indirectArguments == 0 && found->second.packet[0] == 0xc0031500u) {
                const auto direct = found->second.packet;
                dispatch(queue, direct, submission);
                return;
            }
        }
        recordQueuedLabelsBeforeRead(submission.queue);
        const auto readStart = std::chrono::steady_clock::now();
        const auto direct = Pm4::ReadDispatchArguments(arguments, initiator);
        static std::atomic<std::uint32_t> reportedLarge{0};
        if (static_cast<std::uint64_t>(direct[1]) * direct[2] * direct[3] > (std::uint64_t{1} << 24u) && reportedLarge.fetch_add(1, std::memory_order_relaxed) < 16u) {
            const auto shaderRegister = [&](std::uint32_t offset) { const auto it = queue.shader.find(offset); return it == queue.shader.end() ? 0u : it->second; };
            const auto program = (static_cast<std::uint64_t>(shaderRegister(0x20c)) << 8u) | (static_cast<std::uint64_t>(shaderRegister(0x20d) & 0xffu) << 40u);
            std::fprintf(stderr, "[gpu] indirect dispatch of 0x%llx reads %ux%ux%u from 0x%llx (base 0x%llx + 0x%x, %zu-word packet, queue 0x%x, path %d, initiator 0x%x)\n", static_cast<unsigned long long>(program), direct[1], direct[2], direct[3], static_cast<unsigned long long>(arguments), static_cast<unsigned long long>(queue.dispatchIndirectBase), packet.size() > 1 ? packet[1] : 0u, packet.size(), submission.queue, path, initiator);
        }
        countIndirect(path, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - readStart).count());
        dispatch(queue, direct, submission);
        return;
    }

    const std::array<std::uint32_t, 5> unresolved{0xc0031500u, 0, 0, 0, initiator};
    dispatch(queue, unresolved, submission, arguments);
}

}
