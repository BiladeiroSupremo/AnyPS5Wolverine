#include "prx/libSceAgcDriver/Graphics/include/Breadcrumbs.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"

#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>

namespace AgcDriver::Graphics::Breadcrumbs {

namespace {

struct Entry {
    std::atomic<std::uint32_t> id{0};
    const char* kind = "";
    std::uint64_t program = 0;
    std::uint32_t a = 0, b = 0, c = 0;
};

constexpr std::size_t RingSize = 1u << 16u;
std::array<Entry, RingSize>& Ring() {
    static auto* ring = new std::array<Entry, RingSize>();
    return *ring;
}

std::atomic<std::uint32_t> next{0};
std::unique_ptr<Buffer> markers;
PFN_vkCmdWriteBufferMarkerAMD writeMarker = nullptr;
thread_local std::uint64_t currentProgram = 0;

}

bool Requested() {
    static const bool requested = std::getenv("APS5_GPU_BREADCRUMBS") != nullptr;
    return requested;
}

void Attach(const Context& context, PFN_vkCmdWriteBufferMarkerAMD write) {
    if (!Requested() || write == nullptr) return;
    markers = std::make_unique<Buffer>(context, 16, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    std::memset(markers->Bytes().data(), 0, 16);
    writeMarker = write;
    std::fprintf(stderr, "[breadcrumbs] on: every draw and dispatch writes a completion marker\n");
}

void SetProgram(std::uint64_t program) {
    currentProgram = program;
}

std::uint64_t Program() {
    return currentProgram;
}

void Mark(VkCommandBuffer commands, const char* kind, std::uint64_t program, std::uint32_t a, std::uint32_t b, std::uint32_t c) {
    if (writeMarker == nullptr) return;
    const auto id = next.fetch_add(1, std::memory_order_relaxed) + 1;
    auto& entry = Ring()[id % RingSize];
    entry.kind = kind;
    entry.program = program;
    entry.a = a;
    entry.b = b;
    entry.c = c;
    entry.id.store(id, std::memory_order_release);
    writeMarker(commands, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, markers->Handle(), 0, id);
}

void Report() {
    if (markers == nullptr) return;
    std::uint32_t done = 0;
    std::memcpy(&done, markers->Bytes().data(), 4);
    const auto last = next.load(std::memory_order_relaxed);
    std::fprintf(stderr, "[breadcrumbs] the GPU finished command %u of %u recorded; the last finished and the next ones:\n", done, last);
    for (std::uint32_t id = done > 8 ? done - 8 : 1; id <= last && id <= done + 24; ++id) {
        const auto& entry = Ring()[id % RingSize];
        if (entry.id.load(std::memory_order_acquire) != id) continue;
        std::fprintf(stderr, "[breadcrumbs] %s %u %s program 0x%llx %u %u %u\n", id <= done ? "done" : "PENDING", id, entry.kind, static_cast<unsigned long long>(entry.program), entry.a, entry.b, entry.c);
    }
}

}
