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
    std::uint32_t slot = 0;
};

// Each recording thread (one per queue in practice) writes its own marker word, so one queue's
// progress does not hide another's hang.
constexpr std::uint32_t Slots = 16;
std::atomic<std::uint32_t> nextSlot{0};
std::uint32_t ThreadSlot() {
    thread_local const std::uint32_t slot = nextSlot.fetch_add(1) % Slots;
    return slot;
}

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
    markers = std::make_unique<Buffer>(context, Slots * 4u, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    std::memset(markers->Bytes().data(), 0, Slots * 4u);
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
    entry.slot = ThreadSlot();
    entry.id.store(id, std::memory_order_release);
    writeMarker(commands, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, markers->Handle(), entry.slot * 4u, id);
}

void Report() {
    if (markers == nullptr) return;
    const auto last = next.load(std::memory_order_relaxed);
    for (std::uint32_t slot = 0; slot < Slots; ++slot) {
        std::uint32_t done = 0;
        std::memcpy(&done, markers->Bytes().data() + slot * 4u, 4);
        if (done == 0) continue;
        // The commands this recording thread recorded after the last one the GPU finished.
        std::uint32_t pending = 0;
        for (std::uint32_t id = done + 1; id <= last && last - id < RingSize; ++id) {
            if (Ring()[id % RingSize].id.load(std::memory_order_acquire) == id && Ring()[id % RingSize].slot == slot) ++pending;
        }
        std::fprintf(stderr, "[breadcrumbs] slot %u: the GPU finished command %u of %u recorded; %u of this slot's commands pending\n", slot, done, last, pending);
        std::uint32_t shownDone = 0, shownPending = 0;
        for (std::uint32_t id = done; id >= 1 && shownDone < 4 && done - id < RingSize; --id) {
            const auto& entry = Ring()[id % RingSize];
            if (entry.id.load(std::memory_order_acquire) != id || entry.slot != slot) continue;
            std::fprintf(stderr, "[breadcrumbs]   done %u %s program 0x%llx %u %u %u\n", id, entry.kind, static_cast<unsigned long long>(entry.program), entry.a, entry.b, entry.c);
            ++shownDone;
        }
        for (std::uint32_t id = done + 1; id <= last && shownPending < 12 && last - id < RingSize; ++id) {
            const auto& entry = Ring()[id % RingSize];
            if (entry.id.load(std::memory_order_acquire) != id || entry.slot != slot) continue;
            std::fprintf(stderr, "[breadcrumbs]   PENDING %u %s program 0x%llx %u %u %u\n", id, entry.kind, static_cast<unsigned long long>(entry.program), entry.a, entry.b, entry.c);
            ++shownPending;
        }
    }
}

}
