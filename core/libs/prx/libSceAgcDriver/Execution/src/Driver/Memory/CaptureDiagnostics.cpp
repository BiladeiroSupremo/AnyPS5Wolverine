#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include <algorithm>
#include <chrono>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <vector>

namespace AgcDriver::DriverDetail {

bool Driver::traceBudget() {
    static std::atomic<int> lines{0};
    return lines.fetch_add(1) < 600;
}

void Driver::traceCapture(const char* what, std::uint64_t program, std::uint32_t queue, std::span<const ShaderRecompiler::MemoryRegion> regions, double waitedMs) {
    static const double minMs = [] { const char* text = std::getenv("APS5_TRACE_CAPSYNC_MIN_MS"); return text ? std::atof(text) : 0.05; }();
    if (waitedMs < minMs) return;
    // Per program and queue, the captures that waited and the time, on a [capsync-top] line every
    // 10 s (the line budget below is spent during loading); a program's first wait of 1 ms or more
    // is described in full once, budget or not (up to 300 programs).
    bool describe = false;
    {
        struct Total {
            std::uint64_t count = 0;
            double ms = 0;
            const char* what = "";
        };
        static std::mutex mutex;
        static std::map<std::pair<std::uint64_t, std::uint32_t>, Total> totals;
        static std::set<std::uint64_t> described;
        static auto last = std::chrono::steady_clock::now();
        std::lock_guard lock(mutex);
        auto& total = totals[{program, queue}];
        ++total.count;
        total.ms += waitedMs;
        total.what = what;
        if (waitedMs >= 1.0 && described.size() < 300) describe = described.insert(program).second;
        const auto now = std::chrono::steady_clock::now();
        if (now - last >= std::chrono::seconds(10)) {
            last = now;
            std::vector<std::pair<std::pair<std::uint64_t, std::uint32_t>, Total>> top(totals.begin(), totals.end());
            std::sort(top.begin(), top.end(), [](const auto& a, const auto& b) { return a.second.ms > b.second.ms; });
            std::string line = "[capsync-top] (10 s) program q count ms:";
            char item[96];
            for (std::size_t index = 0; index < top.size() && index < 16u; ++index) {
                std::snprintf(item, sizeof(item), " %llx q0x%x %s %llu %.0f;", static_cast<unsigned long long>(top[index].first.first), top[index].first.second, top[index].second.what, static_cast<unsigned long long>(top[index].second.count), top[index].second.ms);
                line += item;
            }
            std::fprintf(stderr, "%s\n", line.c_str());
            totals.clear();
        }
    }
    if (!describe && !traceBudget()) return;
    std::fprintf(stderr, "[capsync] %s q0x%x program 0x%llx waited %.1f ms over %zu regions\n", what, queue, static_cast<unsigned long long>(program), waitedMs, regions.size());
    static const bool values = std::getenv("APS5_TRACE_CAPSYNC_VALUES") != nullptr;
    for (const auto& region : regions) {
        const auto begin = region.guestAddress;
        const auto end = begin + region.bytes.size();
        const auto pageBegin = begin & ~static_cast<std::uint64_t>(4095);
        const auto pageEnd = (end + 4095) & ~static_cast<std::uint64_t>(4095);
        if (values) {
            std::string text;
            for (std::size_t offset = 0; offset < std::min<std::size_t>(region.bytes.size(), 64); ++offset) {
                char hex[4];
                std::snprintf(hex, sizeof(hex), "%02x", static_cast<unsigned>(region.bytes[offset]));
                text += hex;
                if (offset % 4 == 3) text += ' ';
            }
            std::fprintf(stderr, "[capsync]   value 0x%llx+0x%zx: %s\n", static_cast<unsigned long long>(begin), region.bytes.size(), text.c_str());
        }
        if (!newestWriter(pageBegin, pageEnd)) continue;
        std::fprintf(stderr, "[capsync]   region 0x%llx+0x%zx: writers of its page(s):%s; of its dwords:%s\n", static_cast<unsigned long long>(begin), region.bytes.size(), describeWriters(pageBegin, pageEnd).c_str(), describeWriters(begin, end).c_str());
    }
}

void Driver::reportValidation(ValidateCounters& counters) {
    const auto count = [](std::uint64_t value) { return static_cast<unsigned long long>(value); };
    std::size_t tracked = 0, eligible = 0;
    {
        std::lock_guard lock(writtenBuffersMutex);
        tracked = dwordEvidence.size();
        for (const auto& [address, evidence] : dwordEvidence) {
            if (evidence.streak >= writeEvidenceAfter()) ++eligible;
        }
    }
    AgcDriver::ProfilePrint_nid_no_patch( "[validate] dispatch-cache compares over pending GPU writes (10 s): %llu; missed without the sync %llu, skipped the sync %llu, synced %llu waiting %.0f ms (no dispatch writer %llu, foreign writer %llu, larger range %llu, label %llu, evidence short %llu, writer changed %llu, sampled %llu, image %llu, shadow %llu, off %llu); known-value reads %llu; verify: %llu would-skip compares synced anyway (%.0f ms), %llu mismatches, %llu misses a sync made hits; dwords observed over waits: %llu unchanged, %llu changed (regions not observed: %llu after no GPU wait, %llu not compared); %zu tracked, %zu with a streak\n", count(counters.pending), count(counters.unsyncedMisses), count(counters.skipped), count(counters.syncedNoWriter + counters.syncedForeign + counters.syncedLargeRange + counters.syncedLabel + counters.syncedEvidence + counters.syncedWriterChanged + counters.syncedSample + counters.syncedImage + counters.syncedShadow + counters.syncedOff), counters.syncedWaitMs, count(counters.syncedNoWriter), count(counters.syncedForeign), count(counters.syncedLargeRange), count(counters.syncedLabel), count(counters.syncedEvidence), count(counters.syncedWriterChanged), count(counters.syncedSample), count(counters.syncedImage), count(counters.syncedShadow), count(counters.syncedOff), count(counters.knownValue), count(counters.verified), counters.verifiedWaitMs, count(counters.mismatches), count(counters.verifiedMissesHit), count(observedUnchanged.exchange(0)), count(observedChanged.exchange(0)), count(observationsNoWait.exchange(0)), count(observationsNotReached.exchange(0)), tracked, eligible);
    counters = ValidateCounters{};
}

}
