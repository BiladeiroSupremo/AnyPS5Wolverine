#include "prx/libc/include/PreciseWait.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <iostream>

int main() {
    static_assert(PreciseHundredNanosToQpcTicks(500, 10000000ULL) == 500);
    static_assert(PreciseHundredNanosToQpcTicks(500, 3579545ULL) == 179);
    static_assert(PreciseHundredNanosToQpcTicks(500, 14318180ULL) == 716);
    static_assert(PreciseHundredNanosToQpcTicks(500, 24000000ULL) == 1200);
    static_assert(PrecisePauseThresholdTicks(3579545ULL) == 1);
    static_assert(PrecisePauseThresholdTicks(10000000ULL) == 2);
    static_assert(PrecisePauseThresholdTicks(24000000ULL) == 5);

    for (const std::uint64_t freq : {3579545ULL, 10000000ULL, 14318180ULL, 19200000ULL, 24000000ULL}) {
        for (const unsigned long long hundredNanos : {1ULL, 5ULL, 100ULL, 300ULL, 500ULL, 10000ULL, 53330ULL}) {
            const std::uint64_t ticks = PreciseHundredNanosToQpcTicks(hundredNanos, freq);
            if (ticks * 10000000ULL < hundredNanos * freq) {
                std::cerr << "PreciseHundredNanosToQpcTicks rounded below requested duration\n";
                return 1;
            }
        }
    }

    PreciseSleepUs(0);

    for (const unsigned long long micros : {50ULL, 1000ULL}) {
        const auto start = std::chrono::steady_clock::now();
        PreciseSleepUs(micros);
        const auto elapsed = std::chrono::steady_clock::now() - start;
        if (elapsed < std::chrono::microseconds(micros)) {
            std::cerr << "PreciseSleepUs returned before its requested duration\n";
            return 1;
        }
    }

    std::array<std::chrono::nanoseconds, 31> samples{};
    for (auto& sample : samples) {
        const auto start = std::chrono::steady_clock::now();
        PreciseSleepUs(50);
        sample = std::chrono::steady_clock::now() - start;
        if (sample < std::chrono::microseconds(50)) {
            std::cerr << "PreciseSleepUs(50) returned before its requested duration\n";
            return 1;
        }
    }
    std::sort(samples.begin(), samples.end());
    const auto median = samples[samples.size() / 2];
    if (median >= std::chrono::microseconds(90)) {
        std::cerr << "PreciseSleepUs(50) median overslept target bound: "
                  << std::chrono::duration_cast<std::chrono::nanoseconds>(median).count() << " ns\n";
        return 1;
    }

    const auto subMicroDeadline = std::chrono::steady_clock::now() + std::chrono::nanoseconds(500);
    PreciseSleepUntil(subMicroDeadline);
    if (std::chrono::steady_clock::now() < subMicroDeadline) {
        std::cerr << "PreciseSleepUntil returned before sub-microsecond deadline\n";
        return 1;
    }

    const auto millisecondDeadline = std::chrono::steady_clock::now() + std::chrono::microseconds(1000);
    PreciseSleepUntil(millisecondDeadline);
    if (std::chrono::steady_clock::now() < millisecondDeadline) {
        std::cerr << "PreciseSleepUntil returned before millisecond deadline\n";
        return 1;
    }

#ifdef _WIN32
    for (const HANDLE badHandle : {static_cast<HANDLE>(nullptr), GetCurrentProcess()}) {
        const auto start = std::chrono::steady_clock::now();
        LARGE_INTEGER counter;
        QueryPerformanceCounter(&counter);
        PreciseSleepFromCounterWithTimer(counter, 10000ULL, badHandle, true);
        const auto elapsed = std::chrono::steady_clock::now() - start;
        if (elapsed < std::chrono::microseconds(1000)) {
            std::cerr << "PreciseSleepFromCounterWithTimer fallback returned early\n";
            return 1;
        }
    }
#endif

#ifdef __linux__
    if (PreciseLinuxLowTimerSlack() && prctl(PR_GET_TIMERSLACK, 0UL, 0UL, 0UL, 0UL) != 1L) {
        std::cerr << "PreciseLinuxLowTimerSlack did not configure 1ns timer slack\n";
        return 1;
    }
    for (const unsigned long long nanos : {80000ULL, 1000000ULL}) {
        const auto start = std::chrono::steady_clock::now();
        PreciseSleepFromNanosWithSlack(PreciseMonotonicNanos(), nanos, false);
        const auto elapsed = std::chrono::steady_clock::now() - start;
        if (elapsed < std::chrono::nanoseconds(nanos)) {
            std::cerr << "PreciseSleepFromNanosWithSlack conservative fallback returned early\n";
            return 1;
        }
    }
#endif

    return 0;
}
