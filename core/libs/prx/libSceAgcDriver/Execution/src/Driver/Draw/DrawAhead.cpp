#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawAhead.hpp"
#include "prx/libSceAgcDriver/Execution/include/PerformanceTrace.hpp"
#include <algorithm>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <thread>

namespace AgcDriver::DriverDetail {

namespace {

struct DrawAheadPool {
    std::mutex mutex;
    std::condition_variable wake;
    std::deque<std::function<void()>> work;
    std::vector<std::thread> threads;
    bool stopping = false;

    void run() {
        if (PerformanceTrace::Configured()) PerformanceTrace::BindThread("draw prepare");
        for (;;) {
            std::function<void()> next;
            {
                std::unique_lock lock(mutex);
                wake.wait(lock, [&] { return stopping || !work.empty(); });
                if (work.empty()) return;
                next = std::move(work.front());
                work.pop_front();
            }
            next();
        }
    }
};

DrawAheadPool& pool() {
    static auto* instance = new DrawAheadPool();
    return *instance;
}

}

DrawAheadTask*& CurrentDrawAhead() {
    static thread_local DrawAheadTask* task = nullptr;
    return task;
}

std::vector<std::pair<std::uint64_t, std::uint64_t>>*& DrawWindowWrites() {
    static thread_local std::vector<std::pair<std::uint64_t, std::uint64_t>>* writes = nullptr;
    return writes;
}

bool& DrawWindowInvalidated() {
    static thread_local bool invalidated = false;
    return invalidated;
}

std::size_t DrawAheadThreads() {
    static const std::size_t threads = [] {
        const char* text = std::getenv("APS5_PREPARE_AHEAD");
        return text != nullptr ? static_cast<std::size_t>(std::min<unsigned long long>(std::strtoull(text, nullptr, 10), 16)) : std::size_t{0};
    }();
    return threads;
}

void RunDrawAhead(std::function<void()> work) {
    auto& instance = pool();
    {
        std::unique_lock lock(instance.mutex);
        if (instance.stopping) {
            lock.unlock();
            work();
            return;
        }
        if (instance.threads.empty()) {
            for (std::size_t i = 0; i < DrawAheadThreads(); ++i) instance.threads.emplace_back([&instance] { instance.run(); });
        }
        instance.work.push_back(std::move(work));
    }
    instance.wake.notify_one();
}

void StopDrawAhead() {
    auto& instance = pool();
    std::vector<std::thread> threads;
    {
        std::lock_guard lock(instance.mutex);
        instance.stopping = true;
        threads = std::move(instance.threads);
    }
    instance.wake.notify_all();
    for (auto& thread : threads) {
        if (thread.joinable()) thread.join();
    }
}

}
