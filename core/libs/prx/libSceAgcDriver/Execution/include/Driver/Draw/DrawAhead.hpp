#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DRAW_DRAWAHEAD_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DRAW_DRAWAHEAD_HPP

#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawCache.hpp"
#include "prx/libSceAgcDriver/Execution/include/QueueState.hpp"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <future>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace AgcDriver::DriverDetail {

// Draw prepare-ahead (APS5_PREPARE_AHEAD=<threads>, with APS5_PIPELINED_DRAWS): while queue 0's
// worker executes a submission, the direct draws that follow the current packet behind nothing but
// register and draw-state packets run Driver::draw on a pool of threads, each against its own copy of
// the queue state as it will be at that draw. Instead of handing the draw to the committer, draw()
// leaves the commit in the task; the worker takes the tasks in packet order and enqueues their
// commits, or runs the draw itself when a task could not prepare it (DrawAheadSerial) or read memory
// that a draw queued before it may write.
struct DrawAheadSerial {};

struct DrawAheadTask {
    QueueState state;
    std::span<const std::uint32_t> packet;
    std::promise<void> done;
    std::shared_future<void> ready;
    bool serial = false;
    std::optional<std::string> error;
    DrawVerdict verdict = DrawVerdict::Drawn;
    std::string rejected;
    std::function<void()> commit;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> writes;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> reads;
};

// The task the calling thread prepares (null on every other thread, the worker included).
DrawAheadTask*& CurrentDrawAhead();
// On the worker while tasks are prepared ahead: the ranges the draws it enqueued since may write (the
// tasks' captures predate them), and whether it recorded a draw inline meanwhile (the remaining tasks
// are then redone). Null / false otherwise.
std::vector<std::pair<std::uint64_t, std::uint64_t>>*& DrawWindowWrites();
bool& DrawWindowInvalidated();
// Pool threads (0 = off).
std::size_t DrawAheadThreads();
void RunDrawAhead(std::function<void()> work);
void StopDrawAhead();

}

#endif
