#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_DRIVER_DRAW_DRAWREGISTERKEY_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_DRIVER_DRAW_DRAWREGISTERKEY_HPP

#include "prx/libSceAgcDriver/Execution/include/QueueState.hpp"
#include <vector>

namespace AgcDriver::DriverDetail {

// skipUserWords: sorted shader register offsets of user words left out of the key (relocated
// pointers); noUserWords leaves every user word out (the draw's shape).
std::uint64_t RegisterStateKey(const QueueState& queue, bool allUserWords = false, bool incremental = true, const std::vector<std::uint32_t>* skipUserWords = nullptr, bool noUserWords = false);

}

#endif
