#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_SHADERS_SHADERCALLS_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_SHADERS_SHADERCALLS_HPP

#include "prx/libSceAgcDriver/Execution/include/Driver/Shaders/ShaderRegistry.hpp"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

namespace AgcDriver::DriverDetail {

// Subroutine calls (s_swappc_b64) are expanded before a program is recompiled: each call whose
// target is known from the program alone or its user data (a pointer passed in user data, copied
// through s_mov, or s_getpc-relative) is replaced by the callee's code, its return (s_setpc_b64
// through the return pair) by a branch back to the caller, with the return address still written
// to the return pair. s_getpc_b64 becomes moves of the original program counter, so the expanded
// code still addresses the original code's constants. The approach follows KytyPS5's
// ShaderFunctions (call tables, which it also resolves, are not supported here).
//
// The expanded program as a snapshot of its own (code at offset 0, its own source-handle memos),
// cached by the snapshot, the code offset, the wave size, the values of the user-data registers
// the targets were resolved from and the callee code as read. Null when the program at
// `codeOffset` has no calls; throws std::runtime_error naming the reason when it has calls that
// cannot be expanded.
std::shared_ptr<const ShaderSnapshot> ExpandShaderCalls(const std::shared_ptr<const ShaderSnapshot>& snapshot, std::size_t codeOffset, std::span<const std::uint32_t> userData, std::uint32_t waveSize);

}

#endif
