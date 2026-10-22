#ifndef CORE_SHADER_RECOMPILIER_OPTIMIZATION_INCLUDE_OPTIMIZATION_MASKEDSELECTELIMINATOR_HPP
#define CORE_SHADER_RECOMPILIER_OPTIMIZATION_INCLUDE_OPTIMIZATION_MASKEDSELECTELIMINATOR_HPP

#include "IntermediateRepresentation/IrProgram.hpp"
#include <cstdint>

namespace ShaderRecompiler {

struct MaskedSelectEliminationStats {
    std::uint32_t removedSelects = 0;
    std::uint64_t steps = 0;
};

// Whether an instruction of this opcode only reads its own lane's operands and has no side effects
// (no cross-lane, derivative or memory access), so where it runs does not change its result.
[[nodiscard]] bool IsLaneLocalOpcode(IrOpcode opcode);

class MaskedSelectEliminator {
public:
    [[nodiscard]] MaskedSelectEliminationStats Eliminate(IrProgram& program) const;
};

}

#endif
