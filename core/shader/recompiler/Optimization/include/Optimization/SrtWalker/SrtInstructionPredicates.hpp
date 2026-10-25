#ifndef CORE_SHADER_RECOMPILIER_OPTIMIZATION_SRTWALKER_SRTINSTRUCTIONPREDICATES_HPP
#define CORE_SHADER_RECOMPILIER_OPTIMIZATION_SRTWALKER_SRTINSTRUCTIONPREDICATES_HPP

#include "IntermediateRepresentation/IrProgram.hpp"
#include <array>

namespace ShaderRecompiler::Detail {

// A T# the GPU resolves (BindlessHeaps::HashedImagesEnabled): the handle's eight dwords are reads of
// one scalar buffer at one constant offset, one dword apart, and every operation on it is a sampled
// image access the heap set serves. The reads stay runtime loads instead of SRT words. `reads` gets
// them, `offset` the first dword's byte offset.
bool HashedImageCandidate(const IrResourcePlan& program, const IrValue& handle, std::array<IrValue*, 8>* reads = nullptr, std::uint32_t* offset = nullptr);
// The same for an S# (four dwords) used only by sampling ops without depth comparison or LOD
// adjustment: the GPU resolves it to a slot of the heap set's sampler array.
bool HashedSamplerCandidate(const IrResourcePlan& program, const IrValue& handle, std::array<IrValue*, 8>* reads = nullptr, std::uint32_t* offset = nullptr);
bool IsRawRead(const IrResourcePlan& program, const IrValue& inst);
bool IsRawRead(const IrResourcePlan& program, IrOpcode op, std::uint32_t memoryIndex);
bool LoadedFromMemory(const IrValue& handle);
bool IsDescriptorHandle(IrOpcode opcode);
bool IsRuntimeSelect(IrOpcode opcode);
bool IsRuntimeUniformOp(IrOpcode opcode);

}

#endif
