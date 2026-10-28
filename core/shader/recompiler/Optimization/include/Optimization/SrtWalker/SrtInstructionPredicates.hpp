#ifndef CORE_SHADER_RECOMPILIER_OPTIMIZATION_SRTWALKER_SRTINSTRUCTIONPREDICATES_HPP
#define CORE_SHADER_RECOMPILIER_OPTIMIZATION_SRTWALKER_SRTINSTRUCTIONPREDICATES_HPP

#include "IntermediateRepresentation/IrProgram.hpp"
#include <array>
#include <span>

namespace ShaderRecompiler::Detail {

// A T# the GPU resolves (BindlessHeaps::HashedImagesEnabled): the handle's eight dwords are reads of
// one scalar buffer at one constant offset, one dword apart, and every operation on it is a sampled
// image access the heap set serves. The reads stay runtime loads instead of SRT words. `reads` gets
// them, `offset` the first dword's byte offset.
bool HashedImageCandidate(const IrResourcePlan& program, const IrValue& handle, std::array<IrValue*, 8>* reads = nullptr, std::uint32_t* offset = nullptr);
// The same for an S# (four dwords) used only by sampling ops without depth comparison or LOD
// adjustment: the GPU resolves it to a slot of the heap set's sampler array.
bool HashedSamplerCandidate(const IrResourcePlan& program, const IrValue& handle, std::array<IrValue*, 8>* reads = nullptr, std::uint32_t* offset = nullptr);
// APS5_ADDRESS_SRT=<hex code address,...>: compute programs whose SRT is left entirely to the shader
// (no flat slots, buffers whose V# is not user data go through the address-based path), for
// programs that read words an earlier GPU pass writes, so the CPU never waits for that pass.
bool AddressSrtProgram(const IrResourcePlan& program);
// APS5_ADDRESS_BOUND_FORMATS=1: an address-SRT program's V#s read by format-converting loads stay on
// the bound path (ResourceTracker::TakeGpuDescriptor; SrtPlanBuilder plans their dwords' reads).
bool AddressBoundFormats();
// APS5_ADDRESS_USER_VSHARPS=1: an address-SRT program reads its user-data V#s through the mirror
// too, unless it writes through them or reads them with a run-time format.
bool AddressUserVsharps();
// Whether an address-SRT program binds the V# a buffer access uses: user-data V#s always, written
// V#s with APS5_ADDRESS_BOUND_WRITES, format-converting reads with APS5_ADDRESS_BOUND_FORMATS.
bool AddressBoundAccess(const IrResourcePlan& program, const IrValue& user);
// FNV-1a over a program's code words: its name for APS5_ADDRESS_SRT across runs.
std::uint64_t CodeContentHash(std::span<const std::uint32_t> code);
// Called for every request before translation: a program whose code hash is listed in
// APS5_ADDRESS_SRT is matched by its code address from then on.
void NoteAddressSrtCode(std::uint64_t codeAddress, std::span<const std::uint32_t> code);
// APS5_ADDRESS_SRT_LEARN=1: a compute capture that waited for GPU-written words reports its
// program's code hash once ([address-srt-learn]), the candidates for the next APS5_ADDRESS_SRT.
void ReportCaptureWait(std::uint64_t codeAddress);
bool IsRawRead(const IrResourcePlan& program, const IrValue& inst);
bool IsRawRead(const IrResourcePlan& program, IrOpcode op, std::uint32_t memoryIndex);
bool LoadedFromMemory(const IrValue& handle);
bool IsDescriptorHandle(IrOpcode opcode);
bool IsRuntimeSelect(IrOpcode opcode);
bool IsRuntimeUniformOp(IrOpcode opcode);

}

#endif
