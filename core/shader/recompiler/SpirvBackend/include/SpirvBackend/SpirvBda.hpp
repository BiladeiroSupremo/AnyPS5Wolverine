#ifndef CORE_SHADER_RECOMPILER_SPIRVBACKEND_SPIRVBDA_HPP
#define CORE_SHADER_RECOMPILER_SPIRVBACKEND_SPIRVBDA_HPP

#include "SpirvBackend/SpirvEmitterHelpers.hpp"
#include "SpirvBackend/SpirvEmitter.hpp"
#include "BdaAbi.hpp"
#include <array>
#include <functional>

namespace ShaderRecompiler {

std::uint32_t BdaConstant(SpirvEmitterState& state, std::uint64_t value);
std::uint32_t BdaWord(SpirvEmitterState& state, std::uint32_t variable, std::uint32_t index);
std::uint32_t BdaLoadWord(SpirvEmitterState& state, std::uint32_t index);
std::uint32_t BdaLoadAddress(SpirvEmitterState& state, std::uint32_t index);
void DefineBdaFaultFunction(SpirvEmitterState& state);
void DefineBdaDwordReadFunctions(SpirvEmitterState& state);
void DefineBdaSpanReadFunctions(SpirvEmitterState& state);
void RecordBdaFault(SpirvEmitterState& state, std::uint32_t address, std::uint32_t bytes, std::uint32_t instruction, BdaAbi::FaultReason reason);
void RecordBdaFaultWords(SpirvEmitterState& state, std::uint32_t addressLow, std::uint32_t addressHigh, std::uint32_t bytes, std::uint32_t instruction, BdaAbi::FaultReason reason);
void ReturnBdaFailureIf(SpirvEmitterState& state, std::uint32_t condition, std::uint32_t address, std::uint32_t bytes, std::uint32_t instruction, BdaAbi::FaultReason reason);
void ValidateBdaTarget(const IrProgram& program, const SpirvTargetOptions& target);
// Whether a faulting BDA access may end its invocation. Programs with workgroup barriers must keep
// every invocation running, so their faulting reads return zero instead.
bool BdaInvocationsMayStop(const IrProgram& program);
void StopBdaInvocationIf(SpirvEmitterState& state, std::uint32_t condition);
std::uint32_t BdaInstructionPc(SpirvEmitterState& state, const IrValue& inst);
std::uint32_t EmitBdaRead(SpirvValueEmitContext& ctx, const IrValue& inst, std::uint32_t address, std::uint32_t bits);
void EmitBdaWrite(SpirvValueEmitContext& ctx, const IrValue& inst, std::uint32_t address, std::uint32_t value, std::uint32_t bits = 32u);
void EmitBdaStore(SpirvValueEmitContext& ctx, const IrValue& inst, std::uint32_t address, std::uint32_t value, std::uint32_t bits);
void DefineBdaByteWriteFunctions(SpirvEmitterState& state);
void EmitBdaDwordWrites(SpirvValueEmitContext& ctx, const IrValue& inst, std::uint32_t address, std::uint32_t offset, std::uint32_t dwords, std::uint32_t value);
std::uint32_t EmitBdaAtomic(SpirvValueEmitContext& ctx, const IrValue& inst, std::uint32_t address, std::uint32_t bytes, const std::function<std::uint32_t(std::uint32_t)>& operation);
// Reads the dwords of a 1-4 dword load at address + offset that the program extracts, with one
// table lookup for the whole span; the per-byte lookups of EmitBdaRead remain the fallback (and
// the only path under APS5_BDA_BYTE_READS=1). Dwords the program never extracts are neither read
std::array<std::uint32_t, 4> EmitBdaDwordReads(SpirvValueEmitContext& ctx, const IrValue& inst, std::uint32_t address, std::uint32_t offset, std::uint32_t dwords, bool everyDword = false);
bool BdaByteReadsForced();
std::uint32_t AddBdaAddress(SpirvValueEmitContext& ctx, const IrValue& inst, std::uint32_t address, std::uint32_t offset, bool subtract);
// Experiment (M4, APS5_BDA_MIRROR=1): the table is the guest mirror alone.
bool BdaMirrorMode();
// Mirror mode: the 32-bit word at the 4-byte aligned guest address `guest` (u64) when `valid`, else
// zero, with no branch and no fault record: an invalid or unmapped address reads the mirror's last
// page, which nothing binds (zeros).
// Mirror mode, a non-coherent read: no fault record and no stop at any of its sites (the invalid-V#,
// overflow and unmapped cases read zero through selects). The real pipelines' ISA (2026-10-27) had
// the inlined fault records at every vector read site: mesh executables 17x larger.
bool BdaMirrorFaultless(const SpirvEmitterState& state, const IrValue& inst);
// Experiment: guest-scalar values read on the address path are made wave-uniform (SpirvBdaRead.cpp).
bool BdaUniformReads();
std::uint32_t BroadcastFirstU32(SpirvEmitterState& state, std::uint32_t value);
std::uint32_t BroadcastFirstU64(SpirvEmitterState& state, std::uint32_t value);
// The mirror's table words (ids): begin and end of the mirrored guest range, its device address.
struct MirrorWords {
    std::uint32_t begin;
    std::uint32_t finish;
    std::uint32_t base;
};
// The words the current body holds (HoistMirrorWords), else loaded here.
MirrorWords LoadMirrorWords(SpirvEmitterState& state);
// Loads them once at the start of a body (the entry block) in mirror mode; clears them otherwise.
void HoistMirrorWords(SpirvEmitterState& state);
std::uint32_t EmitMirrorWordRead(SpirvEmitterState& state, std::uint32_t guest, std::uint32_t valid, std::uint32_t accessMask, const MirrorWords& words);
// `count` (1-5) consecutive words from the 4-aligned guest address with ONE range check and one
// pointer for the whole span (a span crossing the mirror's end reads zero as a whole); four of them
// as one vector load. Unused entries of the result are 0 ids.
std::array<std::uint32_t, 5> EmitMirrorWordsRead(SpirvEmitterState& state, std::uint32_t guest, std::uint32_t count, std::uint32_t valid, const MirrorWords& words);
// The same read as one helper function (read_mirror_word) the reads call: defined with the other
// BDA functions in mirror mode; EmitMirrorWordCall falls back to the inline read without it.
void DefineMirrorWordFunction(SpirvEmitterState& state);
std::uint32_t EmitMirrorWordCall(SpirvEmitterState& state, std::uint32_t guest, std::uint32_t valid);
std::uint32_t AddBdaImmediate(SpirvValueEmitContext& ctx, const IrValue& inst, std::uint32_t address, std::int32_t immediate);

}

#endif
