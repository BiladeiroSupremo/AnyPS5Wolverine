#include "Optimization/SrtWalker/SrtPlanBuilder.hpp"
#include "Optimization/SrtWalker/SrtDiagnostics.hpp"
#include "Optimization/SrtWalker/SrtInstructionPredicates.hpp"
#include "Optimization/SrtWalker/SrtRuntimeValidator.hpp"

#include "IntermediateRepresentation/IrBuilder.hpp"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <vector>

namespace ShaderRecompiler::Detail {

void PlanBuilder::Run() {
    _program.Resources().srtReads.clear();
    _program.Metadata().dynamicReads.clear();
    // An address-SRT program walks its SRT itself, per invocation, through device addresses.
    // Experiment (M4 per-draw resolve, step 1): APS5_ADDRESS_SRT_FLAT=1 plans its flat slots as for
    // any program, so the walk's immediate-offset reads become per-draw flattened SRT words (one
    // uniform load each) while its V#s stay GPU-read (ResourceTracker gpuDescriptor). The CPU fills
    // the slots per draw for now; a GPU evaluator of the same tape is step 2.
    static const bool flatSlots = std::getenv("APS5_ADDRESS_SRT_FLAT") != nullptr;
    // An address-SRT program's WRITTEN V#s stay on the bound path (ResourceTracker::TakeGpuDescriptor:
    // the real pipelines' ISA showed each address-path store as a bounds branch, the store and a
    // written-page note call: mesh executables 17x larger, 2026-10-27 trisa2), so the reads their
    // dwords come from are planned as flat slots: the CPU fills those few words per draw.
    // APS5_ADDRESS_GPU_WRITES=1 keeps every V# of such a program GPU-read as before.
    static const bool boundWrites = std::getenv("APS5_ADDRESS_GPU_WRITES") == nullptr;
    const bool addressProgram = AddressSrtProgram(_program.Resources());
    if (addressProgram && !flatSlots && !boundWrites) return;
    const bool writesOnly = addressProgram && !flatSlots;
    for (auto& block : _program.Blocks()) {
        for (IrValue* inst : block->Instructions()) {
            const auto op = inst->Opcode();
            if (op == IrOpcode::LoadAddressU32 || op == IrOpcode::ReadConstBuffer) {
                const auto flags = inst->Flags<MemoryFlags>();
                if (flags.index < _program.Resources().memoryInfo.size()) {
                    const auto kind = _program.Resources().memoryInfo[flags.index].kind;
                    const bool crosswired = (op == IrOpcode::LoadAddressU32 && kind == ResourceKind::ScalarBuffer) || (op == IrOpcode::ReadConstBuffer && kind == ResourceKind::ScalarAddress);
                    if (crosswired) {
                        Fail(_program.Resources(), flags.pc, std::string(IrOpcodeName(op)) + " has incompatible scalar memory metadata");
                    }
                }
            }
            if (IsDescriptorHandle(inst->Opcode())) {
                // The written V#s only (see above): a buffer handle used by a store or an atomic.
                if (writesOnly) {
                    if (inst->Opcode() != IrOpcode::GetBufferResource) continue;
                    const bool written = std::any_of(inst->Uses().begin(), inst->Uses().end(), [](const IrValue* user) {
                        const auto access = BufferAccessOf(user->Opcode());
                        return access == BufferAccess::Write || access == BufferAccess::Atomic;
                    });
                    if (!written) continue;
                }
                // A GPU-resolved T# or S#: only the V# of the buffer it is read from is planned.
                std::array<IrValue*, 8> reads{};
                if (HashedImageCandidate(_program.Resources(), *inst, &reads) || HashedSamplerCandidate(_program.Resources(), *inst, &reads)) {
                    Collect(reads[0]->Argument(0), 0);
                    continue;
                }
                for (std::size_t index = 0; index < inst->ArgumentCount(); index++) {
                    Collect(inst->Argument(index), 0);
                }
            }
        }
    }
    if (!writesOnly) {
        for (auto& block : _program.Blocks()) {
            for (IrValue* inst : block->Instructions()) {
                if (inst->Opcode() == IrOpcode::LoadAddressU32 && IsRawRead(_program.Resources(), *inst) && inst->Argument(1)->Resolve()->HasImmediate() && RuntimeValidator(_program.Resources(), RuntimeValueType::Any).Run(inst)) {
                    Collect(inst, inst->Flags<MemoryFlags>().pc);
                }
            }
        }
    }
    PatchReads();
}

void PlanBuilder::Collect(IrValue* raw, std::uint32_t usePc) {
    IrValue* value = raw->Resolve();
    if (value->Opcode() == IrOpcode::Void) {
        return;
    }
    IrValue* inst = value;
    const auto cycle = std::find(_visiting.begin(), _visiting.end(), inst);
    if (cycle != _visiting.end()) {
        const auto containsPhi = std::any_of(cycle, _visiting.end(), [](IrValue* candidate) { return candidate->Opcode() == IrOpcode::Phi; });
        if (containsPhi) {
            return;
        }
        Fail(_program.Resources(), usePc, "cyclic typed planning value " + std::string(IrOpcodeName(inst->Opcode())) + " without a phi");
    }
    if (std::find(_visited.begin(), _visited.end(), inst) != _visited.end()) {
        return;
    }
    _visiting.push_back(inst);
    for (std::size_t index = 0; index < inst->ArgumentCount(); index++) {
        Collect(inst->Argument(index), usePc);
    }
    _visiting.pop_back();
    _visited.push_back(inst);
    if (!IsRawRead(_program.Resources(), *inst)) {
        return;
    }
    IrValue* offset = inst->Argument(1)->Resolve();
    const auto foldable = offset->HasImmediate() && offset->Type() == IrType::U32 && RuntimeValidator(_program.Resources(), RuntimeValueType::Any).Run(inst);
    if (!foldable) {
        if (std::find(_program.Metadata().dynamicReads.begin(), _program.Metadata().dynamicReads.end(), value) == _program.Metadata().dynamicReads.end()) {
            _program.Metadata().dynamicReads.push_back(value);
        }
        return;
    }
    for (std::uint32_t slot = 0; slot < _program.Resources().srtReads.size(); slot++) {
        if (EquivalentValue(_program.Resources(), value, _program.Resources().srtReads[slot].value)) {
            _patches.push_back({inst, slot, false});
            return;
        }
    }
    const auto slot = static_cast<std::uint32_t>(_program.Resources().srtReads.size());
    _program.Resources().srtReads.push_back({value, slot});
    _patches.push_back({inst, slot, true});
}

void PlanBuilder::PatchReads() {
    IrBuilder builder(_program);
    for (const auto& patch : _patches) {
        IrBlock* block = patch.inst->Parent();
        if (block == nullptr) {
            throw std::runtime_error("SrtWalker::BuildPlan patched value has no parent block");
        }
        builder.SetInsertionPoint(*block);
        IrValue& resource = builder.Emit(IrOpcode::GetSrtResource, IrType::SrtResource, {});
        block->RemoveInstruction(&resource);
        block->InsertInstructionBefore(patch.inst, &resource);
        IrValue& slotConstant = builder.Constant(patch.slot);
        IrValue& flat = builder.Emit(IrOpcode::ReadConst, IrType::U32, {&resource, &slotConstant});
        block->RemoveInstruction(&flat);
        block->InsertInstructionBefore(patch.inst, &flat);
        patch.inst->ReplaceAllUsesWith(&flat);
        for (auto& info : _program.Metadata().blockInfo) {
            if (info.condition != nullptr && info.condition->Resolve() == patch.inst) {
                info.condition = &flat;
            }
            if (info.indirectTarget != nullptr && info.indirectTarget->Resolve() == patch.inst) {
                info.indirectTarget = &flat;
            }
        }
        if (patch.keep) {
            const auto memory = patch.inst->Flags<MemoryFlags>().index;
            if (memory < _program.Resources().memoryInfo.size()) {
                _program.Resources().memoryInfo[memory].planningOnly = true;
            }
            const auto& irVal = builder.Emit(IrOpcode::ReferenceU32, IrType::Void, {patch.inst});
        }
    }
}

}
