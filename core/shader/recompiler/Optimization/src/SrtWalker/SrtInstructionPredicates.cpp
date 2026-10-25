#include "Optimization/SrtWalker/SrtInstructionPredicates.hpp"
#include "Optimization/BindlessHeaps.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace ShaderRecompiler::Detail {

namespace {

// Diagnostic: APS5_TRACE_HASH_CANDIDATES=1 prints why a T#/S# handle is not GPU-resolved.
bool rejected(const IrResourcePlan& program, const char* reason, std::uint32_t detail = 0u) {
    static const bool trace = std::getenv("APS5_TRACE_HASH_CANDIDATES") != nullptr;
    if (trace) std::fprintf(stderr, "[hash-reject] shader %llx: %s (%u)\n", static_cast<unsigned long long>(program.shaderHash), reason, detail);
    return false;
}

// The handle's `count` dwords are reads of one scalar buffer at one offset (immediate, or one scalar
// value every read shares: a scalar load's offset is wave-uniform, as the heap path's slot), one
// dword apart.
bool ConstantOffsetReads(const IrResourcePlan& program, const IrValue& handle, std::uint32_t count, std::array<IrValue*, 8>* reads, std::uint32_t* offset) {
    if (handle.ArgumentCount() != count) return rejected(program, "dword count", handle.ArgumentCount());
    const IrValue* buffer = nullptr;
    const IrValue* sharedOffset = nullptr;
    std::uint32_t first = 0;
    for (std::uint32_t dword = 0; dword < count; ++dword) {
        IrValue* read = handle.Argument(dword)->Resolve();
        if (read->Opcode() == IrOpcode::LoadAddressU32) return rejected(program, "address load", dword);
        if (read->Opcode() != IrOpcode::ReadConstBuffer) return rejected(program, IrOpcodeName(read->Opcode()).data(), dword);
        if (read->ArgumentCount() != 2u || !IsRawRead(program, *read)) return rejected(program, "not a raw read", dword);
        const auto& memory = program.memoryInfo[read->Flags<MemoryFlags>().index];
        const IrValue* dynamic = read->Argument(1)->Resolve();
        if (memory.dataBits != 32u || memory.dataDwords != 1u) return rejected(program, "read width", memory.dataDwords);
        if (dword == 0u) {
            buffer = read->Argument(0)->Resolve();
            sharedOffset = dynamic;
            first = memory.offset;
        } else if (dynamic != sharedOffset && !(dynamic->HasImmediate() && sharedOffset->HasImmediate() && dynamic->ImmediateU32() == sharedOffset->ImmediateU32())) {
            return rejected(program, "different offsets", dword);
        } else if (read->Argument(0)->Resolve() != buffer) {
            return rejected(program, "two buffers", dword);
        } else if (memory.offset != first + dword * 4u) {
            return rejected(program, "not consecutive", dword);
        }
        if (reads != nullptr) (*reads)[dword] = read;
    }
    if (offset != nullptr) *offset = first;
    return true;
}

}

bool HashedSamplerCandidate(const IrResourcePlan& program, const IrValue& handle, std::array<IrValue*, 8>* reads, std::uint32_t* offset) {
    if (!BindlessHeaps::HashedImagesEnabled() || handle.Opcode() != IrOpcode::GetSamplerResource || !ConstantOffsetReads(program, handle, 4u, reads, offset)) return false;
    if (handle.Uses().empty()) return false;
    for (const IrValue* user : handle.Uses()) {
        const auto info = ImageOpcodeInfoOf(user->Opcode());
        if (!info.needsSampler || user->ArgumentCount() < 2u || user->Argument(1)->Resolve() != &handle) return false;
        const auto index = user->Flags<MemoryFlags>().index;
        if (index >= program.memoryInfo.size()) return false;
        const auto& memory = program.memoryInfo[index];
        // A depth comparison or an LOD adjustment makes the sampler depend on the image or the op.
        if ((memory.imageSampleFlags & (RdnaImageSampleFlagCompare | RdnaImageSampleFlagAdjust)) != 0u) return false;
        if (!BindlessHeaps::HeapEligible(program.stage, static_cast<std::uint32_t>(memory.imageDimension), true, false, memory.imageR128)) return false;
    }
    return true;
}

bool AddressSrtProgram(const IrResourcePlan& program) {
    static const std::vector<std::uint64_t> list = [] {
        std::vector<std::uint64_t> programs;
        if (const char* text = std::getenv("APS5_ADDRESS_SRT")) {
            for (const char* cursor = text; *cursor != '\0';) {
                char* end = nullptr;
                const auto value = std::strtoull(cursor, &end, 16);
                if (end == cursor) break;
                programs.push_back(value);
                cursor = *end == ',' ? end + 1 : end;
            }
        }
        return programs;
    }();
    return program.stage == IrShaderStage::Compute && !list.empty() && std::find(list.begin(), list.end(), program.shaderHash) != list.end();
}

bool HashedImageCandidate(const IrResourcePlan& program, const IrValue& handle, std::array<IrValue*, 8>* reads, std::uint32_t* offset) {
    std::uint32_t first = 0;
    if (!BindlessHeaps::HashedImagesEnabled() || handle.Opcode() != IrOpcode::GetImageResource || !ConstantOffsetReads(program, handle, 8u, reads, &first)) return false;
    if (handle.Uses().empty()) return false;
    for (const IrValue* user : handle.Uses()) {
        const auto info = ImageOpcodeInfoOf(user->Opcode());
        if (info.resourceClass != ImageResourceClass::Sampled || user->ArgumentCount() == 0u || user->Argument(0)->Resolve() != &handle) return rejected(program, "user not a sampled access", static_cast<std::uint32_t>(user->Opcode()));
        const auto index = user->Flags<MemoryFlags>().index;
        if (index >= program.memoryInfo.size()) return false;
        const auto& memory = program.memoryInfo[index];
        const bool compare = (memory.imageSampleFlags & RdnaImageSampleFlagCompare) != 0u;
        if (compare) return rejected(program, "depth compare");
        if (memory.imagePacked || memory.imageByElements != 0u) return rejected(program, "packed access");
        if (!BindlessHeaps::HeapEligible(program.stage, static_cast<std::uint32_t>(memory.imageDimension), true, compare, memory.imageR128)) return rejected(program, "heap ineligible dimension", static_cast<std::uint32_t>(memory.imageDimension));
    }
    if (offset != nullptr) *offset = first;
    return true;
}

bool IsRawRead(const IrResourcePlan& program, const IrValue& inst) {
    return IsRawRead(program, inst.Opcode(), inst.Flags<MemoryFlags>().index);
}

bool IsRawRead(const IrResourcePlan& program, IrOpcode op, std::uint32_t index) {
    if (op != IrOpcode::LoadAddressU32 && op != IrOpcode::ReadConstBuffer) {
        return false;
    }
    if (index >= program.memoryInfo.size()) {
        return false;
    }
    const auto kind = program.memoryInfo[index].kind;
    return (op == IrOpcode::LoadAddressU32 && kind == ResourceKind::ScalarAddress) || (op == IrOpcode::ReadConstBuffer && kind == ResourceKind::ScalarBuffer);
}

bool LoadedFromMemory(const IrValue& handle) {
    std::vector<const IrValue*> pending;
    std::vector<const IrValue*> visited;
    for (std::size_t index = 0; index < handle.ArgumentCount(); ++index) pending.push_back(handle.Argument(index));
    while (!pending.empty()) {
        const IrValue* value = pending.back();
        pending.pop_back();
        if (value == nullptr) continue;
        value = value->Resolve();
        if (value->HasImmediate() || std::find(visited.begin(), visited.end(), value) != visited.end()) continue;
        visited.push_back(value);
        const auto opcode = value->Opcode();
        if (opcode == IrOpcode::ReadConst || opcode == IrOpcode::LoadAddressU32 || opcode == IrOpcode::ReadConstBuffer) return true;
        for (std::size_t index = 0; index < value->ArgumentCount(); ++index) pending.push_back(value->Argument(index));
    }
    return false;
}

bool IsDescriptorHandle(IrOpcode opcode) {
    switch (opcode) {
        case IrOpcode::GetBufferResource:
        case IrOpcode::GetAddressResource:
        case IrOpcode::GetImageResource:
        case IrOpcode::GetSamplerResource: return true;
        default: return false;
    }
}

bool IsRuntimeSelect(IrOpcode opcode) {
    return opcode == IrOpcode::SelectU1 || opcode == IrOpcode::SelectU32 || opcode == IrOpcode::SelectF32;
}

bool IsRuntimeUniformOp(IrOpcode opcode) {
    switch (opcode) {
        case IrOpcode::BitCastU32F32:
        case IrOpcode::BitCastF32U32:
        case IrOpcode::ConvertU32F32:
        case IrOpcode::ConvertF32U32:
        case IrOpcode::CompositeConstructU64:
        case IrOpcode::CompositeExtractU64:
        case IrOpcode::CompositeConstructU32x2:
        case IrOpcode::CompositeExtractU32x2:
        case IrOpcode::BitFieldInsert:
        case IrOpcode::BitFieldUExtract:
        case IrOpcode::BitFieldSExtract:
        case IrOpcode::IAdd32:
        case IrOpcode::IAdd64:
        case IrOpcode::IAddCarry32:
        case IrOpcode::ISub32:
        case IrOpcode::ISub64:
        case IrOpcode::IMul32:
        case IrOpcode::IMul64:
        case IrOpcode::SMin32:
        case IrOpcode::UMin32:
        case IrOpcode::ShiftLeftLogical32:
        case IrOpcode::ShiftLeftLogical64:
        case IrOpcode::ShiftRightLogical32:
        case IrOpcode::ShiftRightLogical64:
        case IrOpcode::ShiftRightArithmetic32:
        case IrOpcode::ShiftRightArithmetic64:
        case IrOpcode::BitwiseAnd32:
        case IrOpcode::BitwiseAnd64:
        case IrOpcode::BitwiseOr32:
        case IrOpcode::BitwiseXor32:
        case IrOpcode::BitwiseNot32:
        case IrOpcode::SelectU1:
        case IrOpcode::SelectU32:
        case IrOpcode::SelectF32:
        case IrOpcode::SLessThan32:
        case IrOpcode::ULessThan32:
        case IrOpcode::IEqual32:
        case IrOpcode::UGreaterThan32:
        case IrOpcode::INotEqual32:
        case IrOpcode::LogicalOr:
        case IrOpcode::LogicalAnd:
        case IrOpcode::LogicalXor:
        case IrOpcode::LogicalNot:
        case IrOpcode::FPOrdLessThanEqual32:
        case IrOpcode::FPOrdGreaterThanEqual32:
        case IrOpcode::FPIsNan32:
        case IrOpcode::FPMul32:
        case IrOpcode::FPTrunc32: return true;
        default: return false;
    }
}

}
