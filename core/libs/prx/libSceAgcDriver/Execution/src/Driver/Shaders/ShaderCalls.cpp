#include "prx/libSceAgcDriver/Execution/include/Driver/Shaders/ShaderCalls.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "RdnaDecoder/RdnaInstructionDecoder.hpp"
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <list>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace AgcDriver::DriverDetail {

namespace {

using ShaderRecompiler::RdnaInstruction;
using ShaderRecompiler::RdnaInstructionFamily;
using ShaderRecompiler::RdnaOpcode;
using ShaderRecompiler::RdnaOperand;
using ShaderRecompiler::RdnaOperandKind;
using ShaderRecompiler::RdnaProgram;

constexpr std::uint32_t MaxFunctionWords = 4096;
constexpr std::size_t MaxExpandedWords = 131072;
constexpr std::uint64_t MaxCallTableBytes = 16u * 1024u * 1024u;
constexpr std::size_t MaxCallTargets = 256;
constexpr std::uint32_t SEndpgm = 0xbf810000u;
constexpr std::uint32_t SBranch = 0xbf820000u;
constexpr std::uint32_t SCbranchScc0 = 0xbf840000u;
constexpr std::uint32_t SCbranchScc1 = 0xbf850000u;

bool traceCalls() {
    static const bool trace = std::getenv("APS5_TRACE_SHADER_CALLS") != nullptr;
    return trace;
}

struct Unexpandable : std::runtime_error {
    using std::runtime_error::runtime_error;
};

[[noreturn]] void reject(const std::string& reason) {
    throw Unexpandable("shader call expansion: " + reason);
}

std::string hex(std::uint64_t value) {
    char text[24];
    std::snprintf(text, sizeof(text), "0x%llx", static_cast<unsigned long long>(value));
    return text;
}

// s_swappc_b64 with a register destination (a null destination is a jump, decoded as s_setpc_b64).
bool isCallWord(std::uint32_t word) {
    return (word & 0xff80ff00u) == 0xbe802100u && ((word >> 16u) & 127u) != 125u;
}

bool isBranch(const RdnaInstruction& instruction) {
    return ShaderRecompiler::IsDirectBranchOpcode(instruction.op);
}

bool isScalarLoad(RdnaOpcode op) {
    return op == RdnaOpcode::SLoadDword || op == RdnaOpcode::SLoadDwordx2 || op == RdnaOpcode::SLoadDwordx4 || op == RdnaOpcode::SLoadDwordx8 || op == RdnaOpcode::SLoadDwordx16;
}

bool isScalarBufferLoad(RdnaOpcode op) {
    return op == RdnaOpcode::SBufferLoadDword || op == RdnaOpcode::SBufferLoadDwordx2 || op == RdnaOpcode::SBufferLoadDwordx4 || op == RdnaOpcode::SBufferLoadDwordx8 || op == RdnaOpcode::SBufferLoadDwordx16;
}

std::optional<std::uint32_t> scalarIndex(const RdnaOperand& operand) {
    switch (operand.kind) {
        case RdnaOperandKind::ScalarRegister: return operand.reg;
        case RdnaOperandKind::VccLo: return 106u;
        case RdnaOperandKind::VccHi: return 107u;
        case RdnaOperandKind::M0: return 124u;
        case RdnaOperandKind::ExecLo: return 126u;
        case RdnaOperandKind::ExecHi: return 127u;
        default: return std::nullopt;
    }
}

// Registers a scalar destination covers. Vector instructions with a scalar destination write lane
// masks (a readlane writes one register, counted as a mask: an over-approximation is safe here, as
// writes only ever refuse an expansion).
std::uint32_t destinationWords(const RdnaInstruction& instruction, std::uint32_t waveSize) {
    switch (instruction.family) {
        case RdnaInstructionFamily::SOP1:
        case RdnaInstructionFamily::SOP2:
        case RdnaInstructionFamily::SOPK:
        case RdnaInstructionFamily::SOPC: return std::max(instruction.dataDwordCount, 1u);
        case RdnaInstructionFamily::SMEM: return std::max(instruction.dataDwordCount, 2u);
        default: return std::max({instruction.dataDwordCount, waveSize / 32u, 1u});
    }
}

bool writes(const RdnaInstruction& instruction, std::uint32_t reg, std::uint32_t waveSize) {
    switch (instruction.op) {
        // Indexed by M0: any register.
        case RdnaOpcode::SMovreldB32:
        case RdnaOpcode::SMovreldB64: return true;
        default: break;
    }
    if (const auto first = scalarIndex(instruction.destination); first && reg >= *first && reg < *first + destinationWords(instruction, waveSize)) return true;
    if (const auto first = scalarIndex(instruction.destination2); first && reg >= *first && reg < *first + std::max(waveSize / 32u, 1u)) return true;
    return false;
}

std::optional<std::uint32_t> immediate(const RdnaOperand& operand) {
    if (operand.kind == RdnaOperandKind::Null) return 0u;
    if (operand.kind == RdnaOperandKind::LiteralConstant || operand.kind == RdnaOperandKind::IntegerInlineConstant) return operand.value;
    return std::nullopt;
}

// Reads guest memory; false when the range is not readable.
bool readGuest(std::uint64_t address, std::span<std::uint32_t> words) {
    const auto bytes = words.size() * sizeof(std::uint32_t);
    if (!GuestMemory::Accessible(reinterpret_cast<const void*>(address), bytes)) return false;
    GuestMemory::Read(address, std::as_writable_bytes(words), alignof(std::uint32_t));
    return true;
}

using Reads = std::vector<std::pair<std::uint64_t, std::vector<std::uint32_t>>>;

struct Function {
    std::uint64_t address = 0;
    std::vector<std::uint32_t> code;
    std::vector<RdnaInstruction> instructions;
};

// The callee at `address`: straight code up to its last return through the return pair, with no
// call, s_endpgm or write of the return pair, and every branch inside it.
Function readFunction(std::uint64_t address, std::uint32_t returnRegister, std::uint32_t waveSize) {
    Function function;
    function.address = address;
    if ((address & 3u) != 0) reject("call target " + hex(address) + " is not DWORD aligned");
    std::uint32_t next = 0;
    std::uint32_t furthestTarget = 0;
    bool returned = false;
    while (next + ShaderRecompiler::MaxRdnaInstructionRawWords <= MaxFunctionWords) {
        while (function.code.size() < next + ShaderRecompiler::MaxRdnaInstructionRawWords) {
            std::array<std::uint32_t, 64> chunk{};
            if (!readGuest(address + function.code.size() * sizeof(std::uint32_t), chunk)) reject("the code of the function at " + hex(address) + " is not mapped");
            function.code.insert(function.code.end(), chunk.begin(), chunk.end());
        }
        RdnaInstruction instruction;
        try {
            instruction = ShaderRecompiler::DecodeRdnaInstruction(next * sizeof(std::uint32_t), function.code, next);
        } catch (const std::exception& error) {
            reject("the function at " + hex(address) + " does not decode: " + error.what());
        }
        if (instruction.op == RdnaOpcode::SSwappcB64 || instruction.op == RdnaOpcode::SEndpgm) reject("the function at " + hex(address) + " makes a nested call or ends the program");
        if (instruction.op != RdnaOpcode::SSetpcB64 && (writes(instruction, returnRegister, waveSize) || writes(instruction, returnRegister + 1u, waveSize))) reject("the function at " + hex(address) + " writes its return address");
        function.instructions.push_back(instruction);
        next += instruction.wordCount;
        if (isBranch(instruction)) furthestTarget = std::max(furthestTarget, instruction.branchTarget);
        if (instruction.op == RdnaOpcode::SSetpcB64) {
            if (instruction.source0.kind != RdnaOperandKind::ScalarRegister || instruction.source0.reg != returnRegister) reject("the function at " + hex(address) + " jumps through another register pair than its return address");
            returned = true;
            if (next * sizeof(std::uint32_t) > furthestTarget) break;
        }
    }
    if (!returned || function.instructions.back().op != RdnaOpcode::SSetpcB64) reject("the function at " + hex(address) + " does not return within " + std::to_string(MaxFunctionWords) + " words");
    function.code.resize(next);
    std::set<std::uint32_t> starts;
    for (const auto& instruction : function.instructions) starts.insert(instruction.programCounter);
    for (const auto& instruction : function.instructions) {
        if (isBranch(instruction) && !starts.contains(instruction.branchTarget)) reject("the function at " + hex(address) + " branches outside itself");
    }
    return function;
}

void emitMove(std::vector<std::uint32_t>& code, std::uint32_t reg, std::uint32_t value) {
    code.push_back(0xbe8003ffu | (reg << 16u));  // s_mov_b32 s[reg], literal
    code.push_back(value);
}

// s_getpc_b64 as moves of the original address after it (moves keep SCC, as s_getpc does).
void emitOriginalPc(std::vector<std::uint32_t>& code, const RdnaInstruction& instruction, std::uint64_t base) {
    const auto reg = (instruction.rawWords[0] >> 16u) & 127u;
    if (reg > 105u) reject("s_getpc_b64 writes a special register pair");
    const auto pc = base + instruction.programCounter + sizeof(std::uint32_t);
    emitMove(code, reg, static_cast<std::uint32_t>(pc));
    emitMove(code, reg + 1u, static_cast<std::uint32_t>(pc >> 32u));
}

using Branches = std::vector<std::pair<std::uint32_t, std::uint32_t>>;

// Writes every branch's offset; an edge beyond the 16-bit range goes through s_branch islands
// placed between instructions (as KytyPS5's RelocateBranches), which clobber no register or SCC.
void relocateBranches(std::vector<std::uint32_t>& code, Branches& branches) {
    for (std::size_t index = 0; index < branches.size();) {
        const auto [at, target] = branches[index];
        const auto delta = static_cast<std::int64_t>(target) - static_cast<std::int64_t>(at) - 1;
        if (delta >= std::numeric_limits<std::int16_t>::min() && delta <= std::numeric_limits<std::int16_t>::max()) {
            ++index;
            continue;
        }
        if (code.size() + 2 > MaxExpandedWords || branches.size() > MaxExpandedWords) reject("branch islands exceed the bounded code size");
        const auto desired = delta > 0 ? at + 16384u : at - 16384u;
        std::uint32_t insertion = 0;
        for (std::uint32_t word = 0; word <= desired && word < code.size();) {
            insertion = word;
            word += ShaderRecompiler::DecodeRdnaInstruction(word * sizeof(std::uint32_t), code, word).wordCount;
        }
        if (insertion <= std::min(at, target) || insertion >= std::max(at, target)) reject("cannot place a branch island between instructions");
        code.insert(code.begin() + insertion, {SBranch, SBranch});
        for (auto& [source, destination] : branches) {
            if (source >= insertion) source += 2;
            if (destination >= insertion) destination += 2;
        }
        const auto onward = branches[index].second;
        branches[index].second = insertion + 1u;
        branches.emplace_back(insertion, insertion + 2u);
        branches.emplace_back(insertion + 1u, onward);
        index = 0;
    }
    for (const auto [at, target] : branches) {
        const auto delta = static_cast<std::int64_t>(target) - static_cast<std::int64_t>(at) - 1;
        code[at] = (code[at] & 0xffff0000u) | (static_cast<std::uint32_t>(delta) & 0xffffu);
    }
}

// The decoded program: everything the expansion needs that depends only on its code.
struct Analysis {
    std::vector<std::uint32_t> code;
    RdnaProgram program;
    std::set<std::uint32_t> labels;
    bool calls = false;
};

std::shared_ptr<const Analysis> analyze(std::span<const std::uint32_t> code) {
    auto analysis = std::make_shared<Analysis>();
    analysis->code.assign(code.begin(), code.end());
    try {
        ShaderRecompiler::DecodeRdnaProgram(analysis->code, analysis->program);
    } catch (const std::exception& error) {
        reject(std::string("the program does not decode: ") + error.what());
    }
    for (const auto& instruction : analysis->program.instructions) {
        if (isBranch(instruction)) analysis->labels.insert(instruction.branchTarget);
        analysis->calls = analysis->calls || instruction.op == RdnaOpcode::SSwappcB64;
    }
    if (analysis->calls && std::any_of(analysis->program.instructions.begin(), analysis->program.instructions.end(), [](const auto& instruction) { return instruction.op == RdnaOpcode::SSetpcB64; })) reject("calls in a program with indirect jumps are unsupported");
    return analysis;
}

// One call and every address it can reach: one when the target is known statically, the entries
// of a call table read through a scalar buffer load otherwise (dynamic: chosen at run time).
struct Site {
    std::uint32_t programCounter = 0;
    std::uint32_t sourceRegister = 0;
    std::uint32_t returnRegister = 0;
    bool dynamic = false;
    std::vector<std::uint64_t> targets;  // ascending

    bool operator==(const Site&) const = default;
};

struct Resolution {
    std::vector<Site> sites;
    std::set<std::uint32_t> dependencies;  // the user-data registers consulted
};

// Resolves each call's targets from the user data and the memory holding call tables (as
// KytyPS5's ResolveCalls). Which registers it consults depends on the code alone.
Resolution resolveSites(const Analysis& analysis, std::uint64_t base, std::span<const std::uint32_t> userData, std::uint32_t waveSize) {
    Resolution result;
    const auto& instructions = analysis.program.instructions;
    const auto& labels = analysis.labels;
    // The value of `reg` before instruction `before`, followed back through moves within the block
    // starting at `first`, to user data no earlier instruction writes, a constant or s_getpc; with
    // `loadMemory`, also through scalar loads whose address resolves.
    const auto resolve = [&](auto&& self, std::uint32_t reg, std::size_t before, std::size_t first, std::uint32_t depth, bool loadMemory) -> std::optional<std::uint32_t> {
        if (reg >= 128u || depth > 32u) return std::nullopt;
        const bool initial = reg < userData.size() && std::none_of(instructions.begin(), instructions.begin() + static_cast<std::ptrdiff_t>(before), [&](const auto& instruction) { return writes(instruction, reg, waveSize); });
        if (initial) {
            result.dependencies.insert(reg);
            return userData[reg];
        }
        for (std::size_t index = before; index > first;) {
            const auto& instruction = instructions[--index];
            if (isBranch(instruction) || instruction.op == RdnaOpcode::SSwappcB64) return std::nullopt;
            if (writes(instruction, reg, waveSize)) {
                const auto destination = scalarIndex(instruction.destination);
                if (!destination) return std::nullopt;
                if (instruction.op == RdnaOpcode::SGetpcB64) {
                    const auto pc = base + instruction.programCounter + sizeof(std::uint32_t);
                    return static_cast<std::uint32_t>(pc >> ((reg - *destination) * 32u));
                }
                if (loadMemory && isScalarLoad(instruction.op)) {
                    const auto addressRegister = scalarIndex(instruction.source0);
                    if (!addressRegister) return std::nullopt;
                    const auto low = self(self, *addressRegister, index, first, depth + 1u, true);
                    const auto high = self(self, *addressRegister + 1u, index, first, depth + 1u, true);
                    const auto offsetRegister = scalarIndex(instruction.source1);
                    const auto offset = offsetRegister ? self(self, *offsetRegister, index, first, depth + 1u, true) : immediate(instruction.source1);
                    if (!low || !high || !offset) return std::nullopt;
                    const auto address = ((static_cast<std::uint64_t>(*high & 0xffffu) << 32u) | *low) + *offset + instruction.memoryOffset + (reg - *destination) * 4u;
                    std::array<std::uint32_t, 1> value{};
                    if (!readGuest(address, value)) return std::nullopt;
                    return value[0];
                }
                if ((instruction.op != RdnaOpcode::SMovB32 && instruction.op != RdnaOpcode::SMovB64) || instruction.destination.kind != RdnaOperandKind::ScalarRegister) return std::nullopt;
                if (const auto source = scalarIndex(instruction.source0); source && instruction.source0.kind == RdnaOperandKind::ScalarRegister) return self(self, *source + reg - *destination, index, first, depth + 1u, loadMemory);
                if (instruction.op == RdnaOpcode::SMovB32) return immediate(instruction.source0);
                return std::nullopt;
            }
            if (labels.contains(instruction.programCounter)) return std::nullopt;
        }
        return std::nullopt;
    };
    // Straight-line copies back to one scalar buffer load: (its index, the dword of its result).
    const auto origin = [&](auto&& self, std::uint32_t reg, std::size_t before, std::size_t first, std::uint32_t depth) -> std::optional<std::pair<std::size_t, std::uint32_t>> {
        if (reg >= 105u || depth > 32u) return std::nullopt;
        for (std::size_t index = before; index > first;) {
            const auto& instruction = instructions[--index];
            if (isBranch(instruction) || instruction.op == RdnaOpcode::SSwappcB64) return std::nullopt;
            if (writes(instruction, reg, waveSize)) {
                const auto destination = scalarIndex(instruction.destination);
                if (!destination) return std::nullopt;
                if (isScalarBufferLoad(instruction.op)) return std::pair{index, reg - *destination};
                if ((instruction.op == RdnaOpcode::SMovB32 || instruction.op == RdnaOpcode::SMovB64) && instruction.source0.kind == RdnaOperandKind::ScalarRegister) return self(self, instruction.source0.reg + reg - *destination, index, first, depth + 1u);
                return std::nullopt;
            }
            if (labels.contains(instruction.programCounter)) return std::nullopt;
        }
        return std::nullopt;
    };
    for (std::size_t index = 0; index < instructions.size(); ++index) {
        const auto& call = instructions[index];
        if (call.op != RdnaOpcode::SSwappcB64) continue;
        if (call.source0.kind != RdnaOperandKind::ScalarRegister || call.destination.kind != RdnaOperandKind::ScalarRegister || call.source0.reg >= 105u || call.destination.reg >= 105u) reject("the call at " + hex(call.programCounter) + " uses a special register pair");
        const auto source = call.source0.reg;
        std::size_t first = index;
        while (first > 0 && !labels.contains(instructions[first].programCounter) && !isBranch(instructions[first - 1])) --first;
        const auto low = resolve(resolve, source, index, first, 0, false);
        const auto high = resolve(resolve, source + 1u, index, first, 0, false);
        Site site{call.programCounter, source, call.destination.reg, !low || !high, {}};
        std::set<std::uint64_t> targets;
        if (!site.dynamic) {
            targets.insert((static_cast<std::uint64_t>(*high) << 32u) | *low);
        } else {
            // A call table: the pair comes from one scalar buffer load whose descriptor resolves.
            const auto lo = origin(origin, source, index, first, 0);
            const auto hi = origin(origin, source + 1u, index, first, 0);
            if (!lo || !hi || lo->first != hi->first || hi->second != lo->second + 1u) {
                if (traceCalls()) {
                    for (std::size_t at = index >= 16 ? index - 16 : 0; at <= index; ++at) std::fprintf(stderr, "[shader-calls]   %s\n", ShaderRecompiler::RdnaInstructionToString(instructions[at]).c_str());
                }
                reject("the target of the call at " + hex(call.programCounter) + " is neither known nor a scalar buffer table load");
            }
            const auto& load = instructions[lo->first];
            const auto descriptorRegister = scalarIndex(load.source0);
            if (!descriptorRegister) reject("the call table descriptor of the call at " + hex(call.programCounter) + " is not a scalar register tuple");
            std::array<std::uint32_t, 4> descriptor{};
            for (std::uint32_t word = 0; word < 4; ++word) {
                const auto value = resolve(resolve, *descriptorRegister + word, lo->first, first, 0, true);
                if (!value) reject("the call table descriptor of the call at " + hex(call.programCounter) + " is not statically readable");
                descriptor[word] = *value;
            }
            // The table stride from the last definition of the dynamic byte offset (a left shift).
            std::uint32_t shift = 0;
            if (const auto offsetRegister = scalarIndex(load.source1)) {
                for (std::size_t at = lo->first; at > first;) {
                    const auto& definition = instructions[--at];
                    if (isBranch(definition) || definition.op == RdnaOpcode::SSwappcB64) break;
                    if (writes(definition, *offsetRegister, waveSize)) {
                        if (definition.op == RdnaOpcode::SLshlB32 && scalarIndex(definition.destination) == offsetRegister) {
                            if (const auto amount = immediate(definition.source1)) shift = *amount & 31u;
                        }
                        break;
                    }
                    if (labels.contains(definition.programCounter)) break;
                }
            }
            const std::uint64_t tableBase = descriptor[0] | (static_cast<std::uint64_t>(descriptor[1] & 0xffffu) << 32u);
            const std::uint32_t stride = (descriptor[1] >> 16u) & 0x3fffu;
            // A scalar load bounds its offset by num_records bytes, times the stride when there is one.
            const std::uint64_t size = stride == 0u ? descriptor[2] : static_cast<std::uint64_t>(stride) * descriptor[2];
            const bool swizzled = (descriptor[1] >> 30u) != 0u;
            const bool addTid = ((descriptor[3] >> 23u) & 1u) != 0u;
            const bool buffer = (descriptor[3] >> 30u) == 0u;
            if (shift < 3u || shift > 16u || size == 0u || size > MaxCallTableBytes || !buffer || swizzled || addTid || tableBase == 0u || (tableBase & 3u) != 0u || (load.memoryOffset & 3u) != 0u) reject("the call table of the call at " + hex(call.programCounter) + " needs unsupported bounds or addressing (shift " + std::to_string(shift) + ", stride " + std::to_string(stride) + ", size " + hex(size) + ", base " + hex(tableBase) + ", load offset " + hex(load.memoryOffset) + ")");
            std::vector<std::uint32_t> table(static_cast<std::size_t>(size / 4u));
            if (!readGuest(tableBase, table)) reject("the call table of the call at " + hex(call.programCounter) + " is not mapped");
            const std::uint64_t step = std::uint64_t{1} << shift;
            const std::uint64_t firstSlot = (load.memoryOffset % step) + lo->second * 4u;
            for (std::uint64_t offset = firstSlot; offset + 8u <= size; offset += step) {
                const std::uint64_t address = table[offset / 4u] | (static_cast<std::uint64_t>(table[offset / 4u + 1u]) << 32u);
                if (address == 0) continue;  // a null entry takes the miss path
                targets.insert(address);
                if (targets.size() > MaxCallTargets) reject("the call table of the call at " + hex(call.programCounter) + " has more than " + std::to_string(MaxCallTargets) + " targets");
            }
        }
        site.targets.assign(targets.begin(), targets.end());
        result.sites.push_back(std::move(site));
    }
    return result;
}

struct Expansion {
    std::vector<std::uint32_t> code;
    Reads callees;  // the callee code as read, for the cache's check that it is unchanged
};

// Reads the callees and emits the expanded program (as KytyPS5's BuildExpansion).
Expansion build(const Analysis& analysis, std::uint64_t base, const Resolution& resolution, std::uint32_t waveSize) {
    const auto& instructions = analysis.program.instructions;
    std::map<std::uint32_t, const Site*> sites;
    for (const auto& site : resolution.sites) sites[site.programCounter] = &site;
    // A loop or a callee could change a register a later call's target was resolved from.
    const bool backedge = std::any_of(instructions.begin(), instructions.end(), [](const auto& instruction) { return isBranch(instruction) && instruction.branchTarget <= instruction.programCounter; });
    if (backedge) {
        for (const auto reg : resolution.dependencies) {
            if (std::any_of(instructions.begin(), instructions.end(), [&](const auto& instruction) { return writes(instruction, reg, waveSize); })) reject("a loop may change the user data register s" + std::to_string(reg) + " a call target comes from");
        }
    }
    std::map<std::pair<std::uint64_t, std::uint32_t>, Function> functions;
    for (const auto& site : resolution.sites) {
        for (const auto target : site.targets) {
            const auto key = std::pair{target, site.returnRegister};
            if (functions.contains(key)) continue;
            auto function = readFunction(target, site.returnRegister, waveSize);
            for (const auto& instruction : function.instructions) {
                for (const auto reg : resolution.dependencies) {
                    if (writes(instruction, reg, waveSize)) reject("the function at " + hex(target) + " changes the user data register s" + std::to_string(reg) + " a call target comes from");
                }
            }
            functions.emplace(key, std::move(function));
        }
    }
    std::vector<std::uint32_t> out;
    std::map<std::uint32_t, std::uint32_t> mainPositions;
    Branches mainBranches;
    Branches relocations;
    for (const auto& instruction : instructions) {
        mainPositions[instruction.programCounter] = static_cast<std::uint32_t>(out.size());
        const auto found = sites.find(instruction.programCounter);
        if (found == sites.end()) {
            if (instruction.op == RdnaOpcode::SEndpgm) {
                out.push_back(SEndpgm);
            } else if (instruction.op == RdnaOpcode::SGetpcB64) {
                emitOriginalPc(out, instruction, base);
            } else {
                if (isBranch(instruction)) mainBranches.emplace_back(static_cast<std::uint32_t>(out.size()), instruction.branchTarget);
                out.insert(out.end(), instruction.rawWords.begin(), instruction.rawWords.begin() + instruction.wordCount);
            }
        } else {
            const auto& site = *found->second;
            const auto returnAddress = base + instruction.programCounter + instruction.wordCount * sizeof(std::uint32_t);
            // The branch words that leave the callee being emitted (its returns).
            std::vector<std::uint32_t> returns;
            const auto emitTarget = [&](std::uint64_t target) {
                const auto& function = functions.at({target, site.returnRegister});
                emitMove(out, site.returnRegister, static_cast<std::uint32_t>(returnAddress));
                emitMove(out, site.returnRegister + 1u, static_cast<std::uint32_t>(returnAddress >> 32u));
                std::map<std::uint32_t, std::uint32_t> localPositions;
                Branches localBranches;
                for (const auto& body : function.instructions) {
                    const auto at = static_cast<std::uint32_t>(out.size());
                    localPositions[body.programCounter] = at;
                    if (body.op == RdnaOpcode::SSetpcB64) {
                        returns.push_back(at);
                        out.push_back(SBranch);
                    } else if (body.op == RdnaOpcode::SGetpcB64) {
                        emitOriginalPc(out, body, function.address);
                    } else {
                        if (isBranch(body)) localBranches.emplace_back(at, body.branchTarget);
                        out.insert(out.end(), body.rawWords.begin(), body.rawWords.begin() + body.wordCount);
                    }
                }
                for (const auto& [at, target] : localBranches) relocations.emplace_back(at, localPositions.at(target));
            };
            if (site.dynamic) {
                // The target is compared with each table entry, high word first; s_swappc keeps
                // SCC, so there are two decision trees, one per incoming SCC value, each restoring
                // it before its callee (no guest SGPR is borrowed). An address the table did not
                // hold ends the program.
                // Every comparison is a selection nested in the previous one's false side. Each
                // gets its own merge, a branch to the merge of the selection it is nested in, so
                // the structurizer sees nested constructs instead of one join with a predecessor
                // per callee return (which it split by cloning the code after the call).
                struct Level {
                    std::size_t parent;  // the enclosing level; Join for the outermost
                    std::vector<std::uint32_t> exits;  // branch words that go to this level's merge
                };
                constexpr std::size_t Join = std::numeric_limits<std::size_t>::max();
                std::vector<Level> levels;
                const auto addLevel = [&](std::size_t parent) {
                    levels.push_back({parent, {}});
                    return levels.size() - 1u;
                };
                const auto sccLevel = addLevel(Join);
                const auto selectScc = static_cast<std::uint32_t>(out.size());
                out.push_back(SCbranchScc1);
                for (std::uint32_t scc = 0; scc < 2; ++scc) {
                    if (scc != 0) relocations.emplace_back(selectScc, static_cast<std::uint32_t>(out.size()));
                    std::map<std::uint32_t, std::vector<std::uint64_t>> groups;
                    for (const auto target : site.targets) groups[static_cast<std::uint32_t>(target >> 32u)].push_back(target);
                    std::size_t groupParent = sccLevel;
                    for (const auto& [high, targets] : groups) {
                        const auto groupLevel = addLevel(groupParent);
                        out.push_back(0xbf06ff00u | (site.sourceRegister + 1u));  // s_cmp_eq_u32 s[src+1], literal
                        out.push_back(high);
                        const auto highMiss = static_cast<std::uint32_t>(out.size());
                        out.push_back(SCbranchScc0);
                        std::size_t targetParent = groupLevel;
                        for (const auto target : targets) {
                            const auto targetLevel = addLevel(targetParent);
                            out.push_back(0xbf06ff00u | site.sourceRegister);  // s_cmp_eq_u32 s[src], literal
                            out.push_back(static_cast<std::uint32_t>(target));
                            const auto lowMiss = static_cast<std::uint32_t>(out.size());
                            out.push_back(SCbranchScc0);
                            out.push_back(scc != 0 ? 0xbf068080u : 0xbf068180u);  // SCC = 1 (0 == 0) or 0 (0 == 1)
                            returns.clear();
                            emitTarget(target);
                            // A callee that ends without a return falls through to its merge.
                            returns.push_back(static_cast<std::uint32_t>(out.size()));
                            out.push_back(SBranch);
                            levels[targetLevel].exits = returns;
                            relocations.emplace_back(lowMiss, static_cast<std::uint32_t>(out.size()));
                            targetParent = targetLevel;
                        }
                        out.push_back(SEndpgm);
                        relocations.emplace_back(highMiss, static_cast<std::uint32_t>(out.size()));
                        groupParent = groupLevel;
                    }
                    out.push_back(SEndpgm);
                }
                returns.clear();
                // The merges, innermost first: a level's parent comes earlier, so its merge is
                // emitted later and every branch is forward. The outermost merge is the join.
                std::vector<std::uint32_t> merges(levels.size(), 0);
                for (std::size_t level = levels.size(); level-- > 0;) {
                    merges[level] = static_cast<std::uint32_t>(out.size());
                    for (const auto at : levels[level].exits) relocations.emplace_back(at, merges[level]);
                    const auto at = static_cast<std::uint32_t>(out.size());
                    out.push_back(SBranch);
                    if (levels[level].parent == Join) returns.push_back(at);
                    else levels[levels[level].parent].exits.push_back(at);
                }
            } else {
                emitTarget(site.targets.front());
            }
            for (const auto at : returns) relocations.emplace_back(at, static_cast<std::uint32_t>(out.size()));
        }
        if (out.size() > MaxExpandedWords) reject("the expanded program exceeds " + std::to_string(MaxExpandedWords) + " words");
    }
    for (const auto& [at, target] : mainBranches) {
        const auto found = mainPositions.find(target);
        if (found == mainPositions.end()) reject("a branch targets " + hex(target) + ", outside the program");
        relocations.emplace_back(at, found->second);
    }
    relocateBranches(out, relocations);
    // The expanded program must decode as one, as the recompiler will.
    try {
        RdnaProgram check;
        ShaderRecompiler::DecodeRdnaProgram(out, check);
        const auto& last = check.instructions.back();
        const auto end = last.programCounter / sizeof(std::uint32_t) + last.wordCount;
        if (end != out.size()) reject("the expanded program decodes to " + std::to_string(end) + " of " + std::to_string(out.size()) + " words");
    } catch (const Unexpandable&) {
        throw;
    } catch (const std::exception& error) {
        reject(std::string("the expanded program does not decode: ") + error.what());
    }
    Expansion result;
    result.code = std::move(out);
    for (auto& [key, function] : functions) result.callees.emplace_back(key.first, std::move(function.code));
    return result;
}

struct CachedExpansion {
    std::vector<Site> sites;
    Reads callees;
    std::shared_ptr<const ShaderSnapshot> expanded;
};

struct CacheEntry {
    std::weak_ptr<const ShaderSnapshot> source;
    const ShaderSnapshot* sourceKey = nullptr;
    std::size_t codeOffset = 0;
    std::uint32_t waveSize = 0;
    std::shared_ptr<const Analysis> analysis;
    std::list<CachedExpansion> expansions;  // most recent first
    // A refused expansion: the reason, for the same user data (all of it, as the refusal may
    // come before the registers a target depends on are known).
    std::shared_ptr<const std::string> failure;
    std::vector<std::uint32_t> failedUserData;
};

bool stillReads(const Reads& reads) {
    return std::all_of(reads.begin(), reads.end(), [](const auto& read) { return GuestMemory::CompareMapped(read.first, std::as_bytes(std::span(read.second))) == GuestMemory::Compare::Equal; });
}

}

std::shared_ptr<const ShaderSnapshot> ExpandShaderCalls(const std::shared_ptr<const ShaderSnapshot>& snapshot, std::size_t codeOffset, std::span<const std::uint32_t> userData, std::uint32_t waveSize) {
    const auto code = std::span(snapshot->code).subspan(codeOffset);
    if (std::none_of(code.begin(), code.end(), isCallWord)) return nullptr;
    // Kill switch: APS5_NO_SHADER_CALLS=1 refuses every program with calls, as before.
    static const bool disabled = std::getenv("APS5_NO_SHADER_CALLS") != nullptr;
    if (disabled) throw std::runtime_error("shader call expansion is disabled (APS5_NO_SHADER_CALLS)");
    static std::mutex mutex;
    static std::list<CacheEntry> cache;
    std::unique_lock lock(mutex);
    auto entry = cache.begin();
    for (; entry != cache.end(); ++entry) {
        if (entry->sourceKey == snapshot.get() && entry->codeOffset == codeOffset && entry->waveSize == waveSize && entry->source.lock() == snapshot) break;
    }
    if (entry != cache.end()) {
        cache.splice(cache.begin(), cache, entry);
    } else {
        cache.push_front(CacheEntry{snapshot, snapshot.get(), codeOffset, waveSize, nullptr, {}, nullptr, {}});
        if (cache.size() > 64) cache.pop_back();
    }
    auto& current = cache.front();
    if (current.failure != nullptr && std::equal(current.failedUserData.begin(), current.failedUserData.end(), userData.begin(), userData.end())) throw std::runtime_error(*current.failure);
    const auto base = snapshot->codeAddress + codeOffset * sizeof(std::uint32_t);
    try {
        if (current.analysis == nullptr) current.analysis = analyze(code);
        if (!current.analysis->calls) return nullptr;
        const auto resolution = resolveSites(*current.analysis, base, userData, waveSize);
        for (auto it = current.expansions.begin(); it != current.expansions.end(); ++it) {
            if (it->sites != resolution.sites || !stillReads(it->callees)) continue;
            current.expansions.splice(current.expansions.begin(), current.expansions, it);
            return current.expansions.front().expanded;
        }
        auto expansion = build(*current.analysis, base, resolution, waveSize);
        // Without a header the expansion is prepared at its first dispatch, as an unregistered
        // compute program is (registration prepared the original code); the request keeps the
        // original header.
        ShaderSnapshot expanded{base, snapshot->headerAddress, 0, std::move(expansion.code), {}};
        if (traceCalls()) {
            std::fprintf(stderr, "[shader-calls] 0x%llx: %zu words expanded to %zu with %zu callee(s):", static_cast<unsigned long long>(base), code.size(), expanded.code.size(), expansion.callees.size());
            for (const auto& [address, words] : expansion.callees) std::fprintf(stderr, " 0x%llx (%zu words)", static_cast<unsigned long long>(address), words.size());
            std::fputc('\n', stderr);
        }
        current.expansions.push_front(CachedExpansion{resolution.sites, std::move(expansion.callees), std::make_shared<const ShaderSnapshot>(std::move(expanded))});
        if (current.expansions.size() > 8) current.expansions.pop_back();
        current.failure = nullptr;
        return current.expansions.front().expanded;
    } catch (const Unexpandable& error) {
        if (traceCalls()) std::fprintf(stderr, "[shader-calls] 0x%llx: %s\n", static_cast<unsigned long long>(base), error.what());
        current.failure = std::make_shared<const std::string>(error.what());
        current.failedUserData.assign(userData.begin(), userData.end());
        throw std::runtime_error(error.what());
    }
}

}
