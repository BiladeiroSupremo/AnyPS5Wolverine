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
constexpr std::uint32_t SNop = 0xbf800000u;
constexpr std::uint32_t SEndpgm = 0xbf810000u;
constexpr std::uint32_t SBranch = 0xbf820000u;

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
    if (operand.kind == RdnaOperandKind::LiteralConstant || operand.kind == RdnaOperandKind::IntegerInlineConstant) return operand.value;
    return std::nullopt;
}

// Reads guest code; false when the range is not readable.
bool readCode(std::uint64_t address, std::span<std::uint32_t> words) {
    const auto bytes = words.size() * sizeof(std::uint32_t);
    if (!GuestMemory::Accessible(reinterpret_cast<const void*>(address), bytes)) return false;
    GuestMemory::Read(address, std::as_writable_bytes(words), alignof(std::uint32_t));
    return true;
}

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
            if (!readCode(address + function.code.size() * sizeof(std::uint32_t), chunk)) reject("the code of the function at " + hex(address) + " is not mapped");
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

struct CallSite {
    std::uint32_t programCounter = 0;
    std::uint32_t returnRegister = 0;
    std::uint64_t target = 0;
};

struct Expansion {
    std::vector<std::uint32_t> code;
    // The user-data registers the call targets came from.
    std::vector<std::uint32_t> consulted;
    // The callee code as read, for the cache's check that it is unchanged.
    std::vector<std::pair<std::uint64_t, std::vector<std::uint32_t>>> callees;
};

// Null code when the program has no calls.
Expansion expand(std::span<const std::uint32_t> code, std::uint64_t base, std::span<const std::uint32_t> userData, std::uint32_t waveSize) {
    Expansion result;
    if (std::none_of(code.begin(), code.end(), isCallWord)) return result;
    RdnaProgram program;
    try {
        ShaderRecompiler::DecodeRdnaProgram(code, program);
    } catch (const std::exception& error) {
        reject(std::string("the program does not decode: ") + error.what());
    }
    const auto& instructions = program.instructions;
    if (std::none_of(instructions.begin(), instructions.end(), [](const auto& instruction) { return instruction.op == RdnaOpcode::SSwappcB64; })) return result;
    if (std::any_of(instructions.begin(), instructions.end(), [](const auto& instruction) { return instruction.op == RdnaOpcode::SSetpcB64; })) reject("calls in a program with indirect jumps are unsupported");
    std::set<std::uint32_t> labels;
    for (const auto& instruction : instructions) {
        if (isBranch(instruction)) labels.insert(instruction.branchTarget);
    }
    std::set<std::uint32_t> dependencies;
    // The value of `reg` before instruction `before`, followed back through moves within the block
    // starting at `first`, to user data no earlier instruction writes, a constant or s_getpc.
    const auto resolve = [&](auto&& self, std::uint32_t reg, std::size_t before, std::size_t first, std::uint32_t depth) -> std::optional<std::uint32_t> {
        if (reg >= 128u || depth > 32u) return std::nullopt;
        const bool initial = reg < userData.size() && std::none_of(instructions.begin(), instructions.begin() + static_cast<std::ptrdiff_t>(before), [&](const auto& instruction) { return writes(instruction, reg, waveSize); });
        if (initial) {
            dependencies.insert(reg);
            return userData[reg];
        }
        for (std::size_t index = before; index > first;) {
            const auto& instruction = instructions[--index];
            if (isBranch(instruction) || instruction.op == RdnaOpcode::SSwappcB64) return std::nullopt;
            if (writes(instruction, reg, waveSize)) {
                const auto destination = scalarIndex(instruction.destination);
                if (!destination || instruction.destination.kind != RdnaOperandKind::ScalarRegister) return std::nullopt;
                if (instruction.op == RdnaOpcode::SGetpcB64) {
                    const auto pc = base + instruction.programCounter + sizeof(std::uint32_t);
                    return static_cast<std::uint32_t>(pc >> ((reg - *destination) * 32u));
                }
                if (instruction.op != RdnaOpcode::SMovB32 && instruction.op != RdnaOpcode::SMovB64) return std::nullopt;
                if (instruction.source0.kind == RdnaOperandKind::ScalarRegister) return self(self, instruction.source0.reg + reg - *destination, index, first, depth + 1u);
                if (instruction.op == RdnaOpcode::SMovB32) return immediate(instruction.source0);
                return std::nullopt;
            }
            if (labels.contains(instruction.programCounter)) return std::nullopt;
        }
        return std::nullopt;
    };
    std::map<std::uint32_t, CallSite> sites;
    for (std::size_t index = 0; index < instructions.size(); ++index) {
        const auto& call = instructions[index];
        if (call.op != RdnaOpcode::SSwappcB64) continue;
        if (call.source0.kind != RdnaOperandKind::ScalarRegister || call.destination.kind != RdnaOperandKind::ScalarRegister || call.source0.reg >= 105u || call.destination.reg >= 105u) reject("the call at " + hex(call.programCounter) + " uses a special register pair");
        std::size_t first = index;
        while (first > 0 && !labels.contains(instructions[first].programCounter) && !isBranch(instructions[first - 1])) --first;
        const auto low = resolve(resolve, call.source0.reg, index, first, 0);
        const auto high = resolve(resolve, call.source0.reg + 1u, index, first, 0);
        if (!low || !high) reject("the target of the call at " + hex(call.programCounter) + " is not known from the program or its user data");
        sites[call.programCounter] = {call.programCounter, call.destination.reg, (static_cast<std::uint64_t>(*high) << 32u) | *low};
    }
    // A loop or a callee could change a register a later call's target was resolved from.
    const bool backedge = std::any_of(instructions.begin(), instructions.end(), [](const auto& instruction) { return isBranch(instruction) && instruction.branchTarget <= instruction.programCounter; });
    if (backedge) {
        for (const auto reg : dependencies) {
            if (std::any_of(instructions.begin(), instructions.end(), [&](const auto& instruction) { return writes(instruction, reg, waveSize); })) reject("a loop may change the user data register s" + std::to_string(reg) + " a call target comes from");
        }
    }
    std::map<std::uint64_t, Function> functions;
    for (const auto& [pc, site] : sites) {
        if (functions.contains(site.target)) continue;
        auto function = readFunction(site.target, site.returnRegister, waveSize);
        for (const auto& instruction : function.instructions) {
            for (const auto reg : dependencies) {
                if (writes(instruction, reg, waveSize)) reject("the function at " + hex(site.target) + " changes the user data register s" + std::to_string(reg) + " a call target comes from");
            }
        }
        functions.emplace(site.target, std::move(function));
    }
    std::vector<std::uint32_t> out;
    std::map<std::uint32_t, std::uint32_t> mainPositions;
    Branches mainBranches;
    Branches relocations;
    for (const auto& instruction : instructions) {
        mainPositions[instruction.programCounter] = static_cast<std::uint32_t>(out.size());
        const auto site = sites.find(instruction.programCounter);
        if (site == sites.end()) {
            if (instruction.op == RdnaOpcode::SEndpgm) {
                out.push_back(SEndpgm);
            } else if (instruction.op == RdnaOpcode::SGetpcB64) {
                emitOriginalPc(out, instruction, base);
            } else {
                if (isBranch(instruction)) mainBranches.emplace_back(static_cast<std::uint32_t>(out.size()), instruction.branchTarget);
                out.insert(out.end(), instruction.rawWords.begin(), instruction.rawWords.begin() + instruction.wordCount);
            }
        } else {
            const auto& function = functions.at(site->second.target);
            const auto returnAddress = base + instruction.programCounter + instruction.wordCount * sizeof(std::uint32_t);
            emitMove(out, site->second.returnRegister, static_cast<std::uint32_t>(returnAddress));
            emitMove(out, site->second.returnRegister + 1u, static_cast<std::uint32_t>(returnAddress >> 32u));
            std::map<std::uint32_t, std::uint32_t> localPositions;
            Branches localBranches;
            std::vector<std::uint32_t> returns;
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
    result.code = std::move(out);
    result.consulted.assign(dependencies.begin(), dependencies.end());
    for (auto& [address, function] : functions) result.callees.emplace_back(address, std::move(function.code));
    return result;
}

struct CacheEntry {
    std::weak_ptr<const ShaderSnapshot> source;
    const ShaderSnapshot* sourceKey = nullptr;
    std::size_t codeOffset = 0;
    std::uint32_t waveSize = 0;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> consulted;
    std::vector<std::pair<std::uint64_t, std::vector<std::uint32_t>>> callees;
    std::shared_ptr<const ShaderSnapshot> expanded;
    // A refused expansion: the reason, for the same user data (all of it, as the refusal may
    // come before the registers a target depends on are known).
    std::shared_ptr<const std::string> failure;
    std::vector<std::uint32_t> failedUserData;
};

bool traceCalls() {
    static const bool trace = std::getenv("APS5_TRACE_SHADER_CALLS") != nullptr;
    return trace;
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
    {
        std::lock_guard lock(mutex);
        for (auto it = cache.begin(); it != cache.end(); ++it) {
            if (it->sourceKey != snapshot.get() || it->codeOffset != codeOffset || it->waveSize != waveSize || it->source.lock() != snapshot) continue;
            if (it->failure != nullptr) {
                if (!std::equal(it->failedUserData.begin(), it->failedUserData.end(), userData.begin(), userData.end())) continue;
                throw std::runtime_error(*it->failure);
            }
            const bool same = std::all_of(it->consulted.begin(), it->consulted.end(), [&](const auto& entry) { return entry.first < userData.size() && userData[entry.first] == entry.second; });
            if (!same) continue;
            const bool unchanged = std::all_of(it->callees.begin(), it->callees.end(), [](const auto& callee) { return GuestMemory::CompareMapped(callee.first, std::as_bytes(std::span(callee.second))) == GuestMemory::Compare::Equal; });
            if (!unchanged) continue;
            cache.splice(cache.begin(), cache, it);
            return cache.front().expanded;
        }
    }
    const auto base = snapshot->codeAddress + codeOffset * sizeof(std::uint32_t);
    Expansion expansion;
    try {
        expansion = expand(code, base, userData, waveSize);
    } catch (const Unexpandable& error) {
        if (traceCalls()) std::fprintf(stderr, "[shader-calls] 0x%llx: %s\n", static_cast<unsigned long long>(base), error.what());
        CacheEntry failed{snapshot, snapshot.get(), codeOffset, waveSize, {}, {}, nullptr, std::make_shared<const std::string>(error.what()), std::vector<std::uint32_t>(userData.begin(), userData.end())};
        std::lock_guard lock(mutex);
        cache.push_front(std::move(failed));
        if (cache.size() > 64) cache.pop_back();
        throw std::runtime_error(error.what());
    }
    CacheEntry entry{snapshot, snapshot.get(), codeOffset, waveSize, {}, std::move(expansion.callees), nullptr};
    for (const auto reg : expansion.consulted) entry.consulted.emplace_back(reg, userData[reg]);
    if (!expansion.code.empty()) {
        // Without a header the expansion is prepared at its first dispatch, as an unregistered
        // compute program is (registration prepared the original code); the request keeps the
        // original header.
        ShaderSnapshot expanded{base, snapshot->headerAddress, 0, std::move(expansion.code), {}};
        if (traceCalls()) {
            std::fprintf(stderr, "[shader-calls] 0x%llx: %zu words expanded to %zu with %zu callee(s):", static_cast<unsigned long long>(base), code.size(), expanded.code.size(), entry.callees.size());
            for (const auto& [address, words] : entry.callees) std::fprintf(stderr, " 0x%llx (%zu words)", static_cast<unsigned long long>(address), words.size());
            std::fputc('\n', stderr);
        }
        entry.expanded = std::make_shared<const ShaderSnapshot>(std::move(expanded));
    }
    std::lock_guard lock(mutex);
    cache.push_front(std::move(entry));
    if (cache.size() > 64) cache.pop_back();
    return cache.front().expanded;
}

}
