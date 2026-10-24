#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_NATIVESHADOW_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_NATIVESHADOW_HPP

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace AgcDriver::DriverDetail {

// Native port, milestone M4 step 0 (C:\wolvie\work\native_port_plan.md): what the title's own draw
// state says about a draw, read when libSceAgc writes the draw packet (AgcDriverNoteEngineDraw), to
// compare with what the driver derives from the PM4 stream. APS5_NATIVE_SHADOW=1; nothing changes
// in rendering. PPSA03671's Agc command buffers carry per-stage records of 0x70 bytes from +0x38
// (record +0: the stage's Agc Shader); the stage records are read only when a record holds a Shader
// whose header checks out.
struct NativeDrawRecord {
    // The Shader (header address) per stage record (0 = empty record). Records that the draw does
    // not use keep their last shader, so the set is a superset of the draw's stages.
    std::array<std::uint64_t, 16> shaders{};
    // The user data of each record holding a shader: the record's +0x10 points at the payload of the
    // title's own SET_SH_REG packet in the command buffer, +0x38 is its word count (tr304).
    std::array<std::vector<std::uint32_t>, 16> userData;
    // The extended user data (record +0x18 -> eud_size_dw words of the Shader's layout; user-data
    // offsets from 32 on address it).
    std::array<std::vector<std::uint32_t>, 16> eud;
    // Raw bytes of the records holding a shader, for the first draws (APS5_NATIVE_SHADOW=2: layout
    // study of the user-data images).
    std::vector<std::pair<std::uint8_t, std::array<std::uint64_t, 14>>> raw;
};

bool NativeShadowActive();
// The record of the draw packet at this guest address (taken at the submit's copy, while the title's
// command memory still holds this frame's packets).
std::optional<NativeDrawRecord> TakeNativeDraw(std::uint64_t packetAddress);
// Counts whether every Shader the driver decoded for the draw (header addresses; 0 = a program
// without a registered header) is among the record's (a null record: the draw had none);
// [native-shadow] line every 10 s.
void CompareNativeDraw(const NativeDrawRecord* record, std::span<const std::uint64_t> decodedShaders);
// A descriptor the recompiler bound for the draw: its role (0 buffer, 1 image, 2 sampler) and first
// four words.
struct DecodedDescriptor {
    std::uint8_t role;
    std::array<std::uint32_t, 4> words;
};
// Whether each bound descriptor is one the record's Shaders place in their user data or extended
// user data (ShaderUserData sharp tables: size 1 = 4 dwords, size 0 = 8 dwords; offsets from 32 in
// the extended user data), by role.
void CompareNativeDescriptors(const NativeDrawRecord& record, std::span<const DecodedDescriptor> decoded);
// Compares each decoded program's user data words with the record's for the same Shader.
void CompareNativeUserData(const NativeDrawRecord& record, std::span<const std::uint64_t> decodedShaders, std::span<const std::vector<std::uint32_t>> decodedUserData);
// APS5_NATIVE_SHADOW=2: prints the raw records of a few draws next to each decoded program's user
// data words (layout study).
void DumpNativeDraw(const NativeDrawRecord& record, std::span<const std::uint64_t> decodedShaders, std::span<const std::vector<std::uint32_t>> decodedUserData);

}

#endif
