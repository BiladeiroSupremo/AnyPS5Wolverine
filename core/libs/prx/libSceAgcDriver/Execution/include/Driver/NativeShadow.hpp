#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_NATIVESHADOW_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_NATIVESHADOW_HPP

#include <array>
#include <cstdint>
#include <optional>
#include <span>

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
};

bool NativeShadowActive();
// The record of the draw packet at this guest address (taken at the submit's copy, while the title's
// command memory still holds this frame's packets).
std::optional<NativeDrawRecord> TakeNativeDraw(std::uint64_t packetAddress);
// Counts whether every Shader the driver decoded for the draw (header addresses; 0 = a program
// without a registered header) is among the record's (a null record: the draw had none);
// [native-shadow] line every 10 s.
void CompareNativeDraw(const NativeDrawRecord* record, std::span<const std::uint64_t> decodedShaders);

}

#endif
