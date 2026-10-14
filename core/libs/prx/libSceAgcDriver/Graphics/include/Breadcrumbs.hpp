#ifndef APS5_AGC_GRAPHICS_BREADCRUMBS_HPP
#define APS5_AGC_GRAPHICS_BREADCRUMBS_HPP

#include "prx/libSceAgcDriver/Graphics/include/Context.hpp"

#include <cstdint>

// Debug aid: APS5_GPU_BREADCRUMBS=1 (needs VK_AMD_buffer_marker) writes an increasing number into a
// host-visible buffer after every recorded draw and dispatch completes on the GPU, and keeps what
// each number was. When the device is lost the last numbers are printed: the last command the GPU
// finished and the ones it had in flight - the hang is among the latter.
namespace AgcDriver::Graphics::Breadcrumbs {

bool Requested();
// Once the device exists (the extension enabled): makes the marker buffer.
void Attach(const Context& context, PFN_vkCmdWriteBufferMarkerAMD write);
// After a draw or dispatch is recorded into `commands`.
void Mark(VkCommandBuffer commands, const char* kind, std::uint64_t program, std::uint32_t a, std::uint32_t b, std::uint32_t c);
// The program address the next Mark of the calling thread names (draws: the pixel/vertex program).
void SetProgram(std::uint64_t program);
std::uint64_t Program();
// From the device-lost report.
void Report();

}

#endif
