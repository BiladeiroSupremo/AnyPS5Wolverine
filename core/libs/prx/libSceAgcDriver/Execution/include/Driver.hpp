#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_HPP

#include <span>
#include "SceTypes.hpp"
#include "SceShaders.hpp"

namespace AgcDriver {

void Submit(const Packet* packet, std::uint32_t queue);

}

extern "C" void AgcDriverWaitIdle_nid_postfix();
extern "C" void AgcDriverRaiseWorkerThreadPriority_nid_postfix(const char* role);
extern "C" void AgcDriverLockVulkanLoader_nid_postfix();
extern "C" void AgcDriverUnlockVulkanLoader_nid_postfix();
extern "C" void AgcDriverShutdown_nid_postfix();
extern "C" void AgcDriverSuspendPoint_nid_postfix();
extern "C" void AgcDriverRequestScreenshot_nid_postfix();
extern "C" void AgcDriverRegisterShader_nid_postfix(const Shader* shader);
extern "C" void AgcDriverResolveShaderAbi_nid_postfix(const Shader* shader, std::span<const ShaderRegister> context, std::span<const ShaderRegister> primitive);
extern "C" void AgcDriverResolveGraphicsStagesAbi_nid_postfix(std::span<const Shader* const> stages, std::span<const ShaderRegister> context, std::span<const ShaderRegister> primitive);
extern "C" void AgcDriverResolveGraphicsAbi_nid_postfix(const Shader* vertex, const Shader* pixel, std::uint32_t primitiveType);
// Native-port shadow (APS5_NATIVE_SHADOW=1, NativeShadow.cpp): libSceAgc reports each draw packet it
// writes, with the command buffer, so the driver can read the title's own draw state at that moment.
extern "C" void AgcDriverNoteEngineDraw_nid_postfix(const CommandBuffer* buffer, const std::uint32_t* packet);

#endif
