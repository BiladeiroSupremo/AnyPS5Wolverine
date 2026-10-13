#include "prx/libSceAgc/DcbState/include/Init.hpp"

#include "prx/libSceAgc/Command/include/Packet.hpp"
#include "prx/libSceAgc/Command/include/RegisterDefaults.hpp"
#include <cstdint>
#include <cstddef>
#include "SceTypes.hpp"
#include "prx/libc/include/General.hpp"

namespace {

constexpr std::uint32_t kRegisterDefaultsVersionCount = 14;
constexpr std::uint32_t kRegisterDefaultsFallbackVersion = kRegisterDefaultsVersionCount - 1;

std::uint32_t ResolveRegisterDefaultsVersion(std::uint32_t version) {
    return version < kRegisterDefaultsVersionCount ? version : kRegisterDefaultsFallbackVersion;
}

}

extern "C" {

int APS5_VABI sceAgcInit(std::uint32_t version) {
    const std::uint32_t defaultsVersion = ResolveRegisterDefaultsVersion(version);
    Agc::Command::GetRegisterDefaults(defaultsVersion, false, __func__);
    Agc::Command::GetRegisterDefaults(defaultsVersion, true, __func__);
    return 0;
}

int APS5_VABI sceAgcInit_0090(std::uint32_t* state, std::uint32_t version) {
    Agc::Command::CheckAddress(reinterpret_cast<std::uintptr_t>(state), alignof(std::uint32_t), __func__);
    return sceAgcInit(version);
}

}
