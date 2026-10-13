#include <cstdint>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#else
#include "prx/libc/include/specifics/linux/HostMapping.hpp"
#endif

#include "SceTypes.hpp"
#include "ModuleTable.hpp"
#include "prx/libc/include/General.hpp"

extern "C" void* APS5_VABI dlopen_nid_postfix(const char* path, int flags);

namespace {

constexpr int SCE_SYSMODULE_ERROR_INVALID_VALUE = static_cast<int>(0x805A1000);
constexpr int SCE_SYSMODULE_ERROR_UNLOADED = static_cast<int>(0x805A1001);
constexpr int kRtldNow = 2;

const char* findModuleName(const std::uint32_t id) {
    const auto it = kModuleTable.find(id);
    return it != kModuleTable.end() ? it->second : nullptr;
}

std::mutex gMutex;
std::unordered_map<std::uint32_t, std::int32_t> gLoadCount;
std::unordered_map<std::uint32_t, KernelModule> gHandles;

void* openHostModule(const char* name) {
    const std::string relative = std::string("libs/") + name + ".prx";
    return dlopen_nid_postfix(relative.c_str(), kRtldNow);
}

#ifdef _WIN32
struct ImageName {
    DWORD stamp = 0;
    DWORD checksum = 0;
    DWORD size = 0;
    std::string name;
};

std::string queryMappedFileName(void* address) {
    char path[4096] = {};
    const DWORD length = GetMappedFileNameA(GetCurrentProcess(), address, path, sizeof(path) - 1);
    return std::string(path, length);
}

void copyMappedFileName(const MEMORY_BASIC_INFORMATION& mbi, ModuleInfoForUnwind* info) {
    std::string name;
    if (mbi.Type != MEM_IMAGE) {
        name = queryMappedFileName(mbi.BaseAddress);
    } else {
        const auto* base = static_cast<const std::uint8_t*>(mbi.AllocationBase);
        const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(base);
        const auto* nt = dos->e_magic == IMAGE_DOS_SIGNATURE ? reinterpret_cast<const IMAGE_NT_HEADERS64*>(base + dos->e_lfanew) : nullptr;
        if (nt == nullptr || nt->Signature != IMAGE_NT_SIGNATURE) {
            name = queryMappedFileName(mbi.BaseAddress);
        } else {
            static std::mutex mutex;
            static auto& names = *new std::unordered_map<const void*, ImageName>();
            std::lock_guard lock(mutex);
            auto& cached = names[mbi.AllocationBase];
            if (cached.name.empty() || cached.stamp != nt->FileHeader.TimeDateStamp || cached.checksum != nt->OptionalHeader.CheckSum ||
                cached.size != nt->OptionalHeader.SizeOfImage) {
                cached.stamp = nt->FileHeader.TimeDateStamp;
                cached.checksum = nt->OptionalHeader.CheckSum;
                cached.size = nt->OptionalHeader.SizeOfImage;
                cached.name = queryMappedFileName(mbi.BaseAddress);
            }
            std::strncpy(info->name, cached.name.c_str(), sizeof(info->name) - 1);
            info->name[sizeof(info->name) - 1] = '\0';
            return;
        }
    }
    std::strncpy(info->name, name.c_str(), sizeof(info->name) - 1);
    info->name[sizeof(info->name) - 1] = '\0';
}
#endif

bool fillModuleInfoForUnwind(std::uint64_t addr, ModuleInfoForUnwind* info) {
#ifdef _WIN32
    MEMORY_BASIC_INFORMATION mbi{};
    if (!VirtualQuery(reinterpret_cast<LPCVOID>(addr), &mbi, sizeof(mbi))) {
        return false;
    }
    info->st_size = sizeof(ModuleInfoForUnwind);
    info->eh_frame_hdr_addr = 0;
    info->eh_frame_addr = 0;
    info->eh_frame_size = 0;
    info->seg0_addr = reinterpret_cast<std::uint64_t>(mbi.BaseAddress);
    info->seg0_size = mbi.RegionSize;
    copyMappedFileName(mbi, info);
    return true;
#else
    const auto mapping = HostMapping::Find(addr, "sceSysmoduleGetModuleInfoForUnwind");
    if (!mapping) {
        return false;
    }
    info->st_size = sizeof(ModuleInfoForUnwind);
    std::strncpy(info->name, mapping->path.c_str(), sizeof(info->name) - 1);
    info->name[sizeof(info->name) - 1] = '\0';
    info->eh_frame_hdr_addr = 0;
    info->eh_frame_addr = 0;
    info->eh_frame_size = 0;
    info->seg0_addr = mapping->start;
    info->seg0_size = mapping->end - mapping->start;
    return true;
#endif
}

}

extern "C" {

int APS5_VABI sceSysmoduleGetModuleInfoForUnwind(std::uint64_t addr, int flags, ModuleInfoForUnwind* info) {
    (void)flags;
    if (!fillModuleInfoForUnwind(addr, info)) {
        throw std::runtime_error("sceSysmoduleGetModuleInfoForUnwind: address not found");
    }
    return 0;
}

int APS5_VABI sceSysmoduleGetModuleHandleInternal(std::uint32_t id, std::int32_t* handle) {
    if ((id & 0x7fffffffu) == 0) {
        throw std::runtime_error("sceSysmoduleGetModuleHandleInternal: invalid id 0");
    }
    if (!handle) {
        throw std::runtime_error("sceSysmoduleGetModuleHandleInternal: null handle");
    }
    const char* name = findModuleName(id);
    if (!name) {
        throw std::runtime_error(std::string("sceSysmoduleGetModuleHandleInternal: unknown id ") + std::to_string(id));
    }
    std::lock_guard<std::mutex> lock(gMutex);
    const auto loaded = gLoadCount.find(id);
    if (loaded == gLoadCount.end() || loaded->second < 1) {
        return SCE_SYSMODULE_ERROR_UNLOADED;
    }
    const auto existing = gHandles.find(id);
    if (existing != gHandles.end()) {
        *handle = existing->second;
        return 0;
    }
    void* module = openHostModule(name);
    if (!module) {
        throw std::runtime_error(std::string("sceSysmoduleGetModuleHandleInternal: failed to open ") + name);
    }
    const auto value = static_cast<KernelModule>(reinterpret_cast<std::intptr_t>(module));
    gHandles.emplace(id, value);
    *handle = value;
    return 0;
}

int APS5_VABI sceSysmoduleIsLoaded(std::uint16_t id) {
    if (id == 0) {
        return SCE_SYSMODULE_ERROR_INVALID_VALUE;
    }
    if (!findModuleName(id)) {
        APS5_LOG_OUT("unknown id: %u", static_cast<unsigned>(id));
        // throw std::runtime_error(std::string("sceSysmoduleIsLoaded: unknown id ") + std::to_string(id));
        return 0;
    }
    std::lock_guard<std::mutex> lock(gMutex);
    auto it = gLoadCount.find(id);
    if (it == gLoadCount.end() || it->second < 1) {
        return SCE_SYSMODULE_ERROR_UNLOADED;
    }
    return 0;
}

int APS5_VABI sceSysmoduleLoadModule(std::uint16_t id) {
    if (id == 0) {
        return SCE_SYSMODULE_ERROR_INVALID_VALUE;
    }
    if (!findModuleName(id)) {
        APS5_LOG_OUT("unknown id: %u", static_cast<unsigned>(id));
        // throw std::runtime_error(std::string("sceSysmoduleLoadModule: unknown id ") + std::to_string(id));
        return 0;
    }
    std::lock_guard<std::mutex> lock(gMutex);
    gLoadCount[id]++;
    return 0;
}

int APS5_VABI sceSysmoduleLoadModuleInternalWithArg(std::uint32_t id, int argc, void* argv, std::uint64_t unk, int* ret) {
    (void)argc;
    (void)argv;
    (void)unk;
    if ((id & 0x7fffffffu) == 0) {
        return SCE_SYSMODULE_ERROR_INVALID_VALUE;
    }
    if (!findModuleName(id)) {
        throw std::runtime_error(std::string("sceSysmoduleLoadModuleInternalWithArg: unknown id ") + std::to_string(id));
    }
    std::lock_guard<std::mutex> lock(gMutex);
    gLoadCount[id]++;
    if (ret) {
        *ret = 0;
    }
    return 0;
}

int APS5_VABI sceSysmoduleUnloadModule(std::uint16_t id) {
    if (id == 0) {
        return SCE_SYSMODULE_ERROR_INVALID_VALUE;
    }
    if (!findModuleName(id)) {
        APS5_LOG_OUT("unknown id: %u", static_cast<unsigned>(id));
        return 0;
    }
    std::lock_guard<std::mutex> lock(gMutex);
    auto it = gLoadCount.find(id);
    if (it == gLoadCount.end() || it->second < 1) {
        return SCE_SYSMODULE_ERROR_UNLOADED;
    }
    it->second--;
    return 0;
}

int APS5_VABI sceSysmoduleLoadModuleInternal(std::uint32_t id) {
    if ((id & 0x7fffffffu) == 0) {
        return SCE_SYSMODULE_ERROR_INVALID_VALUE;
    }
    if (!findModuleName(id)) {
        throw std::runtime_error(std::string("sceSysmoduleLoadModuleInternal: unknown id ") + std::to_string(id));
    }
    std::lock_guard<std::mutex> lock(gMutex);
    gLoadCount[id]++;
    return 0;
}

int APS5_VABI sceSysmoduleUnloadModuleInternal(std::uint32_t id) {
    if ((id & 0x7fffffffu) == 0) {
        return SCE_SYSMODULE_ERROR_INVALID_VALUE;
    }
    if (!findModuleName(id)) {
        throw std::runtime_error(std::string("sceSysmoduleUnloadModuleInternal: unknown id ") + std::to_string(id));
    }
    std::lock_guard<std::mutex> lock(gMutex);
    auto it = gLoadCount.find(id);
    if (it == gLoadCount.end() || it->second < 1) {
        return SCE_SYSMODULE_ERROR_UNLOADED;
    }
    it->second--;
    return 0;
}

}
