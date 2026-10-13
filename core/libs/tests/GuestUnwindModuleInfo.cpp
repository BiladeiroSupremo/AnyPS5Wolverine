#include "SceTypes.hpp"
#include "tests/GuestUnwindModuleInfoFixture.hpp"
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <windows.h>
#include <psapi.h>

extern "C" int APS5_VABI sceKernelGetModuleInfoForUnwind(uint64_t addr, int flags, ModuleInfoForUnwind* info);
extern "C" void* APS5_VABI dlopen_nid_postfix(const char*, int);
extern "C" void* APS5_VABI dlsym_nid_postfix(void*, const char*);
extern "C" int APS5_VABI dlclose_nid_postfix(void*);

static void Require(bool value) { if (!value) std::abort(); }

static bool Throws(std::uint64_t address) {
    ModuleInfoForUnwind info{};
    try {
        sceKernelGetModuleInfoForUnwind(address, 0, &info);
    } catch (const std::runtime_error&) {
        return true;
    }
    return false;
}

static void ReloadedModuleIsRead() {
    HMODULE linked = nullptr;
    Require(GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        reinterpret_cast<LPCWSTR>(GetUnwindFixture()), &linked));
    wchar_t original[MAX_PATH] = {};
    Require(GetModuleFileNameW(linked, original, MAX_PATH) != 0);
    const auto copy = std::filesystem::path(original).replace_filename("unwind_fixture_reload.dll");
    std::filesystem::copy_file(original, copy, std::filesystem::copy_options::overwrite_existing);
    const auto path = copy.string();

    std::uint64_t firstBase = 0;
    for (int round = 0; round < 2; ++round) {
        void* handle = dlopen_nid_postfix(path.c_str(), 2);
        Require(handle != nullptr);
        const auto get = reinterpret_cast<UnwindFixture* (*)()>(dlsym_nid_postfix(handle, "GetUnwindFixture_nid_no_patch"));
        Require(get != nullptr);
        auto* fixture = get();
        const auto address = reinterpret_cast<std::uint64_t>(fixture);
        MEMORY_BASIC_INFORMATION memory{};
        Require(VirtualQuery(reinterpret_cast<LPCVOID>(address), &memory, sizeof(memory)) != 0);
        const auto base = reinterpret_cast<std::uint64_t>(memory.AllocationBase);
        if (round == 0) firstBase = base;
        else Require(base == firstBase);
        if (round == 0) {
            fixture->frames.terminator = 12;
            fixture->frames.spare[0] = 1;
        }
        ModuleInfoForUnwind info{};
        Require(sceKernelGetModuleInfoForUnwind(address, 0, &info) == 0);
        Require(info.eh_frame_size == (round == 0 ? 4 + 12 + 4 + 12 : 4 + 12));
        Require(dlclose_nid_postfix(handle) == 0);
    }
    std::filesystem::remove(copy);
}

int main() {
    auto* fixture = GetUnwindFixture();
    Require(fixture != nullptr);
    const auto address = reinterpret_cast<std::uint64_t>(fixture);
    MEMORY_BASIC_INFORMATION memory{};
    Require(VirtualQuery(reinterpret_cast<LPCVOID>(address), &memory, sizeof(memory)) != 0);
    const auto base = reinterpret_cast<std::uint64_t>(memory.AllocationBase);
    Require(base != reinterpret_cast<std::uint64_t>(GetModuleHandleW(nullptr)));

    ModuleInfoForUnwind info{};
    Require(sceKernelGetModuleInfoForUnwind(address, 0, &info) == 0);
    Require(info.st_size == sizeof(ModuleInfoForUnwind));
    Require(info.eh_frame_hdr_addr == reinterpret_cast<std::uint64_t>(&fixture->header));
    Require(info.eh_frame_addr == reinterpret_cast<std::uint64_t>(&fixture->frames));
    Require(info.eh_frame_size == 4 + 12);
    Require(info.seg0_addr == base);
    const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(base);
    const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS64*>(base + dos->e_lfanew);
    Require(info.seg0_size == nt->OptionalHeader.SizeOfImage);
    char mapped[4096] = {};
    Require(GetMappedFileNameA(GetCurrentProcess(), memory.BaseAddress, mapped, sizeof(mapped) - 1) != 0);
    Require(std::strcmp(info.name, mapped) == 0);
    for (int repeat = 0; repeat < 3; ++repeat) {
        ModuleInfoForUnwind again{};
        Require(sceKernelGetModuleInfoForUnwind(address, 0, &again) == 0);
        Require(std::memcmp(&again, &info, sizeof(info)) == 0);
    }

    fixture->header.version = 2;
    Require(Throws(address));
    fixture->header.version = 1;
    for (const std::uint8_t encoding : {0x3b, 0x9b, 0x0f, 0xff}) {
        fixture->header.framePointerEncoding = encoding;
        Require(Throws(address));
    }
    fixture->header.framePointerEncoding = 0x1b;
    fixture->frames.cieLength = 0xffffffffu;
    Require(Throws(address));
    fixture->frames.cieLength = 12;
    Require(!Throws(address));

    ReloadedModuleIsRead();

    ModuleInfoForUnwind host{};
    Require(sceKernelGetModuleInfoForUnwind(reinterpret_cast<std::uint64_t>(&GetModuleHandleW), 0, &host) == 0);
    Require(host.eh_frame_hdr_addr == 0 && host.eh_frame_addr == 0 && host.eh_frame_size == 0);
}
