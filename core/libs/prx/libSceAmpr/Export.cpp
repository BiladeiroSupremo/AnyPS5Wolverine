#include <cstdint>
#include <cstddef>
#include <ctime>
#include "SceTypes.hpp"
#include "prx/libc/include/General.hpp"
#include "prx/libc/include/SonyCrt.hpp"
#include "prx/libkernel/Apr/include/AprCommandBuffer.hpp"
#include "prx/libkernel/Time/include/Time.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <cstdio>
#include <cwchar>
#ifdef _WIN32
#include <windows.h>
#include <tlhelp32.h>
#endif

static constexpr int SCE_AMPR_ERROR_BUFFER_FULL = 0x8002001C;
static constexpr int SCE_KERNEL_ERROR_EINVAL = 0x80020016;
static constexpr int SCE_KERNEL_ERROR_EPERM = 0x80020001;
static constexpr int SCE_KERNEL_ERROR_ESRCH = 0x80020003;
static constexpr int SCE_KERNEL_ERROR_EBUSY = 0x80020010;

static std::uint32_t RemainingCommandBytes(const Apr::CommandBufferObject& buffer) {
    if (buffer.offset > buffer.size) throw std::invalid_argument("AMPR: command buffer offset exceeds bound memory");
    return buffer.size - buffer.offset;
}

static int Append(Apr::CommandBufferObject* buffer, const void* command, uint32_t bytes) {
    if (!buffer->base || bytes > RemainingCommandBytes(*buffer)) return SCE_AMPR_ERROR_BUFFER_FULL;
    std::memcpy(buffer->base + buffer->offset, command, bytes);
    buffer->offset += bytes;
    ++buffer->numCommands;
    return 0;
}

template<class TCommand>
static int AppendCommand(Apr::CommandBufferObject* buffer, Apr::Opcode opcode, TCommand command) {
    if (!buffer) return static_cast<int>(0x80020016);
    command.header = {opcode, sizeof(command)};
    return Append(buffer, &command, sizeof(command));
}

static bool ValidCounter(std::uint32_t counter) {
    return counter < 128u;
}

static bool ValidWriteAddress(volatile std::uint64_t* address) {
    return address && (reinterpret_cast<std::uintptr_t>(address) & 7u) == 0u;
}

static bool ValidWait(std::uint32_t compare, std::uint32_t flush) {
    return compare < 4u && flush < 2u;
}

static bool ValidWaitOnCounter_04_00(std::uint32_t access, std::uint32_t compare, std::uint32_t maskOperation, std::uint32_t flush) {
    return access < 8u && compare <= 6u && maskOperation < 2u && flush < 2u;
}

static bool ValidWriteCounter_04_00(std::uint32_t counter, std::uint32_t access, std::uint32_t operation) {
    return (counter & 0x80u) == 0u && access < 8u && operation < 5u;
}

static constexpr std::uint64_t MeasureInvalid = static_cast<std::uint32_t>(SCE_KERNEL_ERROR_EINVAL);

static bool ValidReadLength(std::uint64_t length) {
    return length != 0u && length <= 0x100000000ull;
}

static bool ValidReadOffset(std::uint64_t offset) {
    return offset < 0x10000000000ull;
}

static bool ValidRead(const void* destination, std::uint64_t length, std::uint64_t offset) {
    constexpr std::uint64_t userLimit = 0xF00000000000ull;
    const auto address = reinterpret_cast<std::uint64_t>(destination);
    return ValidReadLength(length) && address <= userLimit && length <= userLimit - address && ValidReadOffset(offset);
}

static int AppendRead(Apr::CommandBufferObject* buffer, Apr::Opcode opcode, std::uint32_t fileId, const void* destination, std::uint64_t size, std::uint64_t offset) {
    if (!buffer) return SCE_KERNEL_ERROR_EINVAL;
    if (opcode != Apr::Opcode::ReadFile && (buffer->recording & Apr::ScatterGatherValid) == 0u) return SCE_KERNEL_ERROR_EINVAL;
    Apr::ReadFileCommand command{{opcode, sizeof(command)}, fileId, 0, reinterpret_cast<std::uint64_t>(destination), size, offset};
    const int result = Append(buffer, &command, sizeof(command));
    if (result == 0) buffer->recording |= Apr::ScatterGatherValid;
    return result;
}

static std::uint64_t NopBytes(std::uint32_t dwords) {
    return sizeof(Apr::CommandHeader) + std::uint64_t{dwords} * 4u;
}

static int AppendNop(Apr::CommandBufferObject* buffer, std::uint32_t dwords, const std::uint32_t* data) {
    if (!buffer) return SCE_KERNEL_ERROR_EINVAL;
    const std::uint64_t bytes = NopBytes(dwords);
    if (!buffer->base || bytes > RemainingCommandBytes(*buffer)) return SCE_AMPR_ERROR_BUFFER_FULL;
    const Apr::CommandHeader header{Apr::Opcode::Nop, static_cast<std::uint32_t>(bytes)};
    std::uint8_t* destination = buffer->base + buffer->offset;
    std::memset(destination, 0, bytes);
    std::memcpy(destination, &header, sizeof(header));
    if (data) std::memcpy(destination + sizeof(header), data, std::uint64_t{dwords} * 4u);
    buffer->offset += static_cast<std::uint32_t>(bytes);
    ++buffer->numCommands;
    return 0;
}

static std::uint64_t MarkerBytes(const char* text) {
    return sizeof(Apr::MarkerCommand) + ((std::strlen(text) + 8) & ~std::uint64_t{7});
}

static std::uint64_t MeasureMarker(const char* text) {
    if (!text) return static_cast<std::uint32_t>(SCE_KERNEL_ERROR_EINVAL);
    return MarkerBytes(text);
}

static int AppendMarker(Apr::CommandBufferObject* buffer, Apr::Opcode opcode, const char* text) {
    if (!buffer || !text) return SCE_KERNEL_ERROR_EINVAL;
    const std::uint64_t bytes = MarkerBytes(text);
    if (!buffer->base || bytes > RemainingCommandBytes(*buffer)) return SCE_AMPR_ERROR_BUFFER_FULL;
    const Apr::MarkerCommand command{{opcode, static_cast<std::uint32_t>(bytes)}};
    std::uint8_t* destination = buffer->base + buffer->offset;
    std::memset(destination, 0, bytes);
    std::memcpy(destination, &command, sizeof(command));
    std::memcpy(destination + sizeof(command), text, std::strlen(text));
    buffer->offset += static_cast<std::uint32_t>(bytes);
    ++buffer->numCommands;
    return 0;
}

static bool InMap(const Apr::CommandBufferObject* buffer) {
    return buffer && (buffer->recording & Apr::MapActive) != 0u;
}

static bool ValidMapRange(std::uint64_t address, std::uint64_t size) {
    constexpr std::uint64_t pageMask = 0x3FFFu;
    return address != 0u && size != 0u && ((address | size) & pageMask) == 0u && address + size > address;
}

static bool ValidMap(std::uint64_t address, std::uint64_t directOffset, std::uint64_t size, std::int32_t protection) {
    return ValidMapRange(address, size) && (directOffset & 0x3FFFu) == 0u && (static_cast<std::uint32_t>(protection) & 0xFFFFFC0Cu) == 0u;
}

static constexpr std::int64_t AmmMeasureInvalid = SCE_KERNEL_ERROR_EINVAL;

static int AppendAmm(Apr::CommandBufferObject* buffer, const void* command, std::uint32_t bytes) {
    if (bytes > RemainingCommandBytes(*buffer)) return SCE_KERNEL_ERROR_EBUSY;
    return Append(buffer, command, bytes);
}

static int RecordAmmMap(Apr::CommandBufferObject* buffer, Apr::Opcode opcode, std::uint64_t address, std::uint64_t directOffset, std::uint64_t size, std::int32_t type, std::int32_t protection) {
    if (!buffer) return SCE_KERNEL_ERROR_EINVAL;
    if (!buffer->base) return SCE_KERNEL_ERROR_EPERM;
    if (!ValidMap(address, directOffset, size, protection)) return SCE_KERNEL_ERROR_EINVAL;
    const Apr::AmmMapCommand command{{opcode, sizeof(command)}, address, directOffset, size, type, protection};
    return AppendAmm(buffer, &command, sizeof(command));
}

static bool ValidAmmProtection(std::int32_t protection) {
    return (static_cast<std::uint32_t>(protection) & 0xFFFFFC0Cu) == 0u;
}

static bool ValidRemap(std::uint64_t address, std::uint64_t source, std::uint64_t size, std::int32_t protection) {
    return ValidMapRange(address, size) && ValidMapRange(source, size) && ValidAmmProtection(protection);
}

static bool ValidProtect(std::uint64_t address, std::uint64_t size, std::int32_t protection, std::int32_t mask) {
    return ValidMapRange(address, size) && ValidAmmProtection(protection) && ValidAmmProtection(mask);
}

static int RecordAmmRemap(Apr::CommandBufferObject* buffer, Apr::Opcode opcode, std::uint64_t address, std::uint64_t source, std::uint64_t size, std::int32_t protection) {
    if (!buffer) return SCE_KERNEL_ERROR_EINVAL;
    if (!buffer->base) return SCE_KERNEL_ERROR_EPERM;
    if (!ValidRemap(address, source, size, protection)) return SCE_KERNEL_ERROR_EINVAL;
    const Apr::AmmRemapCommand command{{opcode, sizeof(command)}, address, source, size, protection, 0};
    return AppendAmm(buffer, &command, sizeof(command));
}

static int RecordAmmProtect(Apr::CommandBufferObject* buffer, Apr::Opcode opcode, std::uint64_t address, std::uint64_t size, std::int32_t type, std::int32_t protection, std::int32_t mask) {
    if (!buffer) return SCE_KERNEL_ERROR_EINVAL;
    if (!buffer->base) return SCE_KERNEL_ERROR_EPERM;
    if (!ValidProtect(address, size, protection, mask)) return SCE_KERNEL_ERROR_EINVAL;
    const Apr::AmmProtectCommand command{{opcode, sizeof(command)}, address, size, type, protection, mask, 0};
    return AppendAmm(buffer, &command, sizeof(command));
}

static constexpr std::int32_t PrtAllocationMask = 1019;

static int RecordPrt(Apr::CommandBufferObject* buffer, const void* command, std::uint32_t bytes) {
    if (!buffer) return SCE_KERNEL_ERROR_EINVAL;
    if (bytes > RemainingCommandBytes(*buffer)) return SCE_KERNEL_ERROR_EBUSY;
    if (!buffer->base) return SCE_KERNEL_ERROR_EPERM;
    return Append(buffer, command, bytes);
}

static int RecordMapBegin(Apr::CommandBufferObject* buffer, Apr::Opcode opcode, std::uint64_t address, std::uint64_t directOffset, std::uint64_t size, std::int32_t type, std::int32_t protection) {
    if (!ValidMap(address, directOffset, size, protection)) return SCE_KERNEL_ERROR_EINVAL;
    if (InMap(buffer)) return SCE_KERNEL_ERROR_EPERM;
    const int result = AppendCommand(buffer, opcode, Apr::AmmMapCommand{{}, address, directOffset, size, type, protection});
    if (result == 0) buffer->recording |= Apr::MapActive;
    return result;
}

static int RecordWriteKernelEventQueue(Apr::CommandBufferObject* buffer, std::uint64_t equeue, std::int32_t ident, std::uint64_t data, bool completion) {
    if (!equeue) return SCE_KERNEL_ERROR_EINVAL;
    if (completion && InMap(buffer)) return SCE_KERNEL_ERROR_EPERM;
    return AppendCommand(buffer, Apr::Opcode::WriteKernelEventQueue, Apr::WriteKernelEventQueueCommand{{}, equeue, static_cast<std::uint64_t>(ident), data, 0});
}

static int RecordWriteAddressFromTimeCounter(Apr::CommandBufferObject* buffer, volatile std::uint64_t* address, bool completion) {
    if (!ValidWriteAddress(address)) return SCE_KERNEL_ERROR_EINVAL;
    if (completion && InMap(buffer)) return SCE_KERNEL_ERROR_EPERM;
    return AppendCommand(buffer, Apr::Opcode::WriteAddressFromTimeCounter, Apr::WriteAddressFromCounterCommand{{}, reinterpret_cast<std::uint64_t>(address), 0, 0});
}

static int RecordWriteAddressFromCounter(Apr::CommandBufferObject* buffer, volatile std::uint64_t* address, std::uint8_t counter, bool completion) {
    if (!ValidWriteAddress(address) || !ValidCounter(counter)) return SCE_KERNEL_ERROR_EINVAL;
    if (completion && InMap(buffer)) return SCE_KERNEL_ERROR_EPERM;
    return AppendCommand(buffer, Apr::Opcode::WriteAddressFromCounter, Apr::WriteAddressFromCounterCommand{{}, reinterpret_cast<std::uint64_t>(address), counter, 0});
}

static int RecordWriteAddressFromCounterPair(Apr::CommandBufferObject* buffer, volatile std::uint64_t* address, std::uint8_t counter, bool completion) {
    if (!ValidWriteAddress(address) || (counter & 0x81u) != 0u) return SCE_KERNEL_ERROR_EINVAL;
    if (completion && InMap(buffer)) return SCE_KERNEL_ERROR_EPERM;
    return AppendCommand(buffer, Apr::Opcode::WriteAddressFromCounterPair, Apr::WriteAddressFromCounterCommand{{}, reinterpret_cast<std::uint64_t>(address), counter, counter + 1u});
}

static constexpr int ThrdSuccess = 0;
static constexpr int ThrdNomem = 1;
static constexpr int ThrdTimedout = 2;
static constexpr int ThrdBusy = 3;
static constexpr int ThrdError = 4;

static constexpr int CppDeviceOrResourceBusy = 0;
static constexpr int CppInvalidArgument = 1;
static constexpr int CppResourceUnavailableTryAgain = 6;

static constexpr int MtxPlain = 0x1;

static std::atomic_flag sharedPtrSpinLock;

struct AmmUsageStatsData {
    std::uint64_t sizeInBytes;
    std::uint16_t numPageTablePoolEntries;
    std::uint16_t allocatedPageTablePoolEntries;
    std::uint16_t highWatermarkAllocatedEntries;
    std::uint16_t reserved;
    std::uint32_t ringIdleFlags;
};
static_assert(sizeof(AmmUsageStatsData) == 0x18);
static constexpr std::uint16_t AmmPageTablePoolEntries = 512;
static constexpr std::uint32_t AmmRingIdleFlags = 0x7;

struct GuestPad {
    unsigned (APS5_VABI* const* vtable)(GuestPad*);
    void* condition;
    void* mutex;
    bool started;
};

static_assert(offsetof(GuestPad, condition) == 0x8 && offsetof(GuestPad, mutex) == 0x10 && offsetof(GuestPad, started) == 0x18 && sizeof(GuestPad) == 0x20);

extern "C" {
int APS5_VABI pthread_create_nid_postfix(Pthread* thread, const PthreadAttr* attr, PthreadEntry entry, void* arg);
int APS5_VABI pthread_join_nid_postfix(Pthread thread, void** value);
int APS5_VABI _Mtx_init_nid_postfix(void** handle, int type);
void APS5_VABI _Mtx_destroy_nid_postfix(void** handle);
int APS5_VABI _Mtx_lock_nid_postfix(void** handle);
int APS5_VABI _Mtx_unlock_nid_postfix(void** handle);
int APS5_VABI _Cnd_init_nid_postfix(void** handle);
void APS5_VABI _Cnd_destroy_nid_postfix(void** handle);
int APS5_VABI _Cnd_wait_nid_postfix(void** condition, void** mutex);
int APS5_VABI _Cnd_signal_nid_postfix(void** handle);
int APS5_VABI wcsrtombs_s_nid_postfix(std::size_t* result, char* destination, std::size_t capacity, const std::uint16_t** source, std::size_t limit, void* state);
int APS5_VABI _Cnd_init_nid_postfix(void** handle);
int SignalCondition_nid_no_patch(void** handle);
[[noreturn]] void ThrowCppError_nid_no_patch(int code);
}

static void RequireThrd(int result, const char* what) {
    switch (result) {
    case ThrdSuccess:
        return;
    case ThrdNomem:
    case ThrdTimedout:
        ThrowCppError_nid_no_patch(CppResourceUnavailableTryAgain);
    case ThrdBusy:
        ThrowCppError_nid_no_patch(CppDeviceOrResourceBusy);
    case ThrdError:
        ThrowCppError_nid_no_patch(CppInvalidArgument);
    }
    throw std::runtime_error(std::string(what) + " failed with " + std::to_string(result));
}

static void* APS5_VABI PadEntry_nid_no_patch(void* argument) {
    auto* pad = static_cast<GuestPad*>(argument);
    const auto go = pad->vtable[0];
    return reinterpret_cast<void*>(static_cast<std::uintptr_t>(go(pad)));
}

extern "C" {

int APS5_VABI sceAmprCommandBufferWriteAddressOnCompletion(Apr::CommandBufferObject* buffer, volatile std::uint64_t* address, std::uint64_t value) {
    if (!ValidWriteAddress(address)) return SCE_KERNEL_ERROR_EINVAL;
    if (InMap(buffer)) return SCE_KERNEL_ERROR_EPERM;
    return AppendCommand(buffer, Apr::Opcode::WriteAddress, Apr::WriteAddressCommand{{}, reinterpret_cast<std::uint64_t>(address), value, 0, 0});
}

int APS5_VABI sceAmprCommandBufferWriteCounterOnCompletion(Apr::CommandBufferObject* buffer, std::uint8_t counter, std::uint32_t value) {
    if (!ValidCounter(counter)) return SCE_KERNEL_ERROR_EINVAL;
    if (InMap(buffer)) return SCE_KERNEL_ERROR_EPERM;
    return AppendCommand(buffer, Apr::Opcode::WriteCounter, Apr::WriteCounterCommand{{}, counter, Apr::CounterAccess::Size4, Apr::CounterOperation::Store, 0, value});
}

int APS5_VABI sceAmprCommandBufferWaitOnAddress(Apr::CommandBufferObject* buffer, volatile std::uint64_t* address, std::uint64_t reference, std::uint8_t compare, std::uint8_t flush) {
    if ((reinterpret_cast<std::uintptr_t>(address) & 7u) != 0u || !ValidWait(compare, flush)) return SCE_KERNEL_ERROR_EINVAL;
    return AppendCommand(buffer, Apr::Opcode::WaitOnAddress, Apr::WaitCommand{{}, reinterpret_cast<std::uint64_t>(address), reference, ~0ull, 0, compare});
}

int APS5_VABI sceAmprCommandBufferWaitOnCounter(Apr::CommandBufferObject* buffer, std::uint8_t counter, std::uint32_t reference, std::uint8_t compare, std::uint8_t flush) {
    if (!ValidCounter(counter) || !ValidWait(compare, flush)) return SCE_KERNEL_ERROR_EINVAL;
    return AppendCommand(buffer, Apr::Opcode::WaitOnCounter, Apr::WaitCommand{{}, 0, reference, ~0ull, counter, compare, Apr::CounterAccess::Size4, 0});
}

int APS5_VABI sceAmprCommandBufferWriteKernelEventQueueOnCompletion(Apr::CommandBufferObject* buffer, std::uint64_t equeue, std::int32_t ident, std::uint64_t data) {
    return RecordWriteKernelEventQueue(buffer, equeue, ident, data, true);
}

int APS5_VABI sceAmprCommandBufferWriteAddressFromTimeCounterOnCompletion(Apr::CommandBufferObject* buffer, volatile std::uint64_t* address) {
    return RecordWriteAddressFromTimeCounter(buffer, address, true);
}

int APS5_VABI sceAmprCommandBufferWriteAddressFromCounterOnCompletion(Apr::CommandBufferObject* buffer, volatile std::uint64_t* address, std::uint8_t counter) {
    return RecordWriteAddressFromCounter(buffer, address, counter, true);
}

int APS5_VABI sceAmprCommandBufferWriteAddressFromCounterPairOnCompletion(Apr::CommandBufferObject* buffer, volatile std::uint64_t* address, std::uint8_t counter) {
    return RecordWriteAddressFromCounterPair(buffer, address, counter, true);
}

std::uint64_t APS5_VABI sceAmprMeasureCommandSizeWriteAddressOnCompletion() { return sizeof(Apr::WriteAddressCommand); }
std::uint64_t APS5_VABI sceAmprMeasureCommandSizeWriteCounterOnCompletion() { return sizeof(Apr::WriteCounterCommand); }
std::uint64_t APS5_VABI sceAmprMeasureCommandSizeWaitOnAddress() { return sizeof(Apr::WaitCommand); }
std::uint64_t APS5_VABI sceAmprMeasureCommandSizeWaitOnCounter() { return sizeof(Apr::WaitCommand); }
std::uint64_t APS5_VABI sceAmprMeasureCommandSizeWriteKernelEventQueueOnCompletion() { return sizeof(Apr::WriteKernelEventQueueCommand); }

int APS5_VABI sceAmprAprCommandBufferConstructor(Apr::CommandBufferObject* buffer, uint64_t* gatherState, uint64_t* scatterState) {
    buffer->type = Apr::BufferType::Apr;
    *gatherState = 0;
    *scatterState = 0;
    return 0;
}

int APS5_VABI sceAmprAprCommandBufferDestructor(Apr::CommandBufferObject* buffer, uint64_t* gatherState, uint64_t* scatterState) {
    (void)buffer;
    (void)gatherState;
    (void)scatterState;
    return 0;
}

int APS5_VABI sceAmprAprCommandBufferMapBegin(Apr::CommandBufferObject* buffer, std::uint64_t address, std::uint64_t size, std::int32_t type, std::int32_t protection) {
    return RecordMapBegin(buffer, Apr::Opcode::AmmMap, address, 0, size, type, protection);
}

int APS5_VABI sceAmprAprCommandBufferMapDirectBegin(Apr::CommandBufferObject* buffer, std::uint64_t address, std::uint64_t directOffset, std::uint64_t size, std::int32_t type, std::int32_t protection) {
    return RecordMapBegin(buffer, Apr::Opcode::AmmMapDirect, address, directOffset, size, type, protection);
}

int APS5_VABI sceAmprAprCommandBufferMapEnd(Apr::CommandBufferObject* buffer) {
    if (!buffer) return SCE_KERNEL_ERROR_EINVAL;
    if (!InMap(buffer)) return SCE_KERNEL_ERROR_EPERM;
    const int result = AppendCommand(buffer, Apr::Opcode::MapEnd, Apr::MapEndCommand{});
    if (result == 0) buffer->recording &= static_cast<std::uint16_t>(~Apr::MapActive);
    return result;
}

// Diagnostic: a read into the first 4 GiB is never into guest memory here. PPSA03671 sometimes
// streams texture mips to base-0 addresses (2026-10-14); report who built such reads.
static void ReportLowDestination(const void* destination, std::uint64_t size, std::uint32_t fileId) {
    static std::atomic<int> reports{0};
    if (reports.fetch_add(1) >= 20) return;
    std::string frames;
#ifdef _WIN32
    // The guest call thunk defeats the unwinder: list the stack words that point into app.exe.
    const auto app = reinterpret_cast<std::uintptr_t>(GetModuleHandleW(nullptr));
    const auto* header = reinterpret_cast<const IMAGE_NT_HEADERS*>(app + reinterpret_cast<const IMAGE_DOS_HEADER*>(app)->e_lfanew);
    const auto appEnd = app + header->OptionalHeader.SizeOfImage;
    char text[96];
    std::snprintf(text, sizeof(text), " ret %p;", __builtin_return_address(0));
    frames += text;
    volatile std::uintptr_t marker = 0;
    const auto* words = reinterpret_cast<const std::uintptr_t*>(const_cast<std::uintptr_t*>(&marker));
    // Only the stack's committed region, and only words into the code section (.elf0 of the
    // relinked image: app.exe+0x10000..+0x5f32600).
    MEMORY_BASIC_INFORMATION region{};
    VirtualQuery(words, &region, sizeof(region));
    const auto stackEnd = reinterpret_cast<std::uintptr_t>(region.BaseAddress) + region.RegionSize;
    const auto limit = static_cast<int>(std::min<std::uintptr_t>(16384u, (stackEnd - reinterpret_cast<std::uintptr_t>(words)) / 8u));
    int found = 0;
    for (int index = 0; index < limit && found < 24; ++index) {
        const auto value = words[index];
        if (value < app + 0x10000u || value >= app + 0x5f32600u || value >= appEnd) continue;
        std::snprintf(text, sizeof(text), " [rsp+0x%x] app.exe+0x%llx", index * 8, static_cast<unsigned long long>(value - app));
        frames += text;
        ++found;
    }
#endif
    std::fprintf(stderr, "[ampr] ReadFile into %p (size 0x%llx, file %u) built from:%s\n", destination, static_cast<unsigned long long>(size), fileId, frames.c_str());
    std::fflush(stderr);
}

// Diagnostic, APS5_WATCH_BAD_READ_RECORD=1 (with APS5_APR_SKIP_BAD_DESTINATION=1 to survive the
// bad read): at the first base-0 read, a hardware write watchpoint (DR0, 8 bytes) is armed on the
// destination field of the game's read-request record in every thread; each later write to it
// logs the writing instruction (the RIP after it) and the value, to find the code that produces
// the destination.
#ifdef _WIN32
static std::uint64_t g_watchedAddress = 0;
static std::atomic<int> g_watchHits{0};

static LONG CALLBACK WatchHandler(EXCEPTION_POINTERS* info) {
    if (info->ExceptionRecord->ExceptionCode != EXCEPTION_SINGLE_STEP || (info->ContextRecord->Dr6 & 1u) == 0 || g_watchedAddress == 0) return EXCEPTION_CONTINUE_SEARCH;
    info->ContextRecord->Dr6 = 0;
    const auto value = *reinterpret_cast<const volatile std::uint64_t*>(g_watchedAddress);
    const int hit = g_watchHits.fetch_add(1);
    if (hit < 40 || (value < 0x100000000ull && hit < 400)) {
        const auto app = reinterpret_cast<std::uintptr_t>(GetModuleHandleW(nullptr));
        const auto rip = static_cast<std::uintptr_t>(info->ContextRecord->Rip);
        const auto* record = reinterpret_cast<const volatile std::uint64_t*>(g_watchedAddress);
        std::fprintf(stderr, "[watch] write #%d to 0x%llx = 0x%llx by thread %lu after app.exe+0x%llx (rip %p); record +8 0x%llx +0x10 0x%llx +0x18 0x%llx; rbx 0x%llx rsi 0x%llx rdi 0x%llx r12 0x%llx r13 0x%llx r14 0x%llx r15 0x%llx\n", hit, static_cast<unsigned long long>(g_watchedAddress), static_cast<unsigned long long>(value), GetCurrentThreadId(), static_cast<unsigned long long>(rip - app), reinterpret_cast<void*>(rip), static_cast<unsigned long long>(record[1]), static_cast<unsigned long long>(record[2]), static_cast<unsigned long long>(record[3]), static_cast<unsigned long long>(info->ContextRecord->Rbx), static_cast<unsigned long long>(info->ContextRecord->Rsi), static_cast<unsigned long long>(info->ContextRecord->Rdi), static_cast<unsigned long long>(info->ContextRecord->R12), static_cast<unsigned long long>(info->ContextRecord->R13), static_cast<unsigned long long>(info->ContextRecord->R14), static_cast<unsigned long long>(info->ContextRecord->R15));
        // The producer at app.exe+0x3c886b2 copies entry rcx/16 of a {destination, size} list at
        // [r15 + 0x28] into the record: dump the request object and that list for a bad value.
        static std::atomic<int> dumps{0};
        const auto r15 = static_cast<std::uintptr_t>(info->ContextRecord->R15);
        MEMORY_BASIC_INFORMATION region{};
        if (value < 0x100000000ull && dumps.fetch_add(1) < 6 && VirtualQuery(reinterpret_cast<const void*>(r15), &region, sizeof(region)) == sizeof(region) && region.State == MEM_COMMIT) {
            const auto* object = reinterpret_cast<const std::uint64_t*>(r15);
            std::string text;
            char word[40];
            for (int index = 0; index < 16; ++index) {
                std::snprintf(word, sizeof(word), " %llx", static_cast<unsigned long long>(object[index]));
                text += word;
            }
            std::fprintf(stderr, "[watch]   rcx 0x%llx; request r15 0x%llx words:%s\n", static_cast<unsigned long long>(info->ContextRecord->Rcx), static_cast<unsigned long long>(r15), text.c_str());
            const auto list = object[5];
            if (VirtualQuery(reinterpret_cast<const void*>(list), &region, sizeof(region)) == sizeof(region) && region.State == MEM_COMMIT) {
                const auto* entries = reinterpret_cast<const std::uint64_t*>(list);
                text.clear();
                for (int index = 0; index < 12; ++index) {
                    std::snprintf(word, sizeof(word), " {%llx,%llx}", static_cast<unsigned long long>(entries[index * 2]), static_cast<unsigned long long>(entries[index * 2 + 1]));
                    text += word;
                }
                std::fprintf(stderr, "[watch]   list 0x%llx entries:%s\n", static_cast<unsigned long long>(list), text.c_str());
            }
        }
        std::fflush(stderr);
    }
    return EXCEPTION_CONTINUE_EXECUTION;
}

static void ArmWatchInAllThreads(std::uint64_t address) {
    const auto self = GetCurrentThreadId();
    HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
    if (snapshot == INVALID_HANDLE_VALUE) return;
    THREADENTRY32 entry{};
    entry.dwSize = sizeof(entry);
    int armed = 0;
    for (BOOL more = Thread32First(snapshot, &entry); more; more = Thread32Next(snapshot, &entry)) {
        if (entry.th32OwnerProcessID != GetCurrentProcessId() || entry.th32ThreadID == self) continue;
        HANDLE thread = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_SET_CONTEXT, FALSE, entry.th32ThreadID);
        if (thread == nullptr) continue;
        if (SuspendThread(thread) != static_cast<DWORD>(-1)) {
            CONTEXT context{};
            context.ContextFlags = CONTEXT_DEBUG_REGISTERS;
            if (GetThreadContext(thread, &context)) {
                context.Dr0 = address;
                context.Dr7 = (context.Dr7 & ~0x000f0003ull) | 1ull | (1ull << 16u) | (3ull << 18u);
                if (SetThreadContext(thread, &context)) ++armed;
            }
            ResumeThread(thread);
        }
        CloseHandle(thread);
    }
    CloseHandle(snapshot);
    std::fprintf(stderr, "[watch] armed a write watch on 0x%llx in %d threads\n", static_cast<unsigned long long>(address), armed);
    std::fflush(stderr);
}
#endif

static void WatchBadReadRecord(std::uint64_t record) {
#ifdef _WIN32
    static const bool enabled = std::getenv("APS5_WATCH_BAD_READ_RECORD") != nullptr;
    static std::atomic<bool> armed{false};
    if (!enabled || record == 0 || (record & 7u) != 0 || armed.exchange(true)) return;
    const auto* words = reinterpret_cast<const std::uint64_t*>(record);
    std::fprintf(stderr, "[watch] bad read record at 0x%llx: +0 0x%llx +8 0x%llx +0x10 0x%llx +0x18 0x%llx\n", static_cast<unsigned long long>(record), static_cast<unsigned long long>(words[0]), static_cast<unsigned long long>(words[1]), static_cast<unsigned long long>(words[2]), static_cast<unsigned long long>(words[3]));
    g_watchedAddress = record;
    AddVectoredExceptionHandler(1, WatchHandler);
    // Another thread arms every thread (a thread cannot set its own debug registers this way).
    std::thread([record] { ArmWatchInAllThreads(record); }).join();
#else
    (void)record;
#endif
}

// The caller's r12 and r14 at sceAmprAprCommandBufferReadFile (saved by the naked entry below):
// PPSA03671's builder at app.exe+0x3c8c1b5 keeps its read-request array in r12 and the current
// record's offset in r14, so r12 + r14 is the 32-byte record whose +0 is the destination.
std::uint64_t g_amprReadFileCallerR12 = 0;
std::uint64_t g_amprReadFileCallerR14 = 0;

int APS5_VABI AmprAprCommandBufferReadFileImpl(Apr::CommandBufferObject* buffer, uint64_t* gatherState, uint64_t* scatterState, uint32_t fileId, void* destination, uint64_t size, uint64_t offset) {
    (void)gatherState;
    (void)scatterState;
    if (!ValidRead(destination, size, offset)) return SCE_KERNEL_ERROR_EINVAL;
    if (reinterpret_cast<std::uintptr_t>(destination) < 0x100000000ull) {
        ReportLowDestination(destination, size, fileId);
        WatchBadReadRecord(g_amprReadFileCallerR12 + g_amprReadFileCallerR14);
    }
    return AppendRead(buffer, Apr::Opcode::ReadFile, fileId, destination, size, offset);
}

__attribute__((naked)) int APS5_VABI sceAmprAprCommandBufferReadFile(Apr::CommandBufferObject*, uint64_t*, uint64_t*, uint32_t, void*, uint64_t, uint64_t) {
    asm volatile(
        "movq %r12, g_amprReadFileCallerR12(%rip)\n\t"
        "movq %r14, g_amprReadFileCallerR14(%rip)\n\t"
        "jmp AmprAprCommandBufferReadFileImpl\n\t");
}

int APS5_VABI sceAmprAprCommandBufferReadFileGather(Apr::CommandBufferObject* buffer, std::uint64_t* mapState, std::uint64_t* scatterGatherState, std::uint64_t size, std::uint64_t offset) {
    (void)mapState;
    (void)scatterGatherState;
    if (!ValidReadLength(size) || !ValidReadOffset(offset)) return SCE_KERNEL_ERROR_EINVAL;
    if (!buffer || (buffer->recording & Apr::ScatterGatherValid) == 0u) return SCE_KERNEL_ERROR_EINVAL;
    const Apr::CompactReadFileGatherCommand command{
        static_cast<std::uint32_t>(Apr::Opcode::CompactReadFileGather) | static_cast<std::uint32_t>(offset >> 32u) << 8u,
        static_cast<std::uint32_t>(size - 1u), static_cast<std::uint32_t>(offset)};
    return Append(buffer, &command, sizeof(command));
}

int APS5_VABI sceAmprAprCommandBufferReadFileGatherScatter(Apr::CommandBufferObject* buffer, std::uint64_t* mapState, std::uint64_t* scatterGatherState, void* destination, std::uint64_t size, std::uint64_t offset) {
    (void)mapState;
    (void)scatterGatherState;
    if (!ValidRead(destination, size, offset)) return SCE_KERNEL_ERROR_EINVAL;
    return AppendRead(buffer, Apr::Opcode::ReadFileGatherScatter, 0, destination, size, offset);
}

int APS5_VABI sceAmprAprCommandBufferReadFileScatter(Apr::CommandBufferObject* buffer, std::uint64_t* mapState, std::uint64_t* scatterGatherState, void* destination, std::uint64_t size) {
    (void)mapState;
    (void)scatterGatherState;
    if (!ValidRead(destination, size, 0)) return SCE_KERNEL_ERROR_EINVAL;
    return AppendRead(buffer, Apr::Opcode::ReadFileScatter, 0, destination, size, 0);
}

int APS5_VABI sceAmprAprCommandBufferResetGatherScatterState(Apr::CommandBufferObject* buffer) {
    const int result = AppendCommand(buffer, Apr::Opcode::ResetGatherScatterState, Apr::ResetGatherScatterStateCommand{});
    if (result == 0) buffer->recording &= static_cast<std::uint16_t>(~Apr::ScatterGatherValid);
    return result;
}

void* APS5_VABI sceAmprCommandBufferClearBuffer(Apr::CommandBufferObject* buffer) {
    void* memory = buffer->base;
    buffer->base = nullptr;
    buffer->size = 0;
    buffer->offset = 0;
    buffer->numCommands = 0;
    return memory;
}

int APS5_VABI sceAmprCommandBufferConstructMarker(Apr::CommandBufferObject* buffer, std::uint32_t type, const char* text, const std::uint32_t* color) {
    switch (type) {
        case 1: return AppendMarker(buffer, Apr::Opcode::SetMarker, text);
        case 2: return AppendMarker(buffer, Apr::Opcode::PushMarker, text);
        case 3: return AppendCommand(buffer, Apr::Opcode::PopMarker, Apr::MarkerCommand{});
        case 5: return color ? AppendMarker(buffer, Apr::Opcode::SetMarker, text) : SCE_KERNEL_ERROR_EINVAL;
        case 6: return color ? AppendMarker(buffer, Apr::Opcode::PushMarker, text) : SCE_KERNEL_ERROR_EINVAL;
        default: return SCE_KERNEL_ERROR_EINVAL;
    }
}

int APS5_VABI sceAmprCommandBufferConstructNop(Apr::CommandBufferObject* buffer, std::int16_t type, const void* payload, std::uint32_t payloadBytes, const std::uint32_t* word) {
    (void)type;
    if (payloadBytes > (word ? 56u : 60u)) return SCE_KERNEL_ERROR_EINVAL;
    if (payloadBytes != 0u && !payload) throw std::invalid_argument("sceAmprCommandBufferConstructNop: null payload");
    const std::uint32_t dwords = (payloadBytes + 3u) / 4u + (word ? 1u : 0u);
    const std::uint32_t offset = buffer ? buffer->offset : 0u;
    const int result = AppendNop(buffer, dwords, nullptr);
    if (result != 0) return result;
    std::uint8_t* data = buffer->base + offset + sizeof(Apr::CommandHeader);
    if (word) {
        std::memcpy(data, word, sizeof(*word));
        data += sizeof(*word);
    }
    if (payloadBytes != 0u) std::memcpy(data, payload, payloadBytes);
    return 0;
}

int APS5_VABI sceAmprCommandBufferConstructor(Apr::CommandBufferObject* buffer) {
    *buffer = {Apr::BufferType::Generic, 0, 0, 0, 0, nullptr};
    return 0;
}

int APS5_VABI sceAmprCommandBufferDestructor(Apr::CommandBufferObject* buffer) {
    (void)buffer;
    return 0;
}

void* APS5_VABI sceAmprCommandBufferGetBufferBaseAddress(const Apr::CommandBufferObject* buffer) {
    return buffer->base;
}

uint32_t APS5_VABI sceAmprCommandBufferGetCurrentOffset(const Apr::CommandBufferObject* buffer) {
    return buffer->offset;
}

uint32_t APS5_VABI sceAmprCommandBufferGetNumCommands(const Apr::CommandBufferObject* buffer) {
    return buffer->numCommands;
}

uint32_t APS5_VABI sceAmprCommandBufferGetSize(const Apr::CommandBufferObject* buffer) {
    return buffer->size;
}

uint32_t APS5_VABI sceAmprCommandBufferGetType(const Apr::CommandBufferObject* buffer) {
    return static_cast<uint32_t>(buffer->type);
}

int APS5_VABI sceAmprCommandBufferNop(Apr::CommandBufferObject* buffer, std::uint32_t dwords) {
    if (dwords == 0u || dwords > 16u) return SCE_KERNEL_ERROR_EINVAL;
    return AppendNop(buffer, dwords, nullptr);
}

int APS5_VABI sceAmprCommandBufferNopWithData(Apr::CommandBufferObject* buffer, std::uint32_t dwords, const std::uint32_t* data) {
    if (dwords > 15u) return SCE_KERNEL_ERROR_EINVAL;
    if (dwords != 0u && !data) throw std::invalid_argument("sceAmprCommandBufferNopWithData: null data");
    return AppendNop(buffer, dwords, data);
}

int APS5_VABI sceAmprCommandBufferPopMarker(Apr::CommandBufferObject* buffer) {
    return AppendCommand(buffer, Apr::Opcode::PopMarker, Apr::MarkerCommand{});
}

int APS5_VABI sceAmprCommandBufferPushMarker(Apr::CommandBufferObject* buffer, const char* text) {
    return AppendMarker(buffer, Apr::Opcode::PushMarker, text);
}

int APS5_VABI sceAmprCommandBufferPushMarkerWithColor(Apr::CommandBufferObject* buffer, const char* text, std::uint32_t color) {
    (void)color;
    return AppendMarker(buffer, Apr::Opcode::PushMarker, text);
}

int APS5_VABI sceAmprCommandBufferReset(Apr::CommandBufferObject* buffer) {
    buffer->offset = 0;
    buffer->numCommands = 0;
    return 0;
}

int APS5_VABI sceAmprCommandBufferSetBuffer(Apr::CommandBufferObject* buffer, void* memory, uint32_t size) {
    buffer->base = static_cast<uint8_t*>(memory);
    buffer->size = size;
    buffer->offset = 0;
    buffer->numCommands = 0;
    return 0;
}

int APS5_VABI sceAmprCommandBufferSetMarker(Apr::CommandBufferObject* buffer, const char* text) {
    return AppendMarker(buffer, Apr::Opcode::SetMarker, text);
}

int APS5_VABI sceAmprCommandBufferSetMarkerWithColor(Apr::CommandBufferObject* buffer, const char* text, const std::uint32_t* color) {
    if (!color) return SCE_KERNEL_ERROR_EINVAL;
    return AppendMarker(buffer, Apr::Opcode::SetMarker, text);
}

int APS5_VABI sceAmprCommandBufferWaitOnAddress_04_00(Apr::CommandBufferObject* buffer, volatile std::uint64_t* address, std::uint64_t reference, std::uint8_t compare, std::uint8_t flush) {
    if (!ValidWriteAddress(address) || compare > 6u || flush > 1u) return SCE_KERNEL_ERROR_EINVAL;
    return AppendCommand(buffer, Apr::Opcode::WaitOnAddress, Apr::WaitCommand{{}, reinterpret_cast<std::uint64_t>(address), reference, ~0ull, 0, compare});
}

int APS5_VABI sceAmprCommandBufferWaitOnCounter_04_00(Apr::CommandBufferObject* buffer, std::uint8_t counter, std::uint8_t access, std::uint64_t reference, std::uint8_t compare, std::uint8_t maskOperation, std::uint64_t mask, std::uint8_t flush) {
    if (!ValidWaitOnCounter_04_00(access, compare, maskOperation, flush)) return SCE_KERNEL_ERROR_EINVAL;
    const std::uint64_t applied = maskOperation ? mask : ~0ull;
    return AppendCommand(buffer, Apr::Opcode::WaitOnCounter, Apr::WaitCommand{{}, 0, reference, applied, counter, compare, static_cast<Apr::CounterAccess>(access), 0});
}

int APS5_VABI sceAmprCommandBufferWriteAddressFromCounterPair_04_00(Apr::CommandBufferObject* buffer, volatile std::uint64_t* address, std::uint8_t counter, std::uint64_t atStart) {
    return RecordWriteAddressFromCounterPair(buffer, address, counter, atStart == 0u);
}

int APS5_VABI sceAmprCommandBufferWriteAddressFromCounter_04_00(Apr::CommandBufferObject* buffer, volatile std::uint64_t* address, std::uint8_t counter, std::uint64_t atStart) {
    return RecordWriteAddressFromCounter(buffer, address, counter, atStart == 0u);
}

int APS5_VABI sceAmprCommandBufferWriteAddressFromTimeCounter_04_00(Apr::CommandBufferObject* buffer, volatile std::uint64_t* address, std::uint64_t atStart) {
    return RecordWriteAddressFromTimeCounter(buffer, address, atStart == 0u);
}

int APS5_VABI sceAmprCommandBufferWriteAddress_04_00(Apr::CommandBufferObject* buffer, uint64_t* address, uint64_t value, uint32_t flags) {
    Apr::WriteAddressCommand command{};
    command.header = {Apr::Opcode::WriteAddress, sizeof(command)};
    command.address = reinterpret_cast<uint64_t>(address);
    command.value = value;
    command.flags = flags;
    return Append(buffer, &command, sizeof(command));
}

int APS5_VABI sceAmprCommandBufferWriteCounter_04_00(Apr::CommandBufferObject* buffer, std::uint8_t counter, std::uint8_t access, std::uint64_t value, std::uint8_t operation, std::uint8_t atStart) {
    if (!ValidWriteCounter_04_00(counter, access, operation)) return SCE_KERNEL_ERROR_EINVAL;
    if (atStart == 0u && InMap(buffer)) return SCE_KERNEL_ERROR_EPERM;
    return AppendCommand(buffer, Apr::Opcode::WriteCounter, Apr::WriteCounterCommand{{}, counter, static_cast<Apr::CounterAccess>(access), static_cast<Apr::CounterOperation>(operation), 0, value});
}

int APS5_VABI sceAmprCommandBufferWriteKernelEventQueue_04_00(Apr::CommandBufferObject* buffer, std::uint64_t equeue, std::int32_t ident, std::uint64_t data, std::uint64_t atStart) {
    return RecordWriteKernelEventQueue(buffer, equeue, ident, data, atStart == 0u);
}

std::uint64_t APS5_VABI sceAmprMeasureCommandSizeMapBegin(std::uint64_t address, std::uint64_t size, std::uint32_t type, std::uint32_t protection) {
    (void)type;
    if (!ValidMap(address, 0, size, static_cast<std::int32_t>(protection))) return MeasureInvalid;
    return sizeof(Apr::AmmMapCommand);
}

std::uint64_t APS5_VABI sceAmprMeasureCommandSizeMapDirectBegin(std::uint64_t address, std::uint64_t directOffset, std::uint64_t size, std::uint32_t type, std::uint32_t protection) {
    (void)type;
    if (!ValidMap(address, directOffset, size, static_cast<std::int32_t>(protection))) return MeasureInvalid;
    return sizeof(Apr::AmmMapCommand);
}

std::uint64_t APS5_VABI sceAmprMeasureCommandSizeMapEnd() {
    return sizeof(Apr::MapEndCommand);
}

std::uint64_t APS5_VABI sceAmprMeasureCommandSizeNop(std::uint32_t dwords) {
    if (dwords == 0u || dwords > 16u) return MeasureInvalid;
    return NopBytes(dwords);
}

std::uint64_t APS5_VABI sceAmprMeasureCommandSizeNopWithData(std::uint32_t dwords) {
    if (dwords == 0u || dwords > 16u) return MeasureInvalid;
    return NopBytes(dwords - 1u);
}

std::uint64_t APS5_VABI sceAmprMeasureCommandSizePopMarker() {
    return sizeof(Apr::MarkerCommand);
}

std::uint64_t APS5_VABI sceAmprMeasureCommandSizePushMarker(const char* text) {
    return MeasureMarker(text);
}

std::uint64_t APS5_VABI sceAmprMeasureCommandSizePushMarkerWithColor(const char* text, std::uint32_t color) {
    (void)color;
    return MeasureMarker(text);
}

uint32_t APS5_VABI sceAmprMeasureCommandSizeReadFile(void) {
    return sizeof(Apr::ReadFileCommand);
}

std::uint64_t APS5_VABI sceAmprMeasureCommandSizeReadFileGather(std::uint64_t size, std::uint64_t offset) {
    if (!ValidReadLength(size) || !ValidReadOffset(offset)) return MeasureInvalid;
    return sizeof(Apr::CompactReadFileGatherCommand);
}

std::uint64_t APS5_VABI sceAmprMeasureCommandSizeReadFileGatherScatter(void* destination, std::uint64_t size, std::uint64_t offset) {
    if (!ValidRead(destination, size, offset)) return MeasureInvalid;
    return sizeof(Apr::ReadFileCommand);
}

std::uint64_t APS5_VABI sceAmprMeasureCommandSizeReadFileScatter(void* destination, std::uint64_t size) {
    if (!ValidRead(destination, size, 0)) return MeasureInvalid;
    return sizeof(Apr::ReadFileCommand);
}

std::uint64_t APS5_VABI sceAmprMeasureCommandSizeResetGatherScatterState() {
    return sizeof(Apr::ResetGatherScatterStateCommand);
}

std::uint64_t APS5_VABI sceAmprMeasureCommandSizeSetMarker(const char* text) {
    return MeasureMarker(text);
}

std::uint64_t APS5_VABI sceAmprMeasureCommandSizeSetMarkerWithColor(const char* text, std::uint32_t color) {
    (void)color;
    return MeasureMarker(text);
}

std::uint64_t APS5_VABI sceAmprMeasureCommandSizeWaitOnAddress_04_00(volatile std::uint64_t* address, std::uint64_t, std::uint8_t compare, std::uint8_t flush) {
    if (!ValidWriteAddress(address) || compare > 6u || flush > 1u) return MeasureInvalid;
    return sizeof(Apr::WaitCommand);
}

std::uint64_t APS5_VABI sceAmprMeasureCommandSizeWaitOnCounter_04_00(std::uint8_t, std::uint8_t access, std::uint64_t, std::uint8_t compare, std::uint8_t maskOperation, std::uint64_t, std::uint8_t flush) {
    if (!ValidWaitOnCounter_04_00(access, compare, maskOperation, flush)) return MeasureInvalid;
    return sizeof(Apr::WaitCommand);
}

std::uint64_t APS5_VABI sceAmprMeasureCommandSizeWriteAddressFromCounterPair_04_00(volatile std::uint64_t* address, std::uint8_t counter) {
    if (!ValidWriteAddress(address) || (counter & 1u) != 0u) return MeasureInvalid;
    return sizeof(Apr::WriteAddressFromCounterCommand);
}

std::uint64_t APS5_VABI sceAmprMeasureCommandSizeWriteAddressFromCounter_04_00(volatile std::uint64_t* address, std::uint8_t counter) {
    (void)counter;
    if (!ValidWriteAddress(address)) return MeasureInvalid;
    return sizeof(Apr::WriteAddressFromCounterCommand);
}

std::uint64_t APS5_VABI sceAmprMeasureCommandSizeWriteAddressFromTimeCounter_04_00(volatile std::uint64_t* address) {
    if (!ValidWriteAddress(address)) return MeasureInvalid;
    return sizeof(Apr::WriteAddressFromCounterCommand);
}

uint32_t APS5_VABI sceAmprMeasureCommandSizeWriteAddress_04_00(void) {
    return sizeof(Apr::WriteAddressCommand);
}

std::uint64_t APS5_VABI sceAmprMeasureCommandSizeWriteCounter_04_00(std::uint8_t counter, std::uint8_t access, std::uint64_t, std::uint8_t operation) {
    if (!ValidWriteCounter_04_00(counter, access, operation)) return MeasureInvalid;
    return sizeof(Apr::WriteCounterCommand);
}

std::uint64_t APS5_VABI sceAmprMeasureCommandSizeWriteKernelEventQueue_04_00(std::uint64_t, std::int32_t, std::uint64_t) {
    return sizeof(Apr::WriteKernelEventQueueCommand);
}

struct AmmSubmitResult {
    std::int32_t result;
    std::uint32_t errorOffset;
};

int APS5_VABI sceAmprAmmCommandBufferConstructor(Apr::CommandBufferObject* buffer) {
    (void)buffer;
    return 0;
}

int APS5_VABI sceAmprAmmCommandBufferDestructor(Apr::CommandBufferObject* buffer) {
    (void)buffer;
    return 0;
}

int APS5_VABI sceAmprAmmCommandBufferMap(Apr::CommandBufferObject* buffer, std::uint64_t address, std::uint64_t size, std::int32_t type, std::int32_t protection) {
    return RecordAmmMap(buffer, Apr::Opcode::AmmMap, address, 0, size, type, protection);
}

int APS5_VABI sceAmprAmmCommandBufferMapWithGpuMaskId(Apr::CommandBufferObject* buffer, std::uint64_t address, std::uint64_t size, std::int32_t type, std::int32_t protection, std::uint8_t gpuMaskId) {
    (void)gpuMaskId;
    return RecordAmmMap(buffer, Apr::Opcode::AmmMap, address, 0, size, type, protection);
}

int APS5_VABI sceAmprAmmCommandBufferMapDirect(Apr::CommandBufferObject* buffer, std::uint64_t address, std::uint64_t directOffset, std::uint64_t size, std::int32_t type, std::int32_t protection) {
    return RecordAmmMap(buffer, Apr::Opcode::AmmMapDirect, address, directOffset, size, type, protection);
}

int APS5_VABI sceAmprAmmCommandBufferMapDirectWithGpuMaskId(Apr::CommandBufferObject* buffer, std::uint64_t address, std::uint64_t directOffset, std::uint64_t size, std::int32_t type, std::int32_t protection, std::uint8_t gpuMaskId) {
    (void)gpuMaskId;
    return RecordAmmMap(buffer, Apr::Opcode::AmmMapDirect, address, directOffset, size, type, protection);
}

int APS5_VABI sceAmprAmmCommandBufferUnmap(Apr::CommandBufferObject* buffer, std::uint64_t address, std::uint64_t size) {
    if (!buffer) return SCE_KERNEL_ERROR_EINVAL;
    if (!buffer->base) return SCE_KERNEL_ERROR_EPERM;
    if (!ValidMapRange(address, size)) return SCE_KERNEL_ERROR_EINVAL;
    const Apr::AmmUnmapCommand command{{Apr::Opcode::AmmUnmap, sizeof(command)}, address, size};
    return AppendAmm(buffer, &command, sizeof(command));
}

std::int64_t APS5_VABI sceAmprAmmMeasureAmmCommandSizeMap(std::uint64_t address, std::uint64_t size, std::int32_t type, std::int32_t protection) {
    (void)type;
    return ValidMap(address, 0, size, protection) ? std::int64_t{sizeof(Apr::AmmMapCommand)} : AmmMeasureInvalid;
}

std::int64_t APS5_VABI sceAmprAmmMeasureAmmCommandSizeMapWithGpuMaskId(std::uint64_t address, std::uint64_t size, std::int32_t type, std::int32_t protection, std::uint8_t gpuMaskId) {
    (void)gpuMaskId;
    return sceAmprAmmMeasureAmmCommandSizeMap(address, size, type, protection);
}

std::int64_t APS5_VABI sceAmprAmmMeasureAmmCommandSizeMapDirect(std::uint64_t address, std::uint64_t directOffset, std::uint64_t size, std::int32_t type, std::int32_t protection) {
    (void)type;
    return ValidMap(address, directOffset, size, protection) ? std::int64_t{sizeof(Apr::AmmMapCommand)} : AmmMeasureInvalid;
}

std::int64_t APS5_VABI sceAmprAmmMeasureAmmCommandSizeMapDirectWithGpuMaskId(std::uint64_t address, std::uint64_t directOffset, std::uint64_t size, std::int32_t type, std::int32_t protection, std::uint8_t gpuMaskId) {
    (void)gpuMaskId;
    return sceAmprAmmMeasureAmmCommandSizeMapDirect(address, directOffset, size, type, protection);
}

std::int64_t APS5_VABI sceAmprAmmMeasureAmmCommandSizeUnmap(std::uint64_t address, std::uint64_t size) {
    return ValidMapRange(address, size) ? std::int64_t{sizeof(Apr::AmmUnmapCommand)} : AmmMeasureInvalid;
}

int APS5_VABI sceAmprAmmCommandBufferRemap(Apr::CommandBufferObject* buffer, std::uint64_t address, std::uint64_t source, std::uint64_t size, std::int32_t protection) {
    return RecordAmmRemap(buffer, Apr::Opcode::AmmRemap, address, source, size, protection);
}

int APS5_VABI sceAmprAmmCommandBufferRemapWithGpuMaskId(Apr::CommandBufferObject* buffer, std::uint64_t address, std::uint64_t source, std::uint64_t size, std::int32_t protection, std::uint8_t gpuMaskId) {
    (void)gpuMaskId;
    return RecordAmmRemap(buffer, Apr::Opcode::AmmRemap, address, source, size, protection);
}

int APS5_VABI sceAmprAmmCommandBufferMultiMap(Apr::CommandBufferObject* buffer, std::uint64_t address, std::uint64_t alias, std::uint64_t size, std::int32_t protection) {
    return RecordAmmRemap(buffer, Apr::Opcode::AmmMultiMap, address, alias, size, protection);
}

int APS5_VABI sceAmprAmmCommandBufferMultiMapWithGpuMaskId(Apr::CommandBufferObject* buffer, std::uint64_t address, std::uint64_t alias, std::uint64_t size, std::int32_t protection, std::uint8_t gpuMaskId) {
    (void)gpuMaskId;
    return RecordAmmRemap(buffer, Apr::Opcode::AmmMultiMap, address, alias, size, protection);
}

int APS5_VABI sceAmprAmmCommandBufferModifyProtect(Apr::CommandBufferObject* buffer, std::uint64_t address, std::uint64_t size, std::int32_t protection, std::int32_t mask) {
    return RecordAmmProtect(buffer, Apr::Opcode::AmmModifyProtect, address, size, 0, protection, mask);
}

int APS5_VABI sceAmprAmmCommandBufferModifyProtectWithGpuMaskId(Apr::CommandBufferObject* buffer, std::uint64_t address, std::uint64_t size, std::int32_t protection, std::int32_t mask, std::uint8_t gpuMaskId) {
    (void)gpuMaskId;
    return RecordAmmProtect(buffer, Apr::Opcode::AmmModifyProtect, address, size, 0, protection, mask);
}

int APS5_VABI sceAmprAmmCommandBufferModifyMtypeProtect(Apr::CommandBufferObject* buffer, std::uint64_t address, std::uint64_t size, std::int32_t type, std::int32_t protection, std::int32_t mask) {
    return RecordAmmProtect(buffer, Apr::Opcode::AmmModifyMtypeProtect, address, size, type, protection, mask);
}

int APS5_VABI sceAmprAmmCommandBufferModifyMtypeProtectWithGpuMaskId(Apr::CommandBufferObject* buffer, std::uint64_t address, std::uint64_t size, std::int32_t type, std::int32_t protection, std::int32_t mask, std::uint8_t gpuMaskId) {
    (void)gpuMaskId;
    return RecordAmmProtect(buffer, Apr::Opcode::AmmModifyMtypeProtect, address, size, type, protection, mask);
}

std::int64_t APS5_VABI sceAmprAmmMeasureAmmCommandSizeRemap(std::uint64_t address, std::uint64_t source, std::uint64_t size, std::int32_t protection) {
    return ValidRemap(address, source, size, protection) ? std::int64_t{sizeof(Apr::AmmRemapCommand)} : AmmMeasureInvalid;
}

std::int64_t APS5_VABI sceAmprAmmMeasureAmmCommandSizeRemapWithGpuMaskId(std::uint64_t address, std::uint64_t source, std::uint64_t size, std::int32_t protection, std::uint8_t gpuMaskId) {
    (void)gpuMaskId;
    return sceAmprAmmMeasureAmmCommandSizeRemap(address, source, size, protection);
}

std::int64_t APS5_VABI sceAmprAmmMeasureAmmCommandSizeMultiMap(std::uint64_t address, std::uint64_t alias, std::uint64_t size, std::int32_t protection) {
    return ValidRemap(address, alias, size, protection) ? std::int64_t{sizeof(Apr::AmmRemapCommand)} : AmmMeasureInvalid;
}

std::int64_t APS5_VABI sceAmprAmmMeasureAmmCommandSizeMultiMapWithGpuMaskId(std::uint64_t address, std::uint64_t alias, std::uint64_t size, std::int32_t protection, std::uint8_t gpuMaskId) {
    (void)gpuMaskId;
    return sceAmprAmmMeasureAmmCommandSizeMultiMap(address, alias, size, protection);
}

std::int64_t APS5_VABI sceAmprAmmMeasureAmmCommandSizeModifyProtect(std::uint64_t address, std::uint64_t size, std::int32_t protection, std::int32_t mask) {
    return ValidProtect(address, size, protection, mask) ? std::int64_t{sizeof(Apr::AmmProtectCommand)} : AmmMeasureInvalid;
}

std::int64_t APS5_VABI sceAmprAmmMeasureAmmCommandSizeModifyProtectWithGpuMaskId(std::uint64_t address, std::uint64_t size, std::int32_t protection, std::int32_t mask, std::uint8_t gpuMaskId) {
    (void)gpuMaskId;
    return sceAmprAmmMeasureAmmCommandSizeModifyProtect(address, size, protection, mask);
}

std::int64_t APS5_VABI sceAmprAmmMeasureAmmCommandSizeModifyMtypeProtect(std::uint64_t address, std::uint64_t size, std::int32_t type, std::int32_t protection, std::int32_t mask) {
    (void)type;
    return ValidProtect(address, size, protection, mask) ? std::int64_t{sizeof(Apr::AmmProtectCommand)} : AmmMeasureInvalid;
}

std::int64_t APS5_VABI sceAmprAmmMeasureAmmCommandSizeModifyMtypeProtectWithGpuMaskId(std::uint64_t address, std::uint64_t size, std::int32_t type, std::int32_t protection, std::int32_t mask, std::uint8_t gpuMaskId) {
    (void)gpuMaskId;
    return sceAmprAmmMeasureAmmCommandSizeModifyMtypeProtect(address, size, type, protection, mask);
}

int APS5_VABI sceAmprAmmCommandBufferMapAsPrt(Apr::CommandBufferObject* buffer, std::uint64_t address, std::uint64_t size) {
    if (!buffer) return SCE_KERNEL_ERROR_EINVAL;
    if (!ValidMapRange(address, size)) return SCE_KERNEL_ERROR_EINVAL;
    const Apr::AmmUnmapCommand command{{Apr::Opcode::AmmMapAsPrt, sizeof(command)}, address, size};
    return RecordPrt(buffer, &command, sizeof(command));
}

int APS5_VABI sceAmprAmmCommandBufferAllocatePaForPrt(Apr::CommandBufferObject* buffer, std::uint64_t address, std::uint64_t size, std::int32_t type, std::int32_t protection) {
    if (!ValidAmmProtection(protection)) return SCE_KERNEL_ERROR_EINVAL;
    if (!buffer) return SCE_KERNEL_ERROR_EINVAL;
    if (!ValidMapRange(address, size)) return SCE_KERNEL_ERROR_EINVAL;
    const Apr::AmmProtectCommand command{{Apr::Opcode::AmmAllocatePaForPrt, sizeof(command)}, address, size, type, protection, PrtAllocationMask, 0};
    return RecordPrt(buffer, &command, sizeof(command));
}

int APS5_VABI sceAmprAmmCommandBufferRemapIntoPrt(Apr::CommandBufferObject* buffer, std::uint64_t address, std::uint64_t source, std::uint64_t size, std::int32_t protection, std::uint32_t opcode) {
    if (!buffer) return SCE_KERNEL_ERROR_EINVAL;
    if (!buffer->base) return SCE_KERNEL_ERROR_EPERM;
    if (!ValidRemap(address, source, size, protection)) return SCE_KERNEL_ERROR_EINVAL;
    const Apr::AmmRemapCommand command{{Apr::Opcode::AmmRemapIntoPrt, sizeof(command)}, address, source, size, protection, static_cast<std::int32_t>(opcode ? opcode : 1011u)};
    return AppendAmm(buffer, &command, sizeof(command));
}

int APS5_VABI sceAmprAmmCommandBufferUnmapToPrt(Apr::CommandBufferObject* buffer, std::uint64_t address, std::uint64_t size) {
    if (!buffer) return SCE_KERNEL_ERROR_EINVAL;
    if (!buffer->base) return SCE_KERNEL_ERROR_EPERM;
    if (!ValidMapRange(address, size)) return SCE_KERNEL_ERROR_EINVAL;
    const Apr::AmmUnmapCommand command{{Apr::Opcode::AmmUnmapToPrt, sizeof(command)}, address, size};
    return AppendAmm(buffer, &command, sizeof(command));
}

std::int64_t APS5_VABI sceAmprAmmMeasureAmmCommandSizeMapAsPrt(std::uint64_t address, std::uint64_t size) {
    return ValidMapRange(address, size) ? std::int64_t{sizeof(Apr::AmmUnmapCommand)} : AmmMeasureInvalid;
}

std::int64_t APS5_VABI sceAmprAmmMeasureAmmCommandSizeAllocatePaForPrt(std::uint64_t address, std::uint64_t size, std::int32_t type, std::int32_t protection) {
    (void)type;
    return ValidMapRange(address, size) && ValidAmmProtection(protection) ? std::int64_t{sizeof(Apr::AmmProtectCommand)} : AmmMeasureInvalid;
}

int APS5_VABI sceAmprAmmGiveDirectMemory(std::int64_t searchStart, std::int64_t searchEnd, std::size_t size, std::size_t alignment, int usage, std::int64_t* offset) {
    return AmmGiveDirectMemory_nid_no_patch(searchStart, searchEnd, size, alignment, usage, offset);
}

int APS5_VABI sceAmprAmmGetVirtualAddressRanges(std::uint64_t* start, std::uint64_t* end, std::uint64_t* multimapStart, std::uint64_t* multimapEnd) {
    if (!start || !end || !multimapStart || !multimapEnd) throw std::invalid_argument("sceAmprAmmGetVirtualAddressRanges: null output");
    AmmVirtualAddressRanges_nid_no_patch(start, end, multimapStart, multimapEnd);
    return 0;
}

int APS5_VABI sceAmprAmmSetPageTablePoolOccupancyNotificationThreshold(void) {
    NotImplemented_nid_no_patch(__func__);
    return 0;
}

int APS5_VABI sceAmprAmmSubmitCommandBuffer(void* base, std::uint32_t offset, std::uint32_t priority) {
    if (priority > 2u) return SCE_KERNEL_ERROR_EINVAL;
    if (!base) return SCE_KERNEL_ERROR_EPERM;
    AmmSubmit_nid_no_patch(base, offset);
    return 0;
}

int APS5_VABI sceAmprAmmSubmitCommandBuffer2(void* base, std::uint32_t offset, std::uint32_t priority, AmmSubmitResult* result, std::uint32_t* id) {
    if (priority > 2u) return SCE_KERNEL_ERROR_EINVAL;
    if (!base) return SCE_KERNEL_ERROR_EPERM;
    const auto submitted = AmmSubmit_nid_no_patch(base, offset);
    if (result) *result = {0, 0};
    if (id) *id = submitted;
    return 0;
}

int APS5_VABI sceAmprAmmSubmitCommandBuffer3(void* base, std::uint32_t offset, std::uint32_t priority, std::uint32_t* id) {
    if (priority > 2u) return SCE_KERNEL_ERROR_EINVAL;
    if (!base) return SCE_KERNEL_ERROR_EPERM;
    const auto submitted = AmmSubmit_nid_no_patch(base, offset);
    if (id) *id = submitted;
    return 0;
}

int APS5_VABI sceAmprAmmWaitCommandBufferCompletion(std::uint32_t id) {
    return AmmSubmitted_nid_no_patch(id) ? 0 : SCE_KERNEL_ERROR_ESRCH;
}

[[noreturn]] void APS5_VABI _ZSt16_Throw_Cpp_errori_nid_postfix(int code) {
    ThrowCppError_nid_no_patch(code);
}
int APS5_VABI _Thrd_join_nid_postfix(Pthread thread, int* code) {
    void* result = nullptr;
    if (pthread_join_nid_postfix(thread, &result) != 0) return ThrdError;
    if (code) *code = static_cast<int>(reinterpret_cast<std::intptr_t>(result));
    return ThrdSuccess;
}

void APS5_VABI _ZNSt4_Pad8_ReleaseEv_nid_postfix(GuestPad* pad) {
    if (!pad) APS5_INVALID_ARG_EX;
    RequireThrd(_Mtx_lock_nid_postfix(&pad->mutex), "_Pad::_Release: _Mtx_lock");
    pad->started = true;
    RequireThrd(_Cnd_signal_nid_postfix(&pad->condition), "_Pad::_Release: _Cnd_signal");
    RequireThrd(_Mtx_unlock_nid_postfix(&pad->mutex), "_Pad::_Release: _Mtx_unlock");
}

void APS5_VABI _ZNSt4_PadC2Ev_nid_postfix(GuestPad* pad) {
    if (!pad) APS5_INVALID_ARG_EX;
    RequireThrd(_Cnd_init_nid_postfix(&pad->condition), "_Pad::_Pad: _Cnd_init");
    const int created = _Mtx_init_nid_postfix(&pad->mutex, MtxPlain);
    if (created != ThrdSuccess) {
        _Cnd_destroy_nid_postfix(&pad->condition);
        RequireThrd(created, "_Pad::_Pad: _Mtx_init");
    }
    pad->started = false;
    const int locked = _Mtx_lock_nid_postfix(&pad->mutex);
    if (locked != ThrdSuccess) {
        _Mtx_destroy_nid_postfix(&pad->mutex);
        _Cnd_destroy_nid_postfix(&pad->condition);
        RequireThrd(locked, "_Pad::_Pad: _Mtx_lock");
    }
}

void APS5_VABI _ZNSt4_PadD2Ev_nid_postfix(GuestPad* pad) {
    if (!pad) APS5_INVALID_ARG_EX;
    RequireThrd(_Mtx_unlock_nid_postfix(&pad->mutex), "_Pad::~_Pad: _Mtx_unlock");
    _Mtx_destroy_nid_postfix(&pad->mutex);
    _Cnd_destroy_nid_postfix(&pad->condition);
}

int APS5_VABI _Thrd_id_nid_postfix(void) {
    NotImplemented_nid_no_patch(__func__);
    return 0;
}

void APS5_VABI _ZNSt4_Pad7_LaunchEPP7pthread_nid_postfix(GuestPad* pad, Pthread* thread) {
    if (!pad || !thread) APS5_INVALID_ARG_EX;
    const int created = pthread_create_nid_postfix(thread, nullptr, PadEntry_nid_no_patch, pad);
    if (created != 0) ThrowCppError_nid_no_patch(CppResourceUnavailableTryAgain);
    while (!pad->started) RequireThrd(_Cnd_wait_nid_postfix(&pad->condition, &pad->mutex), "_Pad::_Launch: _Cnd_wait");
}
int APS5_VABI _ZNSt7num_getIcSt19istreambuf_iteratorIcSt11char_traitsIcEEE2idE_nid_postfix(void) {
    NotImplemented_nid_no_patch(__func__);
    return 0;
}

void APS5_VABI _Unlock_shared_ptr_spin_lock_nid_postfix(void) {
    sharedPtrSpinLock.clear(std::memory_order_release);
}

int APS5_VABI _Cnd_init_with_name_nid_postfix(void** handle, const char*) {
    return _Cnd_init_nid_postfix(handle);
}

int APS5_VABI _ZTVSt7num_getIcSt19istreambuf_iteratorIcSt11char_traitsIcEEE_nid_postfix(void) {
    NotImplemented_nid_no_patch(__func__);
    return 0;
}

int APS5_VABI _ZNKSt8time_getIcSt19istreambuf_iteratorIcSt11char_traitsIcEEE3getES3_S3_RSt8ios_baseRNSt5_IosbIiE8_IostateEP2tmPKcSE__nid_postfix(void) {
    NotImplemented_nid_no_patch(__func__);
    return 0;
}

int APS5_VABI _ZNSt8time_getIcSt19istreambuf_iteratorIcSt11char_traitsIcEEE2idE_nid_postfix(void) {
    NotImplemented_nid_no_patch(__func__);
    return 0;
}

int APS5_VABI _ZNSt8time_getIcSt19istreambuf_iteratorIcSt11char_traitsIcEEE7_GetcatEPPKNSt6locale5facetEPKS5__nid_postfix(void) {
    NotImplemented_nid_no_patch(__func__);
    return 0;
}

void APS5_VABI _Lock_shared_ptr_spin_lock_nid_postfix(void) {
    while (sharedPtrSpinLock.test_and_set(std::memory_order_acquire)) std::this_thread::yield();
}


int APS5_VABI wcstombs_s_nid_postfix(std::size_t* result, char* destination, std::size_t capacity, const std::uint16_t* source, std::size_t limit) {
    std::array<std::uint64_t, 16> state{};
    return wcsrtombs_s_nid_postfix(result, destination, capacity, &source, limit, state.data());
}

int APS5_VABI __cxa_call_unexpected_nid_postfix(void) {
    NotImplemented_nid_no_patch(__func__);
    return 0;
}

APS5_EXPORT("0hlfW1O4Aa4", sceAmprLocaleconv);
SonyLconv* APS5_VABI sceAmprLocaleconv(void) {
    return SonyLocaleconv();
}

APS5_EXPORT("802pFCwC9w0", sceAmprUdivti3);
unsigned __int128 APS5_VABI sceAmprUdivti3(unsigned __int128 dividend, unsigned __int128 divisor) {
    return SonyUdivti3(dividend, divisor);
}

APS5_EXPORT("MU25eqxSDTw", sceAmprSinh);
double APS5_VABI sceAmprSinh(double x, double y) {
    return SonySinh(x, y);
}

APS5_EXPORT("-vXEQdRADLI", sceAmprStrtokS);
char* APS5_VABI sceAmprStrtokS(char* str, const char* delim, char** context) {
    return SonyStrtokS(str, delim, context);
}

APS5_EXPORT("2UFh+YKfuzk", sceAmprCtimeS);
int APS5_VABI sceAmprCtimeS(char* buffer, std::size_t size, const std::int64_t* time) {
    return SonyCtimeS(buffer, size, time);
}

APS5_EXPORT("6f5f-qx4ucA", sceAmprWcscpyS);
int APS5_VABI sceAmprWcscpyS(std::uint16_t* dest, std::size_t size, const std::uint16_t* src) {
    return SonyWcscpyS(dest, size, src);
}

APS5_EXPORT("ENLfKJEZTjE", sceAmprSetConstraintHandlerS);
void* APS5_VABI sceAmprSetConstraintHandlerS(void* handler) {
    return SonySetConstraintHandlerS(handler);
}

APS5_EXPORT("Jo9ON-AX9eU", sceAmprFltrounds);
int APS5_VABI sceAmprFltrounds(void) {
    return SonyFltrounds();
}

APS5_EXPORT("ODGONXcSmz4", sceAmprIgnoreHandlerS);
void APS5_VABI sceAmprIgnoreHandlerS(const char* message, void* pointer, int error) {
    SonyIgnoreHandlerS(message, pointer, error);
}

APS5_EXPORT("VUzjXknPPBs", sceAmprMbstowcs);
std::size_t APS5_VABI sceAmprMbstowcs(std::uint16_t* dest, const char* src, std::size_t size) {
    return SonyMbstowcs(dest, src, size);
}

APS5_EXPORT("qPe7-h5Jnuc", sceAmprAsctimeS);
int APS5_VABI sceAmprAsctimeS(char* buffer, std::size_t size, const GuestTm* time) {
    return SonyAsctimeS(buffer, size, time);
}

APS5_EXPORT("qlWiRfOJx1A", sceAmprFpclassifyd);
int APS5_VABI sceAmprFpclassifyd(double value) {
    return SonyFpclassifyd(value);
}

int APS5_VABI sceAmprAmmGetUsageStatsData(AmmUsageStatsData* stats) {
    if (!stats || stats->sizeInBytes > sizeof(AmmUsageStatsData)) return SCE_KERNEL_ERROR_EINVAL;
    AmmUsageStatsData out{};
    out.sizeInBytes = stats->sizeInBytes;
    out.numPageTablePoolEntries = AmmPageTablePoolEntries;
    out.ringIdleFlags = AmmRingIdleFlags;
    std::memcpy(stats, &out, static_cast<std::size_t>(stats->sizeInBytes));
    return 0;
}
}
