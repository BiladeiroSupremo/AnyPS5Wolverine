#include <algorithm>
#include <optional>
#include <deque>
#include <condition_variable>
#include <cstdint>
#include <cstddef>
#include "SceTypes.hpp"
#include "prx/libc/include/General.hpp"
#include "prx/libc/include/FilesystemError.hpp"
#include "prx/libc/include/PreciseWait.hpp"
#include "prx/libc/include/GuestArena.hpp"
#include "prx/libkernel/Apr/include/AprCommandBuffer.hpp"
#include "prx/libkernel/DirectMemory/DirectMemory.hpp"
#include "prx/libkernel/File/include/NativeStat.hpp"
#include "prx/libkernel/File/include/CompressedFile.hpp"
#include "prx/libkernel/Equeue/Equeue.hpp"
#include "prx/libkernel/Time/include/Time.hpp"
#include <array>
#include <atomic>
#include <chrono>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

extern "C" int* APS5_VABI __error_nid_postfix();

// The AMM submission queue's progress (defined with the queues below).
struct AprAmmQueueState {
    std::uint32_t submitted;
    std::uint32_t completed;
    std::uint32_t pending;
    bool running;
};
AprAmmQueueState AprAmmQueueStatus();

namespace {

constexpr int GUEST_ENOENT = 2;
constexpr int GUEST_EINVAL = 22;

struct AprFile {
    std::filesystem::path path;
    std::uint64_t size;
};

std::mutex& g_filesLock = *new std::mutex();
std::vector<AprFile>& g_files = *new std::vector<AprFile>();
std::unordered_map<std::string, std::uint32_t>& g_idsByPath = *new std::unordered_map<std::string, std::uint32_t>();

// File sizes by directory, listed on first use. Titles resolve thousands of package paths at startup,
// and one size query per path took over 20 s; a listing costs one directory read. Names are compared
// case-insensitively, like the host file system.
std::mutex& g_directoriesLock = *new std::mutex();
std::unordered_map<std::string, std::unordered_map<std::string, std::uint64_t>>& g_directories = *new std::unordered_map<std::string, std::unordered_map<std::string, std::uint64_t>>();

std::string _foldCase(std::string text) {
    for (auto& character : text) {
        if (character >= 'A' && character <= 'Z') character = static_cast<char>(character - 'A' + 'a');
    }
    return text;
}

// The size the guest sees: a GDEF container's uncompressed size (its header), the host size else.
std::uint64_t _guestSize(const std::filesystem::path& path, std::uint64_t hostBytes) {
    if (hostBytes < 20) return hostBytes;
    return CompressedFile::UncompressedSize(path).value_or(hostBytes);
}

bool _fileSize(const std::filesystem::path& path, std::uint64_t& bytes) {
    {
        std::lock_guard lock(g_directoriesLock);
        const auto directory = _foldCase(path.parent_path().string());
        auto listed = g_directories.find(directory);
        if (listed == g_directories.end()) {
            std::unordered_map<std::string, std::uint64_t> sizes;
#ifdef _WIN32
            // The find data carries each size; std::filesystem would query every entry again.
            WIN32_FIND_DATAW entry{};
            const HANDLE find = FindFirstFileExW((path.parent_path() / L"*").c_str(), FindExInfoBasic, &entry, FindExSearchNameMatch, nullptr, FIND_FIRST_EX_LARGE_FETCH);
            if (find != INVALID_HANDLE_VALUE) {
                do {
                    if ((entry.dwFileAttributes & (FILE_ATTRIBUTE_DIRECTORY | FILE_ATTRIBUTE_REPARSE_POINT)) != 0) continue;
                    const auto size = (static_cast<std::uint64_t>(entry.nFileSizeHigh) << 32u) | entry.nFileSizeLow;
                    sizes.emplace(_foldCase(std::filesystem::path(entry.cFileName).string()), _guestSize(path.parent_path() / entry.cFileName, size));
                } while (FindNextFileW(find, &entry));
                FindClose(find);
            }
#else
            std::error_code error;
            for (std::filesystem::directory_iterator it(path.parent_path(), error), end; !error && it != end; it.increment(error)) {
                std::error_code entryError;
                if (!it->is_regular_file(entryError)) continue;
                const auto size = it->file_size(entryError);
                if (!entryError) sizes.emplace(_foldCase(it->path().filename().string()), _guestSize(it->path(), size));
            }
#endif
            listed = g_directories.emplace(directory, std::move(sizes)).first;
        }
        if (const auto file = listed->second.find(_foldCase(path.filename().string())); file != listed->second.end()) {
            bytes = file->second;
            return true;
        }
    }
    // Not in the listing (created since, or an unusual name): ask the file system directly.
    std::error_code error;
    if (!std::filesystem::is_regular_file(path, error)) return false;
    FileStat stat{};
    File::FillFileStat(path, &stat);
    bytes = _guestSize(path, static_cast<std::uint64_t>(stat.st_size));
    return true;
}

int _fail(int guestErrno) {
    *__error_nid_postfix() = guestErrno;
    return -1;
}

bool _resolve(const char* guestPath, std::uint32_t* id, std::uint64_t* size) {
    if (!guestPath) return false;
    std::filesystem::path hostPath;
    if (ResolvePathError_nid_no_patch(guestPath, hostPath) != 0) return false;
    std::uint64_t bytes = 0;
    if (!_fileSize(hostPath, bytes)) return false;
    std::lock_guard lock(g_filesLock);
    const auto key = hostPath.string();
    auto found = g_idsByPath.find(key);
    if (found == g_idsByPath.end()) {
        found = g_idsByPath.emplace(key, static_cast<std::uint32_t>(g_files.size())).first;
        g_files.push_back({std::move(hostPath), bytes});
    }
    if (id) *id = found->second;
    if (size) *size = bytes;
    return true;
}

AprFile _file(std::uint32_t id) {
    std::lock_guard lock(g_filesLock);
    if (id >= g_files.size()) throw std::runtime_error("APR: unknown file id " + std::to_string(id));
    return g_files[id];
}

constexpr std::uint32_t INVALID_FILE_ID = 0xFFFFFFFFu;
constexpr int SCE_KERNEL_ERROR_ENOENT = static_cast<int>(0x80020002);

int _resolveForEach(const char* prefix, const char** paths, uint32_t count, uint32_t* ids, uint64_t* sizes, int* results) {
    if (!paths || !ids) return _fail(GUEST_EINVAL);
    std::uint32_t resolvedCount = 0;
    for (uint32_t index = 0; index < count; ++index) {
        const std::string path = prefix ? (paths[index] ? std::string(prefix) + paths[index] : std::string()) : (paths[index] ? std::string(paths[index]) : std::string());
        const bool resolved = !path.empty() && _resolve(path.c_str(), &ids[index], sizes ? &sizes[index] : nullptr);
        if (!resolved) {
            ids[index] = INVALID_FILE_ID;
            if (sizes) sizes[index] = 0;
        }
        if (resolved) ++resolvedCount;
        if (results) results[index] = resolved ? 0 : SCE_KERNEL_ERROR_ENOENT;
    }
    return static_cast<int>(resolvedCount);
}

// Host handles of resolved files, opened on first read and kept: titles stream thousands of small
// reads (Wolverine ~16000 at startup), and opening the file for each took ~8 ms, minutes in all.
// Reads are positional, so the queues' workers share a handle.
#ifdef _WIN32
using HostFile = HANDLE;
const HostFile NoHostFile = INVALID_HANDLE_VALUE;
#else
using HostFile = int;
constexpr HostFile NoHostFile = -1;
#endif
std::mutex& g_handlesLock = *new std::mutex();
std::vector<HostFile>& g_handles = *new std::vector<HostFile>();

HostFile _hostFile(std::uint32_t id, const AprFile& file) {
    std::lock_guard lock(g_handlesLock);
    if (id < g_handles.size() && g_handles[id] != NoHostFile) return g_handles[id];
#ifdef _WIN32
    const HostFile handle = CreateFileW(file.path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
#else
    const HostFile handle = open(file.path.c_str(), O_RDONLY | O_CLOEXEC);
#endif
    if (handle == NoHostFile) throw std::runtime_error("APR: cannot open " + file.path.string());
    if (g_handles.size() <= id) g_handles.resize(id + 1, NoHostFile);
    g_handles[id] = handle;
    return handle;
}

// Reads `bytes` at `offset` into `destination`; returns the bytes read (fewer at the end of the file).
std::uint64_t _readAt(HostFile handle, std::uint64_t offset, std::uint8_t* destination, std::uint64_t bytes) {
    std::uint64_t done = 0;
    while (done < bytes) {
        const auto chunk = static_cast<std::uint32_t>(std::min<std::uint64_t>(bytes - done, 1u << 30u));
#ifdef _WIN32
        OVERLAPPED at{};
        at.Offset = static_cast<DWORD>(offset + done);
        at.OffsetHigh = static_cast<DWORD>((offset + done) >> 32u);
        DWORD read = 0;
        if (!ReadFile(handle, destination + done, chunk, &read, &at) && GetLastError() != ERROR_HANDLE_EOF) throw std::runtime_error("APR: ReadFile failed with error " + std::to_string(GetLastError()));
#else
        const auto read = pread(handle, destination + done, chunk, static_cast<off_t>(offset + done));
        if (read < 0) throw std::runtime_error("APR: pread failed with errno " + std::to_string(errno));
#endif
        if (read == 0) break;
        done += static_cast<std::uint64_t>(read);
    }
    return done;
}

// A file's GDEF index (null for an ordinary file), probed on its first read and kept like its handle.
std::mutex& g_compressedLock = *new std::mutex();
std::vector<std::optional<std::shared_ptr<const CompressedFile::Index>>>& g_compressed = *new std::vector<std::optional<std::shared_ptr<const CompressedFile::Index>>>();

std::shared_ptr<const CompressedFile::Index> _compressed(std::uint32_t id, const AprFile& file, HostFile handle) {
    {
        std::lock_guard lock(g_compressedLock);
        if (id < g_compressed.size() && g_compressed[id].has_value()) return *g_compressed[id];
    }
#ifdef _WIN32
    LARGE_INTEGER hostSize{};
    if (!GetFileSizeEx(handle, &hostSize)) throw std::runtime_error("APR: cannot size " + file.path.string());
    const auto hostBytes = static_cast<std::uint64_t>(hostSize.QuadPart);
#else
    struct stat hostStat {};
    if (fstat(handle, &hostStat) != 0) throw std::runtime_error("APR: cannot size " + file.path.string());
    const auto hostBytes = static_cast<std::uint64_t>(hostStat.st_size);
#endif
    const CompressedFile::RawRead read = [handle](std::uint64_t offset, void* destination, std::size_t bytes) { return _readAt(handle, offset, static_cast<std::uint8_t*>(destination), bytes) == bytes; };
    auto index = CompressedFile::Probe(file.path.string(), hostBytes, read);
    std::lock_guard lock(g_compressedLock);
    if (g_compressed.size() <= id) g_compressed.resize(id + 1);
    g_compressed[id] = index;
    return index;
}

// APS5_APR_TIMING=1: every 10 s, where the reads' time went (opening the guest write, reading and
// decoding, closing the guest write).
struct ReadTiming {
    std::atomic<std::uint64_t> reads{0}, bytes{0}, openNs{0}, readNs{0}, closeNs{0};
    std::atomic<std::int64_t> lastReport{0};
};

ReadTiming* _readTiming() {
    static ReadTiming* const timing = std::getenv("APS5_APR_TIMING") != nullptr ? new ReadTiming() : nullptr;
    return timing;
}

void _reportReadTiming(ReadTiming& timing) {
    const auto now = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    auto last = timing.lastReport.load(std::memory_order_relaxed);
    if (now - last < 10 || !timing.lastReport.compare_exchange_strong(last, now)) return;
    const auto reads = timing.reads.exchange(0);
    if (reads == 0) return;
    const auto ms = [](std::atomic<std::uint64_t>& ns) { return static_cast<double>(ns.exchange(0)) / 1e6; };
    const double bytes = static_cast<double>(timing.bytes.exchange(0)) / 1048576.0;
    const double open = ms(timing.openNs), read = ms(timing.readNs), close = ms(timing.closeNs);
    std::fprintf(stderr, "[apr-timing] %llu reads, %.1f MiB: open %.0f ms, read %.0f ms, close %.0f ms (%.2f ms per read)\n", static_cast<unsigned long long>(reads), bytes, open, read, close, (open + read + close) / static_cast<double>(reads));
}

void _readFile(const Apr::ReadFileCommand& command) {
    const auto file = _file(command.fileId);
    static const bool trace = std::getenv("APS5_TRACE_APR") != nullptr;
    if (trace) std::fprintf(stderr, "[apr] read %s offset=0x%llx size=0x%llx -> 0x%llx at %.3f s\n", file.path.string().c_str(), static_cast<unsigned long long>(command.offset), static_cast<unsigned long long>(command.size), static_cast<unsigned long long>(command.destination), std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count());
    auto* timing = _readTiming();
    const auto started = std::chrono::steady_clock::now();
    auto readStarted = started;
    struct CloseTimer {
        ReadTiming* timing;
        std::chrono::steady_clock::time_point* readStarted;
        std::chrono::steady_clock::time_point readDone{};
        ~CloseTimer() {
            if (timing == nullptr || readDone == std::chrono::steady_clock::time_point{}) return;
            const auto end = std::chrono::steady_clock::now();
            timing->readNs += static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(readDone - *readStarted).count());
            timing->closeNs += static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - readDone).count());
            timing->reads += 1;
            _reportReadTiming(*timing);
        }
    } closeTimer{timing, &readStarted};
    const auto handle = _hostFile(command.fileId, file);
    const GuestArena::HostWrite destination(reinterpret_cast<void*>(command.destination), command.size);
    if (!destination.Open()) throw std::runtime_error("APR: the read destination of " + file.path.string() + " is not writable guest memory");
    if (timing != nullptr) {
        readStarted = std::chrono::steady_clock::now();
        timing->openNs += static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(readStarted - started).count());
        timing->bytes += command.size;
    }
    std::uint64_t read = 0;
    std::uint64_t fileBytes = 0;
    if (const auto compressed = _compressed(command.fileId, file, handle)) {
        fileBytes = compressed->uncompressedBytes;
        read = command.offset < fileBytes ? std::min<std::uint64_t>(command.size, fileBytes - command.offset) : 0;
        const CompressedFile::RawRead raw = [handle](std::uint64_t offset, void* target, std::size_t bytes) { return _readAt(handle, offset, static_cast<std::uint8_t*>(target), bytes) == bytes; };
        if (read != 0) CompressedFile::Read(*compressed, raw, command.offset, reinterpret_cast<void*>(command.destination), static_cast<std::size_t>(read));
    } else {
        read = _readAt(handle, command.offset, reinterpret_cast<std::uint8_t*>(command.destination), command.size);
        if (read != command.size) fileBytes = std::filesystem::file_size(file.path);
    }
    if (timing != nullptr) closeTimer.readDone = std::chrono::steady_clock::now();
    static const auto delay = [] { const char* text = std::getenv("APS5_APR_READ_DELAY_US"); return text != nullptr ? std::strtoul(text, nullptr, 10) : 0ul; }();
    if (delay != 0) PreciseSleepUs(delay);
    if (read != command.size && command.offset + read < fileBytes) throw std::runtime_error("APR: read of " + file.path.string() + " at offset " + std::to_string(command.offset) + " returned " + std::to_string(read) + " of " + std::to_string(command.size) + " bytes");
    if (read != command.size) std::memset(reinterpret_cast<char*>(command.destination) + read, 0, static_cast<std::size_t>(command.size - read));
}

void _writeAddress(const Apr::WriteAddressCommand& command) {
    if (command.flags != 0) throw std::runtime_error("APR: WriteAddress flags " + std::to_string(command.flags) + " not implemented");
    std::atomic_ref<std::uint64_t>(*reinterpret_cast<std::uint64_t*>(command.address)).store(command.value, std::memory_order_release);
}

std::array<std::atomic<std::uint32_t>, 256> g_counters{};

std::mutex& g_counterWrites = *new std::mutex();

std::uint32_t _counter(std::uint32_t index) {
    return g_counters[index % g_counters.size()].load(std::memory_order_acquire);
}

struct CounterField {
    std::uint32_t bits;
    std::uint32_t shift;
};

CounterField _counterField(Apr::CounterAccess access) {
    const auto value = static_cast<std::uint32_t>(access);
    if (value == 0) return {64, 0};
    if (value == 1) return {32, 0};
    if (value < 4) return {16, (value - 2) * 16};
    if (value < 8) return {8, (value - 4) * 8};
    throw std::runtime_error("APR: counter access " + std::to_string(value) + " not implemented");
}

std::uint64_t _fieldMask(std::uint32_t bits) {
    return bits == 64 ? ~std::uint64_t{0} : (std::uint64_t{1} << bits) - 1u;
}

std::uint64_t _readCounter(std::uint32_t index, Apr::CounterAccess access) {
    const auto field = _counterField(access);
    if (field.bits == 64) return _counter(index) | static_cast<std::uint64_t>(_counter(index + 1u)) << 32u;
    return (_counter(index) >> field.shift) & _fieldMask(field.bits);
}

std::uint64_t _applyCounterOperation(Apr::CounterOperation operation, std::uint64_t current, std::uint64_t value) {
    switch (operation) {
        case Apr::CounterOperation::Store: return value;
        case Apr::CounterOperation::AtomicOr: return current | value;
        case Apr::CounterOperation::AtomicAndComplement: return current & ~value;
        case Apr::CounterOperation::AtomicXor: return current ^ value;
        case Apr::CounterOperation::AtomicAdd: return current + value;
    }
    throw std::runtime_error("APR: counter operation " + std::to_string(static_cast<std::uint32_t>(operation)) + " not implemented");
}

void _writeCounter(const Apr::WriteCounterCommand& command) {
    const auto field = _counterField(command.access);
    const std::lock_guard lock(g_counterWrites);
    const std::uint64_t current = _readCounter(command.counter, command.access);
    const std::uint64_t next = _applyCounterOperation(command.operation, current, command.value) & _fieldMask(field.bits);
    auto& low = g_counters[command.counter % g_counters.size()];
    if (field.bits == 64) {
        g_counters[(command.counter + 1u) % g_counters.size()].store(static_cast<std::uint32_t>(next >> 32u), std::memory_order_release);
        low.store(static_cast<std::uint32_t>(next), std::memory_order_release);
        return;
    }
    const auto mask = static_cast<std::uint32_t>(_fieldMask(field.bits) << field.shift);
    const auto bits = static_cast<std::uint32_t>(next << field.shift);
    low.store((low.load(std::memory_order_relaxed) & ~mask) | bits, std::memory_order_release);
}

bool _waitSatisfied(std::uint32_t compare, std::uint64_t value, std::uint64_t reference) {
    constexpr std::uint64_t sign = std::uint64_t{1} << 63u;
    switch (compare) {
        case 0: return value == reference;
        case 1: return value > reference;
        case 2: return value < reference;
        case 3: return value != reference;
        case 4: return value - reference < sign;
        case 5: return (value ^ sign) > (reference ^ sign);
        case 6: return (value ^ sign) < (reference ^ sign);
        default: throw std::runtime_error("APR: wait compare function " + std::to_string(compare) + " not implemented");
    }
}

template<class TCommand>
TCommand _read(const Apr::CommandBufferObject& buffer, std::uint32_t cursor) {
    Apr::CommandHeader header;
    std::memcpy(&header, buffer.base + cursor, sizeof(header));
    if (header.bytes < sizeof(TCommand)) throw std::runtime_error("APR: truncated command payload (opcode " + std::to_string(static_cast<std::uint32_t>(header.opcode)) + ", has " + std::to_string(header.bytes) + ", needs " + std::to_string(sizeof(TCommand)) + ")");
    TCommand command;
    std::memcpy(&command, buffer.base + cursor, sizeof(command));
    return command;
}

// The AMM virtual address range the title manages itself (sceKernelAmmGetVirtualAddressRanges):
// APS5_AMM_RANGE_GB (default 256) for the main range, then 32 GiB for multimaps. KytyPS5 hands
// PPSA03671 512 GiB (and no multimap range). At 32 GiB the title's texture VA allocation failed
// and it streamed mips to base-0 destinations (crashed 7 of 7 runs at its slot-4 title); at 256
// none did in 3 runs (2026-10-14/15).
std::uint64_t AmmMainBytes() {
    static const std::uint64_t bytes = [] { const char* text = std::getenv("APS5_AMM_RANGE_GB"); return (text != nullptr ? std::strtoull(text, nullptr, 10) : 256ull) << 30; }();
    return bytes;
}
constexpr std::uint64_t AmmMultimapBytes = 32ull << 30;
std::uint64_t AmmTotalBytes() { return AmmMainBytes() + AmmMultimapBytes; }
constexpr std::uint64_t AmmRangeAlignment = 2ull << 20;
constexpr int GuestMapFixed = 0x10;
constexpr int GuestMapNoOverwrite = 0x80;

struct AmmPage {
    std::uint64_t physical;
    std::int32_t type;
    std::int32_t protection;
    bool pooled;
};

struct AmmState {
    std::mutex lock;
    std::uintptr_t base = 0;
    std::map<std::uint64_t, std::uint64_t> pool;
    std::map<std::uintptr_t, AmmPage> pages;
    std::map<std::uint64_t, std::uint32_t> pooledUses;
    std::set<std::uintptr_t> prt;
    std::atomic<std::uint32_t> lastSubmit{0};
};

AmmState& _amm() {
    static auto& state = *new AmmState();
    return state;
}

std::uintptr_t _ammBase(AmmState& state) {
    if (state.base == 0) {
        std::uintptr_t arenaBase = 0;
        std::size_t arenaBytes = 0;
        GuestArena::GuestArenaRange_nid_postfix(&arenaBase, &arenaBytes);
        if (arenaBase == 0 || arenaBytes < AmmTotalBytes()) throw std::runtime_error("AMM: the guest address space arena cannot hold the virtual address range");
        const auto top = GuestArena::GuestArenaHighestFree_nid_no_patch(AmmTotalBytes(), AmmRangeAlignment);
        if (top == 0) throw std::runtime_error("AMM: the guest address space arena has no free virtual address range");
        void* address = reinterpret_cast<void*>(top);
        if (DoReserveVirtual(&address, AmmTotalBytes(), GuestMapFixed | GuestMapNoOverwrite, 0) != 0) throw std::runtime_error("AMM: cannot reserve the virtual address range");
        state.base = reinterpret_cast<std::uintptr_t>(address);
    }
    return state.base;
}

void _ammCheckRange(AmmState& state, std::uint64_t address, std::uint64_t size) {
    const auto base = _ammBase(state);
    if (address < base || size > AmmTotalBytes() || address - base > AmmTotalBytes() - size) throw std::runtime_error("AMM: range outside the AMM virtual address range");
}

int _ammHostProtection(std::int32_t protection) {
    int host = protection & 0x33;
    if ((protection & 0x140) != 0) host |= 1;
    if ((protection & 0x280) != 0) host |= 3;
    if ((host & 2) != 0) host |= 1;
    return host;
}

void _returnPoolPages(AmmState& state, std::uint64_t offset, std::uint64_t bytes) {
    auto next = state.pool.lower_bound(offset);
    if (next != state.pool.end() && next->first == offset + bytes) {
        bytes += next->second;
        next = state.pool.erase(next);
    }
    if (next != state.pool.begin()) {
        const auto previous = std::prev(next);
        if (previous->first + previous->second == offset) {
            previous->second += bytes;
            return;
        }
    }
    state.pool.emplace(offset, bytes);
}

bool _tryTakePoolPages(AmmState& state, std::uint64_t bytes, std::uint64_t& offset) {
    for (auto it = state.pool.begin(); it != state.pool.end(); ++it) {
        if (it->second < bytes) continue;
        offset = it->first;
        const auto remaining = it->second - bytes;
        state.pool.erase(it);
        if (remaining != 0) state.pool.emplace(offset + bytes, remaining);
        return true;
    }
    return false;
}

std::uint64_t _takePoolPages(AmmState& state, std::uint64_t bytes) {
    std::uint64_t offset = 0;
    if (!_tryTakePoolPages(state, bytes, offset)) throw std::runtime_error("AMM: no free run of " + std::to_string(bytes) + " bytes in the direct memory given to the mapper");
    return offset;
}

// Pool pages for `bytes` taken from several runs (first fit), for a map no single free run holds:
// the PS5 maps the mapper's pages through page tables, so a fragmented pool still serves it
// (PPSA03671 aborted on an 11 MB map ~4 min into a slot-4 load with only contiguous runs).
std::vector<AmmPage> _takeScatteredPages(AmmState& state, std::uint64_t bytes, std::int32_t type, std::int32_t protection) {
    std::uint64_t free = 0;
    for (const auto& [offset, size] : state.pool) free += size;
    if (free < bytes) throw std::runtime_error("AMM: " + std::to_string(bytes) + " bytes requested, " + std::to_string(free) + " free in the direct memory given to the mapper");
    std::vector<AmmPage> pages;
    pages.reserve(bytes / PS5_PAGE_SIZE);
    while (pages.size() * PS5_PAGE_SIZE < bytes) {
        auto it = state.pool.begin();
        const auto take = std::min(it->second, bytes - pages.size() * PS5_PAGE_SIZE);
        const auto offset = it->first;
        const auto remaining = it->second - take;
        state.pool.erase(it);
        if (remaining != 0) state.pool.emplace(offset + take, remaining);
        for (std::uint64_t page = 0; page < take; page += PS5_PAGE_SIZE) pages.push_back({offset + page, type, protection, true});
    }
    return pages;
}

void _addPage(AmmState& state, std::uint64_t address, const AmmPage& page) {
    state.pages[address] = page;
    if (page.pooled) ++state.pooledUses[page.physical];
}

void _releasePage(AmmState& state, const AmmPage& page) {
    if (!page.pooled) return;
    const auto uses = state.pooledUses.find(page.physical);
    if (--uses->second == 0) {
        state.pooledUses.erase(uses);
        _returnPoolPages(state, page.physical, PS5_PAGE_SIZE);
    }
}

class AmmPageReferences {
public:
    AmmPageReferences(AmmState& state, const std::vector<AmmPage>& pages) : state(state), pages(pages) {
        for (const auto& page : pages) {
            if (page.pooled) ++state.pooledUses.find(page.physical)->second;
        }
    }

    AmmPageReferences(const AmmPageReferences&) = delete;
    AmmPageReferences& operator=(const AmmPageReferences&) = delete;

    ~AmmPageReferences() {
        for (const auto& page : pages) _releasePage(state, page);
    }

private:
    AmmState& state;
    const std::vector<AmmPage>& pages;
};

void _forgetPages(AmmState& state, std::uint64_t address, std::uint64_t size) {
    for (auto it = state.pages.lower_bound(address); it != state.pages.end() && it->first < address + size;) {
        _releasePage(state, it->second);
        it = state.pages.erase(it);
    }
}

std::vector<AmmPage> _mappedPages(AmmState& state, std::uint64_t address, std::uint64_t size) {
    std::vector<AmmPage> pages;
    for (std::uint64_t offset = 0; offset < size; offset += PS5_PAGE_SIZE) {
        const auto found = state.pages.find(address + offset);
        if (found == state.pages.end()) throw std::runtime_error("AMM: range is not fully mapped");
        pages.push_back(found->second);
    }
    return pages;
}

void _reserve(std::uint64_t address, std::uint64_t size) {
    void* target = reinterpret_cast<void*>(address);
    if (DoReserveVirtual(&target, size, GuestMapFixed, 0) != 0) throw std::runtime_error("AMM: cannot return the range to the reservation");
}

void _mapRun(std::uint64_t address, std::uint64_t physical, std::uint64_t size, std::int32_t protection) {
    void* target = reinterpret_cast<void*>(address);
    if (DoMapDirect(&target, size, _ammHostProtection(protection), GuestMapFixed, static_cast<std::int64_t>(physical), 0) != 0) throw std::runtime_error("AMM: map failed");
}

void _mapPages(AmmState& state, std::uint64_t address, const std::vector<AmmPage>& pages) {
    for (std::size_t first = 0; first < pages.size();) {
        std::size_t last = first + 1;
        while (last < pages.size() && pages[last].physical == pages[last - 1].physical + PS5_PAGE_SIZE && pages[last].protection == pages[first].protection) ++last;
        _mapRun(address + first * PS5_PAGE_SIZE, pages[first].physical, (last - first) * PS5_PAGE_SIZE, pages[first].protection);
        for (std::size_t index = first; index < last; ++index) _addPage(state, address + index * PS5_PAGE_SIZE, pages[index]);
        first = last;
    }
}

void _clearPrt(AmmState& state, std::uint64_t address, std::uint64_t size) {
    state.prt.erase(state.prt.lower_bound(address), state.prt.lower_bound(address + size));
}

void _requirePrt(AmmState& state, std::uint64_t address, std::uint64_t size) {
    for (std::uint64_t offset = 0; offset < size; offset += PS5_PAGE_SIZE) {
        if (!state.prt.contains(address + offset)) throw std::runtime_error("AMM: range is not a PRT range");
    }
}

void _zeroPrt(std::uint64_t address, std::uint64_t size) {
    constexpr int GpuRead = 0x10;
    void* target = reinterpret_cast<void*>(address);
    if (DoMapZero(&target, size, GpuRead, GuestMapFixed) != 0) throw std::runtime_error("AMM: cannot map the unbacked PRT pages");
}

void _ammUnmap(const Apr::AmmUnmapCommand& command) {
    auto& state = _amm();
    const std::lock_guard lock(state.lock);
    _ammCheckRange(state, command.address, command.size);
    _forgetPages(state, command.address, command.size);
    _clearPrt(state, command.address, command.size);
    _reserve(command.address, command.size);
}

void _ammMapAsPrt(const Apr::AmmUnmapCommand& command) {
    auto& state = _amm();
    const std::lock_guard lock(state.lock);
    _ammCheckRange(state, command.address, command.size);
    _forgetPages(state, command.address, command.size);
    _zeroPrt(command.address, command.size);
    for (std::uint64_t offset = 0; offset < command.size; offset += PS5_PAGE_SIZE) state.prt.insert(command.address + offset);
}

void _ammUnmapToPrt(const Apr::AmmUnmapCommand& command) {
    auto& state = _amm();
    const std::lock_guard lock(state.lock);
    _ammCheckRange(state, command.address, command.size);
    _requirePrt(state, command.address, command.size);
    _forgetPages(state, command.address, command.size);
    _zeroPrt(command.address, command.size);
}

void _ammAllocatePrt(const Apr::AmmProtectCommand& command) {
    auto& state = _amm();
    const std::lock_guard lock(state.lock);
    _ammCheckRange(state, command.address, command.size);
    _requirePrt(state, command.address, command.size);
    for (std::uint64_t offset = 0; offset < command.size;) {
        const auto address = command.address + offset;
        const bool backed = state.pages.contains(address);
        std::uint64_t end = offset + PS5_PAGE_SIZE;
        while (end < command.size && state.pages.contains(command.address + end) == backed) end += PS5_PAGE_SIZE;
        const auto bytes = end - offset;
        if (backed) {
            for (std::uint64_t page = 0; page < bytes; page += PS5_PAGE_SIZE) {
                auto& entry = state.pages[address + page];
                entry.type = command.type;
                entry.protection = command.protection;
            }
            if (DoMprotect(reinterpret_cast<void*>(address), bytes, _ammHostProtection(command.protection)) != 0) throw std::runtime_error("AMM: protection change failed");
        } else {
            std::vector<AmmPage> pages;
            try {
                for (std::uint64_t page = 0; page < bytes; page += PS5_PAGE_SIZE) pages.push_back({_takePoolPages(state, PS5_PAGE_SIZE), command.type, command.protection, true});
            } catch (...) {
                for (const auto& page : pages) _returnPoolPages(state, page.physical, PS5_PAGE_SIZE);
                throw;
            }
            _mapPages(state, address, pages);
        }
        offset = end;
    }
}

// APS5_APR_TIMING=1 also reports, every 10 s, how long each command opcode took in total.
struct OpcodeTiming {
    std::array<std::atomic<std::uint64_t>, 32> ns{};
    std::array<std::atomic<std::uint64_t>, 32> count{};
    std::atomic<std::int64_t> lastReport{0};
};

OpcodeTiming* _opcodeTiming() {
    static OpcodeTiming* const timing = std::getenv("APS5_APR_TIMING") != nullptr ? new OpcodeTiming() : nullptr;
    return timing;
}

void _ammMap(const Apr::AmmMapCommand& command, bool direct) {
    auto& state = _amm();
    const std::lock_guard lock(state.lock);
    _ammCheckRange(state, command.address, command.size);
    auto* timing = _opcodeTiming();
    const auto reserveStarted = timing != nullptr ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    _reserve(command.address, command.size);
    // APS5_APR_TIMING: slot 30 = the reserve, slot 31 = the map of AmmMap.
    const auto mapStarted = timing != nullptr ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    if (timing != nullptr) {
        timing->ns[30] += static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(mapStarted - reserveStarted).count());
        timing->count[30] += 1;
    }
    _forgetPages(state, command.address, command.size);
    _clearPrt(state, command.address, command.size);
    std::uint64_t physical = direct ? command.directOffset : 0;
    if (!direct && !_tryTakePoolPages(state, command.size, physical)) {
        const auto pages = _takeScatteredPages(state, command.size, command.type, command.protection);
        static std::atomic<int> reported{0};
        if (reported.fetch_add(1, std::memory_order_relaxed) < 8) std::fprintf(stderr, "[amm] no free run of %llu bytes: mapping 0x%llx from scattered pool pages\n", static_cast<unsigned long long>(command.size), static_cast<unsigned long long>(command.address));
        std::size_t mapped = 0;
        try {
            for (std::size_t first = 0; first < pages.size();) {
                std::size_t last = first + 1;
                while (last < pages.size() && pages[last].physical == pages[last - 1].physical + PS5_PAGE_SIZE) ++last;
                _mapRun(command.address + first * PS5_PAGE_SIZE, pages[first].physical, (last - first) * PS5_PAGE_SIZE, command.protection);
                for (std::size_t index = first; index < last; ++index) _addPage(state, command.address + index * PS5_PAGE_SIZE, pages[index]);
                mapped = last;
                first = last;
            }
        } catch (...) {
            _forgetPages(state, command.address, command.size);
            for (std::size_t index = mapped; index < pages.size(); ++index) _returnPoolPages(state, pages[index].physical, PS5_PAGE_SIZE);
            _reserve(command.address, command.size);
            throw;
        }
        return;
    }
    try {
        _mapRun(command.address, physical, command.size, command.protection);
        if (timing != nullptr) {
            timing->ns[31] += static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - mapStarted).count());
            timing->count[31] += 1;
        }
    } catch (...) {
        _reserve(command.address, command.size);
        if (!direct) _returnPoolPages(state, physical, command.size);
        throw;
    }
    for (std::uint64_t offset = 0; offset < command.size; offset += PS5_PAGE_SIZE) _addPage(state, command.address + offset, {physical + offset, command.type, command.protection, !direct});
}

void _ammRemap(const Apr::AmmRemapCommand& command, bool alias, bool intoPrt) {
    auto& state = _amm();
    const std::lock_guard lock(state.lock);
    _ammCheckRange(state, command.address, command.size);
    _ammCheckRange(state, command.source, command.size);
    if ((alias || intoPrt) && command.address < command.source + command.size && command.source < command.address + command.size) throw std::runtime_error("AMM: remap onto its own source range");
    if (intoPrt) _requirePrt(state, command.address, command.size);
    auto pages = _mappedPages(state, command.source, command.size);
    for (auto& page : pages) page.protection = command.protection;
    const AmmPageReferences references(state, pages);
    if (!alias) {
        _reserve(command.source, command.size);
        _forgetPages(state, command.source, command.size);
        _clearPrt(state, command.source, command.size);
    }
    _reserve(command.address, command.size);
    _forgetPages(state, command.address, command.size);
    if (!intoPrt) _clearPrt(state, command.address, command.size);
    _mapPages(state, command.address, pages);
}

void _ammProtect(const Apr::AmmProtectCommand& command, bool type) {
    auto& state = _amm();
    const std::lock_guard lock(state.lock);
    _ammCheckRange(state, command.address, command.size);
    auto pages = _mappedPages(state, command.address, command.size);
    for (std::size_t index = 0; index < pages.size(); ++index) {
        auto& page = pages[index];
        page.protection = (page.protection & ~command.mask) | (command.protection & command.mask);
        if (type) page.type = command.type;
        state.pages[command.address + index * PS5_PAGE_SIZE] = page;
    }
    for (std::size_t first = 0; first < pages.size();) {
        const int host = _ammHostProtection(pages[first].protection);
        std::size_t last = first + 1;
        while (last < pages.size() && _ammHostProtection(pages[last].protection) == host) ++last;
        if (DoMprotect(reinterpret_cast<void*>(command.address + first * PS5_PAGE_SIZE), (last - first) * PS5_PAGE_SIZE, host) != 0) throw std::runtime_error("AMM: protection change failed");
        first = last;
    }
}

struct ReadCursor {
    bool valid = false;
    std::uint32_t fileId = 0;
    std::uint64_t nextDestination = 0;
    std::uint64_t nextOffset = 0;
};

bool _ammRangeMapped(AmmState& state, std::uint64_t address, std::uint64_t size) {
    for (auto page = address & ~(std::uint64_t{PS5_PAGE_SIZE} - 1u); page < address + size; page += PS5_PAGE_SIZE) {
        if (state.pages.find(page) == state.pages.end()) return false;
    }
    return true;
}

// A read into AMM pages that are not mapped yet: AMM buffers run on their own queue and can sit
// behind a deferred wait, while a generic buffer runs at once on the submitting thread, so a read
// could overtake the mapping it reads into (2026-10-14: 'Archive Asset IO' wrote to unmapped AMM
// pages). Waits for the AMM queue to map them, for up to APS5_APR_AMM_WAIT_MS (2000).
void _awaitAmmMapping(const Apr::ReadFileCommand& command) {
    auto& state = _amm();
    {
        const std::lock_guard lock(state.lock);
        if (state.base == 0 || command.destination < state.base || command.destination - state.base >= AmmTotalBytes()) return;
        if (_ammRangeMapped(state, command.destination, command.size)) return;
    }
    static const auto limitMs = [] { const char* text = std::getenv("APS5_APR_AMM_WAIT_MS"); return text != nullptr ? std::strtoul(text, nullptr, 10) : 2000ul; }();
    const auto before = AprAmmQueueStatus();
    const auto started = std::chrono::steady_clock::now();
    bool mapped = false;
    while (std::chrono::steady_clock::now() - started < std::chrono::milliseconds(limitMs)) {
        PreciseSleepUs(100);
        const std::lock_guard lock(state.lock);
        if (_ammRangeMapped(state, command.destination, command.size)) {
            mapped = true;
            break;
        }
    }
    const auto after = AprAmmQueueStatus();
    const auto waitedMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    std::fprintf(stderr, "[apr] read into unmapped AMM pages 0x%llx+0x%llx: %s after %.1f ms; AMM queue before %u/%u submitted/completed, %u pending, %s; after %u/%u, %u pending\n", static_cast<unsigned long long>(command.destination), static_cast<unsigned long long>(command.size), mapped ? "mapped" : "STILL UNMAPPED", waitedMs, before.submitted, before.completed, before.pending, before.running ? "running" : "idle", after.submitted, after.completed, after.pending);
}

// Whether [address, address + bytes) is committed memory (its protection may still be read-only
// for write tracking; the read opens the guest write itself).
bool _committed(std::uint64_t address, std::uint64_t bytes) {
#ifdef _WIN32
    for (auto cursor = address; cursor < address + bytes;) {
        MEMORY_BASIC_INFORMATION region{};
        if (VirtualQuery(reinterpret_cast<const void*>(cursor), &region, sizeof(region)) != sizeof(region) || region.State != MEM_COMMIT) return false;
        cursor = reinterpret_cast<std::uint64_t>(region.BaseAddress) + region.RegionSize;
    }
#else
    (void)address;
    (void)bytes;
#endif
    return true;
}

// The command buffer being executed, for the bad-destination report below.
thread_local const Apr::CommandBufferObject* g_executing = nullptr;
thread_local std::uint32_t g_executingCursor = 0;

void _readResolved(Apr::Opcode opcode, Apr::ReadFileCommand command, ReadCursor& read) {
    if (opcode != Apr::Opcode::ReadFile) {
        if (!read.valid) throw std::runtime_error("APR: gather or scatter read without a preceding read");
        command.fileId = read.fileId;
        if (opcode == Apr::Opcode::ReadFileGather) command.destination = read.nextDestination;
        if (opcode == Apr::Opcode::ReadFileScatter) command.offset = read.nextOffset;
    }
    // A destination that is not committed memory: the title sometimes submits reads into base-0
    // texture addresses (0x4000, 0x8000, 0x1560000..., 2026-10-14, with a save loaded). A read
    // into AMM pages first waits for their mapping. Report the command and its buffer.
    bool bad = command.destination < 0x10000u;
    if (!bad) {
        _awaitAmmMapping(command);
        bad = !_committed(command.destination, command.size);
    }
    static std::atomic<std::uint32_t> badReads{0};
    if (bad && badReads.fetch_add(1) >= 64u) {
        static const bool skipQuietly = std::getenv("APS5_APR_SKIP_BAD_DESTINATION") != nullptr;
        if (skipQuietly) {
            read = {true, command.fileId, command.destination + command.size, command.offset + command.size};
            return;
        }
    }
    if (bad) {
        std::fprintf(stderr, "[apr] read into 0x%llx: opcode %u file %u (%s) size 0x%llx offset 0x%llx; cursor valid %d file %u next destination 0x%llx next offset 0x%llx\n", static_cast<unsigned long long>(command.destination), static_cast<unsigned>(opcode), command.fileId, _file(command.fileId).path.string().c_str(), static_cast<unsigned long long>(command.size), static_cast<unsigned long long>(command.offset), read.valid ? 1 : 0, read.fileId, static_cast<unsigned long long>(read.nextDestination), static_cast<unsigned long long>(read.nextOffset));
        {
            // How full the AMM is: pooled physical memory still free, pages mapped.
            auto& state = _amm();
            const std::lock_guard lock(state.lock);
            std::uint64_t poolFree = 0, largest = 0;
            for (const auto& [offset, bytes] : state.pool) {
                poolFree += bytes;
                largest = std::max(largest, bytes);
            }
            std::size_t pooledPages = 0;
            for (const auto& [address, page] : state.pages) pooledPages += page.pooled ? 1u : 0u;
            std::fprintf(stderr, "[apr]   AMM pool: %.1f MiB free in %zu runs (largest %.1f MiB); %zu pages mapped, %zu of them pooled (%.1f MiB)\n", static_cast<double>(poolFree) / 1048576.0, state.pool.size(), static_cast<double>(largest) / 1048576.0, state.pages.size(), pooledPages, static_cast<double>(pooledPages) * PS5_PAGE_SIZE / 1048576.0);
        }
        // APS5_APR_SKIP_BAD_DESTINATION=1: drop the read and go on (diagnostic).
        static const bool skip = std::getenv("APS5_APR_SKIP_BAD_DESTINATION") != nullptr;
        if (skip) {
            read = {true, command.fileId, command.destination + command.size, command.offset + command.size};
            return;
        }
        if (g_executing != nullptr) {
            const auto from = g_executingCursor > 256u ? g_executingCursor - 256u : 0u;
            const auto to = std::min<std::uint32_t>(g_executing->offset, g_executingCursor + 128u);
            std::fprintf(stderr, "[apr] buffer %p of 0x%x bytes, command at 0x%x; bytes 0x%x-0x%x:", static_cast<const void*>(g_executing->base), g_executing->offset, g_executingCursor, from, to);
            for (auto at = from; at < to; ++at) std::fprintf(stderr, "%s%02x", (at - from) % 32u == 0u ? "\n[apr]   " : " ", g_executing->base[at]);
            std::fprintf(stderr, "\n");
        }
        std::fflush(stderr);
        throw std::runtime_error("APR: read destination 0x" + std::to_string(command.destination) + " is not guest memory");
    }
    _readFile(command);
    read = {true, command.fileId, command.destination + command.size, command.offset + command.size};
}

struct Deferred {
    std::uint32_t cursor = 0;
    ReadCursor read;
};

void _noteOpcode(OpcodeTiming& timing, std::uint32_t opcode, std::chrono::steady_clock::time_point started) {
    const auto now = std::chrono::steady_clock::now();
    timing.ns[opcode & 31u] += static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now - started).count());
    timing.count[opcode & 31u] += 1;
    const auto seconds = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();
    auto last = timing.lastReport.load(std::memory_order_relaxed);
    if (seconds - last < 10 || !timing.lastReport.compare_exchange_strong(last, seconds)) return;
    std::string text;
    for (std::uint32_t index = 0; index < 32u; ++index) {
        const auto count = timing.count[index].exchange(0);
        const auto ns = timing.ns[index].exchange(0);
        if (count == 0) continue;
        char entry[96];
        std::snprintf(entry, sizeof(entry), " op%u: %llu in %.0f ms (%.3f ms each);", index, static_cast<unsigned long long>(count), static_cast<double>(ns) / 1e6, static_cast<double>(ns) / 1e6 / static_cast<double>(count));
        text += entry;
    }
    std::fprintf(stderr, "[apr-opcodes] 10 s:%s\n", text.c_str());
}

std::optional<Deferred> _execute(const Apr::CommandBufferObject& buffer, std::uint32_t start = 0, ReadCursor read = {}, bool defer = false) {
    std::uint32_t cursor = start;
    g_executing = &buffer;
    auto* opcodeTiming = _opcodeTiming();
    while (cursor < buffer.offset) {
        g_executingCursor = cursor;
        const auto opcodeStarted = opcodeTiming != nullptr ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        if (cursor + sizeof(Apr::CommandHeader) > buffer.offset) throw std::runtime_error("APR: truncated command buffer");
        Apr::CommandHeader header;
        std::memcpy(&header, buffer.base + cursor, sizeof(header));
        if ((static_cast<std::uint32_t>(header.opcode) & 0xffu) == static_cast<std::uint32_t>(Apr::Opcode::CompactReadFileGather)) {
            if (sizeof(Apr::CompactReadFileGatherCommand) > buffer.offset - cursor) throw std::runtime_error("APR: truncated compact gather command");
            Apr::CompactReadFileGatherCommand command;
            std::memcpy(&command, buffer.base + cursor, sizeof(command));
            const auto offset = static_cast<std::uint64_t>(command.opcodeAndOffsetHigh >> 8u) << 32u | command.offsetLow;
            if (offset >= 0x10000000000ull) throw std::runtime_error("APR: malformed compact gather offset");
            _readResolved(Apr::Opcode::ReadFileGather, Apr::ReadFileCommand{{}, 0, 0, 0, static_cast<std::uint64_t>(command.sizeMinusOne) + 1u, offset}, read);
            cursor += sizeof(command);
            continue;
        }
        if (header.bytes < sizeof(header) || header.bytes > buffer.offset - cursor) throw std::runtime_error("APR: malformed command");
        switch (header.opcode) {
        case Apr::Opcode::Nop:
        case Apr::Opcode::PushMarker:
        case Apr::Opcode::PopMarker:
        case Apr::Opcode::SetMarker:
            break;
        case Apr::Opcode::ReadFile:
        case Apr::Opcode::ReadFileGather:
        case Apr::Opcode::ReadFileScatter:
        case Apr::Opcode::ReadFileGatherScatter:
            _readResolved(header.opcode, _read<Apr::ReadFileCommand>(buffer, cursor), read);
            break;
        case Apr::Opcode::ResetGatherScatterState:
            read = {};
            break;
        case Apr::Opcode::AmmMap:
        case Apr::Opcode::AmmMapDirect:
            _ammMap(_read<Apr::AmmMapCommand>(buffer, cursor), header.opcode == Apr::Opcode::AmmMapDirect);
            break;
        case Apr::Opcode::AmmUnmap:
            _ammUnmap(_read<Apr::AmmUnmapCommand>(buffer, cursor));
            break;
        case Apr::Opcode::AmmRemap:
        case Apr::Opcode::AmmMultiMap:
            _ammRemap(_read<Apr::AmmRemapCommand>(buffer, cursor), header.opcode == Apr::Opcode::AmmMultiMap, false);
            break;
        case Apr::Opcode::AmmRemapIntoPrt:
            _ammRemap(_read<Apr::AmmRemapCommand>(buffer, cursor), false, true);
            break;
        case Apr::Opcode::AmmMapAsPrt:
            _ammMapAsPrt(_read<Apr::AmmUnmapCommand>(buffer, cursor));
            break;
        case Apr::Opcode::AmmUnmapToPrt:
            _ammUnmapToPrt(_read<Apr::AmmUnmapCommand>(buffer, cursor));
            break;
        case Apr::Opcode::AmmAllocatePaForPrt:
            _ammAllocatePrt(_read<Apr::AmmProtectCommand>(buffer, cursor));
            break;
        case Apr::Opcode::AmmModifyProtect:
        case Apr::Opcode::AmmModifyMtypeProtect:
            _ammProtect(_read<Apr::AmmProtectCommand>(buffer, cursor), header.opcode == Apr::Opcode::AmmModifyMtypeProtect);
            break;
        case Apr::Opcode::MapEnd:
            break;
        case Apr::Opcode::WriteAddress: {
            const auto command = _read<Apr::WriteAddressCommand>(buffer, cursor);
            // APS5_TRACE_APR_WRITES=1: every WriteAddress, with the executing thread.
            static const bool traceWrites = std::getenv("APS5_TRACE_APR_WRITES") != nullptr;
            if (traceWrites) std::fprintf(stderr, "[apr-write] 0x%llx <- 0x%llx (thread %zx at %.4f s)\n", static_cast<unsigned long long>(command.address), static_cast<unsigned long long>(command.value), std::hash<std::thread::id>{}(std::this_thread::get_id()) & 0xffffu, std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count());
            _writeAddress(command);
            break;
        }
        case Apr::Opcode::WriteCounter: {
            _writeCounter(_read<Apr::WriteCounterCommand>(buffer, cursor));
            break;
        }
        case Apr::Opcode::WaitOnAddress:
        case Apr::Opcode::WaitOnCounter: {
            const auto command = _read<Apr::WaitCommand>(buffer, cursor);
            const bool counter = header.opcode == Apr::Opcode::WaitOnCounter;
            const std::uint32_t unused = counter ? 64u - _counterField(command.access).bits : 0u;
            const auto current = [&]() -> std::uint64_t {
                if (counter) return _readCounter(command.counter, command.access);
                return std::atomic_ref<std::uint64_t>(*reinterpret_cast<std::uint64_t*>(command.address)).load(std::memory_order_acquire);
            };
            const std::uint64_t reference = (command.reference & command.mask) << unused;
            if (defer && !_waitSatisfied(command.compare, (current() & command.mask) << unused, reference)) return Deferred{cursor, read};
            while (!_waitSatisfied(command.compare, (current() & command.mask) << unused, reference)) PreciseSleepUs(50);
            break;
        }
        case Apr::Opcode::WriteKernelEventQueue: {
            const auto command = _read<Apr::WriteKernelEventQueueCommand>(buffer, cursor);
            const int result = EqueueTriggerEvent_nid_postfix(static_cast<KernelEqueue>(command.equeue), static_cast<uintptr_t>(command.ident), EVFILT_AMPR, reinterpret_cast<void*>(command.data));
            static const bool trace = std::getenv("APS5_TRACE_APR") != nullptr;
            if (result != 0 || trace) std::fprintf(stderr, "[apr] kernel event queue %p ident %llu data 0x%llx -> 0x%x\n", reinterpret_cast<void*>(command.equeue), static_cast<unsigned long long>(command.ident), static_cast<unsigned long long>(command.data), static_cast<unsigned>(result));
            break;
        }
        case Apr::Opcode::WriteAddressFromTimeCounter: {
            const auto command = _read<Apr::WriteAddressFromCounterCommand>(buffer, cursor);
            std::atomic_ref<std::uint64_t>(*reinterpret_cast<std::uint64_t*>(command.address)).store(sceKernelGetProcessTimeCounter(), std::memory_order_release);
            break;
        }
        case Apr::Opcode::WriteAddressFromCounter:
        case Apr::Opcode::WriteAddressFromCounterPair: {
            const auto command = _read<Apr::WriteAddressFromCounterCommand>(buffer, cursor);
            std::uint64_t value = _counter(command.counter0);
            if (header.opcode == Apr::Opcode::WriteAddressFromCounterPair) value |= static_cast<std::uint64_t>(_counter(command.counter1)) << 32u;
            std::atomic_ref<std::uint64_t>(*reinterpret_cast<std::uint64_t*>(command.address)).store(value, std::memory_order_release);
            break;
        }
        default:
            throw std::runtime_error("APR: unknown opcode " + std::to_string(static_cast<std::uint32_t>(header.opcode)));
        }
        if (opcodeTiming != nullptr) _noteOpcode(*opcodeTiming, static_cast<std::uint32_t>(header.opcode), opcodeStarted);
        // Experiment: APS5_APR_AMM_DELAY_US=<us> sleeps after every AMM map/unmap/remap command.
        static const auto ammDelayUs = [] { const char* text = std::getenv("APS5_APR_AMM_DELAY_US"); return text != nullptr ? std::strtoul(text, nullptr, 10) : 0ul; }();
        if (ammDelayUs != 0 && static_cast<std::uint32_t>(header.opcode) >= 17u && static_cast<std::uint32_t>(header.opcode) <= 28u) PreciseSleepUs(ammDelayUs);
        cursor += header.bytes;
    }
    return std::nullopt;
}

}

extern "C" {

int APS5_VABI sceKernelAllocateDirectMemory(int64_t searchStart, int64_t searchEnd, size_t length, size_t alignment, int memoryType, int64_t* offset);

int AmmGiveDirectMemory_nid_no_patch(std::int64_t searchStart, std::int64_t searchEnd, std::size_t size, std::size_t alignment, int usage, std::int64_t* offset) {
    if (!offset || (usage != 0 && usage != 1)) return SCE_KERNEL_ERROR_EINVAL;
    const int result = sceKernelAllocateDirectMemory(searchStart, searchEnd, size, alignment, 0, offset);
    std::fprintf(stderr, "[amm] give direct memory 0x%zx bytes (%.1f MiB) usage %d -> 0x%x offset 0x%llx\n", size, static_cast<double>(size) / 1048576.0, usage, static_cast<unsigned>(result), static_cast<unsigned long long>(result == 0 ? *offset : 0));
    if (result != 0 || usage == 0) return result;
    auto& state = _amm();
    const std::lock_guard lock(state.lock);
    _returnPoolPages(state, static_cast<std::uint64_t>(*offset), size);
    return 0;
}

void AmmVirtualAddressRanges_nid_no_patch(std::uint64_t* start, std::uint64_t* end, std::uint64_t* multimapStart, std::uint64_t* multimapEnd) {
    auto& state = _amm();
    const std::lock_guard lock(state.lock);
    const auto base = _ammBase(state);
    if (start) *start = base;
    if (end) *end = base + AmmMainBytes();
    if (multimapStart) *multimapStart = base + AmmMainBytes();
    if (multimapEnd) *multimapEnd = base + AmmTotalBytes();
}

struct AprQueue {
    std::mutex submit;
    std::mutex mutex;
    std::condition_variable ready;
    std::condition_variable done;
    struct Work {
        std::vector<std::uint8_t> commands;
        Deferred resume;
        std::uint32_t id = 0;
    };
    std::deque<Work> pending;
    bool running = false;
    std::uint32_t submitted = 0;
    std::uint32_t completed = 0;
};

AprQueue* _startQueue(bool amm) {
    auto* created = new AprQueue();
    std::thread([created, amm] {
#ifdef _WIN32
        SetThreadDescription(GetCurrentThread(), amm ? L"APR queue (AMM)" : L"APR queue (generic)");
#endif
        for (;;) {
            AprQueue::Work work;
            {
                std::unique_lock lock(created->mutex);
                created->ready.wait(lock, [&] { return !created->pending.empty(); });
                work = std::move(created->pending.front());
                created->pending.pop_front();
                created->running = true;
            }
            try {
                const auto bytes = static_cast<std::uint32_t>(work.commands.size());
                _execute(Apr::CommandBufferObject{Apr::BufferType::Generic, 0, bytes, 0, bytes, work.commands.data()}, work.resume.cursor, work.resume.read);
            } catch (const std::exception& error) {
                std::fprintf(stderr, "APR: %s\n", error.what());
                std::fflush(stderr);
                std::abort();
            }
            {
                std::lock_guard lock(created->mutex);
                created->running = false;
                created->completed = std::max(created->completed, work.id);
            }
            created->done.notify_all();
        }
    }).detach();
    return created;
}

AprQueue& _queue(bool amm) {
    static AprQueue* const queues[2] = {_startQueue(false), _startQueue(true)};
    return *queues[amm ? 1 : 0];
}

std::uint32_t _submit(const std::uint8_t* base, std::uint32_t bytes, bool amm) {
    // APS5_TRACE_APR_OPS=1: the opcodes of each submitted buffer, with the submitting thread.
    static const bool traceOps = std::getenv("APS5_TRACE_APR_OPS") != nullptr;
    if (traceOps) {
        std::string ops;
        for (std::uint32_t cursor = 0; cursor + sizeof(Apr::CommandHeader) <= bytes;) {
            Apr::CommandHeader header;
            std::memcpy(&header, base + cursor, sizeof(header));
            const auto opcode = static_cast<std::uint32_t>(header.opcode) & 0xffu;
            ops += ' ' + std::to_string(opcode);
            if (opcode == static_cast<std::uint32_t>(Apr::Opcode::CompactReadFileGather)) cursor += sizeof(Apr::CompactReadFileGatherCommand);
            else if (header.bytes < sizeof(header)) break;
            else cursor += header.bytes;
        }
        std::fprintf(stderr, "[apr-ops] %s thread %zx at %.4f s:%s\n", amm ? "amm" : "generic", std::hash<std::thread::id>{}(std::this_thread::get_id()) & 0xffffu, std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(), ops.c_str());
    }
    auto& queue = _queue(amm);
    std::lock_guard submitLock(queue.submit);
    std::uint32_t id;
    bool queued;
    {
        std::lock_guard lock(queue.mutex);
        id = ++queue.submitted;
        queued = queue.running || !queue.pending.empty();
    }
    const auto enqueue = [&](Deferred resume) {
        {
            std::lock_guard lock(queue.mutex);
            queue.pending.push_back({std::vector<std::uint8_t>(base, base + bytes), resume, id});
        }
        queue.ready.notify_one();
    };
    static const bool traceQueue = std::getenv("APS5_TRACE_APR_WRITES") != nullptr;
    if (queued) {
        if (traceQueue) std::fprintf(stderr, "[apr-queue] %s id %u queued behind the running buffer at %.4f s\n", amm ? "amm" : "generic", id, std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count());
        enqueue({});
        return id;
    }
    if (const auto deferred = _execute(Apr::CommandBufferObject{Apr::BufferType::Generic, 0, bytes, 0, bytes, const_cast<std::uint8_t*>(base)}, 0, {}, true)) {
        if (traceQueue) std::fprintf(stderr, "[apr-queue] %s id %u deferred at byte 0x%x (an unmet wait) at %.4f s\n", amm ? "amm" : "generic", id, deferred->cursor, std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count());
        enqueue(*deferred);
        return id;
    }
    {
        std::lock_guard lock(queue.mutex);
        queue.completed = std::max(queue.completed, id);
    }
    queue.done.notify_all();
    return id;
}

void _waitSubmitted(std::uint32_t id, bool amm) {
    auto& queue = _queue(amm);
    std::unique_lock lock(queue.mutex);
    queue.done.wait(lock, [&] { return queue.completed >= id || id > queue.submitted; });
}

std::uint32_t AmmSubmit_nid_no_patch(void* base, std::uint32_t bytes) {
    const auto id = _submit(static_cast<const std::uint8_t*>(base), bytes, true);
    _amm().lastSubmit.store(id);
    return id;
}

bool AmmSubmitted_nid_no_patch(std::uint32_t id) {
    if (id == 0 || id > _amm().lastSubmit.load()) return false;
    _waitSubmitted(id, true);
    return true;
}


int APS5_VABI sceKernelAprResolveFilepathsToIds(const char** paths, uint32_t count, uint32_t* ids, uint32_t* error_index) {
    if (!paths || !ids) return _fail(GUEST_EINVAL);
    for (uint32_t index = 0; index < count; ++index) {
        if (!_resolve(paths[index], &ids[index], nullptr)) {
            ids[index] = INVALID_FILE_ID;
            if (error_index) *error_index = index;
            return _fail(GUEST_ENOENT);
        }
    }
    return 0;
}

int APS5_VABI sceKernelAprResolveFilepathsToIdsAndFileSizes(const char** paths, uint32_t count, uint32_t* ids, uint64_t* sizes, uint32_t* error_index) {
    if (!paths || !ids || !sizes) return _fail(GUEST_EINVAL);
    for (uint32_t index = 0; index < count; ++index) {
        if (!_resolve(paths[index], &ids[index], &sizes[index])) {
            ids[index] = INVALID_FILE_ID;
            if (sizes) sizes[index] = 0;
            if (error_index) *error_index = index;
            return _fail(GUEST_ENOENT);
        }
    }
    return 0;
}

int APS5_VABI sceKernelAprResolveFilepathsWithPrefixToIdsAndFileSizes(const char* prefix, const char** paths, uint32_t count, uint32_t* ids, uint64_t* sizes, uint32_t* error_index) {
    if (!prefix || !paths || !ids) return _fail(GUEST_EINVAL);
    for (uint32_t index = 0; index < count; ++index) {
        const std::string path = paths[index] ? std::string(prefix) + paths[index] : std::string();
        if (path.empty() || !_resolve(path.c_str(), &ids[index], sizes ? &sizes[index] : nullptr)) {
            ids[index] = INVALID_FILE_ID;
            if (sizes) sizes[index] = 0;
            if (error_index) *error_index = index;
            return _fail(GUEST_ENOENT);
        }
    }
    return 0;
}

int APS5_VABI sceKernelAprResolveFilepathsWithPrefixToIds(const char* prefix, const char** paths, uint32_t count, uint32_t* ids, uint32_t* error_index) {
    return sceKernelAprResolveFilepathsWithPrefixToIdsAndFileSizes(prefix, paths, count, ids, nullptr, error_index);
}

int APS5_VABI sceKernelAprGetFileSize(uint32_t id, uint64_t* size) {
    if (!size) return _fail(GUEST_EINVAL);
    *size = _file(id).size;
    return 0;
}

int APS5_VABI sceKernelAprGetFileStat(uint32_t id, FileStat* stat) {
    if (!stat) return _fail(GUEST_EINVAL);
    const auto file = _file(id);
    File::FillFileStat(file.path, stat);
    stat->st_size = static_cast<decltype(stat->st_size)>(file.size);
    return 0;
}

int APS5_VABI sceKernelAprSubmitCommandBuffer(const Apr::CommandBufferObject* buffer, uint32_t priority) {
    (void)priority;
    if (!buffer) return _fail(GUEST_EINVAL);
    _submit(buffer->base, buffer->offset, false);
    return 0;
}

int APS5_VABI sceKernelAprSubmitCommandBufferAndGetId(const Apr::CommandBufferObject* buffer, uint32_t priority, uint32_t* id) {
    if (!id) return _fail(GUEST_EINVAL);
    (void)priority;
    if (!buffer) return _fail(GUEST_EINVAL);
    *id = _submit(buffer->base, buffer->offset, false);
    return 0;
}

int APS5_VABI sceKernelAprSubmitCommandBufferAndGetResult(const Apr::CommandBufferObject* buffer, uint32_t priority, uint32_t* result, uint32_t* id) {
    if (!result) return _fail(GUEST_EINVAL);
    const int submitted = sceKernelAprSubmitCommandBufferAndGetId(buffer, priority, id);
    if (submitted != 0) return submitted;
    *result = 0;
    return 0;
}

int APS5_VABI sceKernelAprWaitCommandBuffer(uint32_t id) {
    _waitSubmitted(id, false);
    return 0;
}

int APS5_VABI sceKernelAprResolveFilepathsToIdsForEach(const char** paths, uint32_t count, uint32_t* ids, int* results) {
    return _resolveForEach(nullptr, paths, count, ids, nullptr, results);
}

int APS5_VABI sceKernelAprResolveFilepathsToIdsAndFileSizesForEach(const char** paths, uint32_t count, uint32_t* ids, uint64_t* sizes, int* results) {
    if (!sizes) return _fail(GUEST_EINVAL);
    return _resolveForEach(nullptr, paths, count, ids, sizes, results);
}

int APS5_VABI sceKernelAprResolveFilepathsWithPrefixToIdsForEach(const char* prefix, const char** paths, uint32_t count, uint32_t* ids, int* results) {
    if (!prefix) return _fail(GUEST_EINVAL);
    return _resolveForEach(prefix, paths, count, ids, nullptr, results);
}

int APS5_VABI sceKernelAprResolveFilepathsWithPrefixToIdsAndFileSizesForEach(const char* prefix, const char** paths, uint32_t count, uint32_t* ids, uint64_t* sizes, int* results) {
    if (!prefix || !sizes) return _fail(GUEST_EINVAL);
    return _resolveForEach(prefix, paths, count, ids, sizes, results);
}

bool _validWriteRange(std::uint64_t address, std::uint64_t size) {
    constexpr std::uint64_t pageMask = 0x3FFFu;
    return address != 0u && size != 0u && ((address | size) & pageMask) == 0u && address + size > address;
}

bool _validWriteProtection(std::uint64_t protection) {
    return (protection & ~0x3F3ULL) == 0u;
}

int APS5_VABI sceKernelWriteModifyMtypeProtectCommand(void* dst, std::uint64_t address, std::uint64_t size, std::uint64_t type, std::uint64_t protection, std::uint64_t mask, std::uint64_t* outSize) {
    if (!_validWriteRange(address, size) || !_validWriteProtection(protection) || !_validWriteProtection(mask)) return SCE_KERNEL_ERROR_EINVAL;
    const Apr::AmmProtectCommand command{{Apr::Opcode::AmmModifyMtypeProtect, sizeof(command)}, address, size, static_cast<std::int32_t>(type), static_cast<std::int32_t>(protection), static_cast<std::int32_t>(mask), 0};
    if (dst) std::memcpy(dst, &command, sizeof(command));
    if (outSize) *outSize = sizeof(command);
    return 0;
}

int APS5_VABI sceKernelWriteRemapIntoPrtCommand(std::uint32_t* dst, std::uint64_t address, std::uint64_t source, std::uint64_t size, int protection, int opcode, std::uint64_t* outSize) {
    if (!_validWriteRange(address, size) || !_validWriteRange(source, size) || !_validWriteProtection(static_cast<std::uint32_t>(protection))) return SCE_KERNEL_ERROR_EINVAL;
    const Apr::AmmRemapCommand command{{Apr::Opcode::AmmRemapIntoPrt, sizeof(command)}, address, source, size, protection, opcode ? opcode : 1011};
    if (dst) std::memcpy(dst, &command, sizeof(command));
    if (outSize) *outSize = sizeof(command);
    return 0;
}

}

AprAmmQueueState AprAmmQueueStatus() {
    auto& queue = _queue(true);
    std::lock_guard lock(queue.mutex);
    return {queue.submitted, queue.completed, static_cast<std::uint32_t>(queue.pending.size()), queue.running};
}
