#include "prx/libkernel/File/include/CompressedFile.hpp"

#include <libdeflate.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace CompressedFile {

namespace {

constexpr std::uint32_t HeaderBytes = 4 + 4 + 4 + 8;
constexpr std::uint32_t RawChunk = 0x80000000u;
// TileStream: u8 id (4 = GDeflate), u8 magic (id ^ 0xff), u16 tile count, u32 tile size index (2
// bits, 1 = 64 KiB) / last tile size (18) / reserved (12).
constexpr std::uint32_t TileHeaderBytes = 8;
constexpr std::uint8_t GDeflateId = 4;

struct Decompressor {
    libdeflate_gdeflate_decompressor* handle = libdeflate_alloc_gdeflate_decompressor();
    ~Decompressor() { libdeflate_free_gdeflate_decompressor(handle); }
};

// The last chunks a thread decoded: guest reads are often smaller than a chunk and sequential.
struct DecodedChunk {
    const Index* index = nullptr;
    std::uint64_t chunk = 0;
    std::vector<std::uint8_t> bytes;
};

struct ThreadState {
    Decompressor decompressor;
    std::array<DecodedChunk, 4> recent;
    std::size_t next = 0;
    std::vector<std::uint8_t> stored;
};

ThreadState& State() {
    thread_local ThreadState state;
    return state;
}

[[noreturn]] void Corrupt(const char* why, std::uint64_t chunk) {
    throw std::runtime_error(std::string("compressed game file: ") + why + " (chunk " + std::to_string(chunk) + ")");
}

const std::vector<std::uint8_t>& Chunk(const Index& index, const RawRead& read, std::uint64_t chunk) {
    auto& state = State();
    for (const auto& entry : state.recent) {
        if (entry.index == &index && entry.chunk == chunk) return entry.bytes;
    }
    const auto expected = static_cast<std::size_t>(std::min<std::uint64_t>(index.chunkBytes, index.uncompressedBytes - chunk * index.chunkBytes));
    const auto at = index.chunkOffsets[chunk];
    std::uint32_t storedWord = 0;
    if (at + 4 > index.fileBytes || !read(at, &storedWord, 4)) Corrupt("chunk header beyond the file", chunk);
    const bool raw = (storedWord & RawChunk) != 0;
    const std::uint32_t storedBytes = storedWord & ~RawChunk;
    if (at + 4 + storedBytes > index.fileBytes) Corrupt("chunk beyond the file", chunk);
    auto& entry = state.recent[state.next];
    state.next = (state.next + 1) % state.recent.size();
    entry.index = nullptr;
    entry.bytes.resize(expected);
    if (raw) {
        if (storedBytes != expected) Corrupt("raw chunk of the wrong size", chunk);
        if (!read(at + 4, entry.bytes.data(), expected)) Corrupt("short read of a raw chunk", chunk);
    } else {
        if (storedBytes < TileHeaderBytes + 4) Corrupt("chunk smaller than a tile stream", chunk);
        state.stored.resize(storedBytes);
        if (!read(at + 4, state.stored.data(), storedBytes)) Corrupt("short read of a compressed chunk", chunk);
        const auto* stream = state.stored.data();
        const std::uint16_t tiles = static_cast<std::uint16_t>(stream[2] | (stream[3] << 8u));
        if (stream[0] != GDeflateId || stream[1] != (GDeflateId ^ 0xffu) || tiles != 1) Corrupt("not a one-tile GDeflate stream", chunk);
        std::uint32_t tileBytes = 0;
        std::memcpy(&tileBytes, stream + TileHeaderBytes, 4);
        if (tileBytes > storedBytes - TileHeaderBytes - 4) Corrupt("tile larger than its chunk", chunk);
        libdeflate_gdeflate_in_page page{stream + TileHeaderBytes + 4, tileBytes};
        std::size_t produced = 0;
        if (libdeflate_gdeflate_decompress(state.decompressor.handle, &page, 1, entry.bytes.data(), expected, &produced) != LIBDEFLATE_SUCCESS || produced != expected) Corrupt("GDeflate data does not decode to the chunk size", chunk);
    }
    entry.index = &index;
    entry.chunk = chunk;
    return entry.bytes;
}

}

bool Enabled() {
    static const bool enabled = std::getenv("APS5_NO_COMPRESSED_FILES") == nullptr;
    return enabled;
}

std::optional<std::uint64_t> UncompressedSize(const std::filesystem::path& path) {
    if (!Enabled()) return std::nullopt;
    std::ifstream in(path, std::ios::binary);
    char header[HeaderBytes] = {};
    if (!in.read(header, sizeof(header)) || std::memcmp(header, "GDEF", 4) != 0) return std::nullopt;
    std::uint64_t bytes = 0;
    std::memcpy(&bytes, header + 12, sizeof(bytes));
    return bytes;
}

void RequireOrdinary(const std::filesystem::path& path, const char* access) {
    if (UncompressedSize(path).has_value()) throw std::runtime_error(std::string(access) + ": " + path.string() + " is a GDEF container, which only APR reads decode");
}

std::shared_ptr<const Index> Probe(const std::string& key, std::uint64_t fileBytes, const RawRead& read) {
    if (!Enabled() || fileBytes < HeaderBytes) return nullptr;
    static std::mutex mutex;
    static std::unordered_map<std::string, std::shared_ptr<const Index>> cache;
    {
        std::lock_guard lock(mutex);
        if (const auto found = cache.find(key); found != cache.end() && found->second->fileBytes == fileBytes) return found->second;
    }
    std::array<std::uint8_t, HeaderBytes> header{};
    if (!read(0, header.data(), header.size()) || std::memcmp(header.data(), "GDEF", 4) != 0) return nullptr;
    auto index = std::make_shared<Index>();
    std::uint32_t chunks = 0;
    std::memcpy(&index->chunkBytes, header.data() + 4, 4);
    std::memcpy(&chunks, header.data() + 8, 4);
    std::memcpy(&index->uncompressedBytes, header.data() + 12, 8);
    index->fileBytes = fileBytes;
    const auto expectedChunks = index->chunkBytes == 0 ? 0 : (index->uncompressedBytes + index->chunkBytes - 1) / index->chunkBytes;
    if (index->chunkBytes == 0 || index->chunkBytes > (1u << 20u) || chunks != expectedChunks || HeaderBytes + std::uint64_t{chunks} * 8u > fileBytes) {
        throw std::runtime_error("compressed game file " + key + ": malformed GDEF header");
    }
    index->chunkOffsets.resize(chunks);
    if (chunks != 0 && !read(HeaderBytes, index->chunkOffsets.data(), std::size_t{chunks} * 8u)) throw std::runtime_error("compressed game file " + key + ": short chunk table");
    static const bool trace = std::getenv("APS5_TRACE_COMPRESSED_FILES") != nullptr;
    if (trace) std::fprintf(stderr, "[gdef] %s: %llu bytes in %llu on disk, %u chunks\n", key.c_str(), static_cast<unsigned long long>(index->uncompressedBytes), static_cast<unsigned long long>(fileBytes), chunks);
    std::lock_guard lock(mutex);
    return cache.insert_or_assign(key, std::move(index)).first->second;
}

void Read(const Index& index, const RawRead& read, std::uint64_t offset, void* destination, std::size_t bytes) {
    if (offset > index.uncompressedBytes || bytes > index.uncompressedBytes - offset) throw std::out_of_range("compressed game file: read beyond the end");
    auto* out = static_cast<std::uint8_t*>(destination);
    while (bytes != 0) {
        const auto chunk = offset / index.chunkBytes;
        const auto within = static_cast<std::size_t>(offset % index.chunkBytes);
        const auto& data = Chunk(index, read, chunk);
        const auto take = std::min(bytes, data.size() - within);
        std::memcpy(out, data.data() + within, take);
        out += take;
        offset += take;
        bytes -= take;
    }
}

}
