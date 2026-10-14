#ifndef APS5_LIBKERNEL_COMPRESSED_FILE_HPP
#define APS5_LIBKERNEL_COMPRESSED_FILE_HPP

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <memory>
#include <string>
#include <vector>

// Game data files stored as GDEF containers (the SM2 GDeflate mod's format, made by its
// GDeflateCompressor): "GDEF", u32 chunk size (65536), u32 chunk count, u64 uncompressed size,
// u64 file offset of every chunk; each chunk is a u32 stored size (bit 31: stored raw) and the
// payload, a one-tile GDeflate stream (8-byte TileStream header, u32 tile size, tile data). The
// guest sees the original file: its size and its bytes at every offset.
namespace CompressedFile {

struct Index {
    std::uint64_t uncompressedBytes = 0;
    std::uint64_t fileBytes = 0;
    std::uint32_t chunkBytes = 0;
    std::vector<std::uint64_t> chunkOffsets;
};

// Reads exactly `bytes` bytes of the host file at `offset`; false on a short or failed read.
using RawRead = std::function<bool(std::uint64_t offset, void* destination, std::size_t bytes)>;

// The container's index, or null for an ordinary file. Throws on a malformed container. The
// index is cached by `key` (the host path): containers are immutable while the game runs.
std::shared_ptr<const Index> Probe(const std::string& key, std::uint64_t fileBytes, const RawRead& read);

// Copies the uncompressed bytes [offset, offset + bytes) into `destination`; the range must lie
// inside the uncompressed file (callers clamp at the end of the file). Throws on corrupt data.
void Read(const Index& index, const RawRead& read, std::uint64_t offset, void* destination, std::size_t bytes);

// APS5_NO_COMPRESSED_FILES=1 treats every file as ordinary.
bool Enabled();

// The uncompressed size of the host file at `path` when it is a GDEF container (reads its header).
std::optional<std::uint64_t> UncompressedSize(const std::filesystem::path& path);

// Throws when `path` is a GDEF container: for the access paths that do not decode them (only APR
// reads do), so a title reading one another way fails loudly instead of seeing compressed bytes.
void RequireOrdinary(const std::filesystem::path& path, const char* access);

}

#endif
