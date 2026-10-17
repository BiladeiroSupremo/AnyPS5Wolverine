// gdefpack: compresses game files into GDEF containers (the SM2 GDeflate mod's format) in place,
// safely: each file is compressed into <file>.gdef.tmp, the temporary is decoded in full with
// AnyPS5's own reader (libkernel CompressedFile) and compared with the original, and only then
// replaces it (MoveFileEx, atomic on one volume). Files that shrink by less than --min-saving
// percent (videos, already-compressed data) stay as they are. At most one temporary exists at a
// time, so the extra disk space needed is one compressed file.
// Usage: gdefpack [--level N] [--min-saving P] [--threads T] <file or directory>...
//   (directories are processed non-recursively, like GDeflateCompressor)
#include "prx/libkernel/File/include/CompressedFile.hpp"

#include <libdeflate.h>
#include <windows.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr std::uint32_t ChunkBytes = 65536;
constexpr std::uint32_t RawChunk = 0x80000000u;

struct Options {
    int level = 6;
    double minSaving = 5.0;
    unsigned threads = std::max(1u, std::thread::hardware_concurrency());
};

bool isGdef(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    char magic[4] = {};
    return in.read(magic, 4) && std::memcmp(magic, "GDEF", 4) == 0;
}

// One chunk's stored form: the u32 stored size and the payload (a one-tile GDeflate stream, or the
// raw bytes when that is not smaller).
std::vector<std::uint8_t> packChunk(libdeflate_gdeflate_compressor* compressor, const std::uint8_t* data, std::size_t bytes) {
    std::size_t pages = 0;
    const auto bound = libdeflate_gdeflate_compress_bound(compressor, bytes, &pages);
    std::vector<std::uint8_t> out;
    if (pages == 1) {
        // libdeflate 1.8's GDEFLATE compressor writes past its own bound on incompressible input
        // (heap corruption on a video file; GDeflateCompressor crashed on the same file), so the
        // buffer has generous slack and only the bound's worth is offered to it.
        std::vector<std::uint8_t> tile(bound * 2 + 65536);
        libdeflate_gdeflate_out_page page{tile.data(), bound};
        const auto produced = libdeflate_gdeflate_compress(compressor, data, bytes, &page, 1);
        // TileStream header (id 4, magic id ^ 0xff, one tile, tile size index 1, last tile size) and
        // the tile offset table (one entry: the last tile's compressed size), as GDeflate::Compress writes.
        const std::size_t streamBytes = 8 + 4 + page.nbytes;
        if (produced != 0 && page.nbytes != 0 && page.nbytes <= bound && streamBytes < bytes) {
            out.resize(4 + streamBytes);
            const std::uint32_t stored = static_cast<std::uint32_t>(streamBytes);
            std::memcpy(out.data(), &stored, 4);
            std::uint8_t* stream = out.data() + 4;
            stream[0] = 4;
            stream[1] = 4 ^ 0xff;
            stream[2] = 1;
            stream[3] = 0;
            const std::uint32_t lastTile = bytes == ChunkBytes ? 0u : static_cast<std::uint32_t>(bytes);
            const std::uint32_t sizeWord = 1u | (lastTile << 2u);
            std::memcpy(stream + 4, &sizeWord, 4);
            const std::uint32_t tileBytes = static_cast<std::uint32_t>(page.nbytes);
            std::memcpy(stream + 8, &tileBytes, 4);
            std::memcpy(stream + 12, tile.data(), page.nbytes);
            return out;
        }
    }
    out.resize(4 + bytes);
    const std::uint32_t stored = static_cast<std::uint32_t>(bytes) | RawChunk;
    std::memcpy(out.data(), &stored, 4);
    std::memcpy(out.data() + 4, data, bytes);
    return out;
}

// Compresses `source` into `target`; returns the target's size.
std::uint64_t compressFile(const fs::path& source, const fs::path& target, const Options& options) {
    std::ifstream in(source, std::ios::binary);
    if (!in) throw std::runtime_error("cannot read " + source.string());
    const std::uint64_t total = fs::file_size(source);
    const std::uint64_t chunks = (total + ChunkBytes - 1) / ChunkBytes;
    if (chunks > 0xffffffffu) throw std::runtime_error("too many chunks in " + source.string());
    std::ofstream out(target, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("cannot create " + target.string());
    out.write("GDEF", 4);
    const std::uint32_t chunkBytes = ChunkBytes;
    const std::uint32_t chunkCount = static_cast<std::uint32_t>(chunks);
    out.write(reinterpret_cast<const char*>(&chunkBytes), 4);
    out.write(reinterpret_cast<const char*>(&chunkCount), 4);
    out.write(reinterpret_cast<const char*>(&total), 8);
    std::vector<std::uint64_t> offsets(chunks, 0);
    const std::uint64_t tableAt = static_cast<std::uint64_t>(out.tellp());
    out.write(reinterpret_cast<const char*>(offsets.data()), static_cast<std::streamsize>(chunks * 8));
    std::uint64_t written = tableAt + chunks * 8;
    const std::uint64_t batch = std::uint64_t{options.threads} * 32u;
    std::vector<std::uint8_t> input;
    std::vector<std::vector<std::uint8_t>> packed;
    for (std::uint64_t first = 0; first < chunks; first += batch) {
        const auto count = std::min<std::uint64_t>(batch, chunks - first);
        const auto bytes = std::min<std::uint64_t>(count * ChunkBytes, total - first * ChunkBytes);
        input.resize(static_cast<std::size_t>(bytes));
        if (!in.read(reinterpret_cast<char*>(input.data()), static_cast<std::streamsize>(bytes))) throw std::runtime_error("short read of " + source.string());
        packed.assign(static_cast<std::size_t>(count), {});
        std::atomic<std::uint64_t> next{0};
        std::vector<std::thread> workers;
        for (unsigned t = 0; t < options.threads; ++t) {
            workers.emplace_back([&] {
                auto* compressor = libdeflate_alloc_gdeflate_compressor(options.level);
                for (std::uint64_t i; (i = next.fetch_add(1)) < count;) {
                    const auto offset = i * ChunkBytes;
                    const auto size = static_cast<std::size_t>(std::min<std::uint64_t>(ChunkBytes, bytes - offset));
                    packed[static_cast<std::size_t>(i)] = packChunk(compressor, input.data() + offset, size);
                }
                libdeflate_free_gdeflate_compressor(compressor);
            });
        }
        for (auto& worker : workers) worker.join();
        for (std::uint64_t i = 0; i < count; ++i) {
            offsets[static_cast<std::size_t>(first + i)] = written;
            const auto& chunk = packed[static_cast<std::size_t>(i)];
            out.write(reinterpret_cast<const char*>(chunk.data()), static_cast<std::streamsize>(chunk.size()));
            written += chunk.size();
        }
        if (!out) throw std::runtime_error("write error on " + target.string() + " (disk full?)");
    }
    out.seekp(static_cast<std::streamoff>(tableAt));
    out.write(reinterpret_cast<const char*>(offsets.data()), static_cast<std::streamsize>(chunks * 8));
    out.close();
    if (!out) throw std::runtime_error("write error on " + target.string());
    return written;
}

// Decodes all of `container` with AnyPS5's reader and compares it with `original`.
void verify(const fs::path& container, const fs::path& original) {
    std::ifstream packed(container, std::ios::binary);
    std::ifstream plain(original, std::ios::binary);
    const auto packedBytes = fs::file_size(container);
    const CompressedFile::RawRead read = [&](std::uint64_t offset, void* destination, std::size_t bytes) {
        packed.clear();
        packed.seekg(static_cast<std::streamoff>(offset));
        return static_cast<bool>(packed.read(static_cast<char*>(destination), static_cast<std::streamsize>(bytes)));
    };
    const auto index = CompressedFile::Probe(container.string(), packedBytes, read);
    if (!index) throw std::runtime_error("verification: " + container.string() + " is not a GDEF container");
    if (index->uncompressedBytes != fs::file_size(original)) throw std::runtime_error("verification: size differs for " + original.string());
    constexpr std::size_t Step = 8u << 20u;
    std::vector<char> expected(Step), actual(Step);
    for (std::uint64_t offset = 0; offset < index->uncompressedBytes; offset += Step) {
        const auto bytes = static_cast<std::size_t>(std::min<std::uint64_t>(Step, index->uncompressedBytes - offset));
        if (!plain.read(expected.data(), static_cast<std::streamsize>(bytes))) throw std::runtime_error("verification: short read of " + original.string());
        CompressedFile::Read(*index, read, offset, actual.data(), bytes);
        if (std::memcmp(expected.data(), actual.data(), bytes) != 0) throw std::runtime_error("verification: MISMATCH in " + original.string() + " near offset " + std::to_string(offset));
    }
}

}

int main(int argc, char** argv) {
    Options options;
    std::vector<fs::path> files;
    // gdefpack --verify <container> <original>: decodes the container in full and compares it.
    if (argc == 4 && std::string(argv[1]) == "--verify") {
        try {
            verify(argv[2], argv[3]);
            std::printf("verified %s\n", argv[2]);
            return 0;
        } catch (const std::exception& error) {
            std::printf("FAIL %s\n", error.what());
            return 1;
        }
    }
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--level" && i + 1 < argc) options.level = std::atoi(argv[++i]);
        else if (arg == "--min-saving" && i + 1 < argc) options.minSaving = std::atof(argv[++i]);
        else if (arg == "--threads" && i + 1 < argc) options.threads = static_cast<unsigned>(std::max(1, std::atoi(argv[++i])));
        else if (fs::is_directory(arg)) {
            for (const auto& entry : fs::directory_iterator(arg)) {
                if (entry.is_regular_file() && entry.path().extension() != ".tmp" && entry.path().extension() != ".backup") files.push_back(entry.path());
            }
        } else if (fs::is_regular_file(arg)) files.push_back(arg);
        else { std::fprintf(stderr, "not a file or directory: %s\n", arg.c_str()); return 2; }
    }
    std::sort(files.begin(), files.end());
    std::uint64_t before = 0, after = 0;
    int failures = 0;
    for (const auto& file : files) {
        const auto original = fs::file_size(file);
        before += original;
        if (original < 4096 || isGdef(file)) { after += original; std::printf("skip  %s\n", file.filename().string().c_str()); continue; }
        auto temporary = file;
        temporary += ".gdef.tmp";
        try {
            const auto start = std::chrono::steady_clock::now();
            const auto packed = compressFile(file, temporary, options);
            const double saving = 100.0 * (1.0 - static_cast<double>(packed) / static_cast<double>(original));
            if (saving < options.minSaving) {
                fs::remove(temporary);
                after += original;
                std::printf("keep  %s (%.1f%% saving)\n", file.filename().string().c_str(), saving);
                continue;
            }
            verify(temporary, file);
            if (!MoveFileExW(temporary.c_str(), file.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) throw std::runtime_error("cannot replace " + file.string() + " (error " + std::to_string(GetLastError()) + ")");
            after += packed;
            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            std::printf("pack  %s %.1f -> %.1f MB (%.1f%% saving, %.1f s)\n", file.filename().string().c_str(), original / 1048576.0, packed / 1048576.0, saving, seconds);
        } catch (const std::exception& error) {
            ++failures;
            std::error_code ignored;
            fs::remove(temporary, ignored);
            after += original;
            std::printf("FAIL  %s: %s (original kept)\n", file.filename().string().c_str(), error.what());
        }
        std::fflush(stdout);
    }
    std::printf("total %.2f GB -> %.2f GB, %d failures\n", before / 1073741824.0, after / 1073741824.0, failures);
    return failures == 0 ? 0 : 1;
}
