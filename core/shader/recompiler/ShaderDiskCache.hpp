#ifndef CORE_SHADER_RECOMPILER_SHADERDISKCACHE_HPP
#define CORE_SHADER_RECOMPILER_SHADERDISKCACHE_HPP

#include "CompiledVariant.hpp"
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace ShaderRecompiler::ShaderDiskCache {

inline constexpr std::uint32_t FormatVersion = 31;

enum class LoadStatus {
    Loaded,
    Absent,
    KeyMismatch,
    Rejected,
};

struct Counters {
    std::uint64_t hits = 0;
    std::uint64_t misses = 0;
    std::uint64_t writes = 0;
    std::uint64_t loadFailures = 0;
    std::uint64_t writeFailures = 0;
    std::uint64_t bytesRead = 0;
    std::uint64_t bytesWritten = 0;
};

[[nodiscard]] std::uint64_t SourceVersion();

void BuildKey(const RecompileRequest& request, std::uint32_t hostSubgroupSize, std::vector<std::byte>& key);

[[nodiscard]] std::string EntryName(std::span<const std::byte> key);

void EncodeResult(const RecompileResult& result, std::vector<std::byte>& out);
[[nodiscard]] bool DecodeResult(std::span<const std::byte> bytes, RecompileResult& result);

[[nodiscard]] std::vector<std::byte> EncodeEntry(std::span<const std::byte> key, const CompiledVariant& variant);
[[nodiscard]] LoadStatus DecodeEntry(std::span<const std::byte> file, std::span<const std::byte> key, CompiledVariant& variant);

[[nodiscard]] bool Enabled();
[[nodiscard]] std::filesystem::path EntryDirectory();

[[nodiscard]] bool Load(std::span<const std::byte> key, CompiledVariant& variant);
void Store(std::vector<std::byte> key, std::shared_ptr<const CompiledVariant> variant);
// Makes a freshly compiled variant's SPIR-V evictable like a loaded one, pinned until Store has
// written its entry (the words are reloaded from that entry once evicted).
void MakeStoredEvictable(std::span<const std::byte> key, CompiledVariant& variant);
void Flush();
[[nodiscard]] Counters Totals();

// Specialized modules (Recompiler's specializeModule): a prepared module with its specialization
// constants folded in and optimized again, ~100-200 ms for the large ones, made on the draw path the
// first time each is used in a session. Stored by content (the caller's key bytes) under
// <entries>/specialized, so later sessions load them instead.
struct StoredSpecialization {
    std::vector<std::uint32_t> spirv;
    std::vector<std::uint32_t> bindings;
    bool pushData = false;
};
// The entries' content hash (for keys that name large inputs by hash).
[[nodiscard]] std::uint64_t ContentHash(std::span<const std::byte> bytes, std::uint64_t seed);
[[nodiscard]] bool LoadSpecialized(std::span<const std::byte> key, StoredSpecialization& module);
// Whether an entry for the key exists (a cheap probe; LoadSpecialized still verifies it).
[[nodiscard]] bool SpecializedStored(std::span<const std::byte> key);
void StoreSpecialized(std::span<const std::byte> key, const StoredSpecialization& module);

}

#endif
