#include "prx/libc/include/ApplicationHeap.hpp"
#include "prx/libc/include/GuestArena.hpp"
#include "prx/libc/include/general/VabiMacros.hpp"
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <limits>

extern "C" {
void* APS5_VABI malloc_nid_postfix(std::size_t);
void* APS5_VABI calloc_nid_postfix(std::size_t, std::size_t);
void* APS5_VABI realloc_nid_postfix(void*, std::size_t);
void* APS5_VABI memalign_nid_postfix(std::size_t, std::size_t);
void* APS5_VABI aligned_alloc_nid_postfix(std::size_t, std::size_t);
int APS5_VABI posix_memalign_nid_postfix(void**, std::size_t, std::size_t);
void APS5_VABI free_nid_postfix(void*);
int* APS5_VABI __error_nid_postfix();
}

namespace {

constexpr int NoMemory = 12;
constexpr int InvalidArgument = 22;
constexpr std::size_t Max = std::numeric_limits<std::size_t>::max();
constexpr std::size_t HalfPlusOne = static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()) + 1;

int failures = 0;

void report(const char* what, const char* detail) {
    std::fprintf(stderr, "%s: %s\n", what, detail);
    ++failures;
}

template<typename TCall>
void expectNull(const char* what, int expectedError, TCall call) {
    *__error_nid_postfix() = 0;
    void* result = nullptr;
    try {
        result = call();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s: threw %s instead of returning NULL\n", what, error.what());
        ++failures;
        return;
    } catch (...) {
        report(what, "threw a non-standard exception instead of returning NULL");
        return;
    }
    if (result != nullptr) {
        report(what, "returned a block");
        free_nid_postfix(result);
        return;
    }
    if (*__error_nid_postfix() != expectedError) {
        std::fprintf(stderr, "%s: errno %d, expected %d\n", what, *__error_nid_postfix(), expectedError);
        ++failures;
    }
}

template<typename TCall>
void* expectBlock(const char* what, TCall call) {
    *__error_nid_postfix() = 77;
    void* result = nullptr;
    try {
        result = call();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s: threw %s\n", what, error.what());
        ++failures;
        return nullptr;
    }
    if (result == nullptr) report(what, "returned NULL");
    if (*__error_nid_postfix() != 77) report(what, "changed errno on success");
    return result;
}

}

int main() {
    void* api[10]{};
    ApplicationHeapRegister_nid_no_patch(api);

    expectNull("malloc(SIZE_MAX)", NoMemory, [] { return malloc_nid_postfix(Max); });
    expectNull("malloc(PTRDIFF_MAX + 1)", NoMemory, [] { return malloc_nid_postfix(HalfPlusOne); });
    expectNull("calloc(SIZE_MAX, 2)", NoMemory, [] { return calloc_nid_postfix(Max, 2); });
    expectNull("calloc(2^33, 2^33)", NoMemory, [] { return calloc_nid_postfix(std::size_t{1} << 33, std::size_t{1} << 33); });
    expectNull("calloc(PTRDIFF_MAX + 1, 1)", NoMemory, [] { return calloc_nid_postfix(HalfPlusOne, 1); });
    expectNull("realloc(NULL, SIZE_MAX)", NoMemory, [] { return realloc_nid_postfix(nullptr, Max); });
    expectNull("aligned_alloc(64, SIZE_MAX)", NoMemory, [] { return aligned_alloc_nid_postfix(64, Max); });
    expectNull("memalign(64, PTRDIFF_MAX + 1)", NoMemory, [] { return memalign_nid_postfix(64, HalfPlusOne); });
    expectNull("aligned_alloc(24, 64)", InvalidArgument, [] { return aligned_alloc_nid_postfix(24, 64); });
    expectNull("aligned_alloc(0, 64)", InvalidArgument, [] { return aligned_alloc_nid_postfix(0, 64); });
    expectNull("memalign(24, 64)", InvalidArgument, [] { return memalign_nid_postfix(24, 64); });

    auto* block = static_cast<unsigned char*>(expectBlock("malloc(40)", [] { return malloc_nid_postfix(40); }));
    if (block != nullptr) {
        std::memset(block, 0x6b, 40);
        expectNull("realloc(block, SIZE_MAX)", NoMemory, [block] { return realloc_nid_postfix(block, Max); });
        expectNull("realloc(block, PTRDIFF_MAX + 1)", NoMemory, [block] { return realloc_nid_postfix(block, HalfPlusOne); });
        for (unsigned index = 0; index < 40; ++index) {
            if (block[index] != 0x6b) report("realloc failure", "damaged the original block");
        }
        free_nid_postfix(block);
    }

    if (GuestArena::GuestArenaAvailable_nid_postfix()) {
        constexpr std::size_t beyondCommitLimit = std::size_t{300} << 30;
        const auto freeBefore = GuestArena::GuestArenaHighestFree_nid_no_patch(beyondCommitLimit, 0x4000);
        *__error_nid_postfix() = 0;
        void* huge = nullptr;
        try {
            huge = malloc_nid_postfix(beyondCommitLimit);
        } catch (const std::exception& error) {
            std::fprintf(stderr, "malloc(300 GiB): threw %s\n", error.what());
            ++failures;
        }
        if (huge != nullptr) {
            free_nid_postfix(huge);
        } else if (*__error_nid_postfix() != NoMemory && failures == 0) {
            report("malloc(300 GiB)", "returned NULL without ENOMEM");
        }
        if (huge == nullptr && GuestArena::GuestArenaHighestFree_nid_no_patch(beyondCommitLimit, 0x4000) != freeBefore) {
            report("malloc(300 GiB)", "left its arena range reserved after failing");
        }
        void* small = malloc_nid_postfix(100);
        if (small == nullptr) report("malloc(100) after a failed huge malloc", "returned NULL");
        free_nid_postfix(small);
        void* large = malloc_nid_postfix(std::size_t{1} << 20);
        if (large == nullptr) report("malloc(1 MiB) after a failed huge malloc", "returned NULL");
        free_nid_postfix(large);
    }

    void* untouched = &failures;
    *__error_nid_postfix() = 77;
    if (posix_memalign_nid_postfix(&untouched, HalfPlusOne, 8) != NoMemory) report("posix_memalign(PTRDIFF_MAX + 1)", "did not return ENOMEM");
    if (posix_memalign_nid_postfix(&untouched, 64, HalfPlusOne) != NoMemory) report("posix_memalign(64, PTRDIFF_MAX + 1)", "did not return ENOMEM");
    if (posix_memalign_nid_postfix(&untouched, 64, Max) != NoMemory) report("posix_memalign(64, SIZE_MAX)", "did not return ENOMEM");
    if (untouched != &failures || *__error_nid_postfix() != 77) report("posix_memalign failure", "touched the output or errno");

    free_nid_postfix(expectBlock("calloc(3, 5)", [] { return calloc_nid_postfix(3, 5); }));
    free_nid_postfix(expectBlock("calloc(0, 5)", [] { return calloc_nid_postfix(0, 5); }));
    free_nid_postfix(expectBlock("aligned_alloc(128, 1)", [] { return aligned_alloc_nid_postfix(128, 1); }));
    free_nid_postfix(expectBlock("memalign(4096, 1)", [] { return memalign_nid_postfix(4096, 1); }));
    void* first = expectBlock("malloc(0)", [] { return malloc_nid_postfix(0); });
    void* second = expectBlock("malloc(0)", [] { return malloc_nid_postfix(0); });
    if (first != nullptr && first == second) report("malloc(0)", "returned the same block twice");
    free_nid_postfix(first);
    free_nid_postfix(second);

    return failures == 0 ? 0 : 1;
}
