if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 3.20)
    project(gdefpack C CXX)
else()
    enable_language(C)
endif()

set(GDEFPACK_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
set(GDEFPACK_LIBDEFLATE ${GDEFPACK_ROOT}/3rdparty/libdeflate-gdeflate)

add_library(gdefpack_libdeflate STATIC
        ${GDEFPACK_LIBDEFLATE}/lib/deflate_compress.c
        ${GDEFPACK_LIBDEFLATE}/lib/gdeflate_compress.c
        ${GDEFPACK_LIBDEFLATE}/lib/gdeflate_decompress.c
        ${GDEFPACK_LIBDEFLATE}/lib/utils.c
        ${GDEFPACK_LIBDEFLATE}/lib/x86/cpu_features.c)
target_include_directories(gdefpack_libdeflate PUBLIC ${GDEFPACK_LIBDEFLATE} PRIVATE ${GDEFPACK_LIBDEFLATE}/common)
set_target_properties(gdefpack_libdeflate PROPERTIES C_STANDARD 99)
if(MINGW)
    target_compile_options(gdefpack_libdeflate PRIVATE "SHELL:-include stdlib.h")
endif()

add_executable(gdefpack
        ${CMAKE_CURRENT_SOURCE_DIR}/gdefpack.cpp
        ${GDEFPACK_ROOT}/core/libs/prx/libkernel/File/src/CompressedFile.cpp)
target_include_directories(gdefpack PRIVATE ${GDEFPACK_ROOT}/core/libs)
target_link_libraries(gdefpack PRIVATE gdefpack_libdeflate)
set_target_properties(gdefpack PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
if(MINGW)
    target_link_options(gdefpack PRIVATE -static)
endif()
