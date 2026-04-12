cmake_minimum_required(VERSION 3.14)
project(ggml-bridge C CXX)

# Build ggml with only CPU backend, static libs, no extras
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE ON CACHE BOOL "" FORCE)
set(GGML_CPU ON CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_METAL OFF CACHE BOOL "" FORCE)
set(GGML_CUDA OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN OFF CACHE BOOL "" FORCE)
set(GGML_SYCL OFF CACHE BOOL "" FORCE)
set(GGML_RPC OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../ggml ${CMAKE_CURRENT_BINARY_DIR}/ggml)

# Bridge library that wraps ggml functions
add_library(ggml-bridge STATIC ggml_bridge.c)
target_include_directories(ggml-bridge PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../ggml/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src
    ${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src/ggml-cpu
)
target_compile_features(ggml-bridge PRIVATE c_std_11)
target_link_libraries(ggml-bridge PRIVATE ggml-cpu ggml-base)
