# set(CMAKE_C_COMPILER "/usr/bin/gcc-13")
# set(CMAKE_CXX_COMPILER "/usr/bin/g++-13")
# # set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.1/bin/nvcc")
# set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.6/bin/nvcc")

# set(CMAKE_CUDA_ARCHITECTURES native)

# openfhe
set(OPENFHE_SOURCE_DIR ${CMAKE_SOURCE_DIR}/phantom/thirdparty/openfhe)
set(OPENFHE_BINARY_DIR ${CMAKE_BINARY_DIR}/phantom/thirdparty/openfhe)
include_directories(${OPENFHE_SOURCE_DIR}/src/core/include)
include_directories(${OPENFHE_SOURCE_DIR}/src/pke/include)
include_directories(${OPENFHE_SOURCE_DIR}/src/binfhe/include)
include_directories(${OPENFHE_SOURCE_DIR}/third-party/cereal/include)
include_directories(${OPENFHE_BINARY_DIR}/src/core)
link_libraries(OPENFHEcore OPENFHEpke OPENFHEbinfhe)

# phantom wordwise
include_directories(${CMAKE_SOURCE_DIR}/phantom/include)
# link_libraries(phantomfhe)

# phantom bitwise
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# use C++20
# set(CMAKE_CXX_STANDARD 20)
# set(CMAKE_CUDA_STANDARD 20)

add_subdirectory(src)


# example
add_subdirectory(examples)

# [option] PHANTOM_ENABLE_EXAMPLE (default: ON)
# option(PHANTOM_ENABLE_EXAMPLE "Enable examples" OFF)
# message(STATUS "Enable examples: ${PHANTOM_ENABLE_EXAMPLE}")
# if (PHANTOM_ENABLE_EXAMPLE)
#     add_subdirectory(examples)
# endif ()
# if (PHANTOM_ENABLE_EXAMPLE)
#     add_subdirectory(examples)
# endif ()

# if (PHANTOM_ENABLE_TEST)
#     add_subdirectory(tests)
# endif ()

# if (PHANTOM_ENABLE_BENCH)
#     add_subdirectory(benchmarks)
# endif ()
