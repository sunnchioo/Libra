# set(CMAKE_C_COMPILER "/usr/bin/gcc-13")
# set(CMAKE_CXX_COMPILER "/usr/bin/g++-13")
# set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.6/bin/nvcc")

# set(CMAKE_CUDA_ARCHITECTURES native)

# use C++20
# set(CMAKE_CXX_STANDARD 20)
# set(CMAKE_CUDA_STANDARD 20)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/benchmark/moti/single)

# openfhe
# set(OPENFHE_SOURCE_DIR ${CMAKE_SOURCE_DIR}/phantom/thirdparty/openfhe)
# set(OPENFHE_BINARY_DIR ${CMAKE_BINARY_DIR}/phantom/thirdparty/openfhe)
# include_directories(${OPENFHE_SOURCE_DIR}/src/core/include)
# include_directories(${OPENFHE_SOURCE_DIR}/src/pke/include)
# include_directories(${OPENFHE_SOURCE_DIR}/src/binfhe/include)
# include_directories(${OPENFHE_SOURCE_DIR}/third-party/cereal/include)
# include_directories(${OPENFHE_BINARY_DIR}/src/core)
# link_libraries(OPENFHEcore OPENFHEpke OPENFHEbinfhe)

include_directories(${CMAKE_SOURCE_DIR}/RLWE/phantom/include)
# include_directories(${CMAKE_SOURCE_DIR}/RLWE/phantom/bitwise/include)
include_directories(${CMAKE_SOURCE_DIR}/RLWE/boot/include)
include_directories(${CMAKE_SOURCE_DIR}/RLWE/boot/include/boostrapping)
include_directories(${CMAKE_SOURCE_DIR}/RLWE/boot/include/boostrapping/common)

include_directories(/usr/local/include)
include_directories(/usr/local/include/NTL)

# link_libraries(phantomfhe bitwise)
link_libraries(boot)

# target_link_libraries(boot PRIVATE ntl gmp m pthread)

# benchmark
# file(GLOB_RECURSE NEXUS_CUDA_SOURCE_FILES CONFIGURE_DEPENDS
#   "../boot/src/gelu.cu"
#   "../boot/src/ckks_evaluator.cu"
#   "../phantom/src/*.cu"
#   "moti/single/ckks-eval.cu"
# )
# add_executable(ckks-eval ${NEXUS_CUDA_SOURCE_FILES})
# target_link_libraries(ckks-eval PRIVATE ntl gmp m pthread)
# target_include_directories(ckks-eval PRIVATE
#   /usr/local/include
#   /usr/local/include/NTL
#   ../boot/include/bootstrapping/common
#   ../boot/include/bootstrapping
#   ../boot/include
# )

add_executable(ckks-eval ckks-eval.cu)
target_link_libraries(ckks-eval PRIVATE ntl gmp m pthread)

# add_executable(tfhe-eval tfhe-eval.cu)

