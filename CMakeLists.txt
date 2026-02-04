cmake_minimum_required(VERSION 3.20)

set(CMAKE_C_COMPILER "${CMAKE_CURRENT_SOURCE_DIR}/Tool/llvm-project/build/bin/clang")
set(CMAKE_CXX_COMPILER "${CMAKE_CURRENT_SOURCE_DIR}/Tool/llvm-project/build/bin/clang++")

set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.4/bin/nvcc")
# set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.6/bin/nvcc")

project(Libra LANGUAGES C CXX)

if(POLICY CMP0116)
    cmake_policy(SET CMP0116 NEW)
endif()
if(POLICY CMP0114)
    cmake_policy(SET CMP0114 NEW)
endif()

set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/install")

# relase version is not check
set(CMAKE_BUILD_TYPE "Debug" CACHE STRING " " FORCE)
add_compile_options(-fPIC)
# add_compile_options(-fno-rtti)

list(APPEND LLVM_CMAKE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/Tool/llvm-project/build/lib/cmake/llvm")
list(APPEND MLIR_CMAKE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/Tool/llvm-project/build/lib/cmake/mlir")

list(APPEND CMAKE_MODULE_PATH ${MLIR_CMAKE_DIR})
list(APPEND CMAKE_MODULE_PATH ${LLVM_CMAKE_DIR})
include(LLVMConfig)
include(MLIRConfig) 
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

add_subdirectory(Libra)
