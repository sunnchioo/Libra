cmake_minimum_required(VERSION 3.20)

# set(CMAKE_C_COMPILER "/usr/bin/gcc-13")
# set(CMAKE_CXX_COMPILER "/usr/bin/g++-13")

set(CMAKE_C_COMPILER "${CMAKE_CURRENT_SOURCE_DIR}/Tool/llvm-project/build/bin/clang")
set(CMAKE_CXX_COMPILER "${CMAKE_CURRENT_SOURCE_DIR}/Tool/llvm-project/build/bin/clang++")

set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.4/bin/nvcc")

project(Libra LANGUAGES C CXX)

if(POLICY CMP0116)
    cmake_policy(SET CMP0116 NEW)
endif()
if(POLICY CMP0114)
    cmake_policy(SET CMP0114 NEW)
endif()

set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/install")

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



# cmake_minimum_required(VERSION 3.20.0)

# project(Libra LANGUAGES CXX C)

# set(CMAKE_C_COMPILER "${CMAKE_CURRENT_SOURCE_DIR}/Tool/llvm-project/build/bin/clang")
# set(CMAKE_CXX_COMPILER "${CMAKE_CURRENT_SOURCE_DIR}/Tool/llvm-project/build/bin/clang++")

# set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.4/bin/nvcc")


# set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)

# set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")

# if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
#   set(LLVM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/Tool/llvm-project/build/lib/cmake/llvm")
#   set(MLIR_DIR "${CMAKE_CURRENT_SOURCE_DIR}/Tool/llvm-project/build/lib/cmake/mlir")

#   find_package(MLIR REQUIRED CONFIG)


#   # Define the default argument to use by `lit` when testing.
#   set(LLVM_LIT_ARGS "-sv" CACHE STRING "Default options for lit")

#   message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
#   message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

#   set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
#   set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
#   set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})



#   list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
#   list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

#   include(TableGen)
#   include(AddLLVM)
#   include(AddMLIR)
#   include(HandleLLVMOptions)
# else()
#   # Build via external projects mechanism
#   set(MLIR_MAIN_SRC_DIR ${LLVM_MAIN_SRC_DIR}/../mlir)
#   set(MLIR_INCLUDE_DIR ${MLIR_MAIN_SRC_DIR}/include)
#   set(MLIR_GENERATED_INCLUDE_DIR ${LLVM_BINARY_DIR}/tools/mlir/include)
#   set(MLIR_INCLUDE_DIRS "${MLIR_INCLUDE_DIR};${MLIR_GENERATED_INCLUDE_DIR}")
# endif()

# set(LIBRA_SOURCE_DIR ${PROJECT_SOURCE_DIR})
# set(LIBRA_BINARY_DIR ${PROJECT_BINARY_DIR})
# include_directories(${LLVM_INCLUDE_DIRS})
# include_directories(${MLIR_INCLUDE_DIRS})
# include_directories(${LIBRA_SOURCE_DIR}/Libra)
# include_directories(${LIBRA_BINARY_DIR}/Libra)
# link_directories(${LLVM_BUILD_LIBRARY_DIR})
# add_definitions(${LLVM_DEFINITIONS})

# add_subdirectory(Libra)
