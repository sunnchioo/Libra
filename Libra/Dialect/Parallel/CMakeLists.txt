include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../SIMD/IR)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/../SIMD/IR)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../SIMD/Common)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../SISD/IR)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/../SISD/IR)

set(LLVM_TARGET_DEFINITIONS ParallelPass.td)
mlir_tablegen(ParallelPass.h.inc -gen-pass-decls -name PARALLELOpt)
add_public_tablegen_target(PARALLEPassesIncGen)

add_mlir_dialect_library(ParallelPasses
  ParallelPass.cpp

  DEPENDS
  PARALLEPassesIncGen

  LINK_LIBS PRIVATE
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRAffineDialect
  MLIRFuncDialect
)