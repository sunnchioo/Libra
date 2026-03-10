include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../SIMD/IR)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/../SIMD/IR)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../SIMD/Common)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../SISD/IR)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/../SISD/IR)

set(LLVM_TARGET_DEFINITIONS ModeSelectPass.td)
mlir_tablegen(ModeSelectPass.h.inc -gen-pass-decls -name MDSELOpt)
add_public_tablegen_target(MLIRMDSELPassesIncGen)

add_mlir_dialect_library(MODESELECTPasses
  ModeSelectPass.cpp
  CostModel.cpp
  ModeRewriter.cpp

  DEPENDS
  MLIRMDSELPassesIncGen

  LINK_LIBS PRIVATE
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRAffineDialect
  MLIRFuncDialect
)