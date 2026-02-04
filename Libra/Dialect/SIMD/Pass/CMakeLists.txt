set(LLVM_TARGET_DEFINITIONS SIMDPass.td)
mlir_tablegen(SIMDPass.h.inc -gen-pass-decls -name SIMDOpt)
add_public_tablegen_target(MLIRSIMDPassesIncGen)

add_mlir_dialect_library(SIMDPasses
  SIMDPass.cpp

  DEPENDS
  SIMDDialectIncGen
  MLIRSIMDPassesIncGen

  LINK_LIBS PRIVATE
  MLIRSIMDDialect
  
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRAffineDialect
  MLIRFuncDialect
)