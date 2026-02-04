set(LLVM_TARGET_DEFINITIONS SISDPass.td)
mlir_tablegen(SISDPass.h.inc -gen-pass-decls -name SISDOpt)
add_public_tablegen_target(MLIRSISDPassesIncGen)

add_mlir_dialect_library(SISDPasses
  SISDPass.cpp

  DEPENDS
  SISDDialectIncGen
  MLIRSISDPassesIncGen

  LINK_LIBS PRIVATE
  MLIRSISDDialect
  
  MLIRIR
  MLIRPass          
  MLIRTransforms
  MLIRAffineDialect
  MLIRFuncDialect
)