set(LLVM_TARGET_DEFINITIONS FlyHEPass.td)
mlir_tablegen(FlyHEPass.h.inc -gen-pass-decls -name FlyHEOpt)
add_public_tablegen_target(MLIRFlyHEPassesIncGen)
# add_dependencies(mlir-headers MLIRFlyHEPassesIncGen)

add_mlir_dialect_library(FlyHEPasses
  FlyHEPass.cpp

  DEPENDS
  FlyHEDialectIncGen
  MLIRFlyHEPassesIncGen

  LINK_LIBS PRIVATE
  MLIRFlyHEDialect
  
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRAffineDialect
  MLIRFuncDialect
)