set(LLVM_TARGET_DEFINITIONS SCFHEPass.td)
mlir_tablegen(SCFHEPass.h.inc -gen-pass-decls -name SCFHEOpt)
add_public_tablegen_target(MLIRSCFHEPassesIncGen)
# add_dependencies(mlir-headers MLIRSCFHEPassesIncGen)

add_mlir_dialect_library(SCFHEPasses
  AutoAnnotatePass.cpp
  SCFHEPass.cpp

  DEPENDS
  SCFHEDialectIncGen
  MLIRSCFHEPassesIncGen

  LINK_LIBS PRIVATE
  MLIRSCFHEDialect

  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRAffineDialect
  MLIRFuncDialect
)