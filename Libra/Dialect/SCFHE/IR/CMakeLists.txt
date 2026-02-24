set(LLVM_TARGET_DEFINITIONS SCFHEOps.td)

mlir_tablegen(SCFHEDialect.h.inc --gen-dialect-decls -dialect=scfhe)
mlir_tablegen(SCFHEDialect.cpp.inc --gen-dialect-defs -dialect=scfhe)

mlir_tablegen(SCFHETypes.h.inc --gen-typedef-decls -dialect=scfhe)
mlir_tablegen(SCFHETypes.cpp.inc --gen-typedef-defs -dialect=scfhe)

mlir_tablegen(SCFHEOps.h.inc --gen-op-decls -dialect=scfhe)
mlir_tablegen(SCFHEOps.cpp.inc --gen-op-defs -dialect=scfhe)

add_public_tablegen_target(SCFHEDialectIncGen)

# add_dependencies(SCFHEDialectIncGen
#   SCFHETypesIncGen
#   # SCFHEOpsIncGen
# )

add_mlir_dialect_library(MLIRSCFHEDialect
  SCFHEDialect.cpp
  # SCFHEOps.cpp
  # SCFHETypes.cpp

  DEPENDS
  SCFHEDialectIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRAffineDialect
  MLIRInferTypeOpInterface
)