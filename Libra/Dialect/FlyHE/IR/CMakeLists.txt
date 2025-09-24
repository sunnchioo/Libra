set(LLVM_TARGET_DEFINITIONS FlyHEOps.td)

mlir_tablegen(FlyHEDialect.h.inc --gen-dialect-decls -dialect=flyhe)
mlir_tablegen(FlyHEDialect.cpp.inc --gen-dialect-defs -dialect=flyhe)

mlir_tablegen(FlyHETypes.h.inc --gen-typedef-decls -dialect=flyhe)
mlir_tablegen(FlyHETypes.cpp.inc --gen-typedef-defs -dialect=flyhe)

mlir_tablegen(FlyHEOps.h.inc --gen-op-decls -dialect=flyhe)
mlir_tablegen(FlyHEOps.cpp.inc --gen-op-defs -dialect=flyhe)

add_public_tablegen_target(FlyHEDialectIncGen)

# add_dependencies(FlyHEDialectIncGen
#   FlyHETypesIncGen
#   # FlyHEOpsIncGen
# )

add_mlir_dialect_library(MLIRFlyHEDialect
  FlyHEDialect.cpp
  FlyHEOps.cpp

  DEPENDS
  FlyHEDialectIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRAffineDialect
  MLIRInferTypeOpInterface
)