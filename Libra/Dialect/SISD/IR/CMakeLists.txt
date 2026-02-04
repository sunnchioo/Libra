set(LLVM_TARGET_DEFINITIONS SISDOps.td)

mlir_tablegen(SISDDialect.h.inc --gen-dialect-decls -dialect=sisd)
mlir_tablegen(SISDDialect.cpp.inc --gen-dialect-defs -dialect=sisd)

mlir_tablegen(SISDTypes.h.inc --gen-typedef-decls -dialect=sisd)
mlir_tablegen(SISDTypes.cpp.inc --gen-typedef-defs -dialect=sisd)

mlir_tablegen(SISDOps.h.inc --gen-op-decls -dialect=sisd)
mlir_tablegen(SISDOps.cpp.inc --gen-op-defs -dialect=sisd)

add_public_tablegen_target(SISDDialectIncGen)

add_mlir_dialect_library(MLIRSISDDialect
  SISDDialect.cpp
  SISDOps.cpp
  SISDTypes.cpp

  DEPENDS
  SISDDialectIncGen
  SCFHEDialectIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRAffineDialect
  MLIRInferTypeOpInterface
)
