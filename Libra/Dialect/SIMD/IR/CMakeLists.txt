set(LLVM_TARGET_DEFINITIONS SIMDOps.td)

mlir_tablegen(SIMDDialect.h.inc --gen-dialect-decls -dialect=simd)
mlir_tablegen(SIMDDialect.cpp.inc --gen-dialect-defs -dialect=simd)

mlir_tablegen(SIMDTypes.h.inc --gen-typedef-decls -dialect=simd)
mlir_tablegen(SIMDTypes.cpp.inc --gen-typedef-defs -dialect=simd)

mlir_tablegen(SIMDOps.h.inc --gen-op-decls -dialect=simd)
mlir_tablegen(SIMDOps.cpp.inc --gen-op-defs -dialect=simd)

add_public_tablegen_target(SIMDDialectIncGen)

add_mlir_dialect_library(MLIRSIMDDialect
  SIMDDialect.cpp
  SIMDOps.cpp
  SIMDTypes.cpp

  DEPENDS
  SIMDDialectIncGen
  SCFHEDialectIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRAffineDialect
  MLIRInferTypeOpInterface
)
