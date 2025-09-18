set(LLVM_TARGET_DEFINITIONS SCFHEDialect.td)

mlir_tablegen(SCFHEDialect.h.inc --gen-dialect-decls -dialect=scfhe)
mlir_tablegen(SCFHEDialect.cpp.inc --gen-dialect-defs -dialect=scfhe)

add_public_tablegen_target(SCFHEDialectIncGen)

add_mlir_dialect_library(MLIRSCFHEDialect
    SCFHEDialect.cpp

    DEPENDS
    SCFHEDialectIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRAffineDialect
)