add_mlir_dialect_library(MLIRSCFHEDialect
    SCFHEDialect.cpp

    DEPENDS
    SCFHEDialectIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRAffineDialect
)