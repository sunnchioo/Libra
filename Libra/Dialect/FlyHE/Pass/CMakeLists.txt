add_mlir_dialect_library(FlyHEPasses
  FlyHEPass.cpp

  DEPENDS
  FlyHEDialectIncGen

  LINK_LIBS PRIVATE
  MLIRFlyHEDialect
  
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRAffineDialect
  MLIRFuncDialect
  MLIRConversion
)