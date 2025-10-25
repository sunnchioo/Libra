include_directories(${PROJECT_SOURCE_DIR}/Libra/Dialect/FlyHE/IR)
include_directories(${PROJECT_BINARY_DIR}/Libra/Dialect/FlyHE/IR)

add_mlir_library(FLYHETranslateLib
  FlyHETranslate.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRParser
  MLIRTransforms
  MLIRFuncDialect

  MLIRSCFHEDialect
)