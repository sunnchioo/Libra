# add_subdirectory(FlyHE)

include_directories(${PROJECT_SOURCE_DIR}/Libra/Dialect/SIMD/IR)
include_directories(${PROJECT_BINARY_DIR}/Libra/Dialect/SIMD/IR)

include_directories(${PROJECT_SOURCE_DIR}/Libra/Dialect/SISD/IR)
include_directories(${PROJECT_BINARY_DIR}/Libra/Dialect/SISD/IR)

add_mlir_library(FLYHETranslateLib
  Translate.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRParser
  MLIRTransforms
  MLIRFuncDialect

  MLIRSCFHEDialect
)