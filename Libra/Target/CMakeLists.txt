# add_subdirectory(FlyHE)
# include_directories(${PROJECT_SOURCE_DIR}/Libra/Dialect/SCFHE/IR)
# include_directories(${PROJECT_BINARY_DIR}/Libra/Dialect/SCFHE/IR)

# include_directories(${PROJECT_SOURCE_DIR}/Libra/Dialect/SIMD/IR)
# include_directories(${PROJECT_BINARY_DIR}/Libra/Dialect/SIMD/IR)

# include_directories(${PROJECT_SOURCE_DIR}/Libra/Dialect/SISD/IR)
# include_directories(${PROJECT_BINARY_DIR}/Libra/Dialect/SISD/IR)

add_mlir_library(FLYHETranslateLib
  LibraTranslate.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRParser
  MLIRTransforms
  MLIRFuncDialect
  MLIRVectorDialect

  MLIRSCFHEDialect
  MLIRSIMDDialect 
  MLIRSISDDialect
)

target_include_directories(FLYHETranslateLib PUBLIC
  # 源码目录 (用于找到 .h)
  ${PROJECT_SOURCE_DIR}/Libra/Dialect/SCFHE/IR
  ${PROJECT_SOURCE_DIR}/Libra/Dialect/SIMD/IR
  ${PROJECT_SOURCE_DIR}/Libra/Dialect/SISD/IR
  
  # 构建目录 (用于找到 TableGen 生成的 .h.inc)
  ${PROJECT_BINARY_DIR}/Libra/Dialect/SCFHE/IR
  ${PROJECT_BINARY_DIR}/Libra/Dialect/SIMD/IR
  ${PROJECT_BINARY_DIR}/Libra/Dialect/SISD/IR
)