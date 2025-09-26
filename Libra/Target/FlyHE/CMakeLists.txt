add_mlir_library(FLYHETranslateLib
  FlyHETranslate.cpp

  # ADDITIONAL_HEADER_DIRS
  # ${PROJECT_SOURCE_DIR}/Tool/llvm-project/mlir/Tools/mlir-translate

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRParser
  MLIRTransforms
  MLIRFuncDialect
)