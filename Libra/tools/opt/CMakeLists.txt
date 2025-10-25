set(LIBS
  ${dialect_libs}
  ${conversion_libs}
  ${extension_libs}

  MLIRArithDialect
  MLIROptLib
  MLIRRegisterAllDialects
  MLIRRegisterAllExtensions
  MLIRRegisterAllPasses

  # TODO: Remove when registerAllGPUToLLVMIRTranslations is no longer
  # registered directly in libra-opt.cpp.
  # MLIRToLLVMIRTranslationRegistration

  MLIRSCFHEDialect
  SCFHEPasses
)

# add_mlir_tool(libra-opt  
#   libra-opt.cpp

#   DEPENDS
#   ${LIBS}
# )
add_llvm_executable(libra-opt 
  libra-opt.cpp
  # ../../Dialect/SCFHE/IR/SCFHEDialect.cpp
  # ../../Dialect/SCFHE/IR/SCFHEOps.cpp
)

target_include_directories(libra-opt
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../../Dialect/SCFHE/IR
  ${CMAKE_CURRENT_SOURCE_DIR}/../../Dialect/SCFHE/Pass
  ${CMAKE_CURRENT_BINARY_DIR}/../../Dialect/SCFHE/IR
  ${CMAKE_CURRENT_BINARY_DIR}/../../Dialect/SCFHE/Pass
)

llvm_update_compile_flags(libra-opt)

# mlir_target_link_libraries(libra-opt PRIVATE ${LIBS})
target_link_libraries(libra-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(libra-opt)
export_executable_symbols_for_plugins(libra-opt)