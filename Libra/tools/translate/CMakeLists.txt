# get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
# get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)
set(LIBS
        ${dialect_libs}
        ${translation_libs}

        MLIROptLib
        MLIRAnalysis
        MLIRCallInterfaces
        MLIRCastInterfaces
        MLIRExecutionEngine
        MLIRIR
        MLIRLLVMCommonConversion
        MLIRLLVMToLLVMIRTranslation
        MLIRMemRefDialect
        MLIRParser
        MLIRPass
        MLIRTargetCpp
        MLIRTranslateLib
        MLIRSideEffectInterfaces
        MLIRTargetLLVMIRExport
        MLIRTransforms
)
add_llvm_executable(libra-translate
        libra-translate.cpp)

llvm_update_compile_flags(libra-translate)
target_link_libraries(libra-translate PUBLIC ${LIBS})

mlir_check_all_link_libraries(libra-translate)