set(LLVM_TARGET_DEFINITIONS SCFHEDialect.td)
# 生成NorthStar Dialect 的声明
mlir_tablegen(SCFHEDialect.h.inc --gen-dialect-decls -dialect=scfhe)
# 生成NorthStar Dialect 的实现
mlir_tablegen(SCFHEDialect.cpp.inc --gen-dialect-defs -dialect=scfhe)

# 将生成的命令们定义为为target
add_public_tablegen_target(SCFHEDialectIncGen)