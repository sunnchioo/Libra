#include "IR/SCFHEDialect.h"

#include "llvm/Support/raw_ostream.h"
#define FIX
#include "IR/SCFHEDialect.cpp.inc"
#undef FIX

namespace mlir::scfhe {
    // 实现方言的初始化方法
    void SCFHEDialect::initialize() {
        llvm::outs() << "initializing " << getDialectNamespace() << "\n";
    }

    // 实现方言的析构函数
    SCFHEDialect::~SCFHEDialect() {
        llvm::outs() << "destroying " << getDialectNamespace() << "\n";
    }

    // 实现在extraClassDeclaration 声明当中生命的方法。
    void SCFHEDialect::sayHello() {
        llvm::outs() << "Hello in " << getDialectNamespace() << "\n";
    }

}  // namespace mlir::scfhe