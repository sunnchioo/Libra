#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"

// scfhe
#include "IR/SCFHEDialect.h"

void SCFHE() {
    // 初始化方言注册器
    mlir::DialectRegistry registry;
    // 初始化上下文环境
    mlir::MLIRContext context(registry);
    // 加载/注册方言
    auto dialect = context.getOrLoadDialect<mlir::scfhe::SCFHEDialect>();
    // 调用方言中的方法
    dialect->sayHello();
}

int main() {
    SCFHE();
}