#include "mlir/IR/DialectImplementation.h"
#include "FlyHEDialect.h"

using namespace mlir;
using namespace mlir::flyhe;

// 注册方言
void FlyHEDialect::initialize() {
    addTypes<
#define GET_TYPEDEF_LIST
#include "FlyHETypes.cpp.inc"
        >();
    addOperations<
#define GET_OP_LIST
#include "FlyHEOps.cpp.inc"
        >();
}
