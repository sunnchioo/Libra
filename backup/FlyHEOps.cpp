#include "mlir/IR/OpImplementation.h"
#include "FlyHEDialect.h"
#include "FlyHEOps.h.inc"  // 由 tablegen 生成的操作声明

using namespace mlir;
using namespace mlir::flyhe;  // 假设你的方言命名空间是 flyhe

// 实现操作的验证逻辑、解析/打印逻辑等
#define GET_OP_CLASSES
#include "FlyHEOps.cpp.inc"  // 由 tablegen 生成的操作定义

// （可选）如果需要自定义操作的解析/打印逻辑，在此实现
// 例如：
// void MyOp::print(OpAsmPrinter &p) { ... }
// ParseResult MyOp::parse(OpAsmParser &parser, OperationState &result) { ... }