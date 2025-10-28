#include "SCFHETypes.h"
#include "SCFHEDialect.h"  // 假设你的 Dialect 头文件是这个
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::libra::scfhe;

#define GET_TYPEDEF_CLASSES
#include "SCFHETypes.cpp.inc"