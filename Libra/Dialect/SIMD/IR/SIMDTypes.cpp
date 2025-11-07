#include "SIMDTypes.h"
#include "SIMDDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::libra::simd;

#define GET_TYPEDEF_CLASSES
#include "SIMDTypes.cpp.inc"