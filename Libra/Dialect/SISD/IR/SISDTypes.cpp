#include "SISDTypes.h"
#include "SISDDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::libra::sisd;

#define GET_TYPEDEF_CLASSES
#include "SISDTypes.cpp.inc"