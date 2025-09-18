#ifndef LIBRA_DIALECT_FLYHE_IR_FLYHEOPS_H_
#define LIBRA_DIALECT_FLYHE_IR_FLYHEOPS_H_

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// 包含方言定义头文件
#include "FlyHEDialect.h"
#include "FlyHETypes.h"

#define GET_OP_CLASSES
#include "FlyHEOps.h.inc"

#endif  // LIBRA_DIALECT_FLYHE_IR_FLYHEOPS_H_
