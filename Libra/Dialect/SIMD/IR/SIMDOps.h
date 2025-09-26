#ifndef DIALECT_SIMD_OPS_H_
#define DIALECT_SIMD_OPS_H_

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "SIMDDialect.h"
#include "SIMDTypes.h"

#define GET_OP_CLASSES
#include "SIMDOps.h.inc"

#endif  // DIALECT_SIMD_OPS_H_
