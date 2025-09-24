#ifndef DIALECT_FlyHE_OPS_H_
#define DIALECT_FlyHE_OPS_H_

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "FlyHEDialect.h"
#include "FlyHETypes.h"

#define GET_OP_CLASSES
#include "FlyHEOps.h.inc"

#endif  // DIALECT_FlyHE_OPS_H_
