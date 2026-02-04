#ifndef DIALECT_SISD_OPS_H_
#define DIALECT_SISD_OPS_H_

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "SISDDialect.h"
#include "SISDTypes.h"

#define GET_OP_CLASSES
#include "SISDOps.h.inc"

#endif  // DIALECT_SISD_OPS_H_
