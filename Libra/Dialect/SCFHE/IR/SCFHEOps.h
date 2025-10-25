#ifndef DIALECT_SCFHE_OPS_H_
#define DIALECT_SCFHE_OPS_H_

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "SCFHEDialect.h"
#include "SCFHETypes.h"

#define GET_OP_CLASSES
#include "SCFHEOps.h.inc"

#endif  // DIALECT_SCFHE_OPS_H_
