#ifndef PASS_FlyHE_PASS_H_
#define PASS_FlyHE_PASS_H_

// #include "Dialect/FlyHE/IR/FlyHEDialect.h"
// #include "Dialect/FlyHE/IR/FlyHEOps.h"
#include "FlyHEDialect.h"
#include "FlyHEOps.h"

#include "mlir/Pass/Pass.h"

namespace mlir {
    namespace flyhe {
#define GEN_PASS_DECL
#include "FlyHEPass.h.inc"

#define GEN_PASS_REGISTRATION
#include "FlyHEPass.h.inc"
    }  // namespace mlir::FlyHE
}
#endif  // PASS_FlyHE_PASS_H_