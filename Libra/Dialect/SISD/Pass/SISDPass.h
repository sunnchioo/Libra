#ifndef PASS_SISD_PASS_H_
#define PASS_SISD_PASS_H_

// #include "Dialect/SISD/IR/SISDDialect.h"
// #include "Dialect/SISD/IR/SISDOps.h"
#include "SISDDialect.h"
#include "SISDOps.h"

#include "mlir/Pass/Pass.h"

namespace mlir {
    namespace sisd {
#define GEN_PASS_DECL
#include "SISDPass.h.inc"

#define GEN_PASS_REGISTRATION
#include "SISDPass.h.inc"
    }  // namespace mlir::SISD
}
#endif  // PASS_SISD_PASS_H_