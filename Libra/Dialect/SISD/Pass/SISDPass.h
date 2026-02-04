#ifndef PASS_SISD_PASS_H_
#define PASS_SISD_PASS_H_

#include "SISDDialect.h"
#include "SISDOps.h"

#include "mlir/Pass/Pass.h"

namespace mlir {
    namespace libra {
        namespace sisd {
#define GEN_PASS_DECL
#include "SISDPass.h.inc"

#define GEN_PASS_REGISTRATION
#include "SISDPass.h.inc"
        }  // namespace mlir::sisd
    }  // namespace mlir::libra
}
#endif  // PASS_SISD_PASS_H_