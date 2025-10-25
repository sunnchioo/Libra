#ifndef PASS_SCFHE_PASS_H_
#define PASS_SCFHE_PASS_H_

// #include "Dialect/SCFHE/IR/SCFHEDialect.h"
// #include "Dialect/SCFHE/IR/SCFHEOps.h"
#include "SCFHEDialect.h"
#include "SCFHEOps.h"

#include "mlir/Pass/Pass.h"

namespace mlir {
    namespace libra {
        namespace scfhe {
#define GEN_PASS_DECL
#include "SCFHEPass.h.inc"

#define GEN_PASS_REGISTRATION
#include "SCFHEPass.h.inc"
        }  // namespace mlir::scfhe
    }  // namespace mlir::libra
}
#endif  // PASS_SCFHE_PASS_H_