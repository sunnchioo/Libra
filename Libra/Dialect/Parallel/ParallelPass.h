#ifndef PASS_ParallelPass_H_
#define PASS_ParallelPass_H_

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
    namespace libra {
        namespace parallel {

#define GEN_PASS_DECL
#include "ParallelPass.h.inc"

#define GEN_PASS_REGISTRATION
#include "ParallelPass.h.inc"
        }  // namespace mlir::libra::parallel
    }  // namespace mlir::libra
}  // namespace mlir

#endif  // PASS_ParallelPass_H_