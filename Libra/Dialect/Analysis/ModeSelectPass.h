#ifndef PASS_MODESELECT_PASS_H_
#define PASS_MODESELECT_PASS_H_

#include "mlir/Pass/Pass.h"

namespace mlir {
    namespace libra {
        namespace mdsel {
#define GEN_PASS_DECL
#include "ModeSelectPass.h.inc"

#define GEN_PASS_REGISTRATION
#include "ModeSelectPass.h.inc"
        } // namespace mlir::libra::mdsel
    } // namespace mlir::libra
}
#endif // PASS_MODESELECT_PASS_H_