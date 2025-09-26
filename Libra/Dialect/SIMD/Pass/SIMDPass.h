#ifndef PASS_SIMD_PASS_H_
#define PASS_SIMD_PASS_H_

// #include "Dialect/SIMD/IR/SIMDDialect.h"
// #include "Dialect/SIMD/IR/SIMDOps.h"
#include "SIMDDialect.h"
#include "SIMDOps.h"

#include "mlir/Pass/Pass.h"

namespace mlir {
    namespace simd {
#define GEN_PASS_DECL
#include "SIMDPass.h.inc"

#define GEN_PASS_REGISTRATION
#include "SIMDPass.h.inc"
    }  // namespace mlir::SIMD
}
#endif  // PASS_SIMD_PASS_H_