#ifndef PASS_SIMD_PASS_TD_
#define PASS_SIMD_PASS_TD_

include "mlir/Pass/PassBase.td"

def ConvertToSIMDPass : Pass<"convert-to-simd", "::mlir::ModuleOp"> {
  let summary = "build-in mlir to SIMD";
  let description = [{
    This Pass converts build-in mlir to SIMD mlir.
  }];

  let dependentDialects = [
    "::mlir::libra::simd::SIMDDialect",
    "::mlir::libra::scfhe::SCFHEDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::arith::ArithDialect"
  ];
}

#endif // PASS_SIMD_PASS_TD_