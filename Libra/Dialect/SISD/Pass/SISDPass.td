#ifndef PASS_SISD_PASS_TD_
#define PASS_SISD_PASS_TD_

include "mlir/Pass/PassBase.td"

def ConvertToSISDPass : Pass<"convert-to-sisd", "::mlir::ModuleOp"> {
  let summary = "build-in mlir to SISD";
  let description = [{
    This Pass converts build-in mlir to SISD mlir.
  }];

  let dependentDialects = [
    "::mlir::libra::sisd::SISDDialect",
    "::mlir::libra::simd::SIMDDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::arith::ArithDialect"
  ];
}

#endif // PASS_SISD_PASS_TD_