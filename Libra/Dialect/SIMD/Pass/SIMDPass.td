#ifndef PASS_SIMD_PASS_TD_
#define PASS_SIMD_PASS_TD_

include "mlir/Pass/PassBase.td"

def ConvertToSIMDPass : Pass<"convert-to-simd", "::mlir::ModuleOp"> {
  let summary = "memref<?xf64> to SIMD_SIMDCipher";
  let description = [{
    This Pass converts all memref<?xf64> types to SIMD_SIMDCipher type and replaces the related memory operations.
  }];

  // let cppNamespace = "::mlir::flyhe";

  let dependentDialects = [
    "::mlir::flyhe::SIMDDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::arith::ArithDialect"
  ];
}

#endif // PASS_SIMD_PASS_TD_