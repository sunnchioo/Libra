#ifndef PASS_FlyHE_PASS_TD_
#define PASS_FlyHE_PASS_TD_

include "mlir/Pass/PassBase.td"

def ConvertToSIMDPass : Pass<"convert-to-simd", "::mlir::ModuleOp"> {
  let summary = "memref<?xf64> to FlyHE_SIMDCipher";
  let description = [{
    This Pass converts all memref<?xf64> types to FlyHE_SIMDCipher type and replaces the related memory operations.
  }];
  let dependentDialects = [
    "::mlir::flyhe::FlyHEDialect",
    "::mlir::memref::MemRefDialect"
  ];
}

#endif // PASS_FlyHE_PASS_TD_