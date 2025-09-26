#ifndef PASS_SISD_PASS_TD_
#define PASS_SISD_PASS_TD_

include "mlir/Pass/PassBase.td"

def ConvertToSISDPass : Pass<"convert-to-sisd", "::mlir::ModuleOp"> {
  let summary = "memref<?xf64> to SISD_SISDCipher";
  let description = [{
    This Pass converts all memref<?xf64> types to SISD_SISDCipher type and replaces the related memory operations.
  }];

  // let cppNamespace = "::mlir::flyhe";

  let dependentDialects = [
    "::mlir::flyhe::SISDDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::arith::ArithDialect"
  ];
}

#endif // PASS_SISD_PASS_TD_