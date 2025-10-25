#ifndef PASS_SCFHE_PASS_TD_
#define PASS_SCFHE_PASS_TD_

include "mlir/Pass/PassBase.td"

def ConvertToSCFHEPass : Pass<"convert-to-scfhe", "::mlir::ModuleOp"> {
  let summary = "build-in mlir to SCFHE";
  let description = [{
    This Pass converts build-in mlir to SCFHE mlir.
  }];

  let dependentDialects = [
    "::mlir::libra::scfhe::SCFHEDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::arith::ArithDialect"
  ];
}

#endif // PASS_SCFHE_PASS_TD_