#ifndef PASS_SCFHE_PASS_TD_
#define PASS_SCFHE_PASS_TD_

include "mlir/Pass/PassBase.td"

def ConvertToSCFHEPass : Pass<"convert-to-scfhe", "::mlir::ModuleOp"> {
  let summary = "build-in mlir to SCFHE";
  let description = [{
    This Pass converts build-in mlir to SCFHE mlir.
  }];

  let dependentDialects = [
    "::mlir::scf::SCFDialect",
    "::mlir::libra::scfhe::SCFHEDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::arith::ArithDialect"
  ];
}

def AutoAnnotateSCFHEPass : Pass<"auto-annotate-scfhe", "::mlir::ModuleOp"> {
  let summary = "Automatically annotates functions for SCFHE encryption";
  let description = [{
    Scans the module and adds the 'scfhe.crypto' attribute to functions 
    that perform heavy arithmetic operations, marking them for HE conversion.
    Skips 'main' and I/O functions.
  }];

  let dependentDialects = [
    "::mlir::libra::scfhe::SCFHEDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::arith::ArithDialect"
  ];

  // 这里的 constructor 必须和你 C++ (SCFHEPass.h) 中声明的函数名完全一致
  // let constructor = "mlir::libra::scfhe::createAutoAnnotateSCFHEPass()";
}

#endif // PASS_SCFHE_PASS_TD_