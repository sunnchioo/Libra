#ifndef PASS_MODESELECT_PASS_TD_
#define PASS_MODESELECT_PASS_TD_

include "mlir/Pass/PassBase.td"

def ConvertToMDSELPass : Pass<"mode-select", "::mlir::ModuleOp"> {
  let summary = "Select SIMD or SISD mode per operation and insert casts accordingly";
  let description = [{
    This pass analyzes the data-dependence DAG of operations in each function,
    runs dynamic programming to minimize execution and cast cost,
    and rewrites IR to insert casts and mode-specific ops.
  }];

  let dependentDialects = [
    "::mlir::libra::sisd::SISDDialect",
    "::mlir::libra::simd::SIMDDialect",
    // "::mlir::libra::scfhe::SCFHEDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::arith::ArithDialect"
  ];
}

#endif // PASS_MODESELECT_PASS_TD_