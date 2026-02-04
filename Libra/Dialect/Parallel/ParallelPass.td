// ParallelPass.td

#ifndef PASS_PARALLELPASS_TD_
#define PASS_PARALLELPASS_TD_

include "mlir/Pass/PassBase.td"

// [FIX] 修改这里：将 "func::FuncOp" 改为 "::mlir::ModuleOp"
def AddBoundedStreamId : Pass<"add-bounded-stream-id", "::mlir::ModuleOp"> {
  let summary = "Assign CUDA stream IDs based on topological levels with resource bounds";
  let description = [{
    This pass analyzes the data dependency graph within a function block.
    It calculates the topological depth (level) of each operation and assigns
    parallel stream IDs. It attempts to reuse streams from predecessors to minimize
    synchronization overhead while strictly respecting the `maxStreams` limit.
  }];

  // ... 其他选项保持不变 ...
  let options = [
    Option<"maxStreams", "max-streams", "int", /*default=*/"16",
           "Maximum number of CUDA streams to use in parallel">
  ];

  let dependentDialects = [
    "::mlir::func::FuncDialect",
    "::mlir::vector::VectorDialect"
  ];
}

#endif // PASS_PARALLELPASS_TD_