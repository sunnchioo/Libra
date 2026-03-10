// SCFHEAnalysis.h
#ifndef LIBRA_SCFHE_ANALYSIS_H
#define LIBRA_SCFHE_ANALYSIS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir::libra::scfhe {

    struct FunctionArgInfo {
        llvm::BitVector inputArgs;
        llvm::BitVector outputArgs;
    };

    class ArgAnalysis {
    public:
        llvm::DenseMap<func::FuncOp, FunctionArgInfo> infoMap;
        void run(ModuleOp module);
    };

} // namespace mlir::libra::scfhe

#endif // LIBRA_SCFHE_ANALYSIS_H