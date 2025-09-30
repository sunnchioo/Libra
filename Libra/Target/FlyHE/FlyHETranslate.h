#pragma once

// #include "mlir/IR/BuiltinOps.h"
// #include "mlir/IR/OpImplementation.h"
// #include "mlir/Pass/Pass.h"
// #include "mlir/Support/LogicalResult.h"
// #include "mlir/include/mlir/Tools/mlir-translate/Translation.h"
// #include "mlir/IR/Value.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseMap.h"

namespace mlir {
    namespace libra {
        namespace flyhe_cuda {

            /// Emitter for translating MLIR with flyhe dialect ops into CUDA CKKS C++ code.

            void registerFlyHECUDATranslation();

            class flyhecudaemitter {
            public:
                flyhecudaemitter(raw_ostream &os) : os(os) {}

                LogicalResult emitModule(ModuleOp module);

            private:
                raw_ostream &os;
                llvm::DenseMap<Value, std::string> nameTable;

                std::string nameFor(Value v);

                LogicalResult emitFunc(func::FuncOp funcOp);
                LogicalResult translate(Operation &op);

                // === Op Printers ===
                LogicalResult printOperation(func::ReturnOp op);
                LogicalResult printOperation(func::FuncOp funcOp);
                LogicalResult printOperation(Operation *op);  // fallback
                LogicalResult printOperation(Operation &op);  // generic dispatcher
                LogicalResult printOperation(ModuleOp module);

                // Specific FlyHE ops
                LogicalResult printOperation(Operation *op, StringRef name);
            };

        }
    }
}  // namespace mlir::flyhecudaemitter
