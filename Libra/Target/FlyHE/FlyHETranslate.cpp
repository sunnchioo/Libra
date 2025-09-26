#include "FlyHETranslate.h"
#include "FlyHETemplates.h"

// #include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/IndentedOstream.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir {
    namespace libra {
        namespace flyhe_cuda {

            std::string flyhecudaemitter::nameFor(Value v) {
                if (nameTable.count(v))
                    return nameTable.lookup(v);
                std::string newName =
                    "v" + std::to_string(reinterpret_cast<uintptr_t>(v.getAsOpaquePointer()));
                nameTable[v] = newName;
                return newName;
            }

            LogicalResult flyhecudaemitter::emitModule(ModuleOp module) {
                os << kCudaPrelude;
                for (Operation &op : module) {
                    if (failed(translate(op))) {
                        return failure();
                    }
                }
                os << kCudaTail;
                return success();
            }

            LogicalResult flyhecudaemitter::emitFunc(func::FuncOp funcOp) {
                for (auto &block : funcOp.getBody()) {
                    for (auto &op : block) {
                        if (failed(translate(op)))
                            return failure();
                    }
                }
                return success();
            }

            // === Dispatcher ===
            // LogicalResult flyhecudaemitter::translate(Operation &op) {
            //     return llvm::TypeSwitch<Operation *, LogicalResult>(&op)
            //         .Case<ModuleOp>([&](auto module) { return printOperation(module); })
            //         .Case<func::FuncOp>([&](auto func) { return printOperation(func); })
            //         .Case<func::ReturnOp>([&](auto ret) { return printOperation(ret); })
            //         .Default([&](Operation *op) {
            //             return printOperation(op);  // fallback string-based handler
            //         });
            // }
            LogicalResult flyhecudaemitter::translate(Operation &op) {
                return llvm::TypeSwitch<Operation *, LogicalResult>(&op)
                    .Case<ModuleOp>([&](ModuleOp module) { return printOperation(module); })
                    .Case<func::FuncOp>([&](func::FuncOp func) { return printOperation(func); })
                    .Case<func::ReturnOp>([&](func::ReturnOp ret) { return printOperation(ret); })
                    .Default([&](Operation *op) { return printOperation(op); });
            }

            // === Printers ===
            LogicalResult flyhecudaemitter::printOperation(ModuleOp module) {
                for (auto func : module.getOps<func::FuncOp>())
                    if (failed(emitFunc(func)))
                        return failure();
                return success();
            }

            LogicalResult flyhecudaemitter::printOperation(func::FuncOp funcOp) {
                return emitFunc(funcOp);
            }

            LogicalResult flyhecudaemitter::printOperation(func::ReturnOp) {
                // do nothing, handled by epilogue
                return success();
            }

            LogicalResult flyhecudaemitter::printOperation(Operation *op) {
                auto opname = op->getName().getStringRef();
                if (opname == "flyhe.simd_load") {
                    auto res = op->getResult(0);
                    auto varName = nameFor(res);
                    os << "    auto " << varName << " = " << nameFor(op->getOperand(0)) << ";\n";
                    return success();
                }
                if (opname == "flyhe.smidadd") {
                    auto lhs = nameFor(op->getOperand(0));
                    auto rhs = nameFor(op->getOperand(1));
                    auto res = nameFor(op->getResult(0));
                    os << "    ckks_evaluator.evaluator.add(" << lhs << ", " << rhs << ", rtn);\n";
                    return success();
                }

                os << "    // TODO: unhandled op: " << opname << "\n";
                return success();
            }

            // === Registration ===
            void registerflyhecudaemitterTranslation() {
                static TranslateFromMLIRRegistration reg(
                    "emit-flyhe-cuda", "emit CUDA CKKS code",
                    [](ModuleOp module, raw_ostream &output) {
                        flyhecudaemitter emitter(output);
                        return emitter.emitModule(module);
                    },
                    [](DialectRegistry &registry) { registry.insert<func::FuncDialect>(); });
            }
        }
    }
}