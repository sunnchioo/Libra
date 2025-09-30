#include "FlyHETranslate.h"
#include "FlyHEDialect.h"
#include "FlyHETemplates.h"

// #include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/IndentedOstream.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir {
    namespace libra {
        namespace flyhe_cuda {

            // std::string flyhecudaemitter::nameFor(Value v) {
            //     if (nameTable.count(v))
            //         return nameTable.lookup(v);
            //     std::string newName =
            //         "v" + std::to_string(reinterpret_cast<uintptr_t>(v.getAsOpaquePointer()));
            //     nameTable[v] = newName;
            //     return newName;
            // }

            // std::string flyhecudaemitter::nameFor(Value v) {
            //     // 已经有名字了，直接返回
            //     if (nameTable.count(v))
            //         return nameTable.lookup(v);

            //     std::string newName;

            //     // 如果是函数参数，按照顺序命名 SIMDCipher0, SIMDCipher1, ...
            //     if (auto arg = dyn_cast<BlockArgument>(v)) {
            //         if (auto func = dyn_cast<func::FuncOp>(arg.getOwner()->getParentOp())) {
            //             unsigned index = arg.getArgNumber();
            //             newName = "SIMDCipher" + std::to_string(index);
            //         }
            //     } else if (auto res = dyn_cast<OpResult>(v)) {
            //         // 如果是某个运算的结果 => SIMDCipher<result_number>
            //         Operation *parentOp = res.getOwner();
            //         unsigned resIndex = res.getResultNumber();
            //         // 可以用 parentOp 的唯一编号来区分，不和参数冲突
            //         // unsigned uniqueId = reinterpret_cast<uintptr_t>(parentOp) & 0xFFFF;
            //         // newName = "SIMDCipher_rtn_" + std::to_string(uniqueId) + "_" + std::to_string(resIndex);
            //         newName = "SIMDCipher_rtn_" + std::to_string(resIndex);
            //     }

            //     // 否则还是默认规则
            //     if (newName.empty()) {
            //         newName = "v" + std::to_string(reinterpret_cast<uintptr_t>(v.getAsOpaquePointer()));
            //     }

            //     nameTable[v] = newName;
            //     return newName;
            // }

            std::string flyhecudaemitter::nameFor(Value v) {
                // 已经有名字了，直接返回
                if (nameTable.count(v))
                    return nameTable.lookup(v);

                std::string newName;

                // 统一用计数器编号
                static unsigned counter = 0;

                if (auto arg = dyn_cast<BlockArgument>(v)) {
                    // 函数参数 -> 直接用流水号
                    unsigned index = arg.getArgNumber();
                    newName = "SIMDCipher" + std::to_string(counter++);
                } else if (auto res = dyn_cast<OpResult>(v)) {
                    // 运算结果 -> 同样用流水号
                    newName = "SIMDCipher" + std::to_string(counter++);
                } else {
                    // fallback
                    newName = "v" + std::to_string(reinterpret_cast<uintptr_t>(v.getAsOpaquePointer()));
                }

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

            // LogicalResult flyhecudaemitter::emitFunc(func::FuncOp funcOp) {
            //     for (auto &block : funcOp.getBody()) {
            //         for (auto &op : block) {
            //             if (failed(translate(op)))
            //                 return failure();
            //         }
            //     }
            //     return success();
            // }

            LogicalResult flyhecudaemitter::emitFunc(func::FuncOp funcOp) {
                for (auto &block : funcOp.getBody()) {
                    for (auto &op : block) {
                        // 只处理 call 和 flyhe.* op
                        auto opname = op.getName().getStringRef();

                        if (isa<func::CallOp>(&op) || opname.starts_with("flyhe.")) {
                            if (failed(translate(op)))
                                return failure();
                        }
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

            // LogicalResult flyhecudaemitter::printOperation(Operation *op) {
            //     auto opname = op->getName().getStringRef();
            //     if (opname == "flyhe.simd_load") {
            //         auto res = op->getResult(0);
            //         auto varName = nameFor(res);
            //         os << "    auto " << varName << " = " << nameFor(op->getOperand(0)) << ";\n";
            //         return success();
            //     }
            //     if (opname == "flyhe.simdadd") {
            //         auto lhs = nameFor(op->getOperand(0));
            //         auto rhs = nameFor(op->getOperand(1));
            //         auto res = nameFor(op->getResult(0));
            //         os << "    ckks_evaluator.evaluator.add(" << lhs << ", " << rhs << ", rtn);\n";
            //         return success();
            //     }

            //     os << "    // TODO: unhandled op: " << opname << "\n";
            //     return success();
            // }

            LogicalResult flyhecudaemitter::printOperation(Operation *op) {
                auto opname = op->getName().getStringRef();

                if (opname == "flyhe.simd_load") {
                    // 直接复用操作数的名字，不再生成 auto 临时变量
                    auto res = op->getResult(0);
                    auto varName = nameFor(op->getOperand(0));
                    nameTable[res] = varName;  // 让结果值共享参数名字
                    return success();
                }

                if (opname == "flyhe.simdadd") {
                    auto lhs = nameFor(op->getOperand(0));
                    auto rhs = nameFor(op->getOperand(1));
                    auto res = nameFor(op->getResult(0));
                    os << kAddCall << "(" << lhs << ", " << rhs << ", " << res << ");\n";
                    return success();
                }

                if (opname == "flyhe.simdmult") {
                    auto lhs = nameFor(op->getOperand(0));
                    auto rhs = nameFor(op->getOperand(1));
                    auto res = nameFor(op->getResult(0));
                    os << kMultCall << "(" << lhs << ", " << rhs << ", " << res << ");\n";
                    return success();
                }

                if (opname == "flyhe.simd_store") {
                    auto value = nameFor(op->getOperand(0));
                    auto dest = nameFor(op->getOperand(1));
                    os << "    // store result\n";
                    os << "    " << dest << " = " << value << ";\n";
                    return success();
                }

                os << "    // TODO: unhandled op: " << opname << "\n";
                return success();
            }

            // === Registration ===
            void registerFlyHECUDATranslation() {
                static TranslateFromMLIRRegistration reg(
                    "emit-flyhe-cuda", "emit CUDA CKKS code",
                    [](ModuleOp module, raw_ostream &output) {
                        flyhecudaemitter emitter(output);
                        return emitter.emitModule(module);
                    },
                    [](DialectRegistry &registry) { 
                        registry.insert<func::FuncDialect>();
                        registry.insert<mlir::flyhe::FlyHEDialect>(); 
                        registry.insert<LLVM::LLVMDialect>(); });
            }
        }
    }
}