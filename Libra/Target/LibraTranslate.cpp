#include "LibraTranslate.h"
#include "BackendDescriptor.h"

#include "mlir/IR/Operation.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::libra::backend;

// ------------------------------------------------------------
// Utility
// ------------------------------------------------------------

/// 为 SSA Value 分配一个名字
std::string LibraBackendEmitter::nameFor(Value v) {
    if (nameTable.count(v))
        return nameTable[v];

    // 通常你会根据 real SSA name 来，但 MLIR SSA 名默认不稳定
    std::string generated = "v" + std::to_string(nameTable.size());
    nameTable[v] = generated;
    return generated;
}

// ------------------------------------------------------------
// emitModule
// ------------------------------------------------------------
LogicalResult LibraBackendEmitter::emitModule(ModuleOp module) {
    os << "// === Libra Backend Output ===\n";

    for (Operation& op : module.getOps()) {
        if (failed(translate(op)))
            return failure();
    }

    return success();
}

// ------------------------------------------------------------
// translate() — 按类型分发
// ------------------------------------------------------------
LogicalResult LibraBackendEmitter::translate(Operation& op) {
    // func.func
    if (auto fn = dyn_cast<func::FuncOp>(op))
        return emitFunc(fn);

    // module level fallback (global, etc.)
    return printOperation(op);
}

// ------------------------------------------------------------
// emitFunc
// ------------------------------------------------------------
LogicalResult LibraBackendEmitter::emitFunc(func::FuncOp funcOp) {
    os << "\n// ---- Function: " << funcOp.getName() << " ----\n";
    os << "func " << funcOp.getName() << "() {\n";

    for (Block& block : funcOp.getBlocks()) {
        for (Operation& op : block) {
            if (failed(translate(op)))
                return failure();
        }
    }

    os << "}\n";
    return success();
}

// ------------------------------------------------------------
// printOperation: func.return
// ------------------------------------------------------------
LogicalResult LibraBackendEmitter::printOperation(func::ReturnOp op) {
    if (op.getNumOperands() == 0) {
        os << "  return;\n";
    } else {
        os << "  return " << nameFor(op.getOperand(0)) << ";\n";
    }
    return success();
}

// ------------------------------------------------------------
// printOperation: func.func (已由 emitFunc 处理，这里仅防御性实现)
// ------------------------------------------------------------
LogicalResult LibraBackendEmitter::printOperation(func::FuncOp funcOp) {
    os << "// [warning] unexpected nested func.func\n";
    return emitFunc(funcOp);
}

// ------------------------------------------------------------
// 通用 fallback (Operation*)：用户自定义的 op 会走这里
// ------------------------------------------------------------
LogicalResult LibraBackendEmitter::printOperation(Operation* op) {
    os << "  // Unhandled op: " << op->getName().getStringRef() << "\n";
    return success();
}

// ------------------------------------------------------------
// fallback: Operation& -> pointer 调用
// ------------------------------------------------------------
LogicalResult LibraBackendEmitter::printOperation(Operation& op) {
    return printOperation(&op);
}

// ------------------------------------------------------------
// printOperation(op, name) — 可以用来处理特定后端 op
// ------------------------------------------------------------
LogicalResult LibraBackendEmitter::printOperation(Operation* op, StringRef name) {
    os << "  " << name << "(";

    bool first = true;
    for (Value v : op->getOperands()) {
        if (!first)
            os << ", ";
        first = false;
        os << nameFor(v);
    }
    os << ");\n";
    return success();
}

// ------------------------------------------------------------
// registerLibraBackendTranslation
// ------------------------------------------------------------
void mlir::libra::backend::registerLibraBackendTranslation() {
    static BackendDescriptor* backend;
    // ↑ 这是你的后端描述符实例（你可以换成 FlyHEBackend、LLVMBackend 等）

    static TranslateFromMLIRRegistration reg(
        "emit-libra-backend",
        "Lower ops into backend-specific code for Libra and emit as text",

        // ---- Translation callback ----
        [&](ModuleOp module, llvm::raw_ostream& output) -> LogicalResult {
            LibraBackendEmitter emitter(output, backend);

            // ★ 这里你可以加 lowering 逻辑
            // 若希望与 pass 行为一致，可直接在 module.walk 中替换操作
            // 例如：
            /*
            module.walk([&](Operation *op) {
                auto name = op->getName().getStringRef();
                OpBuilder b(op);

                if (name == "simd.mult") {
                    auto fn = backend.getSIMDMultFunc();
                    auto call = b.create<LLVM::CallOp>(
                        op->getLoc(),
                        op->getResult(0).getType(),
                        FlatSymbolRefAttr::get(b.getContext(), fn),
                        op->getOperands());
                    op->getResult(0).replaceAllUsesWith(call.getResult());
                    op->erase();
                }
            });
            */

            return emitter.emitModule(module);
        },

        // ---- Dialect Registration ----
        [](DialectRegistry& registry) {
            registry.insert<func::FuncDialect>();
            registry.insert<LLVM::LLVMDialect>();
            registry.insert<mlir::libra::simd::SIMDDialect>();
            registry.insert<mlir::libra::sisd::SISDDialect>();
        });
}
