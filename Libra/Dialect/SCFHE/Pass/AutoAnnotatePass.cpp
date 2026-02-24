#include "SCFHEPass.h"
#include "SCFHEDialect.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;

namespace mlir::libra::scfhe {

#define GEN_PASS_DEF_AUTOANNOTATESCFHEPASS
#include "SCFHEPass.h.inc"

    struct AutoAnnotateSCFHEPass : public impl::AutoAnnotateSCFHEPassBase<AutoAnnotateSCFHEPass> {

        // 辅助函数：判断黑名单
        bool isForbiddenFunction(StringRef name) {
            const char* blackList[] = {
                "rand", "srand", "time", "clock",
                "printf", "puts", "scanf", "open", "read", "write",
                "malloc", "free"};

            for (const char* forbidden : blackList) {
                if (name.contains(forbidden))
                    return true;
            }
            return false;
        }

        void runOnOperation() override {
            ModuleOp module = getOperation();

            // === 第一层遍历：遍历 Module 中的所有 FuncOp ===
            module.walk([&](func::FuncOp funcOp) {
                // 1. [过滤] 跳过 main 和 外部声明
                if (funcOp.getName() == "main" || funcOp.isExternal())
                    return;

                // 定义局部变量（这些就是报错说未定义的变量）
                bool isBlacklisted = false;
                int computeScore = 0;

                // === 第二层遍历：遍历 FuncOp 中的所有 Operation ===
                funcOp.walk([&](Operation* op) -> WalkResult {
                    // --- A. 检查函数调用 (黑名单) ---
                    if (auto callOp = dyn_cast<func::CallOp>(op)) {
                        StringRef callee = callOp.getCallee();
                        if (isForbiddenFunction(callee)) {
                            isBlacklisted = true;
                            return WalkResult::interrupt(); // 发现非法调用，立即停止
                        }
                    }

                    // --- B. 检查 LLVM 调用 (I/O) ---
                    if (isa<LLVM::CallOp>(op)) {
                        isBlacklisted = true;
                        return WalkResult::interrupt();
                    }

                    // --- C. 计算评分 ---
                    if (isa<affine::AffineForOp>(op)) {
                        computeScore += 10;
                    } else if (isa<arith::AddFOp, arith::MulFOp, arith::SubFOp, arith::DivFOp>(op)) {
                        computeScore += 1;
                    }

                    // [重要] 必须返回 advance 以继续遍历
                    return WalkResult::advance();
                });

                // 3. [判定]
                if (!isBlacklisted && computeScore >= 5) {
                    funcOp->setAttr("scfhe.crypto", UnitAttr::get(&getContext()));
                }
            });
        }
    };

} // namespace mlir::libra::scfhe