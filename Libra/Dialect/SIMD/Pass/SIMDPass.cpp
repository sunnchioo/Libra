#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "SCFHEOps.h"
#include "SCFHETypes.h"
#include "SIMDOps.h"
#include "SIMDPass.h"

#include "SIMDCommon.h"

namespace mlir::libra::simd {
#define GEN_PASS_DEF_CONVERTTOSIMDPASS
#include "SIMDPass.h.inc"

    namespace {

        // [Fix] Define default level
        static constexpr int64_t DEFAULT_LEVEL = 31;

        /// 封装通用类型转换：SCFHE → SIMD
        static FailureOr<SIMDCipherType> convertToSIMDType(Type t, MLIRContext* ctx,
                                                           int64_t level = DEFAULT_LEVEL) {
            if (auto st = dyn_cast<SIMDCipherType>(t))
                return st;
            if (auto st = dyn_cast<scfhe::SCFHECipherType>(t))
                // [Fix] Added Scale=1 (Clean), Basis=2 (Standard)
                return SIMDCipherType::get(ctx, level, st.getPlaintextCount(), st.getElementType(), 1, 2);
            return failure();
        }

        /// 根据两个 operand 类型推断 SIMD 结果类型
        static FailureOr<SIMDCipherType> inferSIMDResultType(Value a, Value b, MLIRContext* ctx,
                                                             bool reduceLevel = false) {
            auto ta = convertToSIMDType(a.getType(), ctx);
            auto tb = convertToSIMDType(b.getType(), ctx);
            if (failed(ta) || failed(tb))
                return failure();

            // [Fix] Use arrow operator -> to access FailureOr value
            int64_t newLevel = std::min(ta->getLevel(), tb->getLevel());
            if (reduceLevel)
                newLevel = std::max<int64_t>(0, newLevel - 1);

            int64_t newPC = std::min(ta->getPlaintextCount(), tb->getPlaintextCount());

            // [Fix] Added Scale=1, Basis=2.
            // For Mult/Sub phase 1 conversion, we assume Clean/Standard output initially.
            return SIMDCipherType::get(ctx, newLevel, newPC, ta->getElementType(), 1, 2);
        }

        //===----------------------------------------------------------------------===//
        // Phase 1: SCFHE → SIMD
        //===----------------------------------------------------------------------===//

        struct ConvertSCFHEEncryptToSIMDPattern : OpRewritePattern<scfhe::SCFHEEncryptOp> {
            using OpRewritePattern::OpRewritePattern;
            LogicalResult matchAndRewrite(scfhe::SCFHEEncryptOp op,
                                          PatternRewriter& rewriter) const override {
                auto ctx = rewriter.getContext();
                auto ty = convertToSIMDType(op.getResult().getType(), ctx);
                if (failed(ty))
                    return failure();
                auto newOp = rewriter.create<SIMDEncryptOp>(op.getLoc(), *ty, op.getOperand());
                rewriter.replaceOp(op, newOp);
                return success();
            }
        };

        struct ConvertSCFHESubToSIMDPattern : OpRewritePattern<scfhe::SCFHESubOp> {
            using OpRewritePattern::OpRewritePattern;
            LogicalResult matchAndRewrite(scfhe::SCFHESubOp op,
                                          PatternRewriter& rewriter) const override {
                auto ctx = rewriter.getContext();
                auto ty = inferSIMDResultType(op.getOperand(0), op.getOperand(1), ctx);
                if (failed(ty))
                    return failure();
                auto newOp = rewriter.create<SIMDSubOp>(op.getLoc(), *ty, op.getOperands());
                rewriter.replaceOp(op, newOp);
                return success();
            }
        };

        struct ConvertSCFHEMultToSIMDPattern : OpRewritePattern<scfhe::SCFHEMultOp> {
            using OpRewritePattern::OpRewritePattern;
            LogicalResult matchAndRewrite(scfhe::SCFHEMultOp op,
                                          PatternRewriter& rewriter) const override {
                auto ctx = rewriter.getContext();
                auto ty = inferSIMDResultType(op.getOperand(0), op.getOperand(1), ctx, /*reduceLevel=*/true);
                if (failed(ty))
                    return failure();
                auto newOp = rewriter.create<SIMDMultOp>(op.getLoc(), *ty, op.getOperands());
                rewriter.replaceOp(op, newOp);
                return success();
            }
        };

        struct ConvertSCFHEMinToSIMDPattern : OpRewritePattern<scfhe::SCFHEMinOp> {
            using OpRewritePattern::OpRewritePattern;
            LogicalResult matchAndRewrite(scfhe::SCFHEMinOp op,
                                          PatternRewriter& rewriter) const override {
                auto ctx = rewriter.getContext();
                auto inTy = convertToSIMDType(op.getOperand().getType(), ctx);
                if (failed(inTy))
                    return failure();
                // [Fix] Added Scale=1, Basis=2
                auto ty = SIMDCipherType::get(ctx, inTy->getLevel(), /*pc=*/1, inTy->getElementType(), 1, 2);
                auto newOp = rewriter.create<SIMDMinOp>(op.getLoc(), ty, op.getOperand());
                rewriter.replaceOp(op, newOp);
                return success();
            }
        };

        struct ConvertSCFHEDecryptToSIMDPattern : OpRewritePattern<scfhe::SCFHEDecryptOp> {
            using OpRewritePattern::OpRewritePattern;
            LogicalResult matchAndRewrite(scfhe::SCFHEDecryptOp op,
                                          PatternRewriter& rewriter) const override {
                auto newOp = rewriter.create<SIMDDecryptOp>(op.getLoc(), op.getResult().getType(),
                                                            op.getOperand());
                rewriter.replaceOp(op, newOp);
                return success();
            }
        };

        struct ConvertFuncOpTypesToSIMDPattern : OpRewritePattern<func::FuncOp> {
            using OpRewritePattern::OpRewritePattern;
            LogicalResult matchAndRewrite(func::FuncOp funcOp,
                                          PatternRewriter& rewriter) const override {
                auto ctx = rewriter.getContext();
                bool changed = false;

                SmallVector<Type> newArgs, newResults;
                for (auto t : funcOp.getArgumentTypes()) {
                    auto ty = convertToSIMDType(t, ctx);
                    if (succeeded(ty)) {
                        changed = true;
                        newArgs.push_back(*ty);
                    } else {
                        newArgs.push_back(t);
                    }
                }
                for (auto t : funcOp.getResultTypes()) {
                    auto ty = convertToSIMDType(t, ctx);
                    if (succeeded(ty)) {
                        changed = true;
                        newResults.push_back(*ty);
                    } else {
                        newResults.push_back(t);
                    }
                }

                if (!changed)
                    return failure();

                auto newTy = FunctionType::get(ctx, newArgs, newResults);
                rewriter.modifyOpInPlace(funcOp, [&]() {
                    funcOp.setType(newTy);
                    Block& entry = funcOp.front();
                    for (unsigned i = 0; i < entry.getNumArguments(); ++i)
                        entry.getArgument(i).setType(newArgs[i]);
                });
                return success();
            }
        };

        //===----------------------------------------------------------------------===//
        // Phase 2: 修正 SIMD level / plaintextCount
        //===----------------------------------------------------------------------===//

        struct AdjustSIMDMultLevelPattern : OpRewritePattern<SIMDMultOp> {
            using OpRewritePattern::OpRewritePattern;
            LogicalResult matchAndRewrite(SIMDMultOp op, PatternRewriter& rewriter) const override {
                auto ctx = rewriter.getContext();
                auto ty = inferSIMDResultType(op.getOperand(0), op.getOperand(1), ctx, /*reduceLevel=*/true);
                if (failed(ty))
                    return failure();
                if (*ty == op.getResult().getType())
                    return failure();
                auto newOp = rewriter.create<SIMDMultOp>(op.getLoc(), *ty, op.getOperands());
                rewriter.replaceOp(op, newOp);
                return success();
            }
        };

        struct AdjustSIMDSubLevelPattern : OpRewritePattern<SIMDSubOp> {
            using OpRewritePattern::OpRewritePattern;
            LogicalResult matchAndRewrite(SIMDSubOp op, PatternRewriter& rewriter) const override {
                auto ctx = rewriter.getContext();
                auto ty = inferSIMDResultType(op.getOperand(0), op.getOperand(1), ctx);
                if (failed(ty))
                    return failure();
                if (*ty == op.getResult().getType())
                    return failure();
                auto newOp = rewriter.create<SIMDSubOp>(op.getLoc(), *ty, op.getOperands());
                rewriter.replaceOp(op, newOp);
                return success();
            }
        };

        struct AdjustSIMDMinLevelPattern : OpRewritePattern<SIMDMinOp> {
            using OpRewritePattern::OpRewritePattern;
            LogicalResult matchAndRewrite(SIMDMinOp op, PatternRewriter& rewriter) const override {
                auto inTy = dyn_cast<SIMDCipherType>(op.getOperand().getType());
                if (!inTy)
                    return failure();
                // [Fix] Added Scale=1, Basis=2
                auto newTy = SIMDCipherType::get(rewriter.getContext(), inTy.getLevel(), /*pc=*/1,
                                                 inTy.getElementType(), 1, 2);
                if (newTy == op.getResult().getType())
                    return failure();
                auto newOp = rewriter.create<SIMDMinOp>(op.getLoc(), newTy, op.getOperand());
                rewriter.replaceOp(op, newOp);
                return success();
            }
        };

        //===----------------------------------------------------------------------===//
        // Pass
        //===----------------------------------------------------------------------===//

        struct ConvertSCFHEToSIMDPass
            : public impl::ConvertToSIMDPassBase<ConvertSCFHEToSIMDPass> {
            void runOnOperation() override {
                MLIRContext* ctx = &getContext();
                Operation* module = getOperation();

                // Phase 1
                {
                    RewritePatternSet patterns(ctx);
                    patterns.add<ConvertSCFHEEncryptToSIMDPattern,
                                 ConvertSCFHESubToSIMDPattern,
                                 ConvertSCFHEMultToSIMDPattern,
                                 ConvertSCFHEMinToSIMDPattern,
                                 ConvertSCFHEDecryptToSIMDPattern,
                                 ConvertFuncOpTypesToSIMDPattern>(ctx);
                    (void)applyPatternsGreedily(module, std::move(patterns));
                }

                // Phase 2
                {
                    RewritePatternSet patterns(ctx);
                    patterns.add<AdjustSIMDMultLevelPattern,
                                 AdjustSIMDSubLevelPattern,
                                 AdjustSIMDMinLevelPattern>(ctx);
                    (void)applyPatternsGreedily(module, std::move(patterns));
                }
            }
        };

    }  // namespace
}  // namespace mlir::libra::simd