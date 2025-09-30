#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "FlyHEOps.h"
#include "FlyHEPass.h"

namespace mlir::flyhe {

#define GEN_PASS_DEF_CONVERTTOSIMDPASS
#include "FlyHEPass.h.inc"

    namespace {
        // class ConvertCallToFlyHEPattern : public OpRewritePattern<func::CallOp> {
        // public:
        //     using OpRewritePattern::OpRewritePattern;

        //     LogicalResult matchAndRewrite(func::CallOp call,
        //                                   PatternRewriter &rewriter) const override {
        //         auto callee = call.getCallee().str();

        //         // sum -> simdadd + store
        //         if (callee == "_Z3sumRdS_S_") {
        //             auto lhs = call.getOperand(0);
        //             auto rhs = call.getOperand(1);
        //             auto dst = call.getOperand(2);

        //             auto simdAdd = rewriter.create<flyhe::SIMDAddOp>(
        //                 call.getLoc(),
        //                 lhs.getType(),  // SIMD type
        //                 lhs,
        //                 rhs);

        //             // rewriter.create<flyhe::SIMDStoreOp>(call.getLoc(), simdAdd.getResult(), dst);
        //             rewriter.eraseOp(call);
        //             return success();
        //         }

        //         // mult -> simdmult + store
        //         if (callee == "_Z4multRdS_S_") {
        //             auto lhs = call.getOperand(0);
        //             auto rhs = call.getOperand(1);
        //             auto dst = call.getOperand(2);

        //             auto simdMul = rewriter.create<flyhe::SIMDMultOp>(
        //                 call.getLoc(),
        //                 lhs.getType(),
        //                 lhs,
        //                 rhs);

        //             // rewriter.create<flyhe::SIMDStoreOp>(call.getLoc(), simdMul.getResult(), dst);
        //             rewriter.eraseOp(call);
        //             return success();
        //         }

        //         return failure();
        //     }
        // };

        class ConvertAddfToSIMDAddPattern : public OpRewritePattern<arith::AddFOp> {
        public:
            using OpRewritePattern::OpRewritePattern;

            LogicalResult matchAndRewrite(arith::AddFOp addOp,
                                          PatternRewriter &rewriter) const override {
                auto lhsType = addOp.getOperand(0).getType();
                auto rhsType = addOp.getOperand(1).getType();
                if (!isa<SIMDCipherType>(lhsType) || !isa<SIMDCipherType>(rhsType))
                    return failure();

                auto simdAdd = rewriter.create<flyhe::SIMDAddOp>(
                    addOp.getLoc(),
                    lhsType,
                    addOp.getOperand(0),
                    addOp.getOperand(1));

                rewriter.replaceOp(addOp, simdAdd.getResult());
                return success();
            }
        };

        class ConvertMulfToSIMDMultPattern : public OpRewritePattern<arith::MulFOp> {
        public:
            using OpRewritePattern::OpRewritePattern;

            LogicalResult matchAndRewrite(arith::MulFOp mulOp,
                                          PatternRewriter &rewriter) const override {
                auto lhsType = mulOp.getOperand(0).getType();
                auto rhsType = mulOp.getOperand(1).getType();
                if (!isa<SIMDCipherType>(lhsType) || !isa<SIMDCipherType>(rhsType))
                    return failure();

                // 创建 flyhe.simdmult
                auto simdMul = rewriter.create<flyhe::SIMDMultOp>(
                    mulOp.getLoc(),
                    lhsType,
                    mulOp.getOperand(0),
                    mulOp.getOperand(1));

                rewriter.replaceOp(mulOp, simdMul.getResult());
                return success();
            }
        };

        class ConvertAffineLoadPattern : public OpRewritePattern<affine::AffineLoadOp> {
        public:
            using OpRewritePattern<affine::AffineLoadOp>::OpRewritePattern;

            LogicalResult matchAndRewrite(affine::AffineLoadOp loadOp,
                                          PatternRewriter &rewriter) const override {
                Value memref = loadOp.getMemRef();

                // 只处理 MemRefType
                auto memrefType = dyn_cast<MemRefType>(memref.getType());
                if (!memrefType)
                    return failure();

                auto simdType = SIMDCipherType::get(rewriter.getContext());

                // 创建 SIMDCipher 加载 op
                // auto simdLoad = rewriter.create<flyhe::SIMDLoadOp>(loadOp.getLoc());
                auto safeSimdLoad = rewriter.create<flyhe::SIMDLoadOp>(
                    loadOp.getLoc(),
                    simdType,
                    memref);

                rewriter.replaceOp(loadOp, safeSimdLoad.getResult());
                return success();
            }
        };

        class ConvertAffineStorePattern : public OpRewritePattern<affine::AffineStoreOp> {
        public:
            using OpRewritePattern<affine::AffineStoreOp>::OpRewritePattern;

            LogicalResult matchAndRewrite(affine::AffineStoreOp storeOp,
                                          PatternRewriter &rewriter) const override {
                Value value = storeOp.getValue();
                Value dest = storeOp.getMemRef();

                // Value must be a SIMD cipher (we produce simd stores for SIMD values).
                if (!isa<SIMDCipherType>(value.getType()))
                    return failure();

                // dest can be MemRefType (before memref->simd conversion) OR SIMDCipherType
                if (!(isa<MemRefType>(dest.getType()) || isa<SIMDCipherType>(dest.getType())))
                    return failure();

                // Create simd_store with current dest (memref or simd)
                // We pass the operands explicitly (no result types, simd_store is void).
                rewriter.create<flyhe::SIMDStoreOp>(storeOp.getLoc(), value, dest);

                // Remove original affine.store
                rewriter.eraseOp(storeOp);
                return success();
            }
        };

        class ConvertMemRefToSIMDCipherPattern : public OpRewritePattern<func::FuncOp> {
        public:
            using OpRewritePattern<func::FuncOp>::OpRewritePattern;

            LogicalResult matchAndRewrite(func::FuncOp funcOp,
                                          PatternRewriter &rewriter) const final {
                bool modified = false;

                SmallVector<Type, 4> newArgTypes;
                for (Type argType : funcOp.getArgumentTypes()) {
                    if (auto memrefType = dyn_cast<mlir::MemRefType>(argType)) {
                        newArgTypes.push_back(SIMDCipherType::get(rewriter.getContext()));
                        modified = true;
                    } else {
                        newArgTypes.push_back(argType);
                    }
                }

                SmallVector<Type, 4> newResultTypes;
                for (Type resultType : funcOp.getResultTypes()) {
                    if (dyn_cast<MemRefType>(resultType) || dyn_cast<FloatType>(resultType)) {
                        newResultTypes.push_back(SIMDCipherType::get(rewriter.getContext()));
                        modified = true;
                    } else {
                        newResultTypes.push_back(resultType);
                    }
                }

                if (!modified)
                    return failure();

                FunctionType newFuncType =
                    FunctionType::get(rewriter.getContext(), newArgTypes, newResultTypes);

                rewriter.modifyOpInPlace(funcOp, [&]() { funcOp.setType(newFuncType); });

                // 安全更新每个操作的 operand/result
                funcOp.walk([&](Operation *op) {
                    for (unsigned i = 0; i < op->getNumResults(); ++i) {
                        Type t = op->getResult(i).getType();
                        if (isa<MemRefType>(t))
                            op->getResult(i).setType(SIMDCipherType::get(rewriter.getContext()));
                    }
                    for (unsigned i = 0; i < op->getNumOperands(); ++i) {
                        Type t = op->getOperand(i).getType();
                        if (isa<MemRefType>(t))
                            op->getOperand(i).setType(SIMDCipherType::get(rewriter.getContext()));
                    }
                });

                return success();
            }
        };

        class ConvertMemRefToSIMDCipher
            : public impl::ConvertToSIMDPassBase<ConvertMemRefToSIMDCipher> {
        public:
            using impl::ConvertToSIMDPassBase<ConvertMemRefToSIMDCipher>::ConvertToSIMDPassBase;

            void runOnOperation() final {
                Operation *op = getOperation();

                for (func::FuncOp f : getOperation().getOps<func::FuncOp>()) {
                    if (f.getName() == "_Z3sumRdS_S_" || f.getName() == "_Z4multRdS_S_") {
                        // apply load/add/mul/store patterns only within f
                        RewritePatternSet patterns(f.getContext());
                        patterns.add<ConvertAffineLoadPattern, ConvertAddfToSIMDAddPattern, ConvertMulfToSIMDMultPattern, ConvertAffineStorePattern, ConvertMemRefToSIMDCipherPattern>(&getContext());
                        (void)applyPatternsGreedily(f, std::move(patterns));
                    }
                }

                // 1. affine.load -> flyhe.simd_load
                // {
                //     RewritePatternSet loadPatterns(&getContext());
                //     loadPatterns.add<ConvertAffineLoadPattern>(&getContext());
                //     FrozenRewritePatternSet frozenLoad(std::move(loadPatterns));
                //     if (failed(applyPatternsGreedily(op, frozenLoad))) {
                //         signalPassFailure();
                //         return;
                //     }
                // }

                // 2. arith.addf -> flyhe.smidadd
                // {
                //     RewritePatternSet addPatterns(&getContext());
                //     addPatterns.add<ConvertAddfToSIMDAddPattern, ConvertMulfToSIMDMultPattern>(&getContext());
                //     FrozenRewritePatternSet frozenAdd(std::move(addPatterns));
                //     if (failed(applyPatternsGreedily(op, frozenAdd))) {
                //         signalPassFailure();
                //         return;
                //     }
                // }

                // 3. affine.store -> flyhe.simd_store
                // {
                //     RewritePatternSet storePatterns(&getContext());
                //     storePatterns.add<ConvertAffineStorePattern>(&getContext());
                //     FrozenRewritePatternSet frozenStore(std::move(storePatterns));
                //     if (failed(applyPatternsGreedily(op, frozenStore))) {
                //         signalPassFailure();
                //         return;
                //     }
                // }

                // 4. memref -> SIMDCipherType
                // {
                //     RewritePatternSet funcPatterns(&getContext());
                //     funcPatterns.add<ConvertMemRefToSIMDCipherPattern>(&getContext());
                //     FrozenRewritePatternSet frozenFunc(std::move(funcPatterns));
                //     if (failed(applyPatternsGreedily(op, frozenFunc))) {
                //         signalPassFailure();
                //         return;
                //     }
                // }

                // 5. call -> flyhe ops
                // {
                //     RewritePatternSet callPatterns(&getContext());
                //     callPatterns.add<ConvertCallToFlyHEPattern>(&getContext());
                //     FrozenRewritePatternSet frozenCall(std::move(callPatterns));
                //     if (failed(applyPatternsGreedily(op, frozenCall))) {
                //         signalPassFailure();
                //         return;
                //     }
                // }
            }
        };

    }  // namespace
}  // namespace mlir::flyhe
