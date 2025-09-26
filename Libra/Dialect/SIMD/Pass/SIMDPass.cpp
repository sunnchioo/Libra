#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"

#include "SIMDOps.h"
#include "SIMDPass.h"

namespace mlir::simd {

#define GEN_PASS_DEF_CONVERTTOSIMDPASS
#include "SIMDPass.h.inc"

    namespace {

        class ConvertAddfToSIMDAddPattern : public OpRewritePattern<arith::AddFOp> {
        public:
            using OpRewritePattern::OpRewritePattern;

            LogicalResult matchAndRewrite(arith::AddFOp addOp,
                                          PatternRewriter &rewriter) const override {
                auto lhsType = addOp.getOperand(0).getType();
                auto rhsType = addOp.getOperand(1).getType();
                if (!isa<SIMDCipherType>(lhsType) || !isa<SIMDCipherType>(rhsType))
                    return failure();

                auto simdAdd = rewriter.create<simd::SIMDAddOp>(
                    addOp.getLoc(),
                    lhsType,
                    addOp.getOperand(0),
                    addOp.getOperand(1));

                rewriter.replaceOp(addOp, simdAdd.getResult());
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
                // auto simdLoad = rewriter.create<simd::SIMDLoadOp>(loadOp.getLoc());
                auto safeSimdLoad = rewriter.create<simd::SIMDLoadOp>(
                    loadOp.getLoc(),
                    simdType,
                    memref);

                rewriter.replaceOp(loadOp, safeSimdLoad.getResult());
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

                // 1. affine.load -> simd.simd_load
                {
                    RewritePatternSet loadPatterns(&getContext());
                    loadPatterns.add<ConvertAffineLoadPattern>(&getContext());
                    FrozenRewritePatternSet frozenLoad(std::move(loadPatterns));
                    if (failed(applyPatternsGreedily(op, frozenLoad))) {
                        signalPassFailure();
                        return;
                    }
                }

                // 2. arith.addf -> simd.smidadd
                {
                    RewritePatternSet addPatterns(&getContext());
                    addPatterns.add<ConvertAddfToSIMDAddPattern>(&getContext());
                    FrozenRewritePatternSet frozenAdd(std::move(addPatterns));
                    if (failed(applyPatternsGreedily(op, frozenAdd))) {
                        signalPassFailure();
                        return;
                    }
                }

                // 3. memref -> SIMDCipherType
                {
                    RewritePatternSet funcPatterns(&getContext());
                    funcPatterns.add<ConvertMemRefToSIMDCipherPattern>(&getContext());
                    FrozenRewritePatternSet frozenFunc(std::move(funcPatterns));
                    if (failed(applyPatternsGreedily(op, frozenFunc))) {
                        signalPassFailure();
                        return;
                    }
                }
            }
        };

    }  // namespace
}  // namespace mlir::simd
