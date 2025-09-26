#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"

#include "SISDOps.h"
#include "SISDPass.h"

namespace mlir::sisd {

#define GEN_PASS_DEF_CONVERTTOSISDPASS
#include "SISDPass.h.inc"

    namespace {

        class ConvertAddfToSISDAddPattern : public OpRewritePattern<arith::AddFOp> {
        public:
            using OpRewritePattern::OpRewritePattern;

            LogicalResult matchAndRewrite(arith::AddFOp addOp,
                                          PatternRewriter &rewriter) const override {
                auto lhsType = addOp.getOperand(0).getType();
                auto rhsType = addOp.getOperand(1).getType();
                if (!isa<SISDCipherType>(lhsType) || !isa<SISDCipherType>(rhsType))
                    return failure();

                auto sisdAdd = rewriter.create<sisd::SISDAddOp>(
                    addOp.getLoc(),
                    lhsType,
                    addOp.getOperand(0),
                    addOp.getOperand(1));

                rewriter.replaceOp(addOp, sisdAdd.getResult());
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

                auto sisdType = SISDCipherType::get(rewriter.getContext());

                // 创建 SISDCipher 加载 op
                // auto sisdLoad = rewriter.create<sisd::SISDLoadOp>(loadOp.getLoc());
                auto safeSimdLoad = rewriter.create<sisd::SISDLoadOp>(
                    loadOp.getLoc(),
                    sisdType,
                    memref);

                rewriter.replaceOp(loadOp, safeSimdLoad.getResult());
                return success();
            }
        };

        class ConvertMemRefToSISDCipherPattern : public OpRewritePattern<func::FuncOp> {
        public:
            using OpRewritePattern<func::FuncOp>::OpRewritePattern;

            LogicalResult matchAndRewrite(func::FuncOp funcOp,
                                          PatternRewriter &rewriter) const final {
                bool modified = false;

                SmallVector<Type, 4> newArgTypes;
                for (Type argType : funcOp.getArgumentTypes()) {
                    if (auto memrefType = dyn_cast<mlir::MemRefType>(argType)) {
                        newArgTypes.push_back(SISDCipherType::get(rewriter.getContext()));
                        modified = true;
                    } else {
                        newArgTypes.push_back(argType);
                    }
                }

                SmallVector<Type, 4> newResultTypes;
                for (Type resultType : funcOp.getResultTypes()) {
                    if (dyn_cast<MemRefType>(resultType) || dyn_cast<FloatType>(resultType)) {
                        newResultTypes.push_back(SISDCipherType::get(rewriter.getContext()));
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
                            op->getResult(i).setType(SISDCipherType::get(rewriter.getContext()));
                    }
                    for (unsigned i = 0; i < op->getNumOperands(); ++i) {
                        Type t = op->getOperand(i).getType();
                        if (isa<MemRefType>(t))
                            op->getOperand(i).setType(SISDCipherType::get(rewriter.getContext()));
                    }
                });

                return success();
            }
        };

        class ConvertMemRefToSISDCipher
            : public impl::ConvertToSISDPassBase<ConvertMemRefToSISDCipher> {
        public:
            using impl::ConvertToSISDPassBase<ConvertMemRefToSISDCipher>::ConvertToSISDPassBase;

            void runOnOperation() final {
                Operation *op = getOperation();

                // 1. affine.load -> sisd.sisd_load
                {
                    RewritePatternSet loadPatterns(&getContext());
                    loadPatterns.add<ConvertAffineLoadPattern>(&getContext());
                    FrozenRewritePatternSet frozenLoad(std::move(loadPatterns));
                    if (failed(applyPatternsGreedily(op, frozenLoad))) {
                        signalPassFailure();
                        return;
                    }
                }

                // 2. arith.addf -> sisd.smidadd
                {
                    RewritePatternSet addPatterns(&getContext());
                    addPatterns.add<ConvertAddfToSISDAddPattern>(&getContext());
                    FrozenRewritePatternSet frozenAdd(std::move(addPatterns));
                    if (failed(applyPatternsGreedily(op, frozenAdd))) {
                        signalPassFailure();
                        return;
                    }
                }

                // 3. memref -> SISDCipherType
                {
                    RewritePatternSet funcPatterns(&getContext());
                    funcPatterns.add<ConvertMemRefToSISDCipherPattern>(&getContext());
                    FrozenRewritePatternSet frozenFunc(std::move(funcPatterns));
                    if (failed(applyPatternsGreedily(op, frozenFunc))) {
                        signalPassFailure();
                        return;
                    }
                }
            }
        };

    }  // namespace
}  // namespace mlir::sisd
