#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "FlyHEPass.h"

namespace mlir::flyhe {
#define GEN_PASS_DEF_CONVERTTOSIMDPASS
#include "FlyHEPass.h.inc"

    namespace {

        class ConvertMemRefToSIMDCipherPattern : public OpRewritePattern<func::FuncOp> {
        public:
            using OpRewritePattern<func::FuncOp>::OpRewritePattern;

            LogicalResult matchAndRewrite(func::FuncOp funcOp,
                                          PatternRewriter &rewriter) const final {
                bool modified = false;

                SmallVector<Type, 4> newArgTypes;
                for (Type argType : funcOp.getArgumentTypes()) {
                    if (auto memrefType = mlir::dyn_cast<mlir::MemRefType>(argType)) {
                        Type newType = convertMemRefToSIMDCipher(memrefType, rewriter.getContext());
                        newArgTypes.push_back(newType);
                        modified = true;
                    } else {
                        newArgTypes.push_back(argType);
                    }
                }

                SmallVector<Type, 4> newResultTypes;
                for (Type resultType : funcOp.getResultTypes()) {
                    if (auto memrefType = mlir::dyn_cast<mlir::MemRefType>(resultType)) {
                        Type newType = convertMemRefToSIMDCipher(memrefType, rewriter.getContext());
                        newResultTypes.push_back(newType);
                        modified = true;
                    } else {
                        newResultTypes.push_back(resultType);
                    }
                }

                if (!modified)
                    return failure();

                FunctionType newFuncType = FunctionType::get(
                    rewriter.getContext(), newArgTypes, newResultTypes);

                rewriter.modifyOpInPlace(funcOp, [&]() {
                    funcOp.setType(newFuncType);
                });

                funcOp.walk([&](Operation *op) {
                    for (unsigned i = 0; i < op->getNumResults(); ++i) {
                        Type resultType = op->getResultTypes()[i];
                        if (auto memrefType = mlir::dyn_cast<mlir::MemRefType>(resultType)) {
                            Type newType = convertMemRefToSIMDCipher(memrefType, rewriter.getContext());
                            op->getResult(i).setType(newType);
                        }
                    }
                    for (unsigned i = 0; i < op->getNumOperands(); ++i) {
                        Value operand = op->getOperand(i);
                        Type operandType = operand.getType();
                        if (auto memrefType = mlir::dyn_cast<mlir::MemRefType>(operandType)) {
                            Type newType = convertMemRefToSIMDCipher(memrefType, rewriter.getContext());
                            operand.setType(newType);
                        }
                    }
                });

                return success();
            }

        private:
            Type convertMemRefToSIMDCipher(mlir::MemRefType memrefType, MLIRContext *ctx) const {
                ArrayRef<int64_t> shape = memrefType.getShape();
                Type elementType = memrefType.getElementType();

                return SIMDCipherType::get(ctx);
            }
        };

        class ConvertMemRefToSIMDCipher
            : public impl::ConvertToSIMDPassBase<ConvertMemRefToSIMDCipher> {
        public:
            using impl::ConvertToSIMDPassBase<
                ConvertMemRefToSIMDCipher>::ConvertToSIMDPassBase;

            void runOnOperation() final {
                Operation *op = getOperation();

                RewritePatternSet patterns(&getContext());
                patterns.add<ConvertMemRefToSIMDCipherPattern>(&getContext());

                FrozenRewritePatternSet frozenPatterns(std::move(patterns));

                if (failed(applyPatternsGreedily(op, frozenPatterns))) {
                    signalPassFailure();
                }
            }
        };

    }  // namespace
}  // namespace mlir::flyhe