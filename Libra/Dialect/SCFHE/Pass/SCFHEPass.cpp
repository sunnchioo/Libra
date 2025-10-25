#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "SCFHEOps.h"
#include "SCFHEPass.h"

namespace mlir::libra::scfhe {

#define GEN_PASS_DEF_CONVERTTOSCFHEPASS
#include "SCFHEPass.h.inc"

    namespace {

        // pass add
        class ConvertAddfToSCFHEAddPattern : public OpRewritePattern<arith::AddFOp> {
        public:
            using OpRewritePattern::OpRewritePattern;

            LogicalResult matchAndRewrite(arith::AddFOp addOp,
                                          PatternRewriter &rewriter) const override {
                auto lhsType = addOp.getOperand(0).getType();
                auto rhsType = addOp.getOperand(1).getType();
                if (!isa<SCFHECipherType>(lhsType) || !isa<SCFHECipherType>(rhsType))
                    return failure();

                auto simdAdd = rewriter.create<scfhe::SCFHEAddOp>(
                    addOp.getLoc(),
                    lhsType,
                    addOp.getOperand(0),
                    addOp.getOperand(1));

                rewriter.replaceOp(addOp, simdAdd.getResult());
                return success();
            }
        };

        // pass mult
        class ConvertMulfToSCFHEMultPattern : public OpRewritePattern<arith::MulFOp> {
        public:
            using OpRewritePattern::OpRewritePattern;

            LogicalResult matchAndRewrite(arith::MulFOp mulOp,
                                          PatternRewriter &rewriter) const override {
                auto lhsType = mulOp.getOperand(0).getType();
                auto rhsType = mulOp.getOperand(1).getType();
                if (!isa<SCFHECipherType>(lhsType) || !isa<SCFHECipherType>(rhsType))
                    return failure();

                // 创建 scfhe.simdmult
                auto simdMul = rewriter.create<scfhe::SCFHEMultOp>(
                    mulOp.getLoc(),
                    lhsType,
                    mulOp.getOperand(0),
                    mulOp.getOperand(1));

                rewriter.replaceOp(mulOp, simdMul.getResult());
                return success();
            }
        };

        // pass InsertEncrypt
        class InsertEncryptAfterAffineLoadPattern : public OpRewritePattern<affine::AffineLoadOp> {
        public:
            using OpRewritePattern<affine::AffineLoadOp>::OpRewritePattern;

            LogicalResult matchAndRewrite(affine::AffineLoadOp loadOp,
                                          PatternRewriter &rewriter) const override {
                auto resultType = loadOp.getResult().getType();
                if (!resultType.isF64() && !isa<FloatType>(resultType))
                    return failure();

                Value plainVal = loadOp.getResult();

                for (Operation *user : plainVal.getUsers()) {
                    if (isa<scfhe::SCFHEEncryptOp>(user)) {
                        return failure();
                    }
                }

                auto loc = loadOp.getLoc();
                auto cipherType = scfhe::SCFHECipherType::get(rewriter.getContext());

                rewriter.setInsertionPointAfter(loadOp);
                auto encrpyted = rewriter.create<scfhe::SCFHEEncryptOp>(
                    loc, cipherType, plainVal);

                rewriter.replaceAllUsesExcept(plainVal, encrpyted.getResult(), encrpyted);

                return success();
            }
        };

        class InsertEncryptAfterVectorTransferReadPattern
            : public OpRewritePattern<vector::TransferReadOp> {
        public:
            using OpRewritePattern<vector::TransferReadOp>::OpRewritePattern;

            LogicalResult matchAndRewrite(vector::TransferReadOp readOp,
                                          PatternRewriter &rewriter) const override {
                Type resTy = readOp.getResult().getType();

                auto isFloatOrVecFloat = [](Type t) -> bool {
                    if (isa<FloatType>(t))
                        return true;
                    if (auto vt = dyn_cast<VectorType>(t))
                        return isa<FloatType>(vt.getElementType());
                    return false;
                };
                if (!isFloatOrVecFloat(resTy))
                    return failure();

                Value plain = readOp.getResult();

                for (Operation *user : plain.getUsers())
                    if (isa<scfhe::SCFHEEncryptOp>(user))
                        return failure();

                auto loc = readOp.getLoc();
                auto cipherTy = scfhe::SCFHECipherType::get(rewriter.getContext());

                rewriter.setInsertionPointAfter(readOp);
                auto enc = rewriter.create<scfhe::SCFHEEncryptOp>(loc, cipherTy, plain);

                rewriter.replaceAllUsesExcept(plain, enc.getResult(), enc);

                return success();
            }

            // pass InsertDecrypt
            class InsertDecryptBeforePrintfPattern : public OpRewritePattern<LLVM::CallOp> {
            public:
                using OpRewritePattern<LLVM::CallOp>::OpRewritePattern;

                LogicalResult matchAndRewrite(LLVM::CallOp callOp,
                                              PatternRewriter &rewriter) const override {
                    auto calleeAttr = callOp.getCalleeAttr();
                    if (!calleeAttr || !calleeAttr.getValue().contains("printf"))
                        return failure();

                    SmallVector<Value> newOperands;
                    bool modified = false;

                    for (auto operand : callOp.getOperands()) {
                        if (isa<scfhe::SCFHECipherType>(operand.getType())) {
                            auto decrypted = rewriter.create<scfhe::SCFHEDecryptOp>(
                                callOp.getLoc(),
                                rewriter.getF64Type(),
                                operand);
                            newOperands.push_back(decrypted.getResult());
                            modified = true;
                        } else {
                            newOperands.push_back(operand);
                        }
                    }

                    if (!modified)
                        return failure();
                    Operation *newCallOp = rewriter.clone(*callOp);

                    newCallOp->setOperands(newOperands);
                    rewriter.replaceOp(callOp, newCallOp->getResults());

                    return success();
                }
            };

            // pass affine.load
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

                    auto simdType = SCFHECipherType::get(rewriter.getContext());

                    // 创建 SCFHECipher 加载 op
                    // auto simdLoad = rewriter.create<scfhe::SCFHELoadOp>(loadOp.getLoc());
                    auto safeSimdLoad = rewriter.create<scfhe::SCFHELoadOp>(
                        loadOp.getLoc(),
                        simdType,
                        memref);

                    rewriter.replaceOp(loadOp, safeSimdLoad.getResult());
                    return success();
                }
            };

            // pass affine.store
            class ConvertAffineStorePattern : public OpRewritePattern<affine::AffineStoreOp> {
            public:
                using OpRewritePattern<affine::AffineStoreOp>::OpRewritePattern;

                LogicalResult matchAndRewrite(affine::AffineStoreOp storeOp,
                                              PatternRewriter &rewriter) const override {
                    Value value = storeOp.getValue();
                    Value dest = storeOp.getMemRef();

                    // Value must be a SCFHE cipher (we produce simd stores for SCFHE values).
                    if (!isa<SCFHECipherType>(value.getType()))
                        return failure();

                    // dest can be MemRefType (before memref->simd conversion) OR SCFHECipherType
                    if (!(isa<MemRefType>(dest.getType()) || isa<SCFHECipherType>(dest.getType())))
                        return failure();

                    // Create simd_store with current dest (memref or simd)
                    // We pass the operands explicitly (no result types, simd_store is void).
                    rewriter.create<scfhe::SCFHEStoreOp>(storeOp.getLoc(), value, dest);

                    // Remove original affine.store
                    rewriter.eraseOp(storeOp);
                    return success();
                }
            };

            // pass memref
            class ConvertMemRefToSCFHECipherPattern : public OpRewritePattern<func::FuncOp> {
            public:
                using OpRewritePattern<func::FuncOp>::OpRewritePattern;

                LogicalResult matchAndRewrite(func::FuncOp funcOp,
                                              PatternRewriter &rewriter) const final {
                    bool modified = false;

                    SmallVector<Type, 4> newArgTypes;
                    for (Type argType : funcOp.getArgumentTypes()) {
                        if (auto memrefType = dyn_cast<mlir::MemRefType>(argType)) {
                            newArgTypes.push_back(SCFHECipherType::get(rewriter.getContext()));
                            modified = true;
                        } else {
                            newArgTypes.push_back(argType);
                        }
                    }

                    SmallVector<Type, 4> newResultTypes;
                    for (Type resultType : funcOp.getResultTypes()) {
                        if (dyn_cast<MemRefType>(resultType) || dyn_cast<FloatType>(resultType)) {
                            newResultTypes.push_back(SCFHECipherType::get(rewriter.getContext()));
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
                                op->getResult(i).setType(SCFHECipherType::get(rewriter.getContext()));
                        }
                        for (unsigned i = 0; i < op->getNumOperands(); ++i) {
                            Type t = op->getOperand(i).getType();
                            if (isa<MemRefType>(t))
                                op->getOperand(i).setType(SCFHECipherType::get(rewriter.getContext()));
                        }
                    });

                    return success();
                }
            };

            class ConvertToSCFHEIR
                : public impl::ConvertToSCFHEPassBase<ConvertToSCFHEIR> {
            public:
                using impl::ConvertToSCFHEPassBase<ConvertToSCFHEIR>::ConvertToSCFHEPassBase;

                void runOnOperation() final {
                    auto module = getOperation(); // ModuleOp

                    for (auto f : module.getOps<func::FuncOp>()) {
                        if (f.getName().starts_with("polygeist."))
                            continue;

                        RewritePatternSet patterns(f.getContext());
                        patterns.add<
                            ConvertAddfToSCFHEAddPattern,
                            ConvertMulfToSCFHEMultPattern,
                            InsertEncryptAfterAffineLoadPattern,
                            InsertEncryptAfterVectorTransferReadPattern,
                            InsertDecryptBeforePrintfPattern>(&getContext());

                        (void)applyPatternsGreedily(f, std::move(patterns));
                    }
                }
            };

        } // namespace
    } // namespace mlir::libra::scfhe
