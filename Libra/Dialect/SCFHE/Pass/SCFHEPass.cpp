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

        //===================== Arith. OPS Begin =====================//

        // pass add
        class ConvertAddfToSCFHEAddPattern : public OpRewritePattern<arith::AddFOp> {
        public:
            using OpRewritePattern::OpRewritePattern;

            LogicalResult matchAndRewrite(arith::AddFOp addOp,
                                          PatternRewriter& rewriter) const override {
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

        // pass sub
        class ConvertSubfToSCFHESubPattern : public OpRewritePattern<arith::SubFOp> {
        public:
            using OpRewritePattern<arith::SubFOp>::OpRewritePattern;

            LogicalResult matchAndRewrite(arith::SubFOp subOp,
                                          PatternRewriter& rewriter) const override {
                auto lhsType = subOp.getOperand(0).getType();
                auto rhsType = subOp.getOperand(1).getType();

                if (!isa<SCFHECipherType>(lhsType) || !isa<SCFHECipherType>(rhsType))
                    return failure();

                auto simdSub = rewriter.create<scfhe::SCFHESubOp>(
                    subOp.getLoc(),
                    lhsType,
                    subOp.getOperand(0),
                    subOp.getOperand(1));

                rewriter.replaceOp(subOp, simdSub.getResult());

                return success();
            }
        };

        // pass mult
        class ConvertMulfToSCFHEMultPattern : public OpRewritePattern<arith::MulFOp> {
        public:
            using OpRewritePattern::OpRewritePattern;

            LogicalResult matchAndRewrite(arith::MulFOp mulOp,
                                          PatternRewriter& rewriter) const override {
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

        //===================== Arith. OPS End =====================//

        //===================== En/Decrypt Begin =====================//
        // pass InsertEncrypt
        class InsertEncryptAfterAffineLoadPattern : public OpRewritePattern<affine::AffineLoadOp> {
        public:
            using OpRewritePattern<affine::AffineLoadOp>::OpRewritePattern;

            LogicalResult matchAndRewrite(affine::AffineLoadOp loadOp,
                                          PatternRewriter& rewriter) const override {
                mlir::Type resultType = loadOp.getResult().getType();
                int64_t plaintextCount;
                mlir::Type elementType;

                if (auto vecType = dyn_cast<VectorType>(resultType)) {
                    plaintextCount = vecType.getNumElements();
                    elementType = vecType.getElementType();

                    if (!isa<FloatType>(elementType))
                        return failure();
                } else if (isa<FloatType>(resultType)) {
                    plaintextCount = 1;
                    elementType = resultType;
                } else {
                    return failure();
                }

                Value plainVal = loadOp.getResult();

                for (Operation* user : plainVal.getUsers()) {
                    if (isa<scfhe::SCFHEEncryptOp>(user)) {
                        return failure();
                    }
                }

                auto loc = loadOp.getLoc();

                auto cipherType = scfhe::SCFHECipherType::get(rewriter.getContext(),
                                                              plaintextCount,
                                                              elementType);
                if (!cipherType) {
                    return failure();
                }

                rewriter.setInsertionPointAfter(loadOp);
                auto encryptOp = rewriter.create<scfhe::SCFHEEncryptOp>(
                    loc, cipherType, plainVal);

                rewriter.replaceAllUsesExcept(plainVal, encryptOp.getResult(), encryptOp);

                return success();
            }
        };

        class InsertEncryptAfterVectorTransferReadPattern
            : public OpRewritePattern<vector::TransferReadOp> {
        public:
            using OpRewritePattern<vector::TransferReadOp>::OpRewritePattern;

            LogicalResult matchAndRewrite(vector::TransferReadOp readOp,
                                          PatternRewriter& rewriter) const override {
                auto resultType = readOp.getResult().getType();

                int64_t plaintextCount;
                Type elementType;

                // 仍然从 vector<8xf64> 中获取元素个数
                if (auto vecType = dyn_cast<VectorType>(resultType)) {
                    plaintextCount = vecType.getNumElements();
                    // 但忽略原来的元素类型，强制用 i64
                    elementType = rewriter.getI64Type();
                } else if (isa<FloatType>(resultType) || isa<IntegerType>(resultType)) {
                    plaintextCount = 1;
                    elementType = rewriter.getI64Type();
                } else {
                    return failure();
                }

                Value plainVal = readOp.getResult();

                // 如果已经有 encrypt 了就不重复插入
                for (Operation* user : plainVal.getUsers()) {
                    if (isa<scfhe::SCFHEEncryptOp>(user))
                        return failure();
                }

                auto loc = readOp.getLoc();

                // 构造 !scfhe.scfhecipher<8 x i64>
                auto cipherType = scfhe::SCFHECipherType::get(
                    rewriter.getContext(), plaintextCount, elementType);

                rewriter.setInsertionPointAfter(readOp);
                auto encrypted = rewriter.create<scfhe::SCFHEEncryptOp>(
                    loc, cipherType, plainVal);

                rewriter.replaceAllUsesExcept(plainVal, encrypted.getResult(), encrypted);

                return success();
            }
        };

        // pass InsertDecrypt
        class InsertDecryptBeforePrintfPattern : public OpRewritePattern<LLVM::CallOp> {
        public:
            using OpRewritePattern<LLVM::CallOp>::OpRewritePattern;

            LogicalResult matchAndRewrite(LLVM::CallOp callOp,
                                          PatternRewriter& rewriter) const override {
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
                Operation* newCallOp = rewriter.clone(*callOp);

                newCallOp->setOperands(newOperands);
                rewriter.replaceOp(callOp, newCallOp->getResults());

                return success();
            }
        };

        // 在 vector.transfer_write 之前插入 scfhe.decrypt
        class InsertDecryptBeforeVectorTransferWritePattern
            : public OpRewritePattern<vector::TransferWriteOp> {
        public:
            using OpRewritePattern<vector::TransferWriteOp>::OpRewritePattern;

            LogicalResult matchAndRewrite(vector::TransferWriteOp writeOp,
                                          PatternRewriter& rewriter) const override {
                Value vec = writeOp.getVector();

                // 仅当 vector.transfer_write 的输入是加密类型时匹配
                if (!isa<scfhe::SCFHECipherType>(vec.getType()))
                    return failure();

                // 从 memref 类型推断目标向量类型
                Value memref = writeOp.getOperand(1);
                auto memrefType = dyn_cast<MemRefType>(memref.getType());
                if (!memrefType)
                    return failure();

                // 仅支持静态一维 memref，如 memref<8xf64>
                if (memrefType.getRank() != 1 || memrefType.getDimSize(0) <= 0)
                    return failure();

                Type elemType = memrefType.getElementType();
                if (!isa<FloatType>(elemType))
                    return failure();

                // 构造对应向量类型 (例如 vector<8xf64>)
                auto vectorTy =
                    VectorType::get(ArrayRef<int64_t>{memrefType.getDimSize(0)}, elemType);

                // 在写操作前插入解密
                rewriter.setInsertionPoint(writeOp);
                auto decryptOp = rewriter.create<scfhe::SCFHEDecryptOp>(
                    writeOp.getLoc(), vectorTy, vec);

                // 替换 vector.transfer_write 的输入为解密结果
                writeOp.getOperation()->setOperand(0, decryptOp.getResult());

                return success();
            }
        };

        //===================== En/Decrypt End =====================//

        //===================== Mem. OPS Begin =====================//
        // pass affine.load
        class ConvertAffineLoadPattern : public OpRewritePattern<affine::AffineLoadOp> {
        public:
            using OpRewritePattern<affine::AffineLoadOp>::OpRewritePattern;

            LogicalResult matchAndRewrite(affine::AffineLoadOp loadOp,
                                          PatternRewriter& rewriter) const override {
                Value memref = loadOp.getMemRef();
                if (!isa<MemRefType>(memref.getType()))
                    return failure();

                mlir::Type originalResultType = loadOp.getResult().getType();
                int64_t plaintextCount;
                mlir::Type elementType;

                if (auto vecType = dyn_cast<VectorType>(originalResultType)) {
                    plaintextCount = vecType.getNumElements();
                    elementType = vecType.getElementType();
                    if (!isa<FloatType>(elementType))
                        return failure();
                } else if (isa<FloatType>(originalResultType)) {
                    plaintextCount = 1;
                    elementType = originalResultType;
                } else {
                    return failure();
                }

                auto cipherResultType = scfhe::SCFHECipherType::get(rewriter.getContext(),
                                                                    plaintextCount,
                                                                    elementType);
                if (!cipherResultType) {
                    return failure();
                }

                auto safeSimdLoad = rewriter.create<scfhe::SCFHELoadOp>(
                    loadOp.getLoc(),
                    cipherResultType,
                    loadOp.getMemRef());

                rewriter.replaceOp(loadOp, safeSimdLoad.getResult());

                return success();
            }
        };

        // pass affine.store
        class ConvertAffineStorePattern : public OpRewritePattern<affine::AffineStoreOp> {
        public:
            using OpRewritePattern<affine::AffineStoreOp>::OpRewritePattern;

            LogicalResult matchAndRewrite(affine::AffineStoreOp storeOp,
                                          PatternRewriter& rewriter) const override {
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

            FailureOr<scfhe::SCFHECipherType>
            convertToCipherType(Type originalType, PatternRewriter& rewriter) const {
                Type baseType = originalType;

                if (auto memrefType = dyn_cast<MemRefType>(originalType)) {
                    baseType = memrefType.getElementType();
                }

                int64_t plaintextCount;
                Type elementType;

                if (auto vecType = dyn_cast<VectorType>(baseType)) {
                    plaintextCount = vecType.getNumElements();
                    elementType = vecType.getElementType();
                    if (!isa<FloatType>(elementType))
                        return failure();
                } else if (isa<FloatType>(baseType)) {
                    plaintextCount = 1;
                    elementType = baseType;
                } else {
                    return failure();
                }

                return scfhe::SCFHECipherType::get(rewriter.getContext(),
                                                   plaintextCount,
                                                   elementType);
            }

            LogicalResult matchAndRewrite(func::FuncOp funcOp,
                                          PatternRewriter& rewriter) const final {
                bool modified = false;

                SmallVector<Type, 4> newArgTypes;
                for (Type argType : funcOp.getArgumentTypes()) {
                    FailureOr<scfhe::SCFHECipherType> newType = convertToCipherType(argType, rewriter);

                    if (succeeded(newType)) {
                        newArgTypes.push_back(newType.value());
                        modified = true;
                    } else {
                        newArgTypes.push_back(argType);
                    }
                }

                SmallVector<Type, 4> newResultTypes;
                for (Type resultType : funcOp.getResultTypes()) {
                    FailureOr<scfhe::SCFHECipherType> newType = convertToCipherType(resultType, rewriter);

                    if (succeeded(newType)) {
                        newResultTypes.push_back(newType.value());
                        modified = true;
                    } else {
                        newResultTypes.push_back(resultType);
                    }
                }

                if (!modified)
                    return failure();

                FunctionType newFuncType =
                    FunctionType::get(rewriter.getContext(), newArgTypes, newResultTypes);

                rewriter.modifyOpInPlace(funcOp, [&]() {
                    funcOp.setType(newFuncType);

                    Block& entryBlock = funcOp.front();
                    for (size_t i = 0; i < entryBlock.getNumArguments(); ++i) {
                        entryBlock.getArgument(i).setType(newArgTypes[i]);
                    }
                });

                return success();
            }
        };

        class ReplaceVectorTransferWriteWithSCFHEStorePattern
            : public OpRewritePattern<vector::TransferWriteOp> {
        public:
            using OpRewritePattern<vector::TransferWriteOp>::OpRewritePattern;

            LogicalResult matchAndRewrite(vector::TransferWriteOp writeOp,
                                          PatternRewriter& rewriter) const override {
                Value vec = writeOp.getVector();
                Type vecType = vec.getType();

                if (!isa<scfhe::SCFHECipherType>(vecType))
                    return failure();

                auto cipherDestType = vecType;

                rewriter.setInsertionPoint(writeOp);

                auto nameAttr = rewriter.getStringAttr("scfhe_alloca");
                auto nameLoc = mlir::NameLoc::get(nameAttr);

                Value cipherAlloca =
                    rewriter.create<scfhe::SCFHEAllocaOp>(nameLoc, cipherDestType);

                rewriter.create<scfhe::SCFHEStoreOp>(writeOp.getLoc(), vec, cipherAlloca);

                rewriter.eraseOp(writeOp);

                return success();
            }
        };

        //===================== Mem. OPS End =====================//

        //===================== Pattern Matching Begin =====================//

        // min
        class ReplaceAffineMinWithSCFHEMinPattern
            : public OpRewritePattern<affine::AffineForOp> {
        public:
            using OpRewritePattern<affine::AffineForOp>::OpRewritePattern;

            LogicalResult matchAndRewrite(affine::AffineForOp forOp,
                                          PatternRewriter& rewriter) const override {
                // --- 1. 检查循环结构是否像 "find min" ---
                Block* body = forOp.getBody();
                if (std::distance(body->begin(), body->end()) != 4)
                    return failure();

                auto loadOp = dyn_cast<affine::AffineLoadOp>(body->front());
                if (!loadOp)
                    return failure();

                auto* cmpNode = loadOp.getOperation()->getNextNode();
                auto cmpOp = dyn_cast_or_null<arith::CmpFOp>(cmpNode);
                if (!cmpOp)
                    return failure();

                auto selectOp =
                    dyn_cast_or_null<arith::SelectOp>(cmpOp.getOperation()->getNextNode());
                auto yieldOp = dyn_cast_or_null<affine::AffineYieldOp>(
                    selectOp ? selectOp.getOperation()->getNextNode() : nullptr);
                if (!selectOp || !yieldOp)
                    return failure();

                // --- 2. 找到 scfhe.alloca (作为输入密文) ---
                Operation* prevOp = forOp.getOperation()->getPrevNode();
                while (prevOp && !isa<scfhe::SCFHEAllocaOp>(prevOp))
                    prevOp = prevOp->getPrevNode();

                if (!prevOp)
                    return failure();

                auto scfheAlloca = cast<scfhe::SCFHEAllocaOp>(prevOp);
                Value ciphertextInput = scfheAlloca.getResult();

                // --- 3. 构造输出类型: !scfhe.scfhecipher<1 x i64> ---
                auto cipherType = cast<scfhe::SCFHECipherType>(ciphertextInput.getType());
                auto elementType = cipherType.getElementType();
                auto singleCipherType = scfhe::SCFHECipherType::get(
                    rewriter.getContext(), /*plaintextCount=*/1, elementType);

                // --- 4. 创建 scfhe.min ---
                auto scfheMinOp = rewriter.create<scfhe::SCFHEMinOp>(
                    forOp.getLoc(), singleCipherType, ciphertextInput);

                // --- 5. 替换 ---
                rewriter.replaceOp(forOp, scfheMinOp.getResult());
                return success();
            }
        };

        //===----------------------------------------------------------------------===//
        // Fold Alloca + Store Pattern (from FoldSCFHEAllocaStorePass)
        //===----------------------------------------------------------------------===//

        class FoldAllocaStorePattern : public RewritePattern {
        public:
            explicit FoldAllocaStorePattern(MLIRContext* ctx)
                : RewritePattern(MatchAnyOpTypeTag(), 1, ctx) {}

            LogicalResult matchAndRewrite(Operation* op,
                                          PatternRewriter& rewriter) const override {
                // 只处理 scfhe 算术操作
                if (!isa<scfhe::SCFHEMinOp, scfhe::SCFHEAddOp,
                         scfhe::SCFHESubOp, scfhe::SCFHEMultOp>(op))
                    return failure();

                bool modified = false;
                SmallVector<Value, 4> newOperands;
                SmallVector<scfhe::SCFHEStoreOp, 4> storesToErase;
                SmallVector<scfhe::SCFHEAllocaOp, 4> allocasToErase;

                // 遍历操作数
                for (Value operand : op->getOperands()) {
                    // 检查操作数是否由 scfhe.alloca 定义
                    auto allocaOp = operand.getDefiningOp<scfhe::SCFHEAllocaOp>();
                    if (!allocaOp) {
                        newOperands.push_back(operand);
                        continue;
                    }

                    scfhe::SCFHEStoreOp storeOp = nullptr;

                    // 检查 alloca 的用户
                    for (Operation* user : allocaOp->getResult(0).getUsers()) {
                        if (auto s = dyn_cast<scfhe::SCFHEStoreOp>(user)) {
                            if (storeOp)
                                return failure();  // 多个 store，不安全折叠
                            storeOp = s;
                        } else if (user != op) {
                            // alloca 被除目标 op 以外的其它 op 使用 -> 不安全
                            return failure();
                        }
                    }

                    if (!storeOp)
                        return failure();

                    // 拿到 store 的存储值（第一个操作数通常是 value）
                    Value storedVal = storeOp.getValue();
                    newOperands.push_back(storedVal);
                    modified = true;

                    // 记录要删除的 store/alloca —— 注意分组，便于先删 store 再删 alloca
                    storesToErase.push_back(storeOp);
                    allocasToErase.push_back(allocaOp);
                }

                if (!modified)
                    return failure();

                // Build new op (同类型、同属性、但用 newOperands)
                OperationState state(op->getLoc(), op->getName());
                state.addOperands(newOperands);
                state.addTypes(op->getResultTypes());
                state.addAttributes(op->getAttrs());
                Operation* newOp = rewriter.create(state);

                // 用新 op 的结果替换旧 op
                rewriter.replaceOp(op, newOp->getResults());

                // 重要：先删除 store，再删除 alloca（防止 alloca 仍被 store 引用）
                for (auto s : storesToErase) {
                    if (s->use_empty())
                        rewriter.eraseOp(s);
                }
                for (auto a : allocasToErase) {
                    if (a->use_empty())
                        rewriter.eraseOp(a);
                }

                return success();
            }
        };

        //===================== Pattern Matching End =====================//

        class ConvertToSCFHEIR
            : public impl::ConvertToSCFHEPassBase<ConvertToSCFHEIR> {
        public:
            using impl::ConvertToSCFHEPassBase<ConvertToSCFHEIR>::ConvertToSCFHEPassBase;

            void runOnOperation() final {
                auto module = getOperation();  // ModuleOp

                RewritePatternSet patterns(module.getContext());
                patterns.add<
                    InsertEncryptAfterVectorTransferReadPattern,
                    ConvertAddfToSCFHEAddPattern,
                    ConvertSubfToSCFHESubPattern,
                    ConvertMulfToSCFHEMultPattern,
                    ReplaceAffineMinWithSCFHEMinPattern,
                    ReplaceVectorTransferWriteWithSCFHEStorePattern,
                    InsertDecryptBeforePrintfPattern,
                    FoldAllocaStorePattern>(&getContext());

                (void)applyPatternsGreedily(module, std::move(patterns));
            }
        };
    }  // namespace
}  // namespace mlir::libra::scfhe
