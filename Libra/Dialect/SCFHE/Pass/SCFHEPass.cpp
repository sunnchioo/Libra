#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Math/IR/Math.h"

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
        // pass exp
        class ConvertMathExpToSCFHEExpPattern : public OpRewritePattern<math::ExpOp> {
        public:
            using OpRewritePattern<math::ExpOp>::OpRewritePattern;

            LogicalResult matchAndRewrite(math::ExpOp expOp,
                                          PatternRewriter& rewriter) const override {
                Value input = expOp.getOperand();
                Type inputType = input.getType();

                // 1. 检查输入是否为密文，或者是否需要提升为密文
                bool isInputCipher = isa<scfhe::SCFHECipherType>(inputType);
                bool isInputVector = isa<VectorType>(inputType); // 明文向量

                // 如果既不是密文也不是向量，暂时不处理（可能是标量浮点运算）
                if (!isInputCipher && !isInputVector) {
                    return failure();
                }

                // 2. 辅助函数：确保输入是密文 (复用您之前的逻辑)
                auto ensureCipher = [&](Value val) -> Value {
                    if (isa<scfhe::SCFHECipherType>(val.getType()))
                        return val;

                    int64_t count = 1;
                    Type elemType = rewriter.getI64Type(); // 默认底层类型 i64

                    if (auto vecType = dyn_cast<VectorType>(val.getType())) {
                        count = vecType.getNumElements();
                        // 保持 elemType 为 i64
                    }

                    auto cipherType = scfhe::SCFHECipherType::get(
                        rewriter.getContext(), count, elemType);

                    // 插入加密操作
                    auto enc = rewriter.create<scfhe::SCFHEEncryptOp>(
                        expOp.getLoc(), cipherType, val);
                    return enc.getResult();
                };

                Value newInput = ensureCipher(input);

                // 3. 构建结果类型 (与输入密文类型保持一致)
                auto resultType = newInput.getType();

                // 4. 创建 scfhe.exp (假设您的 ODS 中定义了 SCFHEExpOp)
                // 如果您的同态库不支持直接的 exp，这里可能需要展开为多项式逼近(Polynomial Approximation)
                // 但通常作为 High-Level IR，应该先转换为 scfhe.exp，由更底层的 Pass 去做逼近。

                auto scfheExp = rewriter.create<scfhe::SCFHEExpOp>(
                    expOp.getLoc(),
                    resultType,
                    newInput);

                // 5. 替换
                rewriter.replaceOp(expOp, scfheExp.getResult());
                return success();
            }
        };

        // pass div (自动处理混合类型的除法)
        class ConvertDivfToSCFHEDivPattern : public OpRewritePattern<arith::DivFOp> {
        public:
            using OpRewritePattern<arith::DivFOp>::OpRewritePattern;

            LogicalResult matchAndRewrite(arith::DivFOp divOp,
                                          PatternRewriter& rewriter) const override {
                // 1. 获取操作数
                Value lhs = divOp.getOperand(0);
                Value rhs = divOp.getOperand(1);

                auto lhsType = lhs.getType();
                auto rhsType = rhs.getType();

                bool isLhsCipher = isa<SCFHECipherType>(lhsType);
                bool isRhsCipher = isa<SCFHECipherType>(rhsType);

                // 如果两个都是明文，不归我管
                if (!isLhsCipher && !isRhsCipher)
                    return failure();

                // 2. 辅助函数：将明文 Value 提升为密文
                auto ensureCipher = [&](Value val) -> Value {
                    if (isa<SCFHECipherType>(val.getType()))
                        return val;

                    int64_t count = 1;
                    Type elemType = rewriter.getI64Type(); // 默认底层类型

                    if (auto vecType = dyn_cast<VectorType>(val.getType())) {
                        count = vecType.getNumElements();
                        // elemType = vecType.getElementType();
                    }

                    auto cipherType = scfhe::SCFHECipherType::get(
                        rewriter.getContext(), count, elemType);

                    // 插入 scfhe.encrypt
                    auto enc = rewriter.create<scfhe::SCFHEEncryptOp>(
                        divOp.getLoc(), cipherType, val);
                    return enc.getResult();
                };

                // 3. 统一操作数类型为密文
                Value newLhs = ensureCipher(lhs);
                Value newRhs = ensureCipher(rhs);

                // 4. 创建 scfhe.div
                // 结果类型通常跟随 LHS (被除数)
                auto resultType = newLhs.getType();

                auto scfheDiv = rewriter.create<scfhe::SCFHEDivOp>(
                    divOp.getLoc(),
                    resultType,
                    newLhs,
                    newRhs);

                rewriter.replaceOp(divOp, scfheDiv.getResult());
                return success();
            }
        };

        class ReplaceReductionLoopWithSCFHEThresholdCountPattern : public OpRewritePattern<vector::ReductionOp> {
        public:
            using OpRewritePattern<vector::ReductionOp>::OpRewritePattern;

            LogicalResult matchAndRewrite(vector::ReductionOp reduceOp, PatternRewriter& rewriter) const override {
                // 1. 检查 Reduction 类型是否为 ADD (累加)
                if (reduceOp.getKind() != vector::CombiningKind::ADD)
                    return failure();

                // 2. 追踪输入来源：必须来自 affine.for
                // Value reduceInput = reduceOp.getVector();
                Value reduceInput = reduceOp.getOperand(0);
                auto forOp = reduceInput.getDefiningOp<affine::AffineForOp>();
                if (!forOp)
                    return failure();

                // 3. 检查循环体结构
                vector::TransferReadOp readOp = nullptr;
                arith::CmpFOp cmpOp = nullptr;

                for (Operation& op : forOp.getBody()->getOperations()) {
                    if (auto r = dyn_cast<vector::TransferReadOp>(op))
                        readOp = r;
                    if (auto c = dyn_cast<arith::CmpFOp>(op))
                        cmpOp = c;
                }

                if (!readOp || !cmpOp)
                    return failure();

                // 4. 提取关键操作数
                // [修复点 1] 使用 getOperand(0) 替代 getSource()，避免兼容性问题
                Value inputMemRef = readOp.getOperand(0);

                // 获取阈值
                Value thresholdVec = cmpOp.getOperand(1);
                Value thresholdScalar = thresholdVec;

                // 尝试穿透 vector.broadcast 获取原始标量
                if (auto broadcastOp = thresholdVec.getDefiningOp<vector::BroadcastOp>()) {
                    // [修复点 2] 同样使用 getOperand(0) 替代 getSource()
                    thresholdScalar = broadcastOp.getOperand();
                }

                // 5. 准备阈值的加密形式
                Value finalThreshold = thresholdScalar;
                if (isa<FloatType>(thresholdScalar.getType())) {
                    auto scalarCipherType = scfhe::SCFHECipherType::get(
                        rewriter.getContext(), 1, thresholdScalar.getType());

                    rewriter.setInsertionPoint(reduceOp);
                    auto encOp = rewriter.create<scfhe::SCFHEEncryptOp>(
                        reduceOp.getLoc(), scalarCipherType, thresholdScalar);
                    finalThreshold = encOp.getResult();
                }

                // 6. 构造输出类型
                auto resultElemType = rewriter.getI32Type();
                auto resultCipherType = scfhe::SCFHECipherType::get(
                    rewriter.getContext(), 1, resultElemType);

                // 7. 创建 High-Level 算子
                rewriter.setInsertionPoint(reduceOp);
                auto countOp = rewriter.create<scfhe::SCFHEThresholdCountOp>(
                    reduceOp.getLoc(),
                    resultCipherType,
                    inputMemRef,
                    finalThreshold,
                    static_cast<uint64_t>(cmpOp.getPredicate()));

                // 8. 插入解密以适配函数返回类型 (i32)
                auto decryptOp = rewriter.create<scfhe::SCFHEDecryptOp>(
                    reduceOp.getLoc(), reduceOp.getType(), countOp.getResult());

                rewriter.replaceOp(reduceOp, decryptOp.getResult());

                return success();
            }
        };

        // pass add
        // class ConvertAddfToSCFHEAddPattern : public OpRewritePattern<arith::AddFOp> {
        // public:
        //     using OpRewritePattern::OpRewritePattern;

        //     LogicalResult matchAndRewrite(arith::AddFOp addOp,
        //                                   PatternRewriter& rewriter) const override {
        //         auto lhsType = addOp.getOperand(0).getType();
        //         auto rhsType = addOp.getOperand(1).getType();
        //         if (!isa<SCFHECipherType>(lhsType) || !isa<SCFHECipherType>(rhsType))
        //             return failure();

        //         auto simdAdd = rewriter.create<scfhe::SCFHEAddOp>(
        //             addOp.getLoc(),
        //             lhsType,
        //             addOp.getOperand(0),
        //             addOp.getOperand(1));

        //         rewriter.replaceOp(addOp, simdAdd.getResult());
        //         return success();
        //     }
        // };

        // pass add (增强版：支持混合类型)
        class ConvertAddfToSCFHEAddPattern : public OpRewritePattern<arith::AddFOp> {
        public:
            using OpRewritePattern<arith::AddFOp>::OpRewritePattern;

            LogicalResult matchAndRewrite(arith::AddFOp addOp,
                                          PatternRewriter& rewriter) const override {
                Value lhs = addOp.getOperand(0);
                Value rhs = addOp.getOperand(1);

                auto lhsType = lhs.getType();
                auto rhsType = rhs.getType();

                bool isLhsCipher = isa<SCFHECipherType>(lhsType);
                bool isRhsCipher = isa<SCFHECipherType>(rhsType);

                if (!isLhsCipher && !isRhsCipher)
                    return failure();

                auto ensureCipher = [&](Value val) -> Value {
                    if (isa<SCFHECipherType>(val.getType()))
                        return val;
                    int64_t count = 1;
                    Type elemType = rewriter.getI64Type();
                    if (auto vecType = dyn_cast<VectorType>(val.getType())) {
                        count = vecType.getNumElements();
                    }
                    auto cipherType = scfhe::SCFHECipherType::get(
                        rewriter.getContext(), count, elemType);
                    auto enc = rewriter.create<scfhe::SCFHEEncryptOp>(
                        addOp.getLoc(), cipherType, val);
                    return enc.getResult();
                };

                Value newLhs = ensureCipher(lhs);
                Value newRhs = ensureCipher(rhs);
                auto resultType = newLhs.getType();

                auto simdAdd = rewriter.create<scfhe::SCFHEAddOp>(
                    addOp.getLoc(),
                    resultType,
                    newLhs,
                    newRhs);

                rewriter.replaceOp(addOp, simdAdd.getResult());
                return success();
            }
        };

        // pass sub
        // class ConvertSubfToSCFHESubPattern : public OpRewritePattern<arith::SubFOp> {
        // public:
        //     using OpRewritePattern<arith::SubFOp>::OpRewritePattern;

        //     LogicalResult matchAndRewrite(arith::SubFOp subOp,
        //                                   PatternRewriter& rewriter) const override {
        //         auto lhsType = subOp.getOperand(0).getType();
        //         auto rhsType = subOp.getOperand(1).getType();

        //         if (!isa<SCFHECipherType>(lhsType) || !isa<SCFHECipherType>(rhsType))
        //             return failure();

        //         auto simdSub = rewriter.create<scfhe::SCFHESubOp>(
        //             subOp.getLoc(),
        //             lhsType,
        //             subOp.getOperand(0),
        //             subOp.getOperand(1));

        //         rewriter.replaceOp(subOp, simdSub.getResult());

        //         return success();
        //     }
        // };

        // pass sub (增强版：支持混合类型)
        class ConvertSubfToSCFHESubPattern : public OpRewritePattern<arith::SubFOp> {
        public:
            using OpRewritePattern<arith::SubFOp>::OpRewritePattern;

            LogicalResult matchAndRewrite(arith::SubFOp subOp,
                                          PatternRewriter& rewriter) const override {
                Value lhs = subOp.getOperand(0);
                Value rhs = subOp.getOperand(1);

                auto lhsType = lhs.getType();
                auto rhsType = rhs.getType();

                bool isLhsCipher = isa<SCFHECipherType>(lhsType);
                bool isRhsCipher = isa<SCFHECipherType>(rhsType);

                // 如果两个都是明文，不归我管
                if (!isLhsCipher && !isRhsCipher)
                    return failure();

                // 辅助函数：自动加密
                auto ensureCipher = [&](Value val) -> Value {
                    if (isa<SCFHECipherType>(val.getType()))
                        return val;

                    int64_t count = 1;
                    Type elemType = rewriter.getI64Type();

                    if (auto vecType = dyn_cast<VectorType>(val.getType())) {
                        count = vecType.getNumElements();
                    }

                    auto cipherType = scfhe::SCFHECipherType::get(
                        rewriter.getContext(), count, elemType);

                    auto enc = rewriter.create<scfhe::SCFHEEncryptOp>(
                        subOp.getLoc(), cipherType, val);
                    return enc.getResult();
                };

                Value newLhs = ensureCipher(lhs);
                Value newRhs = ensureCipher(rhs);

                // 结果类型跟随 LHS (现在 LHS 肯定是 Cipher)
                auto resultType = newLhs.getType();

                auto simdSub = rewriter.create<scfhe::SCFHESubOp>(
                    subOp.getLoc(),
                    resultType,
                    newLhs,
                    newRhs);

                rewriter.replaceOp(subOp, simdSub.getResult());
                return success();
            }
        };

        // pass mult
        // class ConvertMulfToSCFHEMultPattern : public OpRewritePattern<arith::MulFOp> {
        // public:
        //     using OpRewritePattern::OpRewritePattern;

        //     LogicalResult matchAndRewrite(arith::MulFOp mulOp,
        //                                   PatternRewriter& rewriter) const override {
        //         auto lhsType = mulOp.getOperand(0).getType();
        //         auto rhsType = mulOp.getOperand(1).getType();
        //         if (!isa<SCFHECipherType>(lhsType) || !isa<SCFHECipherType>(rhsType))
        //             return failure();

        //         // 创建 scfhe.simdmult
        //         auto simdMul = rewriter.create<scfhe::SCFHEMultOp>(
        //             mulOp.getLoc(),
        //             lhsType,
        //             mulOp.getOperand(0),
        //             mulOp.getOperand(1));

        //         rewriter.replaceOp(mulOp, simdMul.getResult());
        //         return success();
        //     }
        // };
        // pass mult (增强版：支持混合类型自动加密)
        class ConvertMulfToSCFHEMultPattern : public OpRewritePattern<arith::MulFOp> {
        public:
            using OpRewritePattern<arith::MulFOp>::OpRewritePattern;

            LogicalResult matchAndRewrite(arith::MulFOp mulOp,
                                          PatternRewriter& rewriter) const override {
                // 1. 获取操作数
                Value lhs = mulOp.getOperand(0);
                Value rhs = mulOp.getOperand(1);

                auto lhsType = lhs.getType();
                auto rhsType = rhs.getType();

                bool isLhsCipher = isa<SCFHECipherType>(lhsType);
                bool isRhsCipher = isa<SCFHECipherType>(rhsType);

                // 如果两个都是明文，不归我管，返回 failure
                if (!isLhsCipher && !isRhsCipher)
                    return failure();

                // 2. 辅助函数：将明文 Value 提升为密文
                auto ensureCipher = [&](Value val) -> Value {
                    if (isa<SCFHECipherType>(val.getType()))
                        return val;

                    int64_t count = 1;
                    Type elemType = rewriter.getI64Type(); // 默认底层类型

                    if (auto vecType = dyn_cast<VectorType>(val.getType())) {
                        count = vecType.getNumElements();
                        // elemType = vecType.getElementType();
                    }

                    auto cipherType = scfhe::SCFHECipherType::get(
                        rewriter.getContext(), count, elemType);

                    // 插入 scfhe.encrypt
                    auto enc = rewriter.create<scfhe::SCFHEEncryptOp>(
                        mulOp.getLoc(), cipherType, val);
                    return enc.getResult();
                };

                // 3. 统一操作数类型为密文
                Value newLhs = ensureCipher(lhs);
                Value newRhs = ensureCipher(rhs);

                // 4. 创建 scfhe.mult
                // 结果类型通常跟随 LHS (或者根据密文运算规则推导)
                auto resultType = newLhs.getType();

                auto simdMul = rewriter.create<scfhe::SCFHEMultOp>(
                    mulOp.getLoc(),
                    resultType,
                    newLhs,
                    newRhs);

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

        // class ReplaceVectorTransferWriteWithSCFHEStorePattern
        //     : public OpRewritePattern<vector::TransferWriteOp> {
        // public:
        //     using OpRewritePattern<vector::TransferWriteOp>::OpRewritePattern;

        //     LogicalResult matchAndRewrite(vector::TransferWriteOp writeOp,
        //                                   PatternRewriter& rewriter) const override {
        //         // Value vec = writeOp.getVector();
        //         Value vec = writeOp.getOperand(0);
        //         Type vecType = vec.getType();

        //         if (!isa<scfhe::SCFHECipherType>(vecType))
        //             return failure();

        //         auto cipherDestType = vecType;

        //         rewriter.setInsertionPoint(writeOp);

        //         auto nameAttr = rewriter.getStringAttr("scfhe_alloca");
        //         auto nameLoc = mlir::NameLoc::get(nameAttr);

        //         Value cipherAlloca =
        //             rewriter.create<scfhe::SCFHEAllocaOp>(nameLoc, cipherDestType);

        //         rewriter.create<scfhe::SCFHEStoreOp>(writeOp.getLoc(), vec, cipherAlloca);

        //         rewriter.eraseOp(writeOp);

        //         return success();
        //     }
        // };
        class ReplaceVectorTransferWriteWithSCFHEStorePattern
            : public OpRewritePattern<vector::TransferWriteOp> {
        public:
            using OpRewritePattern<vector::TransferWriteOp>::OpRewritePattern;

            LogicalResult matchAndRewrite(vector::TransferWriteOp writeOp,
                                          PatternRewriter& rewriter) const override {
                // 1. [关键] 获取 Value (密文)
                // 必须使用 getOperand(0)，因为 getVector() 会 Crash
                Value vec = writeOp.getOperand(0);

                // 2. 检查类型：如果不是密文，就不处理，返回 failure (交给其他 pattern 或保留原样)
                if (!isa<scfhe::SCFHECipherType>(vec.getType()))
                    return failure();

                // 3. 获取目标内存 (MemRef)
                Value dest = writeOp.getOperand(1);

                // 4. 获取索引 (Indices)
                // vector.transfer_write %vec, %dest[%idx1, %idx2...]
                // 如果你的 scfhe.store 需要索引，请在这里获取并传给 create
                ValueRange indices = writeOp.getIndices();

                // 5. 创建 scfhe.store
                // 注意：这里假设你的 scfhe.store 定义在 TableGen 中支持 (Cipher, MemRef, Indices...)
                // 如果你的 scfhe.store 还没支持索引，可能需要修改 TableGen。
                // 暂时这里只传 (vec, dest)，假设 store 是对齐的或者你内部处理了

                // 这种写法假设 SCFHEStoreOp 的 build 函数签名类似： build(builder, state, value, memref)
                // 如果需要 indices，请改为: rewriter.create<scfhe::SCFHEStoreOp>(writeOp.getLoc(), vec, dest, indices);

                rewriter.create<scfhe::SCFHEStoreOp>(writeOp.getLoc(), vec, dest);

                // 6. 擦除旧 Op
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

        // pass min
        class ReplaceAffineMinWithSCFHEArgMinPattern
            : public OpRewritePattern<affine::AffineForOp> {
        public:
            using OpRewritePattern<affine::AffineForOp>::OpRewritePattern;

            LogicalResult matchAndRewrite(affine::AffineForOp forOp,
                                          PatternRewriter& rewriter) const override {
                // 1. 基础检查
                if (forOp.getNumRegionIterArgs() == 0)
                    return failure();

                Block* body = forOp.getBody();

                // 2. 寻找 cmpf (核心特征：这是一个比较操作)
                arith::CmpFOp cmpOp = nullptr;
                for (auto& op : *body) {
                    if (auto c = dyn_cast<arith::CmpFOp>(op)) {
                        if (c.getPredicate() == arith::CmpFPredicate::OLT ||
                            c.getPredicate() == arith::CmpFPredicate::OLE) {
                            cmpOp = c;
                            break;
                        }
                    }
                }
                if (!cmpOp)
                    return failure();

                // 3. 确定哪个 iter_arg 是 "Min Value" 累加器
                // 比较操作通常是 cmpf(current_val, accumulator)
                Value cmpLhs = cmpOp.getOperand(0);
                Value cmpRhs = cmpOp.getOperand(1);

                BlockArgument iterArgBlockVal = nullptr;
                int iterArgIndex = -1;

                // 检查 operand 0
                for (auto arg : forOp.getRegionIterArgs()) {
                    if (arg == cmpLhs) {
                        iterArgBlockVal = arg;
                        break;
                    }
                }
                // 如果 operand 0 不是，检查 operand 1
                if (!iterArgBlockVal) {
                    for (auto arg : forOp.getRegionIterArgs()) {
                        if (arg == cmpRhs) {
                            iterArgBlockVal = arg;
                            break;
                        }
                    }
                }

                if (!iterArgBlockVal)
                    return failure();

                iterArgIndex = iterArgBlockVal.getArgNumber() - 1;

                // 4. 获取循环的初始输入值 (Init Args)
                // 不再通过 prevNode 瞎找，直接拿 forOp 的输入！
                Value initOperand = forOp.getInits()[iterArgIndex];

                // 5. 准备 Input Ciphertext
                Value ciphertextInput;

                // 情况 A: 输入已经是 Cipher (可能是被其他 Pattern 处理过了，或者是 scfhe.alloca 的结果)
                if (isa<scfhe::SCFHECipherType>(initOperand.getType())) {
                    ciphertextInput = initOperand;
                }
                // 情况 B: 输入是 Plaintext (例如 Constant, VectorType)，我们需要插入加密
                else if (auto vecType = dyn_cast<VectorType>(initOperand.getType())) {
                    // 构造 Cipher Type
                    auto cipherType = scfhe::SCFHECipherType::get(
                        rewriter.getContext(),
                        vecType.getNumElements(),
                        rewriter.getI64Type() // 假设底层用 i64
                    );

                    // 插入 Encrypt Op
                    // 务必设置插入点在 forOp 之前
                    rewriter.setInsertionPoint(forOp);
                    auto encryptOp = rewriter.create<scfhe::SCFHEEncryptOp>(
                        forOp.getLoc(), cipherType, initOperand);
                    ciphertextInput = encryptOp.getResult();
                } else {
                    // 不支持的类型
                    return failure();
                }

                // 6. 构造 scfhe.argmin
                auto inputCipherType = cast<scfhe::SCFHECipherType>(ciphertextInput.getType());
                Type elementType = inputCipherType.getElementType();

                // Result 0: Min Value (Cipher)
                auto resultValType = scfhe::SCFHECipherType::get(rewriter.getContext(), 1, elementType);
                // Result 1: Min Index (Cipher i64/i32)
                auto resultIdxType = scfhe::SCFHECipherType::get(rewriter.getContext(), 1, rewriter.getI64Type());

                // 创建 ArgMin 替代整个 Loop
                rewriter.setInsertionPoint(forOp); // 确保替换位置正确
                auto argMinOp = rewriter.create<scfhe::SCFHEArgMinOp>(
                    forOp.getLoc(),
                    TypeRange{resultValType, resultIdxType},
                    ciphertextInput);

                Value minVal = argMinOp.getResult(0);
                Value minIdx = argMinOp.getResult(1);

                // 7. 替换结果
                SmallVector<Value, 2> replacements;
                for (auto res : forOp.getResults()) {
                    Type resType = res.getType();
                    Type innerType = getElementTypeOrSelf(resType);

                    if (isa<FloatType>(innerType)) {
                        replacements.push_back(minVal);
                    } else {
                        replacements.push_back(minIdx);
                    }
                }

                if (replacements.size() != forOp.getNumResults())
                    return failure();

                rewriter.replaceOp(forOp, replacements);
                return success();
            }

            Type getElementTypeOrSelf(Type type) const {
                if (auto vec = dyn_cast<VectorType>(type))
                    return vec.getElementType();
                if (auto mem = dyn_cast<MemRefType>(type))
                    return mem.getElementType();
                if (auto cipher = dyn_cast<scfhe::SCFHECipherType>(type))
                    return cipher.getElementType();
                return type;
            }
        };

        // pass cmpf
        class ConvertCmpfToSCFHECmpPattern : public OpRewritePattern<arith::CmpFOp> {
        public:
            using OpRewritePattern<arith::CmpFOp>::OpRewritePattern;

            LogicalResult matchAndRewrite(arith::CmpFOp cmpOp,
                                          PatternRewriter& rewriter) const override {
                // 使用 getOperand 避免 crash
                Value lhs = cmpOp.getOperand(0);
                Value rhs = cmpOp.getOperand(1);

                auto lhsType = lhs.getType();
                auto rhsType = rhs.getType();

                bool isLhsCipher = isa<SCFHECipherType>(lhsType);
                bool isRhsCipher = isa<SCFHECipherType>(rhsType);

                // 1. 如果两个都不是密文，不处理 (保持原样作为明文计算)
                if (!isLhsCipher && !isRhsCipher)
                    return failure();

                // 2. 准备转换：确保两个操作数都是密文
                Value newLhs = lhs;
                Value newRhs = rhs;

                // 辅助函数：构造加密操作
                auto encryptValue = [&](Value val) -> Value {
                    // 获取上下文
                    MLIRContext* ctx = rewriter.getContext();
                    Type valType = val.getType();

                    int64_t count = 1;
                    // 这里我们假设密文底层总是处理为 i64 (根据你之前的 vector read pattern)
                    Type elemType = rewriter.getI64Type();

                    if (auto vecType = dyn_cast<VectorType>(valType)) {
                        count = vecType.getNumElements();
                        // 如果需要保留原元素类型，可以用 vecType.getElementType();
                        // 但为了和你之前的 Pattern 保持一致，这里用 i64
                    }

                    auto cipherType = scfhe::SCFHECipherType::get(ctx, count, elemType);

                    // 插入加密 Op
                    auto encOp = rewriter.create<scfhe::SCFHEEncryptOp>(
                        cmpOp.getLoc(), cipherType, val);
                    return encOp.getResult();
                };

                // 如果 LHS 是明文，加密它
                if (!isLhsCipher) {
                    newLhs = encryptValue(lhs);
                }

                // 如果 RHS 是明文，加密它
                if (!isRhsCipher) {
                    newRhs = encryptValue(rhs);
                }

                // 3. 确定结果类型 (通常与输入密文类型一致)
                auto resultType = newLhs.getType();

                // 4. 获取 Predicate 并强转为 uint64_t
                auto predicate = cmpOp.getPredicate();

                // 5. 创建 scfhe.cmp
                auto scfheCmp = rewriter.create<scfhe::SCFHECmpOp>(
                    cmpOp.getLoc(),
                    resultType,
                    newLhs,
                    newRhs,
                    static_cast<uint64_t>(predicate));

                // 6. 替换旧 Op
                rewriter.replaceOp(cmpOp, scfheCmp.getResult());
                return success();
            }
        };

        // 放在 namespace 中
        class RemoveExtUIOnCipherPattern : public OpRewritePattern<arith::ExtUIOp> {
        public:
            using OpRewritePattern<arith::ExtUIOp>::OpRewritePattern;

            LogicalResult matchAndRewrite(arith::ExtUIOp extOp,
                                          PatternRewriter& rewriter) const override {
                Value input = extOp.getOperand();

                // 如果输入已经是密文 (来自 scfhe.cmp)
                if (isa<scfhe::SCFHECipherType>(input.getType())) {
                    // HE 中通常不需要显式的 bit 扩展，因为密文已经是大整数(ciphertext)了
                    // 直接用输入替换输出
                    // 注意：可能需要 update result type，但在 replaceOp 中 value 会被传递
                    rewriter.replaceOp(extOp, input);
                    return success();
                }
                return failure();
            }
        };

        // pass select
        class ConvertSelectToSCFHESelectPattern : public OpRewritePattern<arith::SelectOp> {
        public:
            using OpRewritePattern<arith::SelectOp>::OpRewritePattern;

            LogicalResult matchAndRewrite(arith::SelectOp selectOp,
                                          PatternRewriter& rewriter) const override {
                // 1. 获取通用操作数 (避免 crash)
                Value cond = selectOp.getOperand(0);
                Value trueVal = selectOp.getOperand(1);
                Value falseVal = selectOp.getOperand(2);

                bool isCondCipher = isa<SCFHECipherType>(cond.getType());
                bool isTrueCipher = isa<SCFHECipherType>(trueVal.getType());
                bool isFalseCipher = isa<SCFHECipherType>(falseVal.getType());

                // 2. 如果全都是明文，不归我管，返回 failure
                if (!isCondCipher && !isTrueCipher && !isFalseCipher)
                    return failure();

                // 3. 辅助函数：将 Value 提升为密文
                auto ensureCipher = [&](Value val) -> Value {
                    if (isa<SCFHECipherType>(val.getType()))
                        return val;

                    // 计算加密所需的 count
                    int64_t count = 1;
                    Type elemType = rewriter.getI64Type(); // 默认底层类型

                    if (auto vecType = dyn_cast<VectorType>(val.getType())) {
                        count = vecType.getNumElements();
                        // 如果需要保留 float/int 区别，可以用 vecType.getElementType()
                        // 这里为了兼容性，统一转为后端处理的 encrypt
                    }

                    auto cipherType = scfhe::SCFHECipherType::get(
                        rewriter.getContext(), count, elemType);

                    // 插入加密指令
                    auto enc = rewriter.create<scfhe::SCFHEEncryptOp>(
                        selectOp.getLoc(), cipherType, val);
                    return enc.getResult();
                };

                // 4. 统一所有操作数为密文
                Value newCond = ensureCipher(cond);
                Value newTrue = ensureCipher(trueVal);
                Value newFalse = ensureCipher(falseVal);

                // 5. 创建 scfhe.select
                // 结果类型与 true 分支一致
                auto resultType = newTrue.getType();

                auto scfheSelect = rewriter.create<scfhe::SCFHESelectOp>(
                    selectOp.getLoc(),
                    resultType,
                    newCond,
                    newTrue,
                    newFalse);

                rewriter.replaceOp(selectOp, scfheSelect.getResult());
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
                                return failure(); // 多个 store，不安全折叠
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

        //===----------------------------------------------------------------------===//
        // Control logic
        //===----------------------------------------------------------------------===//

        // class ConvertAffineForPattern : public OpRewritePattern<affine::AffineForOp> {
        // public:
        //     using OpRewritePattern<affine::AffineForOp>::OpRewritePattern;

        //     LogicalResult matchAndRewrite(affine::AffineForOp forOp,
        //                                   PatternRewriter& rewriter) const override {
        //         if (forOp.getInits().empty())
        //             return failure();

        //         bool needsConversion = false;
        //         SmallVector<Value, 4> newInitArgs;

        //         for (auto operand : forOp.getInits()) {
        //             Type originalType = operand.getType();
        //             if (auto vecType = dyn_cast<VectorType>(originalType)) {
        //                 if (isa<FloatType>(vecType.getElementType())) {
        //                     needsConversion = true;
        //                     auto cipherType = scfhe::SCFHECipherType::get(
        //                         rewriter.getContext(), vecType.getNumElements(), rewriter.getI64Type());

        //                     rewriter.setInsertionPoint(forOp);
        //                     auto encryptOp = rewriter.create<scfhe::SCFHEEncryptOp>(forOp.getLoc(), cipherType, operand);
        //                     newInitArgs.push_back(encryptOp.getResult());
        //                     continue;
        //                 }
        //             }
        //             newInitArgs.push_back(operand);
        //         }

        //         if (!needsConversion)
        //             return failure();

        //         // === 修复点在这里 ===
        //         // getStep() 返回 APInt，必须调用 .getSExtValue() 转为 int64_t
        //         int64_t step = forOp.getStep().getSExtValue();

        //         rewriter.setInsertionPoint(forOp);
        //         auto newForOp = rewriter.create<affine::AffineForOp>(
        //             forOp.getLoc(),
        //             forOp.getLowerBoundOperands(), forOp.getLowerBoundMap(),
        //             forOp.getUpperBoundOperands(), forOp.getUpperBoundMap(),
        //             step,
        //             newInitArgs,
        //             [&](OpBuilder& b, Location loc, Value iv, ValueRange args) {
        //             });

        //         SmallVector<Value, 4> newBlockArgs;
        //         newBlockArgs.push_back(newForOp.getInductionVar());
        //         newBlockArgs.append(newForOp.getRegionIterArgs().begin(), newForOp.getRegionIterArgs().end());

        //         Block* oldBody = forOp.getBody();
        //         Block* newBody = newForOp.getBody();
        //         newBody->clear();

        //         rewriter.mergeBlocks(oldBody, newBody, newBlockArgs);

        //         for (size_t i = 0; i < newForOp.getRegionIterArgs().size(); ++i) {
        //             Value arg = newForOp.getRegionIterArgs()[i];
        //             if (isa<scfhe::SCFHECipherType>(newInitArgs[i].getType())) {
        //                 arg.setType(newInitArgs[i].getType());
        //             }
        //         }

        //         rewriter.replaceOp(forOp, newForOp.getResults());
        //         return success();
        //     }
        // };

        // class ConvertAffineForPattern : public OpRewritePattern<affine::AffineForOp> {
        // public:
        //     using OpRewritePattern<affine::AffineForOp>::OpRewritePattern;

        //     LogicalResult matchAndRewrite(affine::AffineForOp forOp,
        //                                   PatternRewriter& rewriter) const override {
        //         // 1. 检查是否有初始参数
        //         if (forOp.getInits().empty())
        //             return failure();

        //         bool needsConversion = false;
        //         SmallVector<Value, 4> newInitArgs;

        //         // 2. 加密初始参数 (Loop Carry Args)
        //         for (auto operand : forOp.getInits()) {
        //             Type originalType = operand.getType();
        //             if (auto vecType = dyn_cast<VectorType>(originalType)) {
        //                 Type elemType = vecType.getElementType();
        //                 // [关键修改]：同时支持 FloatType 和 IntegerType
        //                 // 在 KMeans 中，argMin 的索引是整数，但也需要加密
        //                 if (isa<FloatType>(elemType) || isa<IntegerType>(elemType)) {
        //                     needsConversion = true;
        //                     // 统一使用 i64 作为底层密文类型 (因为 scfhe.encrypt 通常会提升 i32->i64)
        //                     auto cipherType = scfhe::SCFHECipherType::get(
        //                         rewriter.getContext(), vecType.getNumElements(), rewriter.getI64Type());

        //                     rewriter.setInsertionPoint(forOp);
        //                     auto encryptOp = rewriter.create<scfhe::SCFHEEncryptOp>(forOp.getLoc(), cipherType, operand);
        //                     newInitArgs.push_back(encryptOp.getResult());
        //                     continue;
        //                 }
        //             }
        //             newInitArgs.push_back(operand);
        //         }

        //         if (!needsConversion)
        //             return failure();

        //         // 3. 创建新的 Loop
        //         int64_t step = forOp.getStep().getSExtValue();

        //         rewriter.setInsertionPoint(forOp);
        //         auto newForOp = rewriter.create<affine::AffineForOp>(
        //             forOp.getLoc(),
        //             forOp.getLowerBoundOperands(), forOp.getLowerBoundMap(),
        //             forOp.getUpperBoundOperands(), forOp.getUpperBoundMap(),
        //             step,
        //             newInitArgs,
        //             [&](OpBuilder& b, Location loc, Value iv, ValueRange args) {
        //                 // Body creation logic is handled by mergeBlocks later
        //             });

        //         // 4. 合并 Block 并更新参数类型
        //         SmallVector<Value, 4> newBlockArgs;
        //         newBlockArgs.push_back(newForOp.getInductionVar());
        //         newBlockArgs.append(newForOp.getRegionIterArgs().begin(), newForOp.getRegionIterArgs().end());

        //         Block* oldBody = forOp.getBody();
        //         Block* newBody = newForOp.getBody();
        //         newBody->clear();

        //         rewriter.mergeBlocks(oldBody, newBody, newBlockArgs);

        //         // 更新 Block Argument 的类型为 Cipher
        //         for (size_t i = 0; i < newForOp.getRegionIterArgs().size(); ++i) {
        //             Value arg = newForOp.getRegionIterArgs()[i];
        //             if (isa<scfhe::SCFHECipherType>(newInitArgs[i].getType())) {
        //                 arg.setType(newInitArgs[i].getType());
        //             }
        //         }

        //         // ========================== 修复 Yield 类型 (Step 5) ==========================
        //         Operation* terminator = newBody->getTerminator();
        //         if (auto yieldOp = dyn_cast<affine::AffineYieldOp>(terminator)) {
        //             rewriter.setInsertionPoint(yieldOp);
        //             for (unsigned i = 0; i < yieldOp.getNumOperands(); ++i) {
        //                 Value yieldedVal = yieldOp.getOperand(i);
        //                 Type expectedType = newInitArgs[i].getType(); // Loop 期望的类型 (Cipher)

        //                 // 如果 Loop 期望 Cipher，但 Yield 提供的是 Vector (Plaintext)
        //                 if (isa<scfhe::SCFHECipherType>(expectedType) && !isa<scfhe::SCFHECipherType>(yieldedVal.getType())) {
        //                     // 创建一个临时的 Encrypt Op 桥接类型
        //                     auto encryptOp = rewriter.create<scfhe::SCFHEEncryptOp>(
        //                         yieldOp.getLoc(),
        //                         expectedType,
        //                         yieldedVal);
        //                     yieldOp.setOperand(i, encryptOp.getResult());
        //                 }
        //             }
        //         }
        //         // ============================================================================

        //         // ========================== 替换下游 User (Step 6) ==========================
        //         for (auto item : llvm::zip(forOp.getResults(), newForOp.getResults())) {
        //             Value oldResult = std::get<0>(item);
        //             Value newResult = std::get<1>(item);

        //             if (!isa<scfhe::SCFHECipherType>(newResult.getType()))
        //                 continue;

        //             for (Operation* user : llvm::make_early_inc_range(oldResult.getUsers())) {
        //                 // 处理 vector.transfer_write
        //                 if (auto writeOp = dyn_cast<vector::TransferWriteOp>(user)) {
        //                     if (writeOp.getOperand(0) == oldResult) {
        //                         rewriter.setInsertionPoint(writeOp);
        //                         rewriter.create<scfhe::SCFHEStoreOp>(
        //                             writeOp.getLoc(), newResult, writeOp.getOperand(1));
        //                         rewriter.eraseOp(writeOp);
        //                     }
        //                 }
        //                 // 处理 vector.reduction
        //                 else if (auto reduceOp = dyn_cast<vector::ReductionOp>(user)) {
        //                     if (reduceOp.getKind() == vector::CombiningKind::ADD) {
        //                         auto vecCipherType = cast<scfhe::SCFHECipherType>(newResult.getType());
        //                         auto scalarCipherType = scfhe::SCFHECipherType::get(
        //                             rewriter.getContext(), 1, vecCipherType.getElementType());

        //                         rewriter.setInsertionPoint(reduceOp);
        //                         auto scfheReduce = rewriter.create<scfhe::SCFHEReduceAddOp>(
        //                             reduceOp.getLoc(), scalarCipherType, newResult);
        //                         Value reduceRes = scfheReduce.getResult();

        //                         // 替换 reduction 结果的使用者 (通常是 affine.store)
        //                         for (Operation* reduceUser : llvm::make_early_inc_range(reduceOp.getResult().getUsers())) {
        //                             if (auto storeOp = dyn_cast<affine::AffineStoreOp>(reduceUser)) {
        //                                 if (storeOp.getValue() == reduceOp.getResult()) {
        //                                     rewriter.setInsertionPoint(storeOp);
        //                                     rewriter.create<scfhe::SCFHEStoreOp>(
        //                                         storeOp.getLoc(), reduceRes, storeOp.getMemRef());
        //                                     rewriter.eraseOp(storeOp);
        //                                 }
        //                             }
        //                         }
        //                         rewriter.eraseOp(reduceOp);
        //                     } else {
        //                         return failure();
        //                     }
        //                 }
        //             }
        //         }
        //         // ============================================================================

        //         rewriter.replaceOp(forOp, newForOp.getResults());
        //         return success();
        //     }
        // };

        class ConvertAffineForPattern : public OpRewritePattern<affine::AffineForOp> {
        public:
            using OpRewritePattern<affine::AffineForOp>::OpRewritePattern;

            LogicalResult matchAndRewrite(affine::AffineForOp forOp,
                                          PatternRewriter& rewriter) const override {
                // 1. 检查是否有初始参数
                if (forOp.getInits().empty())
                    return failure();

                bool needsConversion = false;
                SmallVector<Value, 4> newInitArgs;

                // 2. 加密初始参数 (处理 Loop Carry Args)
                // 这决定了循环的签名是 (!scfhe.cipher, ...)
                for (auto operand : forOp.getInits()) {
                    Type originalType = operand.getType();
                    if (auto vecType = dyn_cast<VectorType>(originalType)) {
                        Type elemType = vecType.getElementType();
                        // 支持 Float 和 Integer 的向量加密
                        if (isa<FloatType>(elemType) || isa<IntegerType>(elemType)) {
                            needsConversion = true;
                            // 统一使用 i64 作为底层密文类型
                            auto cipherType = scfhe::SCFHECipherType::get(
                                rewriter.getContext(), vecType.getNumElements(), rewriter.getI64Type());

                            rewriter.setInsertionPoint(forOp);
                            auto encryptOp = rewriter.create<scfhe::SCFHEEncryptOp>(forOp.getLoc(), cipherType, operand);
                            newInitArgs.push_back(encryptOp.getResult());
                            continue;
                        }
                    }
                    newInitArgs.push_back(operand);
                }

                if (!needsConversion)
                    return failure();

                // 3. 创建新的 Loop
                int64_t step = forOp.getStep().getSExtValue();

                rewriter.setInsertionPoint(forOp);
                auto newForOp = rewriter.create<affine::AffineForOp>(
                    forOp.getLoc(),
                    forOp.getLowerBoundOperands(), forOp.getLowerBoundMap(),
                    forOp.getUpperBoundOperands(), forOp.getUpperBoundMap(),
                    step,
                    newInitArgs,
                    [&](OpBuilder& b, Location loc, Value iv, ValueRange args) {
                        // Body 暂时留空，由 mergeBlocks 填充
                    });

                // 4. 合并 Block 并更新参数类型
                SmallVector<Value, 4> newBlockArgs;
                newBlockArgs.push_back(newForOp.getInductionVar());
                newBlockArgs.append(newForOp.getRegionIterArgs().begin(), newForOp.getRegionIterArgs().end());

                Block* oldBody = forOp.getBody();
                Block* newBody = newForOp.getBody();
                newBody->clear();

                rewriter.mergeBlocks(oldBody, newBody, newBlockArgs);

                // 更新 Block Argument 的类型为 Cipher
                for (size_t i = 0; i < newForOp.getRegionIterArgs().size(); ++i) {
                    Value arg = newForOp.getRegionIterArgs()[i];
                    if (isa<scfhe::SCFHECipherType>(newInitArgs[i].getType())) {
                        arg.setType(newInitArgs[i].getType());
                    }
                }

                // ============================================================================
                // Step 5: "转化" Affine.yield (核心修复)
                // 我们不替换 Op 本身，而是修正它的操作数，使其符合 scfhe 规范。
                // 这样既满足了 affine.for 的约束，又解决了类型不匹配问题。
                // ============================================================================
                Operation* terminator = newBody->getTerminator();
                if (auto yieldOp = dyn_cast<affine::AffineYieldOp>(terminator)) {
                    rewriter.setInsertionPoint(yieldOp);
                    for (unsigned i = 0; i < yieldOp.getNumOperands(); ++i) {
                        Value yieldedVal = yieldOp.getOperand(i);
                        Type expectedType = newInitArgs[i].getType(); // 这是 Loop 签名要求的类型 (Cipher)

                        // 检查类型是否匹配
                        if (yieldedVal.getType() != expectedType) {
                            // 如果 Loop 期望 Cipher，但 Yield 提供的是 Vector (明文)
                            // 我们在这里插入 scfhe.encrypt，相当于把 yield 变成了 "scfhe 操作"
                            if (isa<scfhe::SCFHECipherType>(expectedType)) {
                                auto encryptOp = rewriter.create<scfhe::SCFHEEncryptOp>(
                                    yieldOp.getLoc(),
                                    expectedType,
                                    yieldedVal);
                                // 修改 yield 的操作数
                                yieldOp.setOperand(i, encryptOp.getResult());
                            }
                        }
                    }
                }
                // ============================================================================

                // 6. 替换下游 User (TransferWrite / Reduction)
                // 这一步必须做，否则旧的 User 还在引用 Loop 结果，会导致 Crash
                for (auto item : llvm::zip(forOp.getResults(), newForOp.getResults())) {
                    Value oldResult = std::get<0>(item);
                    Value newResult = std::get<1>(item);

                    if (!isa<scfhe::SCFHECipherType>(newResult.getType()))
                        continue;

                    for (Operation* user : llvm::make_early_inc_range(oldResult.getUsers())) {
                        if (auto writeOp = dyn_cast<vector::TransferWriteOp>(user)) {
                            if (writeOp.getOperand(0) == oldResult) {
                                rewriter.setInsertionPoint(writeOp);
                                rewriter.create<scfhe::SCFHEStoreOp>(
                                    writeOp.getLoc(), newResult, writeOp.getOperand(1));
                                rewriter.eraseOp(writeOp);
                            }
                        } else if (auto reduceOp = dyn_cast<vector::ReductionOp>(user)) {
                            // 这里需要你定义了 scfhe.reduce_add
                            // 如果还没定义，暂时 return failure() 避免崩溃
                            if (reduceOp.getKind() == vector::CombiningKind::ADD) {
                                auto vecCipherType = cast<scfhe::SCFHECipherType>(newResult.getType());
                                auto scalarCipherType = scfhe::SCFHECipherType::get(
                                    rewriter.getContext(), 1, vecCipherType.getElementType());

                                rewriter.setInsertionPoint(reduceOp);
                                auto scfheReduce = rewriter.create<scfhe::SCFHEReduceAddOp>(
                                    reduceOp.getLoc(), scalarCipherType, newResult);

                                // 级联替换 reduce 的 store user
                                Value reduceRes = scfheReduce.getResult();

                                // for (Operation* reduceUser : llvm::make_early_inc_range(reduceOp.getResult().getUsers())) {
                                //     if (auto storeOp = dyn_cast<affine::AffineStoreOp>(reduceUser)) {
                                //         rewriter.setInsertionPoint(storeOp);
                                //         rewriter.create<scfhe::SCFHEStoreOp>(
                                //             storeOp.getLoc(), reduceRes, storeOp.getMemRef());
                                //         rewriter.eraseOp(storeOp);
                                //     }
                                // }
                                // rewriter.eraseOp(reduceOp);

                                SmallVector<Operation*> storesToDelete;
                                for (Operation* reduceUser : reduceOp.getResult().getUsers()) {
                                    if (auto storeOp = dyn_cast<affine::AffineStoreOp>(reduceUser)) {
                                        rewriter.setInsertionPoint(storeOp);
                                        rewriter.create<scfhe::SCFHEStoreOp>(
                                            storeOp.getLoc(), reduceRes, storeOp.getMemRef());
                                        storesToDelete.push_back(storeOp);
                                    }
                                }

                                for (auto* op : storesToDelete) {
                                    rewriter.eraseOp(op);
                                }

                                // [关键修复] 检查是否还有 User
                                if (reduceOp.use_empty()) {
                                    rewriter.eraseOp(reduceOp);
                                } else {

                                    auto decrypt = rewriter.create<scfhe::SCFHEDecryptOp>(
                                        reduceOp.getLoc(), reduceOp.getType(), reduceRes);
                                    rewriter.replaceOp(reduceOp, decrypt.getResult());
                                }

                            } else {
                                return failure();
                            }
                        }
                    }
                }

                rewriter.replaceOp(forOp, newForOp.getResults());
                return success();
            }
        };

        //===================== Pattern Matching End =====================//

        class ConvertToSCFHEIR
            : public impl::ConvertToSCFHEPassBase<ConvertToSCFHEIR> {
        public:
            using impl::ConvertToSCFHEPassBase<ConvertToSCFHEIR>::ConvertToSCFHEPassBase;

            void runOnOperation() final {
                auto module = getOperation(); // ModuleOp

                RewritePatternSet patterns(module.getContext());
                patterns.add<
                    ReplaceReductionLoopWithSCFHEThresholdCountPattern,

                    InsertEncryptAfterVectorTransferReadPattern,
                    ConvertAddfToSCFHEAddPattern,
                    ConvertSubfToSCFHESubPattern,
                    ConvertMulfToSCFHEMultPattern,
                    ConvertDivfToSCFHEDivPattern,
                    ConvertMathExpToSCFHEExpPattern,

                    ConvertCmpfToSCFHECmpPattern,
                    RemoveExtUIOnCipherPattern,
                    ConvertSelectToSCFHESelectPattern,

                    // ReplaceAffineMinWithSCFHEMinPattern,
                    ReplaceAffineMinWithSCFHEArgMinPattern,
                    ConvertAffineForPattern,
                    ReplaceVectorTransferWriteWithSCFHEStorePattern,
                    InsertDecryptBeforePrintfPattern,
                    FoldAllocaStorePattern>(&getContext());

                (void)applyPatternsGreedily(module, std::move(patterns));
            }
        };
    } // namespace
} // namespace mlir::libra::scfhe
