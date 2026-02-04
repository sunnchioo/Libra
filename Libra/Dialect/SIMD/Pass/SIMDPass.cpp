#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/IR/BuiltinOps.h"

#include "SCFHEOps.h"
#include "SCFHETypes.h"
#include "SIMDOps.h"
#include "SIMDPass.h"

#include "SIMDCommon.h"

namespace mlir::libra::simd {
#define GEN_PASS_DEF_CONVERTTOSIMDPASS
#include "SIMDPass.h.inc"

    namespace {

        // static constexpr int64_t DEFAULT_LEVEL = 12;

        /// 封装通用类型转换：SCFHE → SIMD
        static FailureOr<SIMDCipherType> convertToSIMDType(Type t, MLIRContext* ctx,
                                                           int64_t level = DEFAULT_LEVEL) {
            if (auto st = dyn_cast<SIMDCipherType>(t))
                return st;
            if (auto st = dyn_cast<scfhe::SCFHECipherType>(t))
                return SIMDCipherType::get(ctx, level, st.getPlaintextCount(), st.getElementType());
            return failure();
        }

        /// 根据两个 operand 类型推断 SIMD 结果类型
        static FailureOr<SIMDCipherType> inferSIMDResultType(Value a, Value b, MLIRContext* ctx,
                                                             bool reduceLevel = false) {
            auto ta = convertToSIMDType(a.getType(), ctx);
            auto tb = convertToSIMDType(b.getType(), ctx);
            if (failed(ta) || failed(tb))
                return failure();
            int64_t newLevel = std::min(ta->getLevel(), tb->getLevel());
            if (reduceLevel)
                newLevel = std::max<int64_t>(0, newLevel - 1);
            int64_t newPC = std::min(ta->getPlaintextCount(), tb->getPlaintextCount());
            return SIMDCipherType::get(ctx, newLevel, newPC, ta->getElementType());
        }

        //===----------------------------------------------------------------------===//
        // Phase 1: SCFHE → SIMD
        //===----------------------------------------------------------------------===//

        struct ConvertSCFHEEncryptToSIMDPattern : OpRewritePattern<scfhe::SCFHEEncryptOp> {
            using OpRewritePattern::OpRewritePattern;
            LogicalResult matchAndRewrite(scfhe::SCFHEEncryptOp op, PatternRewriter& rewriter) const override {

                if (isa<scfhe::SCFHECipherType, SIMDCipherType>(op.getOperand().getType())) {
                    rewriter.replaceOp(op, op.getOperand());
                    return success();
                }

                auto ty = convertToSIMDType(op.getResult().getType(), rewriter.getContext());
                if (failed(ty))
                    return failure();

                rewriter.replaceOpWithNewOp<SIMDEncryptOp>(op, *ty, op.getOperand());
                return success();
            }

            // LogicalResult matchAndRewrite(scfhe::SCFHEEncryptOp op,
            //                               PatternRewriter& rewriter) const override {
            //     auto ctx = rewriter.getContext();
            //     auto ty = convertToSIMDType(op.getResult().getType(), ctx);
            //     if (failed(ty))
            //         return failure();
            //     auto newOp = rewriter.create<SIMDEncryptOp>(op.getLoc(), *ty, op.getOperand());
            //     rewriter.replaceOp(op, newOp);
            //     return success();
            // }
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

        // Add
        struct ConvertSCFHEAddToSIMDPattern : OpRewritePattern<scfhe::SCFHEAddOp> {
            using OpRewritePattern::OpRewritePattern;
            LogicalResult matchAndRewrite(scfhe::SCFHEAddOp op, PatternRewriter& rewriter) const override {
                auto ctx = rewriter.getContext();
                // Add 通常不降低 Level (reduceLevel=false)
                auto ty = inferSIMDResultType(op.getOperand(0), op.getOperand(1), ctx, /*reduceLevel=*/false);
                if (failed(ty))
                    return failure();

                auto newOp = rewriter.create<SIMDAddOp>(op.getLoc(), *ty, op.getOperands());
                rewriter.replaceOp(op, newOp);
                return success();
            }
        };

        // Div
        struct ConvertSCFHEDivToSIMDPattern : OpRewritePattern<scfhe::SCFHEDivOp> {
            using OpRewritePattern::OpRewritePattern;
            LogicalResult matchAndRewrite(scfhe::SCFHEDivOp op, PatternRewriter& rewriter) const override {
                auto ctx = rewriter.getContext();
                // Div 可能会大幅消耗 Level，视具体同态库而定，这里暂设为 true
                auto ty = inferSIMDResultType(op.getOperand(0), op.getOperand(1), ctx, /*reduceLevel=*/true);
                if (failed(ty))
                    return failure();

                auto newOp = rewriter.create<SIMDDivOp>(op.getLoc(), *ty, op.getOperands());
                rewriter.replaceOp(op, newOp);
                return success();
            }
        };

        // Exp
        struct ConvertSCFHEExpToSIMDPattern : OpRewritePattern<scfhe::SCFHEExpOp> {
            using OpRewritePattern::OpRewritePattern;

            LogicalResult matchAndRewrite(scfhe::SCFHEExpOp op, PatternRewriter& rewriter) const override {
                auto ctx = rewriter.getContext();

                // 1. 获取输入 (应该是已经被转换过的 SIMD 类型)
                Value input = op.getOperand();
                auto inputTy = dyn_cast<simd::SIMDCipherType>(input.getType());

                if (!inputTy) {
                    // 如果输入还没转成 SIMD，说明前序 Pass 有问题，或者顺序不对
                    return failure();
                }

                // 2. 构造结果类型 (保持 Level 一致或根据 exp 消耗调整)
                // 假设 exp 消耗特定的 Level，这里暂时保持一致或使用 infer
                // 如果你的 scfhe.exp 定义了结果类型，尝试转换它

                // 简单起见，继承输入的 SIMD 类型参数
                auto resultTy = simd::SIMDCipherType::get(
                    ctx,
                    inputTy.getLevel(), // 注意：如果 EXP 消耗 Level，这里需要 -消耗值
                    inputTy.getPlaintextCount(),
                    inputTy.getElementType());

                // 3. 创建 simd.exp (假设你有定义这个 Op)
                // 如果没有定义 simd.exp，你需要在这里展开为多项式 (Add/Mult)
                auto newOp = rewriter.create<simd::SIMDExpOp>(op.getLoc(), resultTy, input);

                rewriter.replaceOp(op, newOp.getResult());
                return success();
            }
        };

        // Select
        struct ConvertSCFHESelectToSIMDPattern : OpRewritePattern<scfhe::SCFHESelectOp> {
            using OpRewritePattern::OpRewritePattern;
            LogicalResult matchAndRewrite(scfhe::SCFHESelectOp op, PatternRewriter& rewriter) const override {
                auto ctx = rewriter.getContext();
                // Select 有 3 个操作数：Condition, TrueVal, FalseVal
                // 我们根据 TrueVal 和 FalseVal 推断结果类型
                auto ty = inferSIMDResultType(op.getTrueValue(), op.getFalseValue(), ctx);
                if (failed(ty))
                    return failure();

                // 还需要确保 Condition 也是 SIMD 类型 (通常由 Cmp 产生)
                if (failed(convertToSIMDType(op.getCondition().getType(), ctx)))
                    return failure();

                auto newOp = rewriter.create<SIMDSelectOp>(
                    op.getLoc(), *ty, op.getCondition(), op.getTrueValue(), op.getFalseValue());
                rewriter.replaceOp(op, newOp);
                return success();
            }
        };

        // Cmp (Compare)
        struct ConvertSCFHECmpToSIMDPattern : OpRewritePattern<scfhe::SCFHECmpOp> {
            using OpRewritePattern::OpRewritePattern;
            LogicalResult matchAndRewrite(scfhe::SCFHECmpOp op, PatternRewriter& rewriter) const override {
                auto ctx = rewriter.getContext();
                // 推断输入的操作数类型 (LHS, RHS) 以确定 Level
                auto inTy = inferSIMDResultType(op.getLhs(), op.getRhs(), ctx);
                if (failed(inTy))
                    return failure();

                // Cmp 的结果通常也是密文 (Mask)，保持相同的 PC，Level 可能不变
                auto resultTy = SIMDCipherType::get(ctx, inTy->getLevel(), inTy->getPlaintextCount(), inTy->getElementType());

                auto newOp = rewriter.create<SIMDCmpOp>(
                    op.getLoc(), resultTy, op.getLhs(), op.getRhs(), op.getPredicate());
                rewriter.replaceOp(op, newOp);
                return success();
            }
        };

        // ReduceAdd
        struct ConvertSCFHEReduceAddToSIMDPattern : OpRewritePattern<scfhe::SCFHEReduceAddOp> {
            using OpRewritePattern::OpRewritePattern;
            LogicalResult matchAndRewrite(scfhe::SCFHEReduceAddOp op, PatternRewriter& rewriter) const override {
                auto ctx = rewriter.getContext();
                auto inTy = convertToSIMDType(op.getInput().getType(), ctx);
                if (failed(inTy))
                    return failure();

                // 结果应该是标量 SIMD (PC=1)
                auto resultTy = SIMDCipherType::get(ctx, inTy->getLevel(), 1, inTy->getElementType());

                auto newOp = rewriter.create<SIMDReduceAddOp>(
                    op.getLoc(), resultTy, op.getInput());
                rewriter.replaceOp(op, newOp);
                return success();
            }
        };

        // Store
        // struct ConvertSCFHEStoreToSIMDPattern : OpRewritePattern<scfhe::SCFHEStoreOp> {
        //     using OpRewritePattern::OpRewritePattern;
        //     LogicalResult matchAndRewrite(scfhe::SCFHEStoreOp op, PatternRewriter& rewriter) const override {
        //         // Store 没有返回值，只需要把 Value 转换为 SIMD 并创建新 Op
        //         // 这里假设 SIMDStoreOp 的定义与 SCFHEStoreOp 类似
        //         if (failed(convertToSIMDType(op.getValue().getType(), rewriter.getContext())))
        //             return failure();

        //         rewriter.create<SIMDStoreOp>(op.getLoc(), op.getValue(), op.getMemRef());
        //         rewriter.eraseOp(op);
        //         return success();
        //     }
        // };

        // Store
        struct ConvertSCFHEStoreToSIMDPattern : OpRewritePattern<scfhe::SCFHEStoreOp> {
            using OpRewritePattern::OpRewritePattern;
            LogicalResult matchAndRewrite(scfhe::SCFHEStoreOp op, PatternRewriter& rewriter) const override {
                // Store 没有返回值，只需要把 Value 转换为 SIMD 并创建新 Op
                if (failed(convertToSIMDType(op.getValue().getType(), rewriter.getContext())))
                    return failure();

                // 修正点：将 op.getMemRef() 改为 op.getDest()
                // 如果 SCFHEStoreOp 的 TableGen 定义中参数名为 $dest，则访问器为 getDest()
                rewriter.create<SIMDStoreOp>(op.getLoc(), op.getValue(), op.getDest());

                rewriter.eraseOp(op);
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
                auto ty = SIMDCipherType::get(ctx, inTy->getLevel(), /*pc=*/1, inTy->getElementType());
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
                auto newTy = SIMDCipherType::get(rewriter.getContext(), inTy.getLevel(), /*pc=*/1,
                                                 inTy.getElementType());
                if (newTy == op.getResult().getType())
                    return failure();
                auto newOp = rewriter.create<SIMDMinOp>(op.getLoc(), newTy, op.getOperand());
                rewriter.replaceOp(op, newOp);
                return success();
            }
        };

        // 处理 affine.for 的类型转换
        // 处理 affine.for 的类型转换 (SCFHE/Vector -> SIMD)
        struct ConvertAffineForToSIMDPattern : public OpRewritePattern<affine::AffineForOp> {
            using OpRewritePattern::OpRewritePattern;

            LogicalResult matchAndRewrite(affine::AffineForOp forOp, PatternRewriter& rewriter) const override {
                // 如果没有携带参数 (init_args)，不需要转换类型
                if (forOp.getInits().empty())
                    return failure();

                bool needsConversion = false;
                SmallVector<Value, 4> newInitArgs;
                MLIRContext* ctx = rewriter.getContext();

                // ====================================================
                // 1. 准备新的 init_args，决定新 Loop 的签名
                // ====================================================
                for (auto operand : forOp.getInits()) {
                    auto res = convertToSIMDType(operand.getType(), ctx);

                    if (succeeded(res)) {
                        Type targetType = *res;

                        // [关键修复]: 只有当类型确实不一致时，才标记需要转换
                        // 如果 operand 已经是 SIMD 类型，则不触发重写，防止死循环
                        if (operand.getType() != targetType) {
                            needsConversion = true;

                            if (isa<VectorType>(operand.getType())) {
                                // 情况 A: Vector -> SIMD (使用 SIMDEncrypt)
                                auto encrypt = rewriter.create<SIMDEncryptOp>(forOp.getLoc(), targetType, operand);
                                newInitArgs.push_back(encrypt.getResult());
                            } else {
                                // 情况 B: SCFHE -> SIMD (使用 UnrealizedConversionCast)
                                // 这是一个临时 Cast，强制让新 Loop 的输入变为 SIMD 类型。
                                // 这样新 Loop 的 Block Argument 和 Result 都会变成 SIMD 类型。
                                auto cast = rewriter.create<UnrealizedConversionCastOp>(
                                    forOp.getLoc(), targetType, operand);
                                newInitArgs.push_back(cast.getResult(0));
                            }
                        } else {
                            // 已经是目标类型，直接使用
                            newInitArgs.push_back(operand);
                        }
                    } else {
                        // 不相关的类型（如 index），保持原样
                        newInitArgs.push_back(operand);
                    }
                }

                if (!needsConversion) {
                    if (auto yieldOp = dyn_cast<affine::AffineYieldOp>(forOp.getBody()->getTerminator())) {
                        for (unsigned i = 0; i < yieldOp.getNumOperands(); ++i) {
                            // newInitArgs[i] 此时就是 Loop 预期的类型 (例如 L31)
                            Type expectedType = newInitArgs[i].getType();
                            Type actualYieldType = yieldOp.getOperand(i).getType();

                            // 如果 Yield 的类型 (例如 L30) 与 签名 (L31) 不一致，强制触发重写
                            if (expectedType != actualYieldType) {
                                needsConversion = true;
                                break;
                            }
                        }
                    }
                }

                // 如果没有任何参数需要类型升级，说明此 Loop 已经被处理过或无关，退出
                if (!needsConversion)
                    return failure();

                // ====================================================
                // 2. 创建新的 Loop (使用 SIMD 类型参数)
                // ====================================================
                auto newForOp = rewriter.create<affine::AffineForOp>(
                    forOp.getLoc(),
                    forOp.getLowerBoundOperands(), forOp.getLowerBoundMap(),
                    forOp.getUpperBoundOperands(), forOp.getUpperBoundMap(),
                    forOp.getStep().getSExtValue(),
                    newInitArgs,
                    [&](OpBuilder& b, Location loc, Value iv, ValueRange args) {
                    });

                Block* oldBody = forOp.getBody();
                Block* newBody = newForOp.getBody();

                SmallVector<Value, 4> newBlockArgs;
                newBlockArgs.push_back(newForOp.getInductionVar());
                for (auto arg : newForOp.getRegionIterArgs()) {
                    newBlockArgs.push_back(arg);
                }

                newBody->clear();
                rewriter.mergeBlocks(oldBody, newBody, newBlockArgs);

                // 修正 Block 参数类型
                for (size_t i = 0; i < newForOp.getRegionIterArgs().size(); ++i) {
                    BlockArgument arg = newForOp.getRegionIterArgs()[i];
                    Value initVal = newInitArgs[i];
                    if (arg.getType() != initVal.getType()) {
                        arg.setType(initVal.getType());
                    }
                }
                // 在 ConvertAffineForToSIMDPattern 的 matchAndRewrite 中，替换 Step 4 部分：

                // ====================================================
                // 4. 修复 affine.yield 类型不匹配 (使用 replaceOpWithNewOp)
                // ====================================================
                Operation* terminator = newBody->getTerminator();
                if (auto yieldOp = dyn_cast<affine::AffineYieldOp>(terminator)) {
                    rewriter.setInsertionPoint(yieldOp);
                    SmallVector<Value, 4> newYieldOperands;
                    bool typesChanged = false;

                    for (unsigned i = 0; i < yieldOp.getNumOperands(); ++i) {
                        Value val = yieldOp.getOperand(i);
                        // Loop 签名期望的类型 (L31)
                        Type expectedType = newInitArgs[i].getType();

                        if (val.getType() != expectedType) {
                            typesChanged = true;
                            if (auto simdTy = dyn_cast<SIMDCipherType>(expectedType)) {
                                // 插入 Cast: ValType -> SIMD L31
                                // 即使 val 目前是 scfhe 类型，这也会插入一个 UnrealizedConversionCast
                                // 当 scfhe 后来被转换为 simd L30 时，这个 Cast 变成 L30 -> L31，保持 IR 合法
                                auto castOp = rewriter.create<UnrealizedConversionCastOp>(
                                    yieldOp.getLoc(), simdTy, val);
                                newYieldOperands.push_back(castOp.getResult(0));
                            } else {
                                // 其他类型不匹配情况，尝试强转或报错，这里暂且保留原值或插入 Cast
                                auto castOp = rewriter.create<UnrealizedConversionCastOp>(
                                    yieldOp.getLoc(), expectedType, val);
                                newYieldOperands.push_back(castOp.getResult(0));
                            }
                        } else {
                            newYieldOperands.push_back(val);
                        }
                    }

                    // 只有在确实需要修改时才替换，或者为了保险起见总是替换
                    // 这里建议：只要是在重构 Loop，总是重建 terminator 以确保干净
                    rewriter.replaceOpWithNewOp<affine::AffineYieldOp>(yieldOp, newYieldOperands);
                }

                // ====================================================
                // 5. 替换旧 Loop
                // ====================================================
                // 注意：newForOp 返回的是 SIMD 类型，而旧 forOp 的 User 可能还在期待 SCFHE/Vector。
                // 为了保证替换后的 IR 合法，我们需要对结果进行反向 Cast (SIMD -> OldType)。
                // 这样后续的 Pattern 才能继续处理 User。

                rewriter.setInsertionPointAfter(newForOp);

                SmallVector<Value, 4> finalResults;
                for (auto item : llvm::zip(forOp.getResults(), newForOp.getResults())) {
                    Value oldRes = std::get<0>(item);
                    Value newRes = std::get<1>(item);

                    if (oldRes.getType() != newRes.getType()) {
                        // 创建反向 Cast (SIMD -> SCFHE/Vector) 以满足旧 User
                        auto castBack = rewriter.create<UnrealizedConversionCastOp>(
                            forOp.getLoc(), oldRes.getType(), newRes);
                        finalResults.push_back(castBack.getResult(0));
                    } else {
                        finalResults.push_back(newRes);
                    }
                }

                rewriter.replaceOp(forOp, finalResults);
                return success();
            }
        };

        // [新增] 专门用于修复 affine.yield 类型不匹配的简单 Pattern
        // 将此结构体放在您的匿名命名空间中 (namespace { ... })

        struct FixAffineYieldTypeMismatch : OpRewritePattern<affine::AffineYieldOp> {
            using OpRewritePattern::OpRewritePattern;

            LogicalResult matchAndRewrite(affine::AffineYieldOp yieldOp, PatternRewriter& rewriter) const override {
                // 1. 获取父循环
                auto forOp = yieldOp->getParentOfType<affine::AffineForOp>();
                if (!forOp)
                    return failure();

                // 2. 检查是否有类型不匹配
                bool needsFix = false;
                auto iterArgs = forOp.getRegionIterArgs();
                if (yieldOp.getNumOperands() != iterArgs.size())
                    return failure();

                for (unsigned i = 0; i < yieldOp.getNumOperands(); ++i) {
                    if (yieldOp.getOperand(i).getType() != iterArgs[i].getType()) {
                        needsFix = true;
                        break;
                    }
                }

                if (!needsFix)
                    return failure();

                // 3. 准备修复：插入 Cast
                rewriter.setInsertionPoint(yieldOp); // 确保插入位置正确
                SmallVector<Value> newOperands;

                for (unsigned i = 0; i < yieldOp.getNumOperands(); ++i) {
                    Value val = yieldOp.getOperand(i);
                    Type expectedTy = iterArgs[i].getType(); // Loop 期望的类型 (例如 Level 31)

                    if (val.getType() != expectedTy) {
                        // [关键修复] 创建 Cast: Level 30 -> Level 31
                        // 这只是为了让 MLIR 校验通过，后续 Pass 会处理这个 Cast
                        auto cast = rewriter.create<UnrealizedConversionCastOp>(
                            yieldOp.getLoc(), expectedTy, val);
                        newOperands.push_back(cast.getResult(0));
                    } else {
                        newOperands.push_back(val);
                    }
                }

                // 4. 原地修改 affine.yield
                rewriter.modifyOpInPlace(yieldOp, [&]() {
                    yieldOp->setOperands(newOperands);
                });

                return success();
            }
        };

        struct ConvertVectorWriteToSIMDStorePattern : public OpRewritePattern<vector::TransferWriteOp> {
            using OpRewritePattern::OpRewritePattern;
            LogicalResult matchAndRewrite(vector::TransferWriteOp writeOp, PatternRewriter& rewriter) const override {
                // Operand 0: Vector to write
                // Operand 1: Source/Dest Memref
                Value vec = writeOp.getOperand(0);

                Value destMemRef = writeOp.getOperand(1);

                // Case 1: Operand is a Cast from SIMD (Loop conversion happened)
                if (auto castOp = vec.getDefiningOp<UnrealizedConversionCastOp>()) {
                    if (auto simdTy = dyn_cast<SIMDCipherType>(castOp.getOperand(0).getType())) {
                        rewriter.create<SIMDStoreOp>(writeOp.getLoc(), castOp.getOperand(0), destMemRef);
                        rewriter.eraseOp(writeOp);
                        return success();
                    }
                }

                // Case 2: Operand is already SIMD (rare, verifier usually catches this earlier)
                if (isa<SIMDCipherType>(vec.getType())) {
                    rewriter.create<SIMDStoreOp>(writeOp.getLoc(), vec, destMemRef);
                    rewriter.eraseOp(writeOp);
                    return success();
                }
                return failure();
            }
        };

        // [新增] 自动对齐 Add/Sub 操作数的 Level (插入 Rescale/ModSwitch)
        template <typename OpTy>
        struct AlignSIMDOperandsPattern : OpRewritePattern<OpTy> {
            using OpRewritePattern<OpTy>::OpRewritePattern;

            LogicalResult matchAndRewrite(OpTy op, PatternRewriter& rewriter) const override {
                Value lhs = op.getOperand(0);
                Value rhs = op.getOperand(1);

                auto lhsTy = dyn_cast<SIMDCipherType>(lhs.getType());
                auto rhsTy = dyn_cast<SIMDCipherType>(rhs.getType());

                if (!lhsTy || !rhsTy)
                    return failure();

                int64_t l1 = lhsTy.getLevel();
                int64_t l2 = rhsTy.getLevel();

                if (l1 == l2)
                    return failure(); // 已对齐，无需处理

                int64_t targetLevel = std::min(l1, l2);

                Value newLhs = lhs;
                Value newRhs = rhs;

                // 降级 LHS
                if (l1 > targetLevel) {
                    auto newTy = SIMDCipherType::get(rewriter.getContext(), targetLevel, lhsTy.getPlaintextCount(), lhsTy.getElementType());
                    // 使用 Rescale 或 ModSwitch (视语义而定，Add通常用ModSwitch)
                    // 假设我们用 Rescale 来代表通用的降级
                    newLhs = rewriter.create<SIMDRescaleOp>(op.getLoc(), newTy, lhs).getResult();
                }

                // 降级 RHS
                if (l2 > targetLevel) {
                    auto newTy = SIMDCipherType::get(rewriter.getContext(), targetLevel, rhsTy.getPlaintextCount(), rhsTy.getElementType());
                    newRhs = rewriter.create<SIMDRescaleOp>(op.getLoc(), newTy, rhs).getResult();
                }

                // 创建新 Op (类型与输入一致)
                // 注意：结果类型也必须更新为 targetLevel
                auto resTy = SIMDCipherType::get(rewriter.getContext(), targetLevel, lhsTy.getPlaintextCount(), lhsTy.getElementType());

                rewriter.replaceOpWithNewOp<OpTy>(op, resTy, newLhs, newRhs);
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
                                 ConvertFuncOpTypesToSIMDPattern,
                                 ConvertSCFHEAddToSIMDPattern,
                                 ConvertSCFHEDivToSIMDPattern,
                                 ConvertSCFHEExpToSIMDPattern,
                                 ConvertSCFHESelectToSIMDPattern,
                                 ConvertSCFHECmpToSIMDPattern,
                                 ConvertSCFHEReduceAddToSIMDPattern,
                                 ConvertSCFHEStoreToSIMDPattern,
                                 ConvertAffineForToSIMDPattern,
                                 ConvertVectorWriteToSIMDStorePattern>(ctx);
                    (void)applyPatternsGreedily(module, std::move(patterns));
                }

                // Phase 2
                {
                    RewritePatternSet patterns(ctx);
                    patterns.add<AdjustSIMDMultLevelPattern,
                                 //  AdjustSIMDSubLevelPattern,
                                 AdjustSIMDMinLevelPattern,
                                 //  AdjustSIMDAddLevelPattern,
                                 AlignSIMDOperandsPattern<SIMDAddOp>,
                                 AlignSIMDOperandsPattern<SIMDSubOp>,
                                 FixAffineYieldTypeMismatch>(ctx);
                    (void)applyPatternsGreedily(module, std::move(patterns));
                }
            }
        };

    } // namespace
} // namespace mlir::libra::simd