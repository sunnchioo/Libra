#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h" // 必须引入
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/Debug.h"

#include "SCFHEOps.h"
#include "SCFHETypes.h"
#include "SIMDOps.h"
#include "SIMDPass.h"
#include "SIMDCommon.h"

#define DEBUG_TYPE "convert-to-simd"

namespace mlir::libra::simd {
#define GEN_PASS_DEF_CONVERTTOSIMDPASS
#include "SIMDPass.h.inc"

    namespace {

        //===----------------------------------------------------------------------===//
        // 1. Type Converter (核心：定义类型映射关系)
        //===----------------------------------------------------------------------===//
        struct SIMDTypeConverter : public TypeConverter {
            SIMDTypeConverter(MLIRContext* ctx) {
                // 默认规则：保留所有原生类型（如 f64, i32, index）
                addConversion([](Type type) { return type; });

                // 核心规则：scfhe.cipher -> simd.cipher
                addConversion([ctx](scfhe::SCFHECipherType t) -> Type {
                    return SIMDCipherType::get(ctx, DEFAULT_LEVEL, t.getPlaintextCount(), t.getElementType());
                });

                // 递归转换 MemRef 内部的元素类型
                addConversion([this](MemRefType type) -> Type {
                    Type convertedElement = convertType(type.getElementType());
                    return MemRefType::get(type.getShape(), convertedElement, type.getLayout(), type.getMemorySpace());
                });

                // 物化规则：当框架遇到无法立解除的类型冲突时，自动插入 Cast
                addTargetMaterialization([](OpBuilder& builder, Type resultType, ValueRange inputs, Location loc) -> Value {
                    return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs).getResult(0);
                });
                addSourceMaterialization([](OpBuilder& builder, Type resultType, ValueRange inputs, Location loc) -> Value {
                    return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs).getResult(0);
                });
            }
        };

        //===----------------------------------------------------------------------===//
        // 2. Conversion Patterns (将 OpRewritePattern 改为 OpConversionPattern)
        //===----------------------------------------------------------------------===//

        // 通用算术转换模板
        template <typename SourceOp, typename TargetOp>
        struct SIMDArithmeticConversion : public OpConversionPattern<SourceOp> {
            using OpConversionPattern<SourceOp>::OpConversionPattern;
            using Adaptor = typename SourceOp::Adaptor;

            LogicalResult matchAndRewrite(SourceOp op, Adaptor adaptor,
                                          ConversionPatternRewriter& rewriter) const override {
                LLVM_DEBUG(llvm::dbgs() << "Converting Arithmetic Op: " << op->getName() << "\n");

                auto ctx = rewriter.getContext();
                // 注意：在 ConversionPattern 中，adaptor.getOperands() 拿到的已经是转换后的 SIMD 类型 Value
                auto lhs = adaptor.getOperands()[0];
                auto rhs = adaptor.getOperands()[1];

                // 这里的推断逻辑保持不变，但输入类型已经是 SIMD 类型了
                auto ta = dyn_cast<SIMDCipherType>(lhs.getType());
                auto tb = dyn_cast<SIMDCipherType>(rhs.getType());
                if (!ta || !tb)
                    return failure();

                int64_t newLevel = std::min(ta.getLevel(), tb.getLevel());
                if (std::is_same_v<SourceOp, scfhe::SCFHEMultOp> || std::is_same_v<SourceOp, scfhe::SCFHEDivOp>)
                    newLevel = std::max<int64_t>(0, newLevel - 1);

                auto resTy = SIMDCipherType::get(ctx, newLevel, std::min(ta.getPlaintextCount(), tb.getPlaintextCount()), ta.getElementType());

                rewriter.replaceOpWithNewOp<TargetOp>(op, resTy, lhs, rhs);
                return success();
            }
        };

        // 专门用于转换 func.return 的操作数类型
        struct ConvertReturnOpPattern : public OpConversionPattern<func::ReturnOp> {
            using OpConversionPattern::OpConversionPattern;
            LogicalResult matchAndRewrite(func::ReturnOp op, OpAdaptor adaptor,
                                          ConversionPatternRewriter& rewriter) const override {
                // adaptor.getOperands() 里装的已经是转换成 SIMD 类型的值了
                rewriter.replaceOpWithNewOp<func::ReturnOp>(op, adaptor.getOperands());
                return success();
            }
        };

        // Encrypt 转换
        struct ConvertSCFHEEncryptToSIMDPattern : public OpConversionPattern<scfhe::SCFHEEncryptOp> {
            using OpConversionPattern::OpConversionPattern;
            LogicalResult matchAndRewrite(scfhe::SCFHEEncryptOp op, OpAdaptor adaptor,
                                          ConversionPatternRewriter& rewriter) const override {
                LLVM_DEBUG(llvm::dbgs() << "Converting Encrypt: " << op.getLoc() << "\n");
                Type resTy = getTypeConverter()->convertType(op.getResult().getType());
                Value plaintext = adaptor.getInput();
                rewriter.replaceOpWithNewOp<SIMDEncryptOp>(op, resTy, plaintext);
                return success();
            }
        };

        // Decrypt 转换
        struct ConvertSCFHEDecryptToSIMDPattern : public OpConversionPattern<scfhe::SCFHEDecryptOp> {
            using OpConversionPattern::OpConversionPattern;
            LogicalResult matchAndRewrite(scfhe::SCFHEDecryptOp op, OpAdaptor adaptor,
                                          ConversionPatternRewriter& rewriter) const override {
                Value cipher = adaptor.getCipher();
                rewriter.replaceOpWithNewOp<SIMDDecryptOp>(op, op.getResult().getType(), cipher);
                return success();
            }
        };

        // AffineFor 循环转换
        struct ConvertAffineForToSIMDPattern : public OpConversionPattern<affine::AffineForOp> {
            using OpConversionPattern::OpConversionPattern;
            LogicalResult matchAndRewrite(affine::AffineForOp op, OpAdaptor adaptor,
                                          ConversionPatternRewriter& rewriter) const override {
                LLVM_DEBUG(llvm::dbgs() << "Converting AffineFor Loop...\n");

                SmallVector<Value> inits = op.getInits();

                auto newForOp = rewriter.create<affine::AffineForOp>(
                    op.getLoc(), op.getLowerBoundOperands(), op.getLowerBoundMap(),
                    op.getUpperBoundOperands(), op.getUpperBoundMap(),
                    op.getStep().getSExtValue(), inits);

                // 迁移旧 Body 到新 Body
                rewriter.inlineRegionBefore(op.getRegion(), newForOp.getRegion(), newForOp.getRegion().end());

                // 关键：转换 Block 参数类型
                if (failed(rewriter.convertRegionTypes(&newForOp.getRegion(), *getTypeConverter())))
                    return failure();

                rewriter.replaceOp(op, newForOp.getResults());
                return success();
            }
        };
        //===----------------------------------------------------------------------===//
        // 3. Phase 2 Patterns (Level 对齐，依然使用 Greedy)
        //===----------------------------------------------------------------------===//

        template <typename OpTy>
        struct AlignSIMDOperandsPattern : OpRewritePattern<OpTy> {
            using OpRewritePattern<OpTy>::OpRewritePattern;
            LogicalResult matchAndRewrite(OpTy op, PatternRewriter& rewriter) const override {
                Value lhs = op.getOperand(0);
                Value rhs = op.getOperand(1);
                auto lTy = dyn_cast<SIMDCipherType>(lhs.getType());
                auto rTy = dyn_cast<SIMDCipherType>(rhs.getType());

                if (!lTy || !rTy || lTy.getLevel() == rTy.getLevel())
                    return failure();

                LLVM_DEBUG(llvm::dbgs() << "Aligning levels for " << op->getName() << "\n");
                int64_t target = std::min(lTy.getLevel(), rTy.getLevel());
                auto newTy = SIMDCipherType::get(rewriter.getContext(), target, lTy.getPlaintextCount(), lTy.getElementType());

                Value newL = (lTy.getLevel() > target) ? rewriter.create<SIMDRescaleOp>(op.getLoc(), newTy, lhs) : lhs;
                Value newR = (rTy.getLevel() > target) ? rewriter.create<SIMDRescaleOp>(op.getLoc(), newTy, rhs) : rhs;

                rewriter.replaceOpWithNewOp<OpTy>(op, newTy, newL, newR);
                return success();
            }
        };

        //===----------------------------------------------------------------------===//
        // 4. Pass Implementation
        //===----------------------------------------------------------------------===//

        struct ConvertSCFHEToSIMDPass : public impl::ConvertToSIMDPassBase<ConvertSCFHEToSIMDPass> {
            void runOnOperation() override {
                auto module = getOperation();
                auto* ctx = &getContext();

                LLVM_DEBUG(llvm::dbgs() << "--- Starting SCFHE to SIMD Conversion ---\n");

                // --- Phase 1: 结构化转换 ---
                SIMDTypeConverter typeConverter(ctx);
                ConversionTarget target(*ctx);

                // 定义哪些方言是合法的，哪些必须转换
                // 【补充了 LLVM::LLVMDialect，因为你的输入 IR 里面有 llvm.call 和 llvm.ptr】
                target.addLegalDialect<mlir::libra::simd::SIMDDialect, arith::ArithDialect,
                                       affine::AffineDialect, memref::MemRefDialect,
                                       LLVM::LLVMDialect>();
                target.addIllegalDialect<scfhe::SCFHEDialect>();

                // =========================================================
                // 【核心修复：补充基础 Op 和 Cast 的合法性】
                // 必须允许临时 Cast 存在，否则 TypeConverter 无法在类型不匹配时插桩
                target.addLegalOp<ModuleOp, mlir::UnrealizedConversionCastOp>();

                // 【核心修复：配置 func::ReturnOp 和 func::CallOp 的动态合法性】
                target.addDynamicallyLegalOp<func::ReturnOp>([&](func::ReturnOp op) {
                    return typeConverter.isLegal(op.getOperandTypes());
                });
                target.addDynamicallyLegalOp<func::CallOp>([&](func::CallOp op) {
                    return typeConverter.isSignatureLegal(op.getCalleeType());
                });
                // =========================================================

                // 函数签名必须转换完成后才合法
                target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
                    return typeConverter.isSignatureLegal(op.getFunctionType());
                });

                target.addDynamicallyLegalOp<affine::AffineForOp>([&](affine::AffineForOp op) {
                    SmallVector<Type> argTypes;
                    for (auto arg : op.getRegionIterArgs()) {
                        argTypes.push_back(arg.getType());
                    }
                    return typeConverter.isLegal(argTypes);
                });

                RewritePatternSet patterns(ctx);

                populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns, typeConverter);

                patterns.add<
                    ConvertReturnOpPattern, // <--- 现在它会被正确触发了
                    ConvertSCFHEEncryptToSIMDPattern,
                    ConvertSCFHEDecryptToSIMDPattern,
                    ConvertAffineForToSIMDPattern,
                    SIMDArithmeticConversion<scfhe::SCFHEAddOp, SIMDAddOp>,
                    SIMDArithmeticConversion<scfhe::SCFHESubOp, SIMDSubOp>,
                    SIMDArithmeticConversion<scfhe::SCFHEMultOp, SIMDMultOp>,
                    SIMDArithmeticConversion<scfhe::SCFHEDivOp, SIMDDivOp>>(typeConverter, ctx);

                if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
                    llvm::errs() << "Error: Partial conversion failed.\n";
                    signalPassFailure();
                    return;
                }

                LLVM_DEBUG(llvm::dbgs() << "--- Phase 1 Finished. Starting Level Alignment ---\n");

                // --- Phase 2: Level 对齐和清理 ---
                RewritePatternSet levelPatterns(ctx);
                levelPatterns.add<AlignSIMDOperandsPattern<SIMDAddOp>,
                                  AlignSIMDOperandsPattern<SIMDSubOp>>(ctx);

                if (failed(applyPatternsGreedily(module, std::move(levelPatterns)))) {
                    signalPassFailure();
                }

                LLVM_DEBUG(llvm::dbgs() << "--- SIMD Conversion Complete ---\n");
            }
        };

    } // namespace
} // namespace mlir::libra::simd