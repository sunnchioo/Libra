#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

#include "mlir/IR/PatternMatch.h"

#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h" // 必须引入
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#include "SCFHEOps.h"
#include "SCFHETypes.h"
#include "SIMDCommon.h"
#include "SIMDOps.h"
#include "SIMDPass.h"

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

        // 辅助函数：穿透框架自动插入的类型强转 Cast，获取带有真实 Level 的底层密文
        static Value peelCast(Value v) {
            if (auto castOp = v.getDefiningOp<UnrealizedConversionCastOp>()) {
                if (castOp.getInputs().size() == 1 && isa<SIMDCipherType>(castOp.getInputs()[0].getType())) {
                    return castOp.getInputs()[0];
                }
            }
            return v;
        }

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

                auto lhs = peelCast(adaptor.getOperands()[0]);
                auto rhs = peelCast(adaptor.getOperands()[1]);

                // === 辅助 lambda：自动加密 ===
                auto ensureCipher = [&](Value val, Value other) -> Value {
                    if (isa<FloatType>(val.getType()) || isa<IntegerType>(val.getType())) {
                        int64_t targetLevel = 0;
                        if (auto otherTy = dyn_cast<SIMDCipherType>(other.getType())) {
                            targetLevel = otherTy.getLevel();
                        }
                        Type elemType = val.getType();
                        if (auto castTy = dyn_cast<SIMDCipherType>(other.getType()))
                            elemType = castTy.getElementType();

                        auto constCipherTy = SIMDCipherType::get(ctx, targetLevel, 1, elemType);
                        LLVM_DEBUG(llvm::dbgs() << "  -> Auto-encrypting plaintext operand to: " << constCipherTy << "\n");
                        return rewriter.create<SIMDEncryptOp>(op.getLoc(), constCipherTy, val).getResult();
                    }
                    return val;
                };

                // === [修改点 1]：有条件地处理 RHS ===
                // LHS 总是尝试转为密文 (Cipher / Plain 是合法的，但 Plain / Cipher 通常需提升)
                lhs = ensureCipher(lhs, rhs);

                // 检查是否为 Div 且 RHS 为明文
                bool isDivOptimized = std::is_same_v<TargetOp, SIMDDivOp> &&
                                      (isa<FloatType>(rhs.getType()) || isa<IntegerType>(rhs.getType()));

                // 如果不是 Div 优化场景，才强制加密 RHS
                if (!isDivOptimized) {
                    rhs = ensureCipher(rhs, lhs);
                } else {
                    LLVM_DEBUG(llvm::dbgs() << "  -> Optimization: Keeping SIMDDivOp divisor as plaintext.\n");
                }

                // === [修改点 2]：类型检查与结果推导 ===
                auto ta = dyn_cast<SIMDCipherType>(lhs.getType());
                auto tb = dyn_cast<SIMDCipherType>(rhs.getType()); // tb 可能为空(如果是明文)

                // 1. LHS 必须是密文
                if (!ta) {
                    LLVM_DEBUG(llvm::dbgs() << "  -> Failed: LHS is not a cipher.\n");
                    return failure();
                }

                // 2. 如果不是优化场景，RHS 也必须是密文
                if (!tb && !isDivOptimized) {
                    LLVM_DEBUG(llvm::dbgs() << "  -> Failed: RHS is not a cipher and not optimized Div.\n");
                    return failure();
                }

                // 3. 计算 Level 和 Count
                int64_t currentLevel = ta.getLevel();
                int64_t currentCount = ta.getPlaintextCount();

                // 如果 RHS 也是密文，取两者的最小值/对齐值
                if (tb) {
                    currentLevel = std::min(ta.getLevel(), tb.getLevel());
                    currentCount = std::min(ta.getPlaintextCount(), tb.getPlaintextCount());
                }

                // 4. 应用 Level 消耗规则 (Mult/Div 消耗一层)
                int64_t newLevel = currentLevel;
                if (std::is_same_v<SourceOp, scfhe::SCFHEMultOp> || std::is_same_v<SourceOp, scfhe::SCFHEDivOp>)
                    newLevel = std::max<int64_t>(0, currentLevel - 1);

                auto resTy = SIMDCipherType::get(ctx, newLevel, currentCount, ta.getElementType());

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

                // 【核心修改】：穿透强制转换，拿到真实的 Level 30 密文
                Value cipher = peelCast(adaptor.getCipher());

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

        // 1. Cast 转换 (修复你现在的报错)
        struct ConvertSCFHECastToSIMDPattern : public OpConversionPattern<scfhe::SCFHECastOp> {
            using OpConversionPattern::OpConversionPattern;
            LogicalResult matchAndRewrite(scfhe::SCFHECastOp op, OpAdaptor adaptor,
                                          ConversionPatternRewriter& rewriter) const override {
                Type resTy = getTypeConverter()->convertType(op.getResult().getType());
                // 如果你的 SIMD 方案里有 simd.cast 则用 simd::CastOp，否则用内置 Cast 暂时代替
                rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(op, resTy, adaptor.getOperands()[0]);
                return success();
            }
        };

        // 2. Func Call 转换 (因为函数签名变了，调用处也得变)
        struct ConvertCallOpPattern : public OpConversionPattern<func::CallOp> {
            using OpConversionPattern::OpConversionPattern;
            LogicalResult matchAndRewrite(func::CallOp op, OpAdaptor adaptor,
                                          ConversionPatternRewriter& rewriter) const override {
                SmallVector<Type> newResultTypes;
                if (failed(getTypeConverter()->convertTypes(op.getResultTypes(), newResultTypes)))
                    return failure();
                rewriter.replaceOpWithNewOp<func::CallOp>(op, op.getCallee(), newResultTypes, adaptor.getOperands());
                return success();
            }
        };

        // 3. SCF 循环转换 (因为前一个 Pass 生成了带密文的 scf.for)
        struct ConvertSCFForToSIMDPattern : public OpConversionPattern<scf::ForOp> {
            using OpConversionPattern::OpConversionPattern;
            LogicalResult matchAndRewrite(scf::ForOp op, OpAdaptor adaptor,
                                          ConversionPatternRewriter& rewriter) const override {
                // 1. 创建新的 scf.for。
                // 注意：Builder 会自动创建一个全新的 Block，且该 Block 的参数已经被自动替换为 SIMD 类型！
                auto newForOp = rewriter.create<scf::ForOp>(
                    op.getLoc(), adaptor.getLowerBound(), adaptor.getUpperBound(),
                    adaptor.getStep(), adaptor.getInitArgs());

                // 2. 将旧 Block 中的所有指令“内联（转移）”到新 Block 的末尾。
                // 这个神仙函数不仅会转移指令，还会自动把旧指令中用到的旧参数（如 f64），
                // 自动重定向绑定到新 Block 的参数（即 SIMD Cipher）上！
                rewriter.inlineBlockBefore(op.getBody(), newForOp.getBody(), newForOp.getBody()->end(),
                                           newForOp.getBody()->getArguments());

                // 3. 替换掉外层的旧 scf.for
                rewriter.replaceOp(op, newForOp.getResults());
                return success();
            }
        };

        // 4. SCF Yield 转换
        struct ConvertSCFYieldToSIMDPattern : public OpConversionPattern<scf::YieldOp> {
            using OpConversionPattern::OpConversionPattern;
            LogicalResult matchAndRewrite(scf::YieldOp op, OpAdaptor adaptor,
                                          ConversionPatternRewriter& rewriter) const override {
                rewriter.replaceOpWithNewOp<scf::YieldOp>(op, adaptor.getOperands());
                return success();
            }
        };

        // --- 将 scfhe.load 转换为 simd.load ---
        struct ConvertSCFHELoadToSIMDPattern : public OpConversionPattern<scfhe::SCFHELoadOp> {
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(scfhe::SCFHELoadOp op, OpAdaptor adaptor,
                                          ConversionPatternRewriter& rewriter) const override {
                // 1. 获取目标 SIMD 类型 (例如从 !scfhe.cipher<1xi64> 转为 !simd.cipher<31x1xi64>)
                Type resultType = getTypeConverter()->convertType(op.getResult().getType());

                // 2. 用新方言的 SIMDLoadOp 替换
                // adaptor.getOperands()[0] 是已经转换为 SIMD 类型的数组
                // adaptor.getOperands()[1] 是未发生改变的 Index 下标
                rewriter.replaceOpWithNewOp<SIMDLoadOp>(
                    op, resultType, adaptor.getOperands()[0], adaptor.getOperands()[1]);

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
                target.addLegalDialect<mlir::libra::simd::SIMDDialect, arith::ArithDialect,
                                       affine::AffineDialect, memref::MemRefDialect,
                                       LLVM::LLVMDialect, scf::SCFDialect>();

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

                target.addDynamicallyLegalOp<scf::ForOp>([&](scf::ForOp op) {
                    return typeConverter.isLegal(op.getResultTypes()) &&
                           typeConverter.isLegal(op.getRegionIterArgs());
                });

                target.addDynamicallyLegalOp<scf::YieldOp>([&](scf::YieldOp op) {
                    return typeConverter.isLegal(op.getOperandTypes());
                });

                RewritePatternSet patterns(ctx);

                populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns, typeConverter);

                patterns.add<
                    ConvertReturnOpPattern,
                    ConvertCallOpPattern,
                    ConvertSCFHECastToSIMDPattern,
                    ConvertSCFForToSIMDPattern,
                    ConvertSCFYieldToSIMDPattern,
                    ConvertSCFHEEncryptToSIMDPattern,
                    ConvertSCFHEDecryptToSIMDPattern,
                    ConvertAffineForToSIMDPattern,
                    ConvertSCFHELoadToSIMDPattern,
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