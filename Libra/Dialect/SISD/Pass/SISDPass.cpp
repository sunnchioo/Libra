#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/Support/Debug.h"

#include "SIMDCommon.h"
#include "SIMDOps.h"
#include "SIMDTypes.h"
#include "SISDDialect.h"
#include "SISDOps.h"
#include "SISDPass.h"

// 触发日志的宏定义，使用时带上 --debug-only=sisd-pass
#define DEBUG_TYPE "sisd-pass"

using namespace mlir;
using namespace mlir::libra;

namespace mlir::libra::sisd {

#define GEN_PASS_DEF_CONVERTTOSISDPASS
#include "SISDPass.h.inc"

    namespace {

        // ============================================================================
        // 1. 类型转换器 (TypeConverter)
        // 负责将所有的 SIMD 类型转换为 SISD 类型，这是消除 cast 的核心！
        // ============================================================================
        class SISDTypeConverter : public TypeConverter {
        public:
            SISDTypeConverter(MLIRContext* ctx) {
                // 默认规则：其他类型保持不变（如 f64, index, i32）
                addConversion([](Type type) { return type; });

                // 核心规则：将 SIMDCipherType 转换为 SISDCipherType
                addConversion([ctx](simd::SIMDCipherType type) -> Type {
                    auto resType = sisd::SISDCipherType::get(ctx, type.getPlaintextCount(), type.getElementType());
                    LLVM_DEBUG(llvm::dbgs() << "[TypeConverter] Converted " << type << " -> " << resType << "\n");
                    return resType;
                });
            }
        };

        // ============================================================================
        // 2. 转换 Patterns
        // ============================================================================

        // --- Convert simd.load -> sisd.load ---
        class ConvertSIMDLoadToSISDPattern : public OpConversionPattern<simd::SIMDLoadOp> {
        public:
            using OpConversionPattern::OpConversionPattern;
            LogicalResult matchAndRewrite(simd::SIMDLoadOp op, OpAdaptor adaptor,
                                          ConversionPatternRewriter& rewriter) const override {
                LLVM_DEBUG(llvm::dbgs() << "\n[SISD Conversion] Inspecting simd.load at " << op.getLoc() << "\n");

                Type resultType = getTypeConverter()->convertType(op.getResult().getType());
                LLVM_DEBUG(llvm::dbgs() << "  -> Converted Result Type: " << resultType << "\n");

                rewriter.replaceOpWithNewOp<sisd::SISDLoadOp>(
                    op, resultType, adaptor.getOperands()[0], adaptor.getOperands()[1]);

                LLVM_DEBUG(llvm::dbgs() << "  -> [Success] Converted to sisd.load\n");
                return success();
            }
        };

        // --- 通用二元算数转换模板 (simd.add -> sisd.add, etc.) ---
        template <typename SourceOp, typename TargetOp>
        struct ConvertSIMDBinaryToSISDPattern : public OpConversionPattern<SourceOp> {
            using OpConversionPattern<SourceOp>::OpConversionPattern;

            LogicalResult matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                                          ConversionPatternRewriter& rewriter) const override {
                LLVM_DEBUG(llvm::dbgs() << "\n[SISD Conversion] Inspecting " << op->getName() << " at " << op.getLoc() << "\n");

                Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
                LLVM_DEBUG(llvm::dbgs() << "  -> LHS Type: " << adaptor.getOperands()[0].getType() << "\n");
                LLVM_DEBUG(llvm::dbgs() << "  -> RHS Type: " << adaptor.getOperands()[1].getType() << "\n");
                LLVM_DEBUG(llvm::dbgs() << "  -> Target Result Type: " << resultType << "\n");

                rewriter.replaceOpWithNewOp<TargetOp>(
                    op, resultType, adaptor.getOperands()[0], adaptor.getOperands()[1]);

                LLVM_DEBUG(llvm::dbgs() << "  -> [Success] Converted to SISD binary operation.\n");
                return success();
            }
        };

        // --- 专属乘法转换规则：拦截非法的密文-密文乘法 ---
        class ConvertSIMDMultToSISDPattern : public OpConversionPattern<simd::SIMDMultOp> {
        public:
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(simd::SIMDMultOp op, OpAdaptor adaptor,
                                          ConversionPatternRewriter& rewriter) const override {
                LLVM_DEBUG(llvm::dbgs() << "\n[SISD Conversion] Inspecting simd.mul at " << op.getLoc() << "\n");

                // 获取转换后的操作数类型
                Type lhsType = adaptor.getOperands()[0].getType();
                Type rhsType = adaptor.getOperands()[1].getType();

                // 检查是否两边都是密文 (SISDCipherType)
                bool isLhsCipher = isa<sisd::SISDCipherType>(lhsType);
                bool isRhsCipher = isa<sisd::SISDCipherType>(rhsType);

                if (isLhsCipher && isRhsCipher) {
                    LLVM_DEBUG(llvm::dbgs() << "  -> [Error] SISD does not support Cipher-Cipher Multiplication!\n");
                    // 发出编译错误，并在对应的源代码位置标红
                    return op.emitError("Illegal Operation: SISD backend does not support Cipher-Cipher Multiplication.");
                }

                // 如果是密文-明文，或明文-密文，则正常转换
                Type resultType = getTypeConverter()->convertType(op.getResult().getType());
                rewriter.replaceOpWithNewOp<sisd::SISDMultOp>(
                    op, resultType, adaptor.getOperands()[0], adaptor.getOperands()[1]);

                LLVM_DEBUG(llvm::dbgs() << "  -> [Success] Converted to sisd.mul (Cipher-Plain computation).\n");
                return success();
            }
        };

        // --- Convert simd.min -> sisd.min ---
        class ConvertSIMDMinToSISDPattern : public OpConversionPattern<simd::SIMDMinOp> {
        public:
            using OpConversionPattern::OpConversionPattern;
            LogicalResult matchAndRewrite(simd::SIMDMinOp op, OpAdaptor adaptor,
                                          ConversionPatternRewriter& rewriter) const override {
                LLVM_DEBUG(llvm::dbgs() << "\n[SISD Conversion] Inspecting simd.min at " << op.getLoc() << "\n");

                Type resultType = getTypeConverter()->convertType(op.getResult().getType());

                // 注意：这里删除了原来强行穿回去的 castBackToSIMD 马甲！保持纯粹的 SISD 输出！
                rewriter.replaceOpWithNewOp<sisd::SISDMinOp>(
                    op, resultType, adaptor.getOperands()[0]);

                LLVM_DEBUG(llvm::dbgs() << "  -> [Success] Converted to sisd.min.\n");
                return success();
            }
        };

        // --- Convert simd.reduce_add -> sisd.reduce_add ---
        class ConvertSIMDReduceAddToSISDPattern : public OpConversionPattern<simd::SIMDReduceAddOp> {
        public:
            using OpConversionPattern::OpConversionPattern;
            LogicalResult matchAndRewrite(simd::SIMDReduceAddOp op, OpAdaptor adaptor,
                                          ConversionPatternRewriter& rewriter) const override {
                LLVM_DEBUG(llvm::dbgs() << "\n[SISD Conversion] Inspecting simd.reduce_add at " << op.getLoc() << "\n");

                // 获取转换后的结果类型 (SISD Cipher)
                Type resultType = getTypeConverter()->convertType(op.getResult().getType());

                // 创建 sisd.reduce_add，adaptor.getOperands()[0] 是已转换为 SISD 类型的输入
                rewriter.replaceOpWithNewOp<sisd::SISDReduceAddOp>(
                    op, resultType, adaptor.getOperands()[0]);

                LLVM_DEBUG(llvm::dbgs() << "  -> [Success] Converted to sisd.reduce_add.\n");
                return success();
            }
        };

        // ============================================================================
        // 手动处理 scf.for 和 scf.yield 的类型转换 (完美避开 MLIR 版本 API 变动)
        // ============================================================================

        struct SCFForOpConversion : public OpConversionPattern<scf::ForOp> {
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(scf::ForOp op, OpAdaptor adaptor,
                                          ConversionPatternRewriter& rewriter) const override {
                // 1. 获取转换后的 iter_args (此时它们已经是 SISD 类型了)
                SmallVector<Value> newInitArgs = adaptor.getInitArgs();

                // 2. 创建一个全新的 scf.for，并塞入新的 SISD iter_args
                auto newOp = rewriter.create<scf::ForOp>(op.getLoc(),
                                                         adaptor.getLowerBound(),
                                                         adaptor.getUpperBound(),
                                                         adaptor.getStep(),
                                                         newInitArgs);

                // 3. 把老循环体里的代码原封不动搬过来
                rewriter.inlineRegionBefore(op.getRegion(), newOp.getRegion(), newOp.getRegion().end());

                // 4. 让 TypeConverter 自动去修改循环内部 Block 参数的类型
                if (failed(rewriter.convertRegionTypes(&newOp.getRegion(), *getTypeConverter()))) {
                    return failure();
                }

                // 5. 替换掉旧的循环
                rewriter.replaceOp(op, newOp.getResults());
                return success();
            }
        };

        struct SCFYieldOpConversion : public OpConversionPattern<scf::YieldOp> {
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(scf::YieldOp op, OpAdaptor adaptor,
                                          ConversionPatternRewriter& rewriter) const override {
                // 把 scf.yield 原本的 SIMD 参数，替换为转换后的 SISD 参数
                rewriter.replaceOpWithNewOp<scf::YieldOp>(op, adaptor.getResults());
                return success();
            }
        };

        // ============================================================================
        // 3. 主 Pass 入口
        // ============================================================================
        class ConvertToSISDIR : public impl::ConvertToSISDPassBase<ConvertToSISDIR> {
        public:
            using impl::ConvertToSISDPassBase<ConvertToSISDIR>::ConvertToSISDPassBase;

            void runOnOperation() final {
                auto module = getOperation();
                MLIRContext* context = &getContext();

                LLVM_DEBUG(llvm::dbgs() << "====== Starting ConvertToSISD Pass ======\n");

                // 1. 初始化类型转换器
                SISDTypeConverter typeConverter(context);

                // 2. 注册转换模式
                RewritePatternSet patterns(context);
                patterns.add<ConvertSIMDLoadToSISDPattern>(typeConverter, context);
                patterns.add<ConvertSIMDMinToSISDPattern>(typeConverter, context);

                patterns.add<ConvertSIMDReduceAddToSISDPattern>(typeConverter, context);

                // 在原本添加 add, sub 等 pattern 的地方，加上这两个：
                patterns.add<SCFForOpConversion>(typeConverter, context);
                patterns.add<SCFYieldOpConversion>(typeConverter, context);

                // 注册核心算数指令的降级
                patterns.add<ConvertSIMDBinaryToSISDPattern<simd::SIMDAddOp, sisd::SISDAddOp>>(typeConverter, context);
                patterns.add<ConvertSIMDBinaryToSISDPattern<simd::SIMDSubOp, sisd::SISDSubOp>>(typeConverter, context);
                patterns.add<ConvertSIMDBinaryToSISDPattern<simd::SIMDDivOp, sisd::SISDDivOp>>(typeConverter, context);

                patterns.add<ConvertSIMDMultToSISDPattern>(typeConverter, context);

                // 4. 定义转换目标 (Conversion Target)
                ConversionTarget target(*context);

                // 我们认为这些基础方言是合法的
                target.addLegalDialect<sisd::SISDDialect, arith::ArithDialect, scf::SCFDialect, affine::AffineDialect>();

                target.addDynamicallyLegalOp<scf::ForOp>([&](scf::ForOp op) {
                    // 如果结果类型和循环体里的参数类型都被成功转换了，它才是合法的
                    return typeConverter.isLegal(op.getResultTypes()) &&
                           typeConverter.isLegal(&op.getRegion());
                });

                target.addDynamicallyLegalOp<scf::YieldOp>([&](scf::YieldOp op) {
                    // yield 的操作数必须合法
                    return typeConverter.isLegal(op.getOperandTypes());
                });

                // 将 SIMD 方言整体标为非法！强制框架转换掉它
                target.addIllegalDialect<simd::SIMDDialect>();

                // 对函数的签名也实施合法化检查
                target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
                    return typeConverter.isSignatureLegal(op.getFunctionType()) &&
                           typeConverter.isLegal(&op.getBody());
                });

                // 5. 执行 Partial Conversion
                LLVM_DEBUG(llvm::dbgs() << "\n[Execution] Running Partial Conversion to eradicate SIMD operations...\n");
                if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
                    LLVM_DEBUG(llvm::dbgs() << "\n!!! [Failed] SISD Conversion aborted due to illegal operations remaining. !!!\n");
                    signalPassFailure();
                    return;
                }

                LLVM_DEBUG(llvm::dbgs() << "\n====== ConvertToSISD Pass Finished Successfully ======\n");
            }
        };

    } // namespace
} // namespace mlir::libra::sisd