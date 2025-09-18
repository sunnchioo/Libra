#include "FlyHEDialect.h"
#include "FlyHEOps.h"
#include "FlyHETypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace flyhe;

// 将memref类型转换为FlyHE_SIMDCipher类型
struct FlyHETypeConverter : public TypeConverter {
    FlyHETypeConverter() {
        // 转换memref类型到FlyHE_SIMDCipher
        addConversion([](MemRefType memrefType) -> Type {
            // 假设FlyHE_SIMDCipher需要原始元素类型作为参数
            return FlyHE_SIMDCipherType::get(memrefType.getContext(),
                                             memrefType.getElementType());
        });

        // 保持其他类型不变
        addConversion([](Type type) { return type; });
    }
};

// struct ConvertAffineLoad : public OpRewritePattern<AffineLoadOp> {
//     using OpRewritePattern::OpRewritePattern;

//     LogicalResult matchAndRewrite(AffineLoadOp op,
//                                   PatternRewriter &rewriter) const override {
//         // 创建新的FlyHE_SIMDCipher加载操作
//         auto newOp = rewriter.create<PhantomLoadOp>(
//             op.getLoc(),
//             op.getType(),
//             op.getMemRef(),
//             op.getIndices());
//         rewriter.replaceOp(op, newOp.getResult());
//         return success();
//     }
// };

// 模式重写：处理函数参数
struct ConvertFuncOp : public OpRewritePattern<FuncOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(FuncOp op,
                                  PatternRewriter &rewriter) const override {
        FlyHETypeConverter typeConverter;
        auto funcType = op.getFunctionType();

        // 转换函数参数类型
        SmallVector<Type, 4> newInputs;
        if (!typeConverter.convertTypes(funcType.getInputs(), newInputs))
            return failure();

        // 转换返回值类型
        SmallVector<Type, 4> newResults;
        if (!typeConverter.convertTypes(funcType.getResults(), newResults))
            return failure();

        // 创建新的函数类型
        auto newFuncType = FunctionType::get(op.getContext(), newInputs, newResults);

        // 创建新函数
        auto newFunc = rewriter.create<FuncOp>(op.getLoc(), op.getName(), newFuncType, op.getAttrs());

        // 复制函数体
        rewriter.inlineRegionBefore(op.getBody(), newFunc.getBody(), newFunc.getBody().end());

        // 更新函数参数名称
        for (auto [oldArg, newArg] : zip(op.getArguments(), newFunc.getArguments()))
            newArg.setName(oldArg.getName());

        rewriter.eraseOp(op);
        return success();
    }
};

// 配置转换目标
struct FlyHEConversionTarget : public ConversionTarget {
    FlyHEConversionTarget(MLIRContext &ctx) : ConversionTarget(ctx) {
        // 标记需要转换的操作
        addDynamicallyLegalOp<AffineLoadOp>([](AffineLoadOp op) {
            return !op.getMemRef().getType().isa<MemRefType>();
        });

        addDynamicallyLegalOp<FuncOp>([&](FuncOp op) {
            FlyHETypeConverter converter;
            return converter.isSignatureLegal(op.getFunctionType());
        });

        // 其他操作保持合法
        markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    }
};

// Pass实现
struct ConvertToFlyHEPass
    : public PassWrapper<ConvertToFlyHEPass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertToFlyHEPass)

    void runOnOperation() override {
        ModuleOp module = getOperation();
        MLIRContext *ctx = &getContext();

        FlyHETypeConverter typeConverter;
        FlyHEConversionTarget target(*ctx);

        // 配置重写模式
        RewritePatternSet patterns(ctx);
        patterns.add<ConvertAffineLoad, ConvertFuncOp>(ctx);

        // 执行转换
        if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
            signalPassFailure();
        }
    }
};

// 注册Pass
namespace mlir {
    namespace flyhe {
        void registerConvertToFlyHEPass() {
            PassRegistration<ConvertToFlyHEPass>(
                "convert-to-flyhe",
                "Convert standard MLIR types to FlyHE types");
        }
    } // namespace flyhe
} // namespace mlir
