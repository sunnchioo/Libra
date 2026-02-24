#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Math/IR/Math.h" // 必须包含 Math 方言处理 Exp
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Debug.h"

#include "SCFHEDialect.h"
#include "SCFHEOps.h"
#include "SCFHETypes.h"
#include "SCFHEPass.h"

// 定义 Debug 标签，运行时使用 -debug-only=scfhe-pass 开启
#define DEBUG_TYPE "scfhe-pass"

using namespace mlir;
using namespace mlir::libra::scfhe;

namespace mlir::libra::scfhe {

#define GEN_PASS_DEF_CONVERTTOSCFHEPASS
#include "SCFHEPass.h.inc"

    namespace {

        // ============================================================================
        // 1. Analysis: 参数角色分析
        // ============================================================================
        struct FunctionArgInfo {
            llvm::BitVector inputArgs;
            llvm::BitVector outputArgs;
        };

        class ArgAnalysis {
        public:
            DenseMap<func::FuncOp, FunctionArgInfo> infoMap;

            void run(ModuleOp module) {
                LLVM_DEBUG(llvm::dbgs() << "\n=== [Analysis] Starting Parameter Analysis ===\n");

                module.walk([&](func::FuncOp func) {
                    if (!func->hasAttr("scfhe.crypto"))
                        return;

                    LLVM_DEBUG(llvm::dbgs() << "Analyzing Function: @" << func.getName() << "\n");

                    FunctionArgInfo info;
                    unsigned numArgs = func.getNumArguments();
                    info.inputArgs.resize(numArgs);
                    info.outputArgs.resize(numArgs);

                    for (unsigned i = 0; i < numArgs; ++i) {
                        BlockArgument arg = func.getArgument(i);

                        LLVM_DEBUG(llvm::dbgs() << "  Arg[" << i << "] Type: " << arg.getType());

                        if (!isa<MemRefType>(arg.getType())) {
                            LLVM_DEBUG(llvm::dbgs() << " -> [Ignored] (Not MemRef)\n");
                            continue;
                        }

                        bool isWritten = false;
                        for (Operation* user : arg.getUsers()) {
                            if (auto storeOp = dyn_cast<affine::AffineStoreOp>(user)) {
                                if (storeOp.getMemRef() == arg) {
                                    isWritten = true;
                                    break;
                                }
                            }
                        }

                        if (isWritten) {
                            info.outputArgs.set(i);
                            LLVM_DEBUG(llvm::dbgs() << " -> [Output] (Will be Decrypted)\n");
                        } else {
                            info.inputArgs.set(i);
                            LLVM_DEBUG(llvm::dbgs() << " -> [Input] (Will be Encrypted)\n");
                        }
                    }
                    infoMap[func] = info;
                });
                LLVM_DEBUG(llvm::dbgs() << "=== [Analysis] Finished ===\n\n");
            }
        };

        // ============================================================================
        // 2. TypeConverter
        // ============================================================================
        class SCFHETypeConverter : public TypeConverter {
        public:
            SCFHETypeConverter(MLIRContext* ctx) {
                addConversion([](Type type) { return type; });

                // 处理 MemRef -> Cipher (保持你之前的逻辑)
                addConversion([ctx](MemRefType type) -> std::optional<Type> {
                    Type cipherElemType = IntegerType::get(ctx, 64);
                    int64_t packedSize = type.hasStaticShape() ? type.getNumElements() : ShapedType::kDynamic;
                    return SCFHECipherType::get(ctx, packedSize, cipherElemType);
                });

                // 【新增】处理标量 f64 -> Cipher
                // 当 addf 操作的结果类型是 f64 时，我们需要将其转换为加密类型
                addConversion([ctx](Float64Type type) -> std::optional<Type> {
                    Type cipherElemType = IntegerType::get(ctx, 64);
                    // 标量通常对应大小为 1 的 Cipher，或者使用 kDynamic 适配所有情况
                    return SCFHECipherType::get(ctx, ShapedType::kDynamic, cipherElemType);
                });

                addSourceMaterialization([](OpBuilder& b, Type t, ValueRange i, Location l) -> Value {
                    return b.create<UnrealizedConversionCastOp>(l, t, i).getResult(0);
                });
                addTargetMaterialization([](OpBuilder& b, Type t, ValueRange i, Location l) -> Value {
                    return b.create<UnrealizedConversionCastOp>(l, t, i).getResult(0);
                });
            }
        };

        // ============================================================================
        // 3. Patterns: 转换模式
        // ============================================================================

        // --- 函数签名转换 ---
        struct FuncSignaturePattern : public OpConversionPattern<func::FuncOp> {
            const ArgAnalysis& analysis;
            FuncSignaturePattern(SCFHETypeConverter& converter, MLIRContext* ctx, const ArgAnalysis& analysis)
                : OpConversionPattern(converter, ctx), analysis(analysis) {}

            LogicalResult matchAndRewrite(func::FuncOp funcOp, OpAdaptor adaptor,
                                          ConversionPatternRewriter& rewriter) const override {
                if (!funcOp->hasAttr("scfhe.crypto"))
                    return failure();

                const auto& info = analysis.infoMap.lookup(funcOp);
                auto funcType = funcOp.getFunctionType();
                unsigned numInputs = funcType.getNumInputs();

                TypeConverter::SignatureConversion signatureConverter(numInputs);
                SmallVector<Type> newResultTypes;

                Block& entryBlock = funcOp.getBody().front();
                OpBuilder::InsertionGuard guard(rewriter);
                rewriter.setInsertionPointToStart(&entryBlock);

                for (unsigned i = 0; i < numInputs; ++i) {
                    Type oldType = funcType.getInput(i);

                    if (info.inputArgs.test(i)) {
                        signatureConverter.addInputs(i, typeConverter->convertType(oldType));
                    } else if (info.outputArgs.test(i)) {
                        Value dummy = rewriter.create<UnrealizedConversionCastOp>(
                                                  funcOp.getLoc(), oldType, ValueRange{})
                                          .getResult(0);
                        signatureConverter.remapInput(i, dummy);
                        newResultTypes.push_back(typeConverter->convertType(oldType));
                    } else {
                        Value c0;
                        if (isa<IntegerType>(oldType)) {
                            c0 = rewriter.create<arith::ConstantOp>(
                                funcOp.getLoc(), oldType, rewriter.getIntegerAttr(oldType, 0));
                        } else {
                            c0 = rewriter.create<arith::ConstantIndexOp>(funcOp.getLoc(), 0);
                        }
                        signatureConverter.remapInput(i, c0);
                    }
                }

                rewriter.modifyOpInPlace(funcOp, [&] {
                    auto newType = rewriter.getFunctionType(signatureConverter.getConvertedTypes(), newResultTypes);
                    funcOp.setFunctionType(newType);

                    funcOp.setVisibility(mlir::SymbolTable::Visibility::Private);
                });

                if (failed(rewriter.convertRegionTypes(&funcOp.getBody(), *typeConverter, &signatureConverter))) {
                    return failure();
                }

                return success();
            }
        };

        // --- 通用二元算术转换模板 (Add, Sub, Mult, Div) ---
        template <typename SourceOp, typename TargetOp>
        struct BinaryOpRewritePattern : public OpConversionPattern<SourceOp> {
            using OpConversionPattern<SourceOp>::OpConversionPattern;
            LogicalResult matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                                          ConversionPatternRewriter& rewriter) const override {
                // 直接使用第一个操作数（已经转换过的）的加密类型作为结果类型
                // 这样可以确保类型的一致性，避免 f64 的干扰
                Type resType = adaptor.getOperands()[0].getType();

                if (!isa<SCFHECipherType>(resType))
                    return failure();

                rewriter.replaceOpWithNewOp<TargetOp>(op, resType,
                                                      adaptor.getOperands()[0],
                                                      adaptor.getOperands()[1]);
                return success();
            }
        };

        // --- Math Exp 转换 ---
        struct ExpRewritePattern : public OpConversionPattern<math::ExpOp> {
            using OpConversionPattern<math::ExpOp>::OpConversionPattern;

            LogicalResult matchAndRewrite(math::ExpOp op, OpAdaptor adaptor,
                                          ConversionPatternRewriter& rewriter) const override {
                Type resultType = getTypeConverter()->convertType(op.getType());
                rewriter.replaceOpWithNewOp<scfhe::SCFHEExpOp>(op, resultType, adaptor.getOperand());
                return success();
            }
        };

        // --- 循环折叠 (向量化/批处理化) ---
        struct LoopCollapsePattern : public OpConversionPattern<affine::AffineForOp> {
            using OpConversionPattern::OpConversionPattern;
            LogicalResult matchAndRewrite(affine::AffineForOp op, OpAdaptor adaptor,
                                          ConversionPatternRewriter& rewriter) const override {
                if (!op->getParentOfType<func::FuncOp>()->hasAttr("scfhe.crypto"))
                    return failure();

                LLVM_DEBUG(llvm::dbgs() << "[Pattern] Collapsing AffineForOp\n");
                Block* body = op.getBody();
                rewriter.eraseOp(body->getTerminator());
                Value c0 = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
                rewriter.inlineBlockBefore(body, op, {c0});
                rewriter.eraseOp(op);
                return success();
            }
        };

        // --- 移除 Load (直接映射为加密 Value) ---
        struct RemoveLoadPattern : public OpConversionPattern<affine::AffineLoadOp> {
            using OpConversionPattern::OpConversionPattern;
            LogicalResult matchAndRewrite(affine::AffineLoadOp op, OpAdaptor adaptor,
                                          ConversionPatternRewriter& rewriter) const override {
                Value memref = adaptor.getMemref();
                if (memref && isa<SCFHECipherType>(memref.getType())) {
                    rewriter.replaceOp(op, memref);
                    return success();
                }
                return failure();
            }
        };

        // --- 将 Store 转换为 Return ---
        struct StoreToReturnPattern : public OpConversionPattern<affine::AffineStoreOp> {
            using OpConversionPattern::OpConversionPattern;
            LogicalResult matchAndRewrite(affine::AffineStoreOp op, OpAdaptor adaptor,
                                          ConversionPatternRewriter& rewriter) const override {
                // 修复点：使用 getOperands()[0] 获取要存储的值（Value to store）
                Value val = adaptor.getOperands()[0];

                if (val && isa<SCFHECipherType>(val.getType())) {
                    Operation* term = op->getBlock()->getTerminator();
                    if (auto retOp = dyn_cast<func::ReturnOp>(term)) {
                        rewriter.setInsertionPoint(term);
                        rewriter.replaceOpWithNewOp<func::ReturnOp>(term, val);
                        rewriter.eraseOp(op);
                        return success();
                    }
                }
                return failure();
            }
        };

        struct RemoveRedundantCopy : public OpRewritePattern<memref::CopyOp> {
            using OpRewritePattern<memref::CopyOp>::OpRewritePattern;

            LogicalResult matchAndRewrite(memref::CopyOp op, PatternRewriter& rewriter) const override {
                Value source = op.getSource();
                Value target = op.getTarget();

                // 【安全检查 1】: 目标必须是刚刚 alloc 出来的
                auto allocOp = target.getDefiningOp<memref::AllocOp>();
                if (!allocOp) {
                    return failure();
                }

                // 【安全检查 2】: 类型必须严格一致 (包含 Shape, Element Type 和 Layout)
                if (source.getType() != target.getType()) {
                    return failure();
                }

                // 遍历所有使用这个 alloc 的操作，进行严格审查
                for (Operation* user : target.getUsers()) {
                    if (user == op)
                        continue; // 忽略当前的 Copy
                    if (isa<memref::DeallocOp>(user))
                        continue; // 忽略释放操作

                    // 查找 User 在当前 Copy 所在 Block 的“祖先节点”
                    Operation* ancestor = user;
                    while (ancestor && ancestor->getBlock() != op->getBlock()) {
                        ancestor = ancestor->getParentOp();
                    }

                    // 【安全检查 3】: 时序检查 (Dominance)
                    // 如果在 Copy 之前就使用了内存，说明逻辑有异，不能替换
                    if (ancestor && ancestor->isBeforeInBlock(op)) {
                        return failure();
                    }

                    // 【安全检查 4】: 唯一写入者检查
                    // 确保没有其他的 store 或 copy 覆盖这块内存
                    if (isa<memref::StoreOp, affine::AffineStoreOp>(user)) {
                        return failure();
                    }
                    if (auto otherCopy = dyn_cast<memref::CopyOp>(user)) {
                        if (otherCopy.getTarget() == target) {
                            return failure();
                        }
                    }
                }

                // 所有安全检查通过：将所有使用 alloc 的地方替换为 source (Decrypt的结果)
                rewriter.replaceAllUsesWith(target, source);

                // 删除冗余的 Copy 和 Alloc
                rewriter.eraseOp(op);
                rewriter.eraseOp(allocOp);

                return success();
            }
        };

        // ============================================================================
        // 4. Pass 主体
        // ============================================================================
        struct ConvertToSCFHEPass : public impl::ConvertToSCFHEPassBase<ConvertToSCFHEPass> {
            using impl::ConvertToSCFHEPassBase<ConvertToSCFHEPass>::ConvertToSCFHEPassBase;

            void runOnOperation() final {
                ModuleOp module = getOperation();
                MLIRContext* ctx = &getContext();
                OpBuilder builder(ctx);

                LLVM_DEBUG(llvm::dbgs() << "====== SCFHE Pass Started ======\n");

                // --- Phase 1: Analysis ---
                ArgAnalysis argAnalysis;
                argAnalysis.run(module);

                // --- Phase 2: Client Rewrite (Call Sites) ---
                module.walk([&](func::CallOp callOp) {
                    auto callee = module.lookupSymbol<func::FuncOp>(callOp.getCallee());
                    if (!callee || !callee->hasAttr("scfhe.crypto"))
                        return;

                    const auto& info = argAnalysis.infoMap.lookup(callee);
                    builder.setInsertionPoint(callOp);
                    SmallVector<Value> newOperands;
                    SmallVector<Value> originalOutputMemRefs;

                    for (unsigned i = 0; i < callOp.getNumOperands(); ++i) {
                        Value arg = callOp.getOperand(i);
                        if (info.inputArgs.test(i)) {
                            auto memRefType = cast<MemRefType>(arg.getType());
                            auto cipherType = SCFHECipherType::get(ctx, memRefType.hasStaticShape() ? memRefType.getNumElements() : ShapedType::kDynamic, IntegerType::get(ctx, 64));
                            newOperands.push_back(builder.create<SCFHEEncryptOp>(callOp.getLoc(), cipherType, arg));
                        } else if (info.outputArgs.test(i)) {
                            originalOutputMemRefs.push_back(arg);
                        }
                    }

                    SmallVector<Type> newResultTypes;
                    for (auto out : originalOutputMemRefs)
                        newResultTypes.push_back(newOperands[0].getType());

                    auto newCallOp = builder.create<func::CallOp>(callOp.getLoc(), callee.getName(), newResultTypes, newOperands);
                    for (auto it : llvm::zip(newCallOp.getResults(), originalOutputMemRefs)) {
                        Value dec = builder.create<SCFHEDecryptOp>(callOp.getLoc(), std::get<1>(it).getType(), std::get<0>(it));
                        builder.create<memref::CopyOp>(callOp.getLoc(), dec, std::get<1>(it));
                    }
                    callOp.erase();
                });

                // --- Phase 3: Server Conversion ---
                SCFHETypeConverter typeConverter(ctx);
                RewritePatternSet patterns(ctx);
                ConversionTarget target(*ctx);

                target.addLegalDialect<SCFHEDialect, arith::ArithDialect, func::FuncDialect, math::MathDialect>();
                target.addLegalOp<mlir::UnrealizedConversionCastOp>();

                patterns.add<FuncSignaturePattern>(typeConverter, ctx, argAnalysis);
                patterns.add<BinaryOpRewritePattern<arith::AddFOp, SCFHEAddOp>>(typeConverter, ctx);
                patterns.add<BinaryOpRewritePattern<arith::SubFOp, SCFHESubOp>>(typeConverter, ctx);
                patterns.add<BinaryOpRewritePattern<arith::MulFOp, SCFHEMultOp>>(typeConverter, ctx);
                patterns.add<BinaryOpRewritePattern<arith::DivFOp, SCFHEDivOp>>(typeConverter, ctx);
                patterns.add<ExpRewritePattern>(typeConverter, ctx);
                patterns.add<LoopCollapsePattern>(typeConverter, ctx);
                patterns.add<RemoveLoadPattern>(typeConverter, ctx);
                patterns.add<StoreToReturnPattern>(typeConverter, ctx);

                patterns.add<RemoveRedundantCopy>(ctx);

                auto isCryptoFunc = [](Operation* op) {
                    auto func = op->getParentOfType<func::FuncOp>();
                    return func && func->hasAttr("scfhe.crypto");
                };

                target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
                    return !op->hasAttr("scfhe.crypto") || (op.getFunctionType().getNumResults() > 0 &&
                                                            llvm::none_of(op.getFunctionType().getInputs(), [](Type t) { return isa<MemRefType>(t); }));
                });

                target.addDynamicallyLegalOp<affine::AffineForOp, affine::AffineLoadOp, affine::AffineStoreOp,
                                             arith::AddFOp, arith::SubFOp, arith::MulFOp, arith::DivFOp, math::ExpOp>(
                    [&](Operation* op) { return !isCryptoFunc(op); });

                if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
                    signalPassFailure();
                    return;
                }

                // --- Phase 4: Cleanup (消除冗余拷贝) ---
                RewritePatternSet cleanupPatterns(ctx);
                cleanupPatterns.add<RemoveRedundantCopy>(ctx);

                // 贪婪地在整个 Module 上应用清理 Pattern
                if (failed(applyPatternsGreedily(module, std::move(cleanupPatterns)))) {
                    signalPassFailure();
                    return;
                }

                LLVM_DEBUG(llvm::dbgs() << "====== SCFHE Pass Finished ======\n");
            }
        };
    }
}