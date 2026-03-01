#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"

#include "mlir/Interfaces/CallInterfaces.h"

#include "mlir/Pass/PassManager.h"

#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/InliningUtils.h"

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Debug.h"

#include "SCFHEDialect.h"
#include "SCFHEOps.h"
#include "SCFHEPass.h"
#include "SCFHETypes.h"

// --debug-only=scfhe-pass
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
                // 1. 默认转换：遇到不认识或无需加密的类型（比如 index），直接原样返回
                addConversion([](Type type) { return type; });

                // 2. 处理 MemRef -> Cipher
                addConversion([ctx](MemRefType type) -> std::optional<Type> {
                    Type cipherElemType = IntegerType::get(ctx, 64);
                    int64_t packedSize = type.hasStaticShape() ? type.getNumElements() : ShapedType::kDynamic;
                    auto resultType = SCFHECipherType::get(ctx, packedSize, cipherElemType);

                    LLVM_DEBUG(llvm::dbgs() << "[TypeConverter] Converted MemRefType: "
                                            << type << " -> " << resultType << "\n");
                    return resultType;
                });

                // 3. 处理浮点标量 f64 -> Cipher
                // 修复：将 kDynamic 改为 1。因为单个 f64 标量对应的一定是大小为 1 的密文！
                addConversion([ctx](Float64Type type) -> std::optional<Type> {
                    Type cipherElemType = IntegerType::get(ctx, 64);
                    auto resultType = SCFHECipherType::get(ctx, 1, cipherElemType);

                    LLVM_DEBUG(llvm::dbgs() << "[TypeConverter] Converted Float64Type: "
                                            << type << " -> " << resultType << "\n");
                    return resultType;
                });

                // 4. 处理整数标量 (比如 i32, i64) -> Cipher
                // 防止遇到整数运算时也退化成 ?
                addConversion([ctx](IntegerType type) -> std::optional<Type> {
                    Type cipherElemType = IntegerType::get(ctx, 64);
                    auto resultType = SCFHECipherType::get(ctx, 1, cipherElemType);

                    LLVM_DEBUG(llvm::dbgs() << "[TypeConverter] Converted IntegerType: "
                                            << type << " -> " << resultType << "\n");
                    return resultType;
                });

                // 5. Source Materialization (反向修补：从密文回退到明文类型)
                addSourceMaterialization([](OpBuilder& b, Type t, ValueRange i, Location l) -> Value {
                    LLVM_DEBUG(llvm::dbgs() << "[TypeConverter] SourceMaterialization: Wrapping with Cast to -> "
                                            << t << "\n");
                    return b.create<UnrealizedConversionCastOp>(l, t, i).getResult(0);
                });

                // 6. Target Materialization (正向修补：从明文强制套上密文马甲)
                addTargetMaterialization([](OpBuilder& b, Type t, ValueRange i, Location l) -> Value {
                    LLVM_DEBUG(llvm::dbgs() << "[TypeConverter] TargetMaterialization: Wrapping with Cast to -> " << t << "\n");

                    // 【核心修复】：如果目标是密文，且输入是 f64/i64 标量，直接自动生成 scfhe.encrypt！
                    if (isa<SCFHECipherType>(t) && i.size() == 1) {
                        Type inType = i[0].getType();
                        if (isa<FloatType>(inType) || isa<IntegerType>(inType)) {
                            LLVM_DEBUG(llvm::dbgs() << "  -> Auto-encrypting plaintext to cipher.\n");
                            return b.create<scfhe::SCFHEEncryptOp>(l, t, i[0]).getResult();
                        }
                    }

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
                        // Value c0;
                        // if (isa<IntegerType>(oldType)) {
                        //     c0 = rewriter.create<arith::ConstantOp>(
                        //         funcOp.getLoc(), oldType, rewriter.getIntegerAttr(oldType, 0));
                        // } else {
                        //     c0 = rewriter.create<arith::ConstantIndexOp>(funcOp.getLoc(), 0);
                        // }
                        // signatureConverter.remapInput(i, c0);

                        signatureConverter.addInputs(i, oldType);
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
        // template <typename SourceOp, typename TargetOp>
        // struct BinaryOpRewritePattern : public OpConversionPattern<SourceOp> {
        //     using OpConversionPattern<SourceOp>::OpConversionPattern;
        //     LogicalResult matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
        //                                   ConversionPatternRewriter& rewriter) const override {
        //         // 直接使用第一个操作数（已经转换过的）的加密类型作为结果类型
        //         // 这样可以确保类型的一致性，避免 f64 的干扰
        //         Type resType = adaptor.getOperands()[0].getType();

        //         if (!isa<SCFHECipherType>(resType))
        //             return failure();

        //         rewriter.replaceOpWithNewOp<TargetOp>(op, resType,
        //                                               adaptor.getOperands()[0],
        //                                               adaptor.getOperands()[1]);
        //         return success();
        //     }
        // };

        // --- 通用二元算术转换模板 (支持纯密文、密文-明文、密文-常数混合运算) ---
        template <typename SourceOp, typename TargetOp>
        struct BinaryOpRewritePattern : public OpConversionPattern<SourceOp> {
            using OpConversionPattern<SourceOp>::OpConversionPattern;

            LogicalResult matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                                          ConversionPatternRewriter& rewriter) const override {
                LLVM_DEBUG(llvm::dbgs() << "\n[BinaryOpRewritePattern] Trying to convert " << op->getName() << " at " << op.getLoc() << "\n");

                Value lhs = adaptor.getOperands()[0];
                Value rhs = adaptor.getOperands()[1];

                bool lhsIsCipher = isa<SCFHECipherType>(lhs.getType());
                bool rhsIsCipher = isa<SCFHECipherType>(rhs.getType());

                LLVM_DEBUG(llvm::dbgs() << "  -> LHS Type: " << lhs.getType() << " (Cipher: " << lhsIsCipher << ")\n");
                LLVM_DEBUG(llvm::dbgs() << "  -> RHS Type: " << rhs.getType() << " (Cipher: " << rhsIsCipher << ")\n");

                // 【混合运算核心逻辑】：只要左右操作数中有一个是密文，就可以转换为 FHE 算子！
                if (!lhsIsCipher && !rhsIsCipher) {
                    LLVM_DEBUG(llvm::dbgs() << "  -> [Failed] Neither LHS nor RHS is a CipherType. Rejecting conversion.\n");
                    return failure();
                }

                // 结果的类型必须是密文。如果一个是明文一个是密文，提取那个密文的类型作为返回类型
                Type resType = lhsIsCipher ? lhs.getType() : rhs.getType();

                LLVM_DEBUG(llvm::dbgs() << "  -> [Success] Conditions met (Mixed Operations Supported!). Converted to SCFHE Binary Op.\n");
                rewriter.replaceOpWithNewOp<TargetOp>(op, resType, lhs, rhs);
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

        // --- 将 AffineYield 转换为 SCFYield ---
        struct AffineYieldToSCFYieldPattern : public OpConversionPattern<affine::AffineYieldOp> {
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(affine::AffineYieldOp op, OpAdaptor adaptor,
                                          ConversionPatternRewriter& rewriter) const override {
                LLVM_DEBUG(llvm::dbgs() << "\n[AffineYieldToSCFYieldPattern] Converting yield...\n");
                // adaptor.getOperands() 会自动拉取到内部算子计算出的 CipherType！
                rewriter.replaceOpWithNewOp<scf::YieldOp>(op, adaptor.getOperands());
                return success();
            }
        };

        // --- 有数据依赖循环 1:1 映射 ---
        struct AffineForToSCFForPattern : public OpConversionPattern<affine::AffineForOp> {
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(affine::AffineForOp op, OpAdaptor adaptor,
                                          ConversionPatternRewriter& rewriter) const override {
                auto funcOp = op->getParentOfType<func::FuncOp>();
                if (!funcOp || !funcOp->hasAttr("scfhe.crypto"))
                    return failure();

                LLVM_DEBUG(llvm::dbgs() << "\n[AffineForToSCFForPattern] Inspecting AffineForOp at " << op.getLoc() << "\n");

                // 【核心拦截】：如果没有 iter_args (无数据依赖)，拒绝匹配，交给策略 A 处理
                if (op.getNumIterOperands() == 0) {
                    LLVM_DEBUG(llvm::dbgs() << "  -> [Rejected] Loop has NO iter_args. Delegating to Batching Pattern.\n");
                    return failure();
                }

                LLVM_DEBUG(llvm::dbgs() << "  -> [Accepted] Data dependency (iter_args) found. Converting to scf.for...\n");

                Location loc = op.getLoc();

                // 1. 获取边界和步长
                Value lowerBound = op.hasConstantLowerBound()
                                       ? rewriter.create<arith::ConstantIndexOp>(loc, op.getConstantLowerBound())
                                       : adaptor.getLowerBoundOperands()[0];

                Value upperBound = op.hasConstantUpperBound()
                                       ? rewriter.create<arith::ConstantIndexOp>(loc, op.getConstantUpperBound())
                                       : adaptor.getUpperBoundOperands()[0];

                Value step = rewriter.create<arith::ConstantIndexOp>(loc, op.getStepAsInt());
                LLVM_DEBUG(llvm::dbgs() << "  -> Extracted loop bounds and step.\n");

                // 2. 获取已经经过 TypeConverter 转换的初始值 (f64 变成了 cipher)
                ValueRange inits = adaptor.getInits();
                LLVM_DEBUG(llvm::dbgs() << "  -> Extracted " << inits.size() << " converted iter_args (inits).\n");
                for (auto [idx, init] : llvm::enumerate(inits)) {
                    LLVM_DEBUG(llvm::dbgs() << "     init[" << idx << "] type is: " << init.getType() << "\n");
                }

                // 3. 创建目标 SCF 循环
                auto scfForOp = rewriter.create<scf::ForOp>(loc, lowerBound, upperBound, step, inits);
                LLVM_DEBUG(llvm::dbgs() << "  -> Created new scf.for operation.\n");

                // 4. 将旧的 AffineFor Block 迁移到 SCF For 中
                Region& oldRegion = op.getRegion();
                Region& newRegion = scfForOp.getRegion();
                if (!newRegion.empty()) {
                    rewriter.eraseBlock(&newRegion.front());
                }
                rewriter.inlineRegionBefore(oldRegion, newRegion, newRegion.end());
                LLVM_DEBUG(llvm::dbgs() << "  -> Inlined old region into new scf.for region.\n");

                // 5. 动态修改 Block 参数的类型，以适配密文 (TypeConverter 的魔法在此生效)
                Block& body = newRegion.front();
                for (size_t i = 1; i < body.getNumArguments(); ++i) {
                    Type oldType = body.getArgument(i).getType();
                    Type newIterType = inits[i - 1].getType();
                    body.getArgument(i).setType(newIterType);
                    LLVM_DEBUG(llvm::dbgs() << "  -> Updated Block Arg " << i << " type: "
                                            << oldType << " -> " << newIterType << "\n");
                }

                // 6. 替换 Yield 算子
                // auto oldYield = cast<affine::AffineYieldOp>(body.getTerminator());
                // rewriter.setInsertionPoint(oldYield);
                // rewriter.replaceOpWithNewOp<scf::YieldOp>(oldYield, oldYield.getOperands());
                // LLVM_DEBUG(llvm::dbgs() << "  -> Replaced affine.yield with scf.yield.\n");

                rewriter.replaceOp(op, scfForOp.getResults());
                LLVM_DEBUG(llvm::dbgs() << "[AffineForToSCFForPattern] Conversion complete.\n");
                return success();
            }
        };

        // --- 移除 Load (直接映射为加密 Value) ---
        // struct RemoveLoadPattern : public OpConversionPattern<affine::AffineLoadOp> {
        //     using OpConversionPattern::OpConversionPattern;
        //     LogicalResult matchAndRewrite(affine::AffineLoadOp op, OpAdaptor adaptor,
        //                                   ConversionPatternRewriter& rewriter) const override {
        //         Value memref = adaptor.getMemref();
        //         if (memref && isa<SCFHECipherType>(memref.getType())) {
        //             rewriter.replaceOp(op, memref);
        //             return success();
        //         }
        //         return failure();
        //     }
        // };

        // --- 将 affine.load 转换为逻辑层的密文提取算子 scfhe.load ---
        struct AffineLoadToSCFHELoadPattern : public OpConversionPattern<affine::AffineLoadOp> {
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(affine::AffineLoadOp op, OpAdaptor adaptor,
                                          ConversionPatternRewriter& rewriter) const override {
                LLVM_DEBUG(llvm::dbgs() << "\n[AffineLoadToSCFHELoadPattern] Inspecting affine.load at " << op.getLoc() << "\n");

                Value memref = adaptor.getMemref();

                // 确保我们正在处理的是已经被转换成密文的数组
                if (!isa<SCFHECipherType>(memref.getType())) {
                    LLVM_DEBUG(llvm::dbgs() << "  -> [Failed] The source memref is NOT a CipherType.\n");
                    return failure();
                }

                auto indices = adaptor.getIndices();
                if (indices.empty()) {
                    LLVM_DEBUG(llvm::dbgs() << "  -> [Failed] No indices found for affine.load.\n");
                    return failure();
                }

                // 核心：从大数组中 Load 出来的是一个“标量”，目标类型固定为 <1 x i64> 的密文
                Type resultType = SCFHECipherType::get(rewriter.getContext(), 1, IntegerType::get(rewriter.getContext(), 64));

                LLVM_DEBUG(llvm::dbgs() << "  -> Array Cipher Type: " << memref.getType() << "\n");
                LLVM_DEBUG(llvm::dbgs() << "  -> Target Result Type: " << resultType << "\n");

                rewriter.replaceOpWithNewOp<SCFHELoadOp>(op, resultType, memref, indices[0]);

                LLVM_DEBUG(llvm::dbgs() << "  -> [Success] Replaced affine.load with scfhe.load.\n");
                return success();
            }
        };

        // --- 将 Store 转换为 Return ---
        struct StoreToReturnPattern : public OpConversionPattern<affine::AffineStoreOp> {
            using OpConversionPattern::OpConversionPattern;
            LogicalResult matchAndRewrite(affine::AffineStoreOp op, OpAdaptor adaptor,
                                          ConversionPatternRewriter& rewriter) const override {
                LLVM_DEBUG(llvm::dbgs() << "\n[StoreToReturnPattern] Inspecting AffineStoreOp...\n");

                Value val = adaptor.getOperands()[0];
                LLVM_DEBUG(llvm::dbgs() << "  -> Value to store Type: " << val.getType() << "\n");

                if (val && isa<SCFHECipherType>(val.getType())) {
                    Operation* term = op->getBlock()->getTerminator();
                    if (auto retOp = dyn_cast<func::ReturnOp>(term)) {

                        // 【新增修复点】：获取这个 store 原本写入的 memref 转换后的目标密文类型
                        // 比如原本写入 memref<?xf64>，这里 expectedRetType 就会是 <? x i64>
                        Type expectedRetType = getTypeConverter()->convertType(op.getMemRef().getType());
                        Value retVal = val;

                        // 如果当前计算出的密文类型（<1 x i64>）与函数签名要求的类型（<? x i64>）不一致
                        if (expectedRetType && retVal.getType() != expectedRetType) {
                            LLVM_DEBUG(llvm::dbgs() << "  -> Type mismatch for return! Route B: Inserting Cast to satisfy signature.\n");

                            // 老老实实贴一个创可贴，把 <1> 强转成 <?> 以满足外部调用者的期望
                            retVal = rewriter.create<UnrealizedConversionCastOp>(
                                                 op.getLoc(), expectedRetType, retVal)
                                         .getResult(0);
                        }

                        LLVM_DEBUG(llvm::dbgs() << "  -> [Success] Replacing store with func.return containing Cipher.\n");
                        rewriter.setInsertionPoint(term);
                        rewriter.replaceOpWithNewOp<func::ReturnOp>(term, retVal);
                        rewriter.eraseOp(op);
                        return success();
                    } else {
                        LLVM_DEBUG(llvm::dbgs() << "  -> [Failed] Block terminator is not func.return.\n");
                    }
                } else {
                    LLVM_DEBUG(llvm::dbgs() << "  -> [Failed] The value to store is NOT a CipherType.\n");
                }
                return failure();
            }
        };

        // --- 清理内联后产生的冗余形状 Cast ---
        struct RemoveDecryptCastPattern : public OpRewritePattern<SCFHEDecryptOp> {
            using OpRewritePattern::OpRewritePattern;

            LogicalResult matchAndRewrite(SCFHEDecryptOp op, PatternRewriter& rewriter) const override {
                // 检查 decrypt 的输入是否是一个 UnrealizedConversionCastOp
                if (auto castOp = op.getOperand().getDefiningOp<UnrealizedConversionCastOp>()) {
                    Value realInput = castOp.getInputs()[0];

                    if (auto cipherType = dyn_cast<SCFHECipherType>(realInput.getType())) {
                        LLVM_DEBUG(llvm::dbgs() << "[Cleanup] Melting away UnrealizedConversionCastOp before decrypt.\n");

                        // 1. 获取准确的静态类型（从输入的 <1 x i64> 密文推导出 memref<1xf64> 明文）
                        auto oldMemRefType = cast<MemRefType>(op.getType());
                        auto newMemRefType = MemRefType::get({cipherType.getPlaintextCount()}, oldMemRefType.getElementType());

                        // 2. 老老实实地用准确的静态类型创建 Decrypt 算子，满足验证器的严格要求
                        auto newDecrypt = rewriter.create<SCFHEDecryptOp>(op.getLoc(), newMemRefType, realInput);

                        // 3. 贴一个标准的 memref.cast，把 <1> 转回老算子期待的 <?>
                        // 这样既满足了 MLIR 严格的类型推断，又满足了后续的 memref.copy
                        rewriter.replaceOpWithNewOp<memref::CastOp>(op, oldMemRefType, newDecrypt);

                        return success();
                    }
                }
                return failure();
            }
        };

        // --- 让 scfhe.load 直接穿透 scfhe.cast，读取底层真实的静态密文 ---
        struct RemoveLoadCastPattern : public OpRewritePattern<SCFHELoadOp> {
            using OpRewritePattern::OpRewritePattern;

            LogicalResult matchAndRewrite(SCFHELoadOp op, PatternRewriter& rewriter) const override {
                // 1. 获取 load 正在读取的数组 (第 0 个操作数)
                Value arrayOperand = op.getOperand(0);

                // 2. 检查它是不是一个 scfhe.cast
                if (auto castOp = arrayOperand.getDefiningOp<SCFHECastOp>()) {
                    LLVM_DEBUG(llvm::dbgs() << "[Cleanup] Melting away SCFHECastOp before load.\n");

                    // 获取 Cast 之前真实的静态数组 (比如 <10 x i64>)
                    Value realArray = castOp.getOperand();

                    // 【修复处】：直接获取第 1 个操作数作为唯一的 index
                    Value index = op.getOperand(1);

                    // 3. 用真实的静态数组和唯一的 index 替换掉老 load
                    rewriter.replaceOpWithNewOp<SCFHELoadOp>(
                        op,
                        op.getType(),
                        realArray,
                        index);
                    return success();
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

        // --- 无数据依赖循环打包 (Plaintext Batching / Flattening) ---
        struct PlaintextBatchingPattern : public OpConversionPattern<affine::AffineForOp> {
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(affine::AffineForOp op, OpAdaptor adaptor,
                                          ConversionPatternRewriter& rewriter) const override {
                auto funcOp = op->getParentOfType<func::FuncOp>();
                if (!funcOp || !funcOp->hasAttr("scfhe.crypto"))
                    return failure();

                LLVM_DEBUG(llvm::dbgs() << "\n[PlaintextBatchingPattern] Inspecting AffineForOp at " << op.getLoc() << "\n");

                // 【核心拦截】：如果有 iter_args (有数据依赖)，拒绝匹配，交给策略 B 处理
                if (op.getNumIterOperands() > 0) {
                    LLVM_DEBUG(llvm::dbgs() << "  -> [Rejected] Loop has iter_args (data dependency). Delegating to Route 1.\n");
                    return failure();
                }

                LLVM_DEBUG(llvm::dbgs() << "  -> [Accepted] No data dependency found. Starting flattening (batching)...\n");

                Block* body = op.getBody();

                // 1. 获取 yield 的返回值并删除 yield 算子
                auto yieldOp = cast<affine::AffineYieldOp>(body->getTerminator());
                SmallVector<Value> yieldedValues;
                for (Value operand : yieldOp.getOperands()) {
                    yieldedValues.push_back(operand);
                }
                LLVM_DEBUG(llvm::dbgs() << "  -> Extracted " << yieldedValues.size() << " yielded values.\n");
                rewriter.eraseOp(yieldOp);

                // 2. 为 Block 准备替身参数（因为没有 iter_args，所以只有 index 需要替换）
                SmallVector<Value, 4> replValues;
                Value c0 = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
                replValues.push_back(c0);
                LLVM_DEBUG(llvm::dbgs() << "  -> Replaced loop induction variable with constant 0.\n");

                // 3. 将循环体内的操作一锅端出来，砸平
                rewriter.inlineBlockBefore(body, op, replValues);
                LLVM_DEBUG(llvm::dbgs() << "  -> Inlined loop body successfully.\n");

                // 4. 处理返回值并删除原循环
                if (yieldedValues.empty()) {
                    rewriter.eraseOp(op);
                    LLVM_DEBUG(llvm::dbgs() << "  -> Erased original loop (no return values).\n");
                } else {
                    rewriter.replaceOp(op, yieldedValues);
                    LLVM_DEBUG(llvm::dbgs() << "  -> Replaced original loop with yielded values.\n");
                }

                LLVM_DEBUG(llvm::dbgs() << "[PlaintextBatchingPattern] Batching complete.\n");
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
                SCFHETypeConverter typeConverter(ctx);

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
                        } else {
                            newOperands.push_back(arg);
                        }
                    }

                    SmallVector<Type> newResultTypes;
                    for (auto out : originalOutputMemRefs) {
                        newResultTypes.push_back(typeConverter.convertType(out.getType()));
                    }

                    auto newCallOp = builder.create<func::CallOp>(callOp.getLoc(), callee.getName(), newResultTypes, newOperands);
                    for (auto it : llvm::zip(newCallOp.getResults(), originalOutputMemRefs)) {
                        Value dec = builder.create<SCFHEDecryptOp>(callOp.getLoc(), std::get<1>(it).getType(), std::get<0>(it));
                        builder.create<memref::CopyOp>(callOp.getLoc(), dec, std::get<1>(it));
                    }

                    callOp.erase();
                });

                // --- Phase 3: Server Conversion ---
                // SCFHETypeConverter typeConverter(ctx);
                RewritePatternSet patterns(ctx);
                ConversionTarget target(*ctx);

                target.addLegalDialect<SCFHEDialect, arith::ArithDialect, func::FuncDialect, math::MathDialect, scf::SCFDialect>();
                target.addLegalOp<mlir::UnrealizedConversionCastOp>();

                patterns.add<FuncSignaturePattern>(typeConverter, ctx, argAnalysis);
                patterns.add<BinaryOpRewritePattern<arith::AddFOp, SCFHEAddOp>>(typeConverter, ctx);
                patterns.add<BinaryOpRewritePattern<arith::SubFOp, SCFHESubOp>>(typeConverter, ctx);
                patterns.add<BinaryOpRewritePattern<arith::MulFOp, SCFHEMultOp>>(typeConverter, ctx);
                patterns.add<BinaryOpRewritePattern<arith::DivFOp, SCFHEDivOp>>(typeConverter, ctx);
                patterns.add<ExpRewritePattern>(typeConverter, ctx);

                patterns.add<PlaintextBatchingPattern>(typeConverter, ctx);
                patterns.add<AffineForToSCFForPattern>(typeConverter, ctx);
                patterns.add<AffineYieldToSCFYieldPattern>(typeConverter, ctx);

                // patterns.add<RemoveLoadPattern>(typeConverter, ctx);
                patterns.add<AffineLoadToSCFHELoadPattern>(typeConverter, ctx);
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

                target.addDynamicallyLegalOp<affine::AffineForOp, affine::AffineYieldOp, affine::AffineLoadOp, affine::AffineStoreOp,
                                             arith::AddFOp, arith::SubFOp, arith::MulFOp, arith::DivFOp, math::ExpOp>(
                    [&](Operation* op) { return !isCryptoFunc(op); });

                LLVM_DEBUG(llvm::dbgs() << "\n--- Executing Phase 3: Dialect Conversion ---\n");
                if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
                    LLVM_DEBUG(llvm::dbgs() << "!!! Phase 3 Conversion Failed (Rollback Occurred) !!!\n");
                    signalPassFailure();
                    return;
                }
                LLVM_DEBUG(llvm::dbgs() << "--- Phase 3 Conversion Succeeded ---\n\n");

                // --- Phase 4: Inlining (调用 MLIR 官方内联管线) ---
                LLVM_DEBUG(llvm::dbgs() << "--- Executing Phase 4: Official Inliner ---\n");

                // 在 Pass 内部创建一个临时的管线，专门用来跑官方 Inliner
                PassManager pm(ctx);
                pm.addPass(createInlinerPass());

                // 运行官方内联！它会自动处理所有复杂的返回值的 Use-Def 链替换
                if (failed(pm.run(module))) {
                    LLVM_DEBUG(llvm::dbgs() << "!!! Inlining Failed !!!\n");
                    signalPassFailure();
                    return;
                }

                // --- Phase 5: Cleanup & Shape Inference (形状推断与清理) ---
                LLVM_DEBUG(llvm::dbgs() << "\n--- Executing Phase 5: Cleanup & Shape Inference ---\n");
                RewritePatternSet cleanupPatterns(ctx);
                cleanupPatterns.add<RemoveRedundantCopy>(ctx);
                cleanupPatterns.add<RemoveDecryptCastPattern>(ctx); // 加入刚才写的融化 Cast 的 Pattern

                cleanupPatterns.add<RemoveLoadCastPattern>(ctx);

                // MLIR 的 GreedyPatternRewriteDriver 会自动调用算子内置的 fold() 方法。
                // 这意味着所有匹配的 cast(<1> to <?>) 和 cast(<?> to <1>) 如果相遇，会自动互相抵消（折叠）！
                if (failed(applyPatternsGreedily(module, std::move(cleanupPatterns)))) {
                    signalPassFailure();
                    return;
                }

                LLVM_DEBUG(llvm::dbgs() << "====== SCFHE Pass Finished ======\n");
            }
        };
    }
}