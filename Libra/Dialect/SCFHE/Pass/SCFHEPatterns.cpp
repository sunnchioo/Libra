// SCFHEPatterns.cpp
#include "SCFHEPatterns.h"
#include "SCFHEOps.h"
#include "SCFHETypes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "scfhe-pass"

using namespace mlir;
using namespace mlir::libra::scfhe;

namespace {

    // ============================================================================
    // Phase 3: Conversion Patterns
    // ============================================================================

    struct FuncSignaturePattern : public OpConversionPattern<func::FuncOp> {
        const ArgAnalysis& analysis;
        FuncSignaturePattern(TypeConverter& converter, MLIRContext* ctx, const ArgAnalysis& analysis)
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

    template <typename SourceOp, typename TargetOp>
    struct BinaryOpRewritePattern : public OpConversionPattern<SourceOp> {
        using OpConversionPattern<SourceOp>::OpConversionPattern;

        Value peelCast(Value v) const {
            if (auto castOp = v.getDefiningOp<UnrealizedConversionCastOp>()) {
                Value input = castOp.getInputs()[0];
                if (auto inputTy = dyn_cast<SCFHECipherType>(input.getType())) {
                    if (inputTy.getPlaintextCount() == ShapedType::kDynamic || inputTy.getPlaintextCount() > 1) {
                        return input;
                    }
                }
            }
            return v;
        }

        LogicalResult matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                                      ConversionPatternRewriter& rewriter) const override {
            LLVM_DEBUG(llvm::dbgs() << "\n[BinaryOpRewritePattern] Trying to convert " << op->getName() << " at " << op.getLoc() << "\n");

            Value lhs = peelCast(adaptor.getOperands()[0]);
            Value rhs;

            if constexpr (std::is_same<TargetOp, SCFHEDivOp>::value) {
                Value originalRhs = op.getOperand(1);
                if (isa<FloatType>(originalRhs.getType())) {
                    LLVM_DEBUG(llvm::dbgs() << "  -> [Optimization] Div op detected. Using ORIGINAL plaintext operand directly (bypassing conversion).\n");
                    rhs = originalRhs;
                } else {
                    rhs = peelCast(adaptor.getOperands()[1]);
                }
            } else {
                rhs = peelCast(adaptor.getOperands()[1]);
            }

            bool lhsIsCipher = isa<SCFHECipherType>(lhs.getType());
            bool rhsIsCipher = isa<SCFHECipherType>(rhs.getType());

            LLVM_DEBUG(llvm::dbgs() << "  -> LHS Type: " << lhs.getType() << "\n");
            LLVM_DEBUG(llvm::dbgs() << "  -> RHS Type: " << rhs.getType() << "\n");

            if (!lhsIsCipher && !rhsIsCipher) {
                return failure();
            }

            Type resType;
            if (lhsIsCipher && rhsIsCipher) {
                auto lTy = cast<SCFHECipherType>(lhs.getType());
                auto rTy = cast<SCFHECipherType>(rhs.getType());
                if (lTy.getPlaintextCount() == ShapedType::kDynamic || lTy.getPlaintextCount() > rTy.getPlaintextCount())
                    resType = lTy;
                else
                    resType = rTy;
            } else {
                resType = lhsIsCipher ? lhs.getType() : rhs.getType();
            }

            LLVM_DEBUG(llvm::dbgs() << "  -> [Success] Converting to SCFHE Binary Op with Result Type: " << resType << "\n");

            rewriter.replaceOpWithNewOp<TargetOp>(op, resType, lhs, rhs);
            return success();
        }
    };

    struct ReduceAddPattern : public OpConversionPattern<affine::AffineForOp> {
        ReduceAddPattern(TypeConverter& converter, MLIRContext* context)
            : OpConversionPattern(converter, context, /*benefit=*/10) {}

        LogicalResult matchAndRewrite(affine::AffineForOp op, OpAdaptor adaptor,
                                      ConversionPatternRewriter& rewriter) const override {
            LLVM_DEBUG(llvm::dbgs() << "\n[ReduceAddPattern] Inspecting loop at " << op.getLoc() << "\n");

            auto funcOp = op->getParentOfType<func::FuncOp>();
            if (!funcOp || !funcOp->hasAttr("scfhe.crypto"))
                return failure();

            if (op.getNumIterOperands() != 1) {
                LLVM_DEBUG(llvm::dbgs() << "  -> Failed: Loop has " << op.getNumIterOperands() << " iter_args (expected 1)\n");
                return failure();
            }

            Block* body = op.getBody();
            auto yieldOp = dyn_cast<affine::AffineYieldOp>(body->getTerminator());
            if (!yieldOp || yieldOp.getNumOperands() != 1)
                return failure();

            Value yieldedVal = yieldOp.getOperand(0);
            auto addOp = yieldedVal.getDefiningOp<arith::AddFOp>();
            if (!addOp) {
                LLVM_DEBUG(llvm::dbgs() << "  -> Failed: Yielded value is not from arith.addf\n");
                return failure();
            }

            Value iterArg = body->getArgument(1);
            Value loadVal;

            if (addOp.getLhs() == iterArg) {
                loadVal = addOp.getRhs();
            } else if (addOp.getRhs() == iterArg) {
                loadVal = addOp.getLhs();
            } else {
                LLVM_DEBUG(llvm::dbgs() << "  -> Failed: AddOp does not use the accumulator (iter_arg)\n");
                return failure();
            }

            auto loadOp = loadVal.getDefiningOp<affine::AffineLoadOp>();
            if (!loadOp) {
                LLVM_DEBUG(llvm::dbgs() << "  -> Failed: The other operand is not from affine.load\n");
                return failure();
            }

            if (loadOp.getIndices().size() != 1 || loadOp.getIndices()[0] != op.getInductionVar()) {
                LLVM_DEBUG(llvm::dbgs() << "  -> Failed: Load index is not the loop IV\n");
                return failure();
            }

            LLVM_DEBUG(llvm::dbgs() << "  -> [MATCHED] Found sum reduction loop!\n");

            Value originalMemref = loadOp.getMemref();
            Value mappedArray = rewriter.getRemappedValue(originalMemref);

            if (!mappedArray) {
                LLVM_DEBUG(llvm::dbgs() << "  -> Warning: Memref not remapped yet. Using original.\n");
                mappedArray = originalMemref;
            }

            if (!isa<SCFHECipherType>(mappedArray.getType())) {
                LLVM_DEBUG(llvm::dbgs() << "  -> Failed: Mapped array is not SCFHECipherType (" << mappedArray.getType() << ")\n");
                return failure();
            }

            auto arrayType = cast<SCFHECipherType>(mappedArray.getType());
            Type resultType = SCFHECipherType::get(getContext(), 1, arrayType.getElementType());

            Value reduceResult = rewriter.create<SCFHEReduceAddOp>(op.getLoc(), resultType, mappedArray);
            Value initVal = adaptor.getInits()[0];
            Value finalResult = rewriter.create<SCFHEAddOp>(op.getLoc(), resultType, reduceResult, initVal);

            rewriter.replaceOp(op, finalResult);
            LLVM_DEBUG(llvm::dbgs() << "  -> [Success] Replaced affine.for with scfhe.reduce_add + scfhe.add.\n");

            return success();
        }
    };

    struct ExpRewritePattern : public OpConversionPattern<math::ExpOp> {
        using OpConversionPattern<math::ExpOp>::OpConversionPattern;

        LogicalResult matchAndRewrite(math::ExpOp op, OpAdaptor adaptor,
                                      ConversionPatternRewriter& rewriter) const override {
            Type resultType = getTypeConverter()->convertType(op.getType());
            rewriter.replaceOpWithNewOp<SCFHEExpOp>(op, resultType, adaptor.getOperand());
            return success();
        }
    };

    struct AffineYieldToSCFYieldPattern : public OpConversionPattern<affine::AffineYieldOp> {
        using OpConversionPattern::OpConversionPattern;

        LogicalResult matchAndRewrite(affine::AffineYieldOp op, OpAdaptor adaptor,
                                      ConversionPatternRewriter& rewriter) const override {
            LLVM_DEBUG(llvm::dbgs() << "\n[AffineYieldToSCFYieldPattern] Converting yield...\n");
            rewriter.replaceOpWithNewOp<scf::YieldOp>(op, adaptor.getOperands());
            return success();
        }
    };

    struct AffineForToSCFForPattern : public OpConversionPattern<affine::AffineForOp> {
        using OpConversionPattern::OpConversionPattern;

        LogicalResult matchAndRewrite(affine::AffineForOp op, OpAdaptor adaptor,
                                      ConversionPatternRewriter& rewriter) const override {
            auto funcOp = op->getParentOfType<func::FuncOp>();
            if (!funcOp || !funcOp->hasAttr("scfhe.crypto"))
                return failure();

            LLVM_DEBUG(llvm::dbgs() << "\n[AffineForToSCFForPattern] Inspecting AffineForOp at " << op.getLoc() << "\n");

            if (op.getNumIterOperands() == 0) {
                LLVM_DEBUG(llvm::dbgs() << "  -> [Rejected] Loop has NO iter_args. Delegating to Batching Pattern.\n");
                return failure();
            }

            LLVM_DEBUG(llvm::dbgs() << "  -> [Accepted] Data dependency (iter_args) found. Converting to scf.for...\n");

            Location loc = op.getLoc();

            Value lowerBound = op.hasConstantLowerBound()
                                   ? rewriter.create<arith::ConstantIndexOp>(loc, op.getConstantLowerBound())
                                   : adaptor.getLowerBoundOperands()[0];

            Value upperBound = op.hasConstantUpperBound()
                                   ? rewriter.create<arith::ConstantIndexOp>(loc, op.getConstantUpperBound())
                                   : adaptor.getUpperBoundOperands()[0];

            Value step = rewriter.create<arith::ConstantIndexOp>(loc, op.getStepAsInt());
            LLVM_DEBUG(llvm::dbgs() << "  -> Extracted loop bounds and step.\n");

            ValueRange inits = adaptor.getInits();
            LLVM_DEBUG(llvm::dbgs() << "  -> Extracted " << inits.size() << " converted iter_args (inits).\n");

            auto scfForOp = rewriter.create<scf::ForOp>(loc, lowerBound, upperBound, step, inits);
            LLVM_DEBUG(llvm::dbgs() << "  -> Created new scf.for operation.\n");

            Region& oldRegion = op.getRegion();
            Region& newRegion = scfForOp.getRegion();
            if (!newRegion.empty()) {
                rewriter.eraseBlock(&newRegion.front());
            }
            rewriter.inlineRegionBefore(oldRegion, newRegion, newRegion.end());
            LLVM_DEBUG(llvm::dbgs() << "  -> Inlined old region into new scf.for region.\n");

            Block& body = newRegion.front();
            for (size_t i = 1; i < body.getNumArguments(); ++i) {
                Type oldType = body.getArgument(i).getType();
                Type newIterType = inits[i - 1].getType();
                body.getArgument(i).setType(newIterType);
                LLVM_DEBUG(llvm::dbgs() << "  -> Updated Block Arg " << i << " type: "
                                        << oldType << " -> " << newIterType << "\n");
            }

            rewriter.replaceOp(op, scfForOp.getResults());
            LLVM_DEBUG(llvm::dbgs() << "[AffineForToSCFForPattern] Conversion complete.\n");
            return success();
        }
    };

    struct AffineLoadToSCFHELoadPattern : public OpConversionPattern<affine::AffineLoadOp> {
        using OpConversionPattern::OpConversionPattern;

        LogicalResult matchAndRewrite(affine::AffineLoadOp op, OpAdaptor adaptor,
                                      ConversionPatternRewriter& rewriter) const override {
            LLVM_DEBUG(llvm::dbgs() << "\n[AffineLoadToSCFHELoadPattern] Inspecting affine.load at " << op.getLoc() << "\n");

            Value memref = adaptor.getMemref();

            if (!isa<SCFHECipherType>(memref.getType())) {
                LLVM_DEBUG(llvm::dbgs() << "  -> [Failed] The source memref is NOT a CipherType.\n");
                return failure();
            }

            auto cipherType = cast<SCFHECipherType>(memref.getType());
            auto indices = adaptor.getIndices();
            if (indices.empty()) {
                LLVM_DEBUG(llvm::dbgs() << "  -> [Failed] No indices found for affine.load.\n");
                return failure();
            }

            bool isBatchingLoad = false;
            if (auto cst = indices[0].getDefiningOp<arith::ConstantIndexOp>()) {
                if (cst.value() == 0 &&
                    (cipherType.getPlaintextCount() > 1 || cipherType.getPlaintextCount() == ShapedType::kDynamic)) {
                    isBatchingLoad = true;
                }
            }

            if (isBatchingLoad) {
                LLVM_DEBUG(llvm::dbgs() << "  -> [SIMD Mode] Detected Batching (Index 0). Bypassing load to use FULL cipher directly: " << cipherType << "\n");
                rewriter.replaceOp(op, memref);
            } else {
                Type resultType = SCFHECipherType::get(rewriter.getContext(), 1, IntegerType::get(rewriter.getContext(), 64));
                LLVM_DEBUG(llvm::dbgs() << "  -> [Scalar Mode] Normal indexing. Creating scfhe.load for single element: " << resultType << "\n");
                rewriter.replaceOpWithNewOp<SCFHELoadOp>(op, resultType, memref, indices[0]);
            }

            return success();
        }
    };

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
                    Type expectedRetType = getTypeConverter()->convertType(op.getMemRef().getType());
                    Value retVal = val;

                    if (expectedRetType && retVal.getType() != expectedRetType) {
                        LLVM_DEBUG(llvm::dbgs() << "  -> Type mismatch for return! Route B: Inserting Cast to satisfy signature.\n");
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

    struct PlaintextBatchingPattern : public OpConversionPattern<affine::AffineForOp> {
        using OpConversionPattern::OpConversionPattern;

        LogicalResult matchAndRewrite(affine::AffineForOp op, OpAdaptor adaptor,
                                      ConversionPatternRewriter& rewriter) const override {
            auto funcOp = op->getParentOfType<func::FuncOp>();
            if (!funcOp || !funcOp->hasAttr("scfhe.crypto"))
                return failure();

            LLVM_DEBUG(llvm::dbgs() << "\n[PlaintextBatchingPattern] Inspecting AffineForOp at " << op.getLoc() << "\n");

            if (op.getNumIterOperands() > 0) {
                LLVM_DEBUG(llvm::dbgs() << "  -> [Rejected] Loop has iter_args (data dependency). Delegating to Route 1.\n");
                return failure();
            }

            LLVM_DEBUG(llvm::dbgs() << "  -> [Accepted] No data dependency found. Starting flattening (batching)...\n");

            Block* body = op.getBody();

            auto yieldOp = cast<affine::AffineYieldOp>(body->getTerminator());
            SmallVector<Value> yieldedValues;
            for (Value operand : yieldOp.getOperands()) {
                yieldedValues.push_back(operand);
            }
            LLVM_DEBUG(llvm::dbgs() << "  -> Extracted " << yieldedValues.size() << " yielded values.\n");
            rewriter.eraseOp(yieldOp);

            SmallVector<Value, 4> replValues;
            Value c0 = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
            replValues.push_back(c0);
            LLVM_DEBUG(llvm::dbgs() << "  -> Replaced loop induction variable with constant 0.\n");

            rewriter.inlineBlockBefore(body, op, replValues);
            LLVM_DEBUG(llvm::dbgs() << "  -> Inlined loop body successfully.\n");

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
    // Phase 5: Cleanup Patterns
    // ============================================================================

    struct RemoveDecryptCastPattern : public OpRewritePattern<SCFHEDecryptOp> {
        using OpRewritePattern::OpRewritePattern;

        LogicalResult matchAndRewrite(SCFHEDecryptOp op, PatternRewriter& rewriter) const override {
            if (auto castOp = op.getOperand().getDefiningOp<UnrealizedConversionCastOp>()) {
                Value realInput = castOp.getInputs()[0];

                if (auto cipherType = dyn_cast<SCFHECipherType>(realInput.getType())) {
                    LLVM_DEBUG(llvm::dbgs() << "[Cleanup] Melting away UnrealizedConversionCastOp before decrypt.\n");

                    auto oldMemRefType = cast<MemRefType>(op.getType());
                    auto newMemRefType = MemRefType::get({cipherType.getPlaintextCount()}, oldMemRefType.getElementType());

                    auto newDecrypt = rewriter.create<SCFHEDecryptOp>(op.getLoc(), newMemRefType, realInput);
                    rewriter.replaceOpWithNewOp<memref::CastOp>(op, oldMemRefType, newDecrypt);

                    return success();
                }
            }
            return failure();
        }
    };

    struct RemoveLoadCastPattern : public OpRewritePattern<SCFHELoadOp> {
        using OpRewritePattern::OpRewritePattern;

        LogicalResult matchAndRewrite(SCFHELoadOp op, PatternRewriter& rewriter) const override {
            Value arrayOperand = op.getOperand(0);

            if (auto castOp = arrayOperand.getDefiningOp<SCFHECastOp>()) {
                LLVM_DEBUG(llvm::dbgs() << "[Cleanup] Melting away SCFHECastOp before load.\n");

                Value realArray = castOp.getOperand();
                Value index = op.getOperand(1);

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

            auto allocOp = target.getDefiningOp<memref::AllocOp>();
            if (!allocOp) {
                return failure();
            }

            if (source.getType() != target.getType()) {
                return failure();
            }

            for (Operation* user : target.getUsers()) {
                if (user == op)
                    continue;
                if (isa<memref::DeallocOp>(user))
                    continue;

                Operation* ancestor = user;
                while (ancestor && ancestor->getBlock() != op->getBlock()) {
                    ancestor = ancestor->getParentOp();
                }

                if (ancestor && ancestor->isBeforeInBlock(op)) {
                    return failure();
                }

                if (isa<memref::StoreOp, affine::AffineStoreOp>(user)) {
                    return failure();
                }
                if (auto otherCopy = dyn_cast<memref::CopyOp>(user)) {
                    if (otherCopy.getTarget() == target) {
                        return failure();
                    }
                }
            }

            rewriter.replaceAllUsesWith(target, source);
            rewriter.eraseOp(op);
            rewriter.eraseOp(allocOp);

            return success();
        }
    };

    struct SCFFullUnrollPattern : public OpRewritePattern<scf::ForOp> {
        using OpRewritePattern<scf::ForOp>::OpRewritePattern;

        LogicalResult matchAndRewrite(scf::ForOp op, PatternRewriter& rewriter) const override {
            auto lbConst = op.getLowerBound().getDefiningOp<arith::ConstantIndexOp>();
            auto ubConst = op.getUpperBound().getDefiningOp<arith::ConstantIndexOp>();
            auto stepConst = op.getStep().getDefiningOp<arith::ConstantIndexOp>();

            if (!lbConst || !ubConst || !stepConst) {
                return failure();
            }

            int64_t lb = lbConst.value();
            int64_t ub = ubConst.value();
            int64_t step = stepConst.value();
            int64_t tripCount = (ub - lb + step - 1) / step;

            if (tripCount > 32) {
                return failure();
            }

            LLVM_DEBUG(llvm::dbgs() << "  -> [Cleanup] Fully unrolling scf.for with trip count: " << tripCount << "\n");

            SmallVector<Value> currentIterArgs = op.getInitArgs();
            Location loc = op.getLoc();

            for (int64_t i = 0; i < tripCount; ++i) {
                int64_t ivVal = lb + i * step;

                IRMapping mapping;
                Value ivConst = rewriter.create<arith::ConstantIndexOp>(loc, ivVal);

                mapping.map(op.getInductionVar(), ivConst);
                for (auto [blockArg, iterVal] : llvm::zip(op.getRegionIterArgs(), currentIterArgs)) {
                    mapping.map(blockArg, iterVal);
                }

                for (Operation& bodyOp : op.getBody()->without_terminator()) {
                    rewriter.clone(bodyOp, mapping);
                }

                auto yieldOp = cast<scf::YieldOp>(op.getBody()->getTerminator());
                currentIterArgs.clear();
                for (Value operand : yieldOp.getOperands()) {
                    currentIterArgs.push_back(mapping.lookupOrDefault(operand));
                }
            }

            rewriter.replaceOp(op, currentIterArgs);
            return success();
        }
    };

} // namespace

void mlir::libra::scfhe::populateSCFHEConversionPatterns(RewritePatternSet& patterns, TypeConverter& typeConverter, MLIRContext* ctx, const ArgAnalysis& analysis) {
    patterns.add<FuncSignaturePattern>(typeConverter, ctx, analysis);
    patterns.add<BinaryOpRewritePattern<arith::AddFOp, SCFHEAddOp>>(typeConverter, ctx);
    patterns.add<BinaryOpRewritePattern<arith::SubFOp, SCFHESubOp>>(typeConverter, ctx);
    patterns.add<BinaryOpRewritePattern<arith::MulFOp, SCFHEMultOp>>(typeConverter, ctx);
    patterns.add<BinaryOpRewritePattern<arith::DivFOp, SCFHEDivOp>>(typeConverter, ctx);
    patterns.add<ExpRewritePattern>(typeConverter, ctx);

    patterns.add<ReduceAddPattern>(typeConverter, ctx);

    patterns.add<PlaintextBatchingPattern>(typeConverter, ctx);
    patterns.add<AffineForToSCFForPattern>(typeConverter, ctx);
    patterns.add<AffineYieldToSCFYieldPattern>(typeConverter, ctx);

    patterns.add<AffineLoadToSCFHELoadPattern>(typeConverter, ctx);
    patterns.add<StoreToReturnPattern>(typeConverter, ctx);

    patterns.add<RemoveRedundantCopy>(ctx);
}

void mlir::libra::scfhe::populateSCFHECleanupPatterns(RewritePatternSet& patterns, MLIRContext* ctx) {
    patterns.add<SCFFullUnrollPattern>(ctx);
    patterns.add<RemoveRedundantCopy>(ctx);
    patterns.add<RemoveDecryptCastPattern>(ctx);
    patterns.add<RemoveLoadCastPattern>(ctx);
}