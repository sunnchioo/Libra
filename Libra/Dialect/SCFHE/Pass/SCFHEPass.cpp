// SCFHEPass.cpp
#include "SCFHEPass.h"
#include "SCFHEAnalysis.h"
#include "SCFHETypeConverter.h"
#include "SCFHEPatterns.h"
#include "SCFHEDialect.h"
#include "SCFHEOps.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "scfhe-pass"

using namespace mlir;
using namespace mlir::libra::scfhe;

namespace mlir::libra::scfhe {

#define GEN_PASS_DEF_CONVERTTOSCFHEPASS
#include "SCFHEPass.h.inc"

    namespace {

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
                RewritePatternSet patterns(ctx);
                ConversionTarget target(*ctx);

                target.addLegalDialect<SCFHEDialect, arith::ArithDialect, func::FuncDialect, math::MathDialect, scf::SCFDialect>();
                target.addLegalOp<mlir::UnrealizedConversionCastOp>();

                populateSCFHEConversionPatterns(patterns, typeConverter, ctx, argAnalysis);

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

                PassManager pm(ctx);
                pm.addPass(createInlinerPass());

                if (failed(pm.run(module))) {
                    LLVM_DEBUG(llvm::dbgs() << "!!! Inlining Failed !!!\n");
                    signalPassFailure();
                    return;
                }

                // --- Phase 5: Cleanup & Shape Inference (形状推断与清理) ---
                LLVM_DEBUG(llvm::dbgs() << "\n--- Executing Phase 5: Cleanup & Shape Inference ---\n");
                RewritePatternSet cleanupPatterns(ctx);

                populateSCFHECleanupPatterns(cleanupPatterns, ctx);

                if (failed(applyPatternsGreedily(module, std::move(cleanupPatterns)))) {
                    signalPassFailure();
                    return;
                }

                LLVM_DEBUG(llvm::dbgs() << "====== SCFHE Pass Finished ======\n");
            }
        };

    } // namespace
} // namespace mlir::libra::scfhe