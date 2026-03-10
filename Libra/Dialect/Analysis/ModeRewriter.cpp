// ModeRewriter.cpp
#include "ModeRewriter.h"
#include "SIMDOps.h"
#include "SISDOps.h"
#include "SIMDCommon.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "mode-select"

using namespace mlir;

// 将实现包裹在命名空间中，这样可以直接访问 mlir::libra 下的 simd 和 sisd
namespace mlir::libra::mdsel {

    // 辅助函数：剥离 Cast，放在这里可以直接被 fixLoopCasts 调用
    static Value peelCast(Value v) {
        if (auto castOp = v.getDefiningOp<sisd::SISDCastSISDCipherToSIMDCipherOp>())
            return castOp.getOperand();
        if (auto castOp = v.getDefiningOp<simd::SIMDCastSIMDCipherToSISDCipherOp>())
            return castOp.getOperand();
        if (auto castOp = v.getDefiningOp<UnrealizedConversionCastOp>())
            return castOp.getOperand(0);
        return v;
    }

    void rewriteOperation(Operation* op,
                          const NodeInfo& nd,
                          llvm::DenseMap<Value, Value>& rewriteMap,
                          IRRewriter& rewriter) {
        rewriter.setInsertionPoint(op);
        SmallVector<Value, 4> newOps;

        int reqInputLevel = nd.finalLevel;
        if (nd.triggerBoot) {
            reqInputLevel = SLIM_BOOT_TRIGGER;
        } else if (isa<simd::SIMDMultOp>(op) && nd.mode == Mode::SIMD) {
            reqInputLevel = nd.finalLevel + 1;
        }

        // 1. 准备操作数 (Cast Insertion)
        for (Value v : op->getOperands()) {
            Value nv = rewriteMap.lookup(v);
            if (!nv)
                nv = v;

            bool isDstSISD = (nd.mode == Mode::SISD);

            if (isa<simd::SIMDCipherType, sisd::SISDCipherType>(nv.getType())) {
                bool isSrcSISD = isa<sisd::SISDCipherType>(nv.getType());

                if (isSrcSISD && !isDstSISD) {
                    auto tt = cast<sisd::SISDCipherType>(nv.getType());
                    auto dstTy = simd::SIMDCipherType::get(op->getContext(), reqInputLevel, tt.getPlaintextCount(), tt.getElementType());
                    nv = rewriter.create<sisd::SISDCastSISDCipherToSIMDCipherOp>(op->getLoc(), dstTy, nv).getResult();
                } else if (!isSrcSISD && isDstSISD) {
                    auto st = cast<simd::SIMDCipherType>(nv.getType());
                    auto dstTy = sisd::SISDCipherType::get(op->getContext(), st.getPlaintextCount(), st.getElementType());
                    nv = rewriter.create<simd::SIMDCastSIMDCipherToSISDCipherOp>(op->getLoc(), dstTy, nv).getResult();
                }

                if (nd.triggerBoot && nd.mode == Mode::SIMD && isa<simd::SIMDCipherType>(nv.getType())) {
                    auto oldTy = cast<simd::SIMDCipherType>(nv.getType());
                    auto bootedTy = simd::SIMDCipherType::get(op->getContext(), BOOT_LEVEL, oldTy.getPlaintextCount(), oldTy.getElementType());
                    nv = rewriter.create<simd::SIMDBootOp>(op->getLoc(), bootedTy, nv).getResult();
                }
            }
            newOps.push_back(nv);
        }

        // 2. SIMD 对齐逻辑 (Alignment)
        if (nd.mode == Mode::SIMD && !newOps.empty()) {
            bool needsAlign = isa<simd::SIMDAddOp, simd::SIMDSubOp, simd::SIMDMinOp,
                                  simd::SIMDSelectOp, simd::SIMDCmpOp, simd::SIMDDivOp>(op);

            if (needsAlign) {
                int64_t minLevel = 10000;
                bool hasSIMD = false;
                for (Value v : newOps) {
                    if (auto st = dyn_cast<simd::SIMDCipherType>(v.getType())) {
                        minLevel = std::min(minLevel, (int64_t)st.getLevel());
                        hasSIMD = true;
                    }
                }
                if (hasSIMD) {
                    for (Value& v : newOps) {
                        if (auto st = dyn_cast<simd::SIMDCipherType>(v.getType())) {
                            if (st.getLevel() > minLevel) {
                                auto targetTy = simd::SIMDCipherType::get(
                                    op->getContext(), minLevel, st.getPlaintextCount(), st.getElementType());
                                v = rewriter.create<simd::SIMDModSwitchOp>(op->getLoc(), targetTy, v).getResult();
                            }
                        }
                    }
                }
            }
        }

        // 3. 创建新 Op
        Operation* newOp = nullptr;
        int64_t vecCnt = nd.vectorCount;
        Type resTy;

        if (nd.mode == Mode::SISD) {
            resTy = sisd::SISDCipherType::get(op->getContext(), vecCnt, rewriter.getI64Type());
        } else {
            int resultLevel = nd.finalLevel;
            if (isa<simd::SIMDAddOp, simd::SIMDSubOp, simd::SIMDMinOp, simd::SIMDSelectOp>(op)) {
                for (Value v : newOps) {
                    if (auto st = dyn_cast<simd::SIMDCipherType>(v.getType())) {
                        resultLevel = st.getLevel();
                        break;
                    }
                }
            }
            resTy = simd::SIMDCipherType::get(op->getContext(), resultLevel, vecCnt, rewriter.getI64Type());
        }

        if (isa<simd::SIMDDecryptOp, sisd::SISDDecryptOp>(op)) {
            if (isa<sisd::SISDCipherType>(newOps[0].getType()))
                newOp = rewriter.create<sisd::SISDDecryptOp>(op->getLoc(), op->getResultTypes(), newOps);
            else
                newOp = rewriter.create<simd::SIMDDecryptOp>(op->getLoc(), op->getResultTypes(), newOps);
        } else if (isa<simd::SIMDAddOp, sisd::SISDAddOp>(op)) {
            if (nd.mode == Mode::SISD)
                newOp = rewriter.create<sisd::SISDAddOp>(op->getLoc(), resTy, newOps);
            else
                newOp = rewriter.create<simd::SIMDAddOp>(op->getLoc(), resTy, newOps);
        } else if (isa<simd::SIMDSubOp, sisd::SISDSubOp>(op)) {
            if (nd.mode == Mode::SISD)
                newOp = rewriter.create<sisd::SISDSubOp>(op->getLoc(), resTy, newOps);
            else
                newOp = rewriter.create<simd::SIMDSubOp>(op->getLoc(), resTy, newOps);
        } else if (isa<simd::SIMDMultOp>(op)) {
            newOp = rewriter.create<simd::SIMDMultOp>(op->getLoc(), resTy, newOps);
        } else if (isa<simd::SIMDMinOp, sisd::SISDMinOp>(op)) {
            if (nd.mode == Mode::SISD)
                newOp = rewriter.create<sisd::SISDMinOp>(op->getLoc(), resTy, newOps);
            else
                newOp = rewriter.create<simd::SIMDMinOp>(op->getLoc(), resTy, newOps);
        } else if (isa<simd::SIMDEncryptOp, sisd::SISDEncryptOp>(op)) {
            if (nd.mode == Mode::SISD)
                newOp = rewriter.create<sisd::SISDEncryptOp>(op->getLoc(), resTy, newOps);
            else
                newOp = rewriter.create<simd::SIMDEncryptOp>(op->getLoc(), resTy, newOps);
        } else if (isa<simd::SIMDSelectOp>(op)) {
            newOp = rewriter.create<simd::SIMDSelectOp>(op->getLoc(), resTy, newOps[0], newOps[1], newOps[2]);
        } else if (isa<simd::SIMDCmpOp>(op)) {
            auto pred = cast<simd::SIMDCmpOp>(op).getPredicate();
            newOp = rewriter.create<simd::SIMDCmpOp>(op->getLoc(), resTy, newOps[0], newOps[1], pred);
        } else if (isa<simd::SIMDDivOp>(op)) {
            if (nd.mode == Mode::SISD)
                newOp = rewriter.create<sisd::SISDDivOp>(op->getLoc(), resTy, newOps);
            else
                newOp = rewriter.create<simd::SIMDDivOp>(op->getLoc(), resTy, newOps);
        } else if (isa<simd::SIMDLoadOp, sisd::SISDLoadOp>(op)) {
            if (nd.mode == Mode::SISD)
                newOp = rewriter.create<sisd::SISDLoadOp>(op->getLoc(), resTy, newOps[0], newOps[1]);
            else
                newOp = rewriter.create<simd::SIMDLoadOp>(op->getLoc(), resTy, newOps[0], newOps[1]);
        } else if (isa<simd::SIMDStoreOp, sisd::SISDStoreOp>(op)) {
            if (nd.mode == Mode::SISD)
                rewriter.create<sisd::SISDStoreOp>(op->getLoc(), newOps[0], newOps[1]);
            else
                rewriter.create<simd::SIMDStoreOp>(op->getLoc(), newOps[0], newOps[1]);
        } else if (isa<simd::SIMDReduceAddOp, sisd::SISDReduceAddOp>(op)) {
            if (nd.mode == Mode::SISD)
                newOp = rewriter.create<sisd::SISDReduceAddOp>(op->getLoc(), resTy, newOps[0]);
            else
                newOp = rewriter.create<simd::SIMDReduceAddOp>(op->getLoc(), resTy, newOps[0]);
        }

        if (newOp)
            rewriteMap[op->getResult(0)] = newOp->getResult(0);
    }

    void fixLoopCasts(func::FuncOp func) {
        SmallVector<scf::ForOp> loopsToFix;
        func.walk([&](scf::ForOp forOp) { if (forOp.getNumResults() > 0) loopsToFix.push_back(forOp); });

        for (scf::ForOp forOp : loopsToFix) {
            IRRewriter r(forOp.getContext());
            r.setInsertionPoint(forOp);
            SmallVector<Value> newInits;
            bool needsFix = false;
            for (Value init : forOp.getInitArgs()) {
                Value realInit = peelCast(init);
                newInits.push_back(realInit);
                if (realInit != init)
                    needsFix = true;
            }
            if (!needsFix)
                continue;

            auto newLoop = r.create<scf::ForOp>(forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(), forOp.getStep(), newInits);
            Region& oldR = forOp.getRegion();
            Region& newR = newLoop.getRegion();
            Block* oldB = &oldR.front();
            Block* newB = &newR.front();

            newB->getArgument(0).setType(oldB->getArgument(0).getType());
            for (size_t i = 0; i < newInits.size(); ++i)
                newB->getArgument(i + 1).setType(newInits[i].getType());
            r.mergeBlocks(oldB, newB, newB->getArguments());

            auto y = cast<scf::YieldOp>(newB->getTerminator());
            r.setInsertionPoint(y);
            SmallVector<Value> ny;
            for (Value v : y.getOperands())
                ny.push_back(peelCast(v));
            r.replaceOpWithNewOp<scf::YieldOp>(y, ny);

            r.setInsertionPointAfter(newLoop);
            for (auto [oldRes, newRes] : llvm::zip(forOp.getResults(), newLoop.getResults())) {
                if (oldRes.getType() != newRes.getType()) {
                    Value cb = r.create<UnrealizedConversionCastOp>(newLoop.getLoc(), oldRes.getType(), newRes).getResult(0);
                    oldRes.replaceAllUsesWith(cb);
                } else {
                    oldRes.replaceAllUsesWith(newRes);
                }
            }
            r.eraseOp(forOp);
        }
    }

    void attachGlobalConfig(ModuleOp module) {
        LLVM_DEBUG(llvm::dbgs() << "\n[Post-Processing] Attaching Global FlyHE Config...\n");
        bool hasSIMD = false, hasSISD = false, hasBoot = false;
        int64_t maxCount = 1;
        int32_t initLevel = 1;

        module.walk([&](Operation* op) {
            StringRef ns = op->getDialect()->getNamespace();
            if (ns == "simd")
                hasSIMD = true;
            if (ns == "sisd")
                hasSISD = true;
            if (isa<simd::SIMDBootOp>(op))
                hasBoot = true;

            auto check = [&](Type t) {
                if (auto st = dyn_cast<simd::SIMDCipherType>(t))
                    maxCount = std::max(maxCount, st.getPlaintextCount());
                else if (auto st = dyn_cast<sisd::SISDCipherType>(t))
                    maxCount = std::max(maxCount, st.getPlaintextCount());
            };
            for (auto t : op->getResultTypes())
                check(t);

            if (isa<simd::SIMDEncryptOp, sisd::SISDCastSISDCipherToSIMDCipherOp>(op)) {
                if (auto st = dyn_cast<simd::SIMDCipherType>(op->getResult(0).getType()))
                    initLevel = std::max<int32_t>(initLevel, st.getLevel());
            }
        });

        module.walk([&](func::FuncOp f) {
            for (auto t : f.getArgumentTypes())
                if (auto st = dyn_cast<simd::SIMDCipherType>(t))
                    initLevel = std::max<int32_t>(initLevel, st.getLevel());
        });

        std::string modeStr = (hasSIMD && hasSISD) ? "CROSS" : (hasSISD ? "SISD" : "SIMD");
        int64_t logn = std::max<int64_t>(1, std::ceil(std::log2(maxCount)));

        OpBuilder builder(module.getContext());
        SmallVector<NamedAttribute, 5> attrs;
        attrs.push_back(builder.getNamedAttr("mode", builder.getStringAttr(modeStr)));
        if (modeStr != "SISD") {
            attrs.push_back(builder.getNamedAttr("logN", builder.getI64IntegerAttr(16)));
            attrs.push_back(builder.getNamedAttr("logn", builder.getI64IntegerAttr(logn)));
            attrs.push_back(builder.getNamedAttr("remaining_levels", builder.getI32IntegerAttr(initLevel)));
            attrs.push_back(builder.getNamedAttr("bootstrapping_enabled", builder.getBoolAttr(hasBoot)));
        }
        module->setAttr("he.config", builder.getDictionaryAttr(attrs));

        LLVM_DEBUG(llvm::dbgs() << "[ModeSel] Finished. Mode=" << modeStr << "\n");
    }

} // namespace mlir::libra::mdsel