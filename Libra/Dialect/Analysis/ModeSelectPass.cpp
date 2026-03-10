#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#include "ModeSelectPass.h"
#include "SIMDOps.h"
#include "SISDOps.h"

#include "CostModel.h"
#include "ModeRewriter.h"

#include <queue>
#include <vector>
#include <string>

#define DEBUG_TYPE "mode-select"

using namespace mlir;
using namespace mlir::libra::mdsel;

static llvm::cl::opt<std::string> CostTablePath(
    "mode-select-cost-table",
    llvm::cl::desc("Path to the cost table JSON file for mode selection"),
    llvm::cl::value_desc("path"),
    llvm::cl::init(""));

namespace mlir::libra::mdsel {

#define GEN_PASS_DEF_CONVERTTOMDSELPASS
#include "ModeSelectPass.h.inc"

    namespace {

        constexpr int MAX_WAVE_ITERATIONS = 100;

        class ConvertToModeSelectIR : public impl::ConvertToMDSELPassBase<ConvertToModeSelectIR> {
        public:
            using impl::ConvertToMDSELPassBase<ConvertToModeSelectIR>::ConvertToMDSELPassBase;

            StringRef getOptionFilePath() {
                if (CostTablePath.empty())
                    return "cost_table.json";
                return CostTablePath;
            }

            std::string getOpName(Operation* op) {
                return op->getName().getStringRef().str();
            }

            void runOnOperation() override {
                ModuleOp module = getOperation();
                CostModel costModel(getOptionFilePath());

                for (func::FuncOp func : module.getOps<func::FuncOp>()) {
                    if (func.getName() != "main")
                        continue;

                    LLVM_DEBUG(llvm::dbgs() << "\n[ModeSel] Processing Function: @" << func.getName() << "\n");

                    SmallVector<Operation*, 64> ops;
                    func.walk([&](Operation* op) {
                        if (!op || !op->getDialect())
                            return;
                        StringRef ns = op->getDialect()->getNamespace();
                        if (ns == "simd" || ns == "sisd")
                            ops.push_back(op);
                    });
                    if (ops.empty())
                        continue;

                    llvm::DenseMap<Operation*, SmallVector<Operation*, 4>> preds, succs;
                    for (Operation* op : ops) {
                        for (Value v : op->getOperands()) {
                            Operation* defOp = nullptr;
                            if (auto* def = v.getDefiningOp()) {
                                defOp = def;
                            } else if (auto arg = dyn_cast<BlockArgument>(v)) {
                                if (auto* ownerOp = arg.getOwner()->getParentOp()) {
                                    defOp = ownerOp;
                                }
                            }
                            if (defOp && std::find(ops.begin(), ops.end(), defOp) != ops.end()) {
                                preds[op].push_back(defOp);
                                succs[defOp].push_back(op);
                            }
                        }
                    }

                    SmallVector<Operation*, 64> topo;
                    {
                        llvm::DenseMap<Operation*, int> indeg;
                        std::queue<Operation*> q;
                        for (Operation* op : ops) {
                            indeg[op] = preds[op].size();
                            if (indeg[op] == 0)
                                q.push(op);
                        }
                        while (!q.empty()) {
                            Operation* x = q.front();
                            q.pop();
                            topo.push_back(x);
                            for (auto* y : succs[x])
                                if (--indeg[y] == 0)
                                    q.push(y);
                        }
                    }

                    llvm::DenseMap<Operation*, NodeInfo> info;
                    for (Operation* op : ops) {
                        info[op].mode = Mode::SIMD;
                        info[op].finalLevel = MAX_SIMD_LEVEL;
                        int64_t count = 8;
                        if (op->getNumResults() > 0) {
                            if (auto t = dyn_cast<simd::SIMDCipherType>(op->getResult(0).getType()))
                                count = t.getPlaintextCount();
                            else if (auto t = dyn_cast<sisd::SISDCipherType>(op->getResult(0).getType()))
                                count = t.getPlaintextCount();
                        }
                        info[op].vectorCount = count;
                    }

                    auto updateNodeState = [&](Operation* op) {
                        auto& nd = info[op];
                        int l_in = MAX_SIMD_LEVEL;
                        if (!preds[op].empty()) {
                            for (auto* p : preds[op])
                                l_in = std::min(l_in, info[p].finalLevel);
                        }
                        bool mustBoot = (l_in <= SLIM_BOOT_TRIGGER);
                        int opDepth = isa<simd::SIMDMultOp>(op) ? 1 : 0;

                        if (nd.mode == Mode::SIMD) {
                            nd.triggerBoot = mustBoot;
                            nd.finalLevel = (mustBoot ? BOOT_LEVEL : l_in) - opDepth;
                        } else {
                            nd.triggerBoot = false;
                            nd.finalLevel = l_in;
                        }
                    };

                    auto calcTotalCost = [&](Operation* op, Mode targetMode) -> double {
                        auto& nd = info[op];
                        int64_t vecCnt = nd.vectorCount;
                        double opCost = 0.0;

                        if (targetMode == Mode::SIMD) {
                            int l_in = MAX_SIMD_LEVEL;
                            for (auto* p : preds[op])
                                l_in = std::min(l_in, info[p].finalLevel);
                            bool mustBoot = (l_in <= SLIM_BOOT_TRIGGER);
                            opCost = costModel.getOpCost(op, Mode::SIMD, l_in, vecCnt);
                            if (mustBoot)
                                opCost += costModel.getBootCost(vecCnt);
                        } else {
                            opCost = costModel.getOpCost(op, Mode::SISD, 0, vecCnt);
                        }

                        double castCost = 0.0;
                        for (auto* p : preds[op]) {
                            castCost += costModel.getCastCost(info[p].mode, targetMode, info[p].vectorCount);
                        }

                        LLVM_DEBUG(llvm::dbgs() << "      Cost(" << getOpName(op) << ", "
                                                << (targetMode == Mode::SIMD ? "SIMD" : "SISD") << ") = "
                                                << "Op(" << opCost << ") + CastIn(" << castCost << ") = "
                                                << (opCost + castCost) << "\n");
                        return opCost + castCost;
                    };

                    auto propagateSISDDown = [&](Operation* startOp) {
                        std::queue<Operation*> q;
                        q.push(startOp);
                        LLVM_DEBUG(llvm::dbgs() << "      >>> [Propagate] Starting SISD Wave from " << getOpName(startOp) << "\n");

                        while (!q.empty()) {
                            Operation* curr = q.front();
                            q.pop();

                            for (auto* child : succs[curr]) {
                                auto& childNd = info[child];
                                if (childNd.mode == Mode::SISD)
                                    continue;

                                double costAsSIMD = calcTotalCost(child, Mode::SIMD);
                                double costAsSISD = calcTotalCost(child, Mode::SISD);

                                if (costAsSISD < costAsSIMD) {
                                    LLVM_DEBUG(llvm::dbgs() << "        -> Child " << getOpName(child) << " flips to SISD (Cascade)\n");
                                    childNd.mode = Mode::SISD;
                                    updateNodeState(child);
                                    q.push(child);
                                }
                            }
                        }
                    };

                    auto propagateSIMDUp = [&](Operation* startOp) {
                        std::queue<Operation*> q;
                        q.push(startOp);
                        LLVM_DEBUG(llvm::dbgs() << "      >>> [Propagate] Starting SIMD Wave from " << getOpName(startOp) << "\n");

                        while (!q.empty()) {
                            Operation* curr = q.front();
                            q.pop();

                            for (auto* p : preds[curr]) {
                                auto& pNd = info[p];
                                if (pNd.mode == Mode::SIMD)
                                    continue;

                                int l_in = MAX_SIMD_LEVEL;
                                for (auto* pp : preds[p])
                                    l_in = std::min(l_in, info[pp].finalLevel);
                                bool mustBoot = (l_in <= SLIM_BOOT_TRIGGER);

                                double costSelf_SIMD = costModel.getOpCost(p, Mode::SIMD, l_in, pNd.vectorCount);
                                if (mustBoot)
                                    costSelf_SIMD += costModel.getBootCost(pNd.vectorCount);
                                double costSelf_SISD = costModel.getOpCost(p, Mode::SISD, 0, pNd.vectorCount);

                                double cOut_SIMD = 0.0, cOut_SISD = 0.0;
                                for (auto* child : succs[p]) {
                                    cOut_SIMD += costModel.getCastCost(Mode::SIMD, info[child].mode, pNd.vectorCount);
                                    cOut_SISD += costModel.getCastCost(Mode::SISD, info[child].mode, pNd.vectorCount);
                                }

                                double total_SIMD = costSelf_SIMD + cOut_SIMD;
                                double total_SISD = costSelf_SISD + cOut_SISD;

                                if (total_SIMD < total_SISD) {
                                    LLVM_DEBUG(llvm::dbgs() << "        -> Parent " << getOpName(p) << " flips to SIMD (Cascade)\n");
                                    pNd.mode = Mode::SIMD;
                                    updateNodeState(p);
                                    q.push(p);
                                }
                            }
                        }
                    };

                    auto propagateSISDUp = [&](Operation* startOp) {
                        std::queue<Operation*> q;
                        q.push(startOp);
                        LLVM_DEBUG(llvm::dbgs() << "      >>> [Propagate] Starting SISD Wave UPWARDS from " << getOpName(startOp) << "\n");

                        while (!q.empty()) {
                            Operation* curr = q.front();
                            q.pop();

                            for (auto* p : preds[curr]) {
                                auto& pNd = info[p];
                                if (pNd.mode == Mode::SISD)
                                    continue;

                                double cSelf_SISD = costModel.getOpCost(p, Mode::SISD, 0, pNd.vectorCount);

                                int l_in = MAX_SIMD_LEVEL;
                                for (auto* pp : preds[p])
                                    l_in = std::min(l_in, info[pp].finalLevel);
                                bool mustBoot = (l_in <= SLIM_BOOT_TRIGGER);

                                double cSelf_SIMD = costModel.getOpCost(p, Mode::SIMD, l_in, pNd.vectorCount);
                                if (mustBoot)
                                    cSelf_SIMD += costModel.getBootCost(pNd.vectorCount);

                                double cOut_SIMD = 0.0, cOut_SISD = 0.0;
                                for (auto* child : succs[p]) {
                                    cOut_SIMD += costModel.getCastCost(Mode::SIMD, info[child].mode, pNd.vectorCount);
                                    cOut_SISD += costModel.getCastCost(Mode::SISD, info[child].mode, pNd.vectorCount);
                                }

                                double total_SIMD = cSelf_SIMD + cOut_SIMD;
                                double total_SISD = cSelf_SISD + cOut_SISD;

                                if (total_SISD < total_SIMD) {
                                    LLVM_DEBUG(llvm::dbgs() << "        -> Parent " << getOpName(p) << " flips to SISD (Cascade Up)\n");
                                    pNd.mode = Mode::SISD;
                                    updateNodeState(p);
                                    q.push(p);
                                }
                            }
                        }
                    };

                    bool changed = true;
                    int iter = 0;

                    while (changed && iter < MAX_WAVE_ITERATIONS) {
                        LLVM_DEBUG(llvm::dbgs() << "\n>>> --- Wave Iteration " << iter << " Start --- <<<\n");
                        changed = false;
                        iter++;

                        LLVM_DEBUG(llvm::dbgs() << "  [Phase 1] Top-Down Scan\n");
                        for (Operation* op : topo) {
                            auto& nd = info[op];
                            LLVM_DEBUG(llvm::dbgs() << "    [TD Eval] " << getOpName(op) << ":\n");

                            double total_SIMD = calcTotalCost(op, Mode::SIMD);
                            double total_SISD = calcTotalCost(op, Mode::SISD);

                            Mode newMode = nd.mode;
                            if (total_SISD < total_SIMD)
                                newMode = Mode::SISD;
                            else if (total_SIMD < total_SISD)
                                newMode = Mode::SIMD;

                            if (newMode != nd.mode) {
                                LLVM_DEBUG(llvm::dbgs() << "    [TD HIT] " << getOpName(op) << " -> " << (newMode == Mode::SIMD ? "SIMD" : "SISD") << "\n");
                                nd.mode = newMode;
                                updateNodeState(op);
                                changed = true;
                                if (nd.mode == Mode::SIMD)
                                    propagateSIMDUp(op);
                                goto start_bottom_up;
                            }
                            updateNodeState(op);
                        }

                    start_bottom_up:;
                        LLVM_DEBUG(llvm::dbgs() << "  [Phase 2] Bottom-Up Scan\n");
                        for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
                            Operation* op = *it;
                            auto& nd = info[op];
                            int64_t vecCnt = nd.vectorCount;

                            int l_in = MAX_SIMD_LEVEL;
                            for (auto* p : preds[op])
                                l_in = std::min(l_in, info[p].finalLevel);
                            bool mustBoot = (l_in <= SLIM_BOOT_TRIGGER);
                            int opDepth = isa<simd::SIMDMultOp>(op) ? 1 : 0;
                            int level_if_simd = (mustBoot ? BOOT_LEVEL : l_in) - opDepth;

                            double cSelf_SIMD = costModel.getOpCost(op, Mode::SIMD, l_in, vecCnt);
                            if (mustBoot)
                                cSelf_SIMD += costModel.getBootCost(vecCnt);
                            double cSelf_SISD = costModel.getOpCost(op, Mode::SISD, 0, vecCnt);

                            double cOut_SIMD = 0.0, cOut_SISD = 0.0;
                            for (auto* child : succs[op]) {
                                int child_l_in = level_if_simd;
                                bool childBoot = (child_l_in <= SLIM_BOOT_TRIGGER);
                                double costChild_SIMD = costModel.getOpCost(child, Mode::SIMD, child_l_in, info[child].vectorCount);
                                if (childBoot)
                                    costChild_SIMD += costModel.getBootCost(info[child].vectorCount);
                                double costChild_SISD = costModel.getOpCost(child, Mode::SISD, 0, info[child].vectorCount);

                                if (costChild_SISD < costChild_SIMD) {
                                    cOut_SIMD += costModel.getCastCost(Mode::SIMD, Mode::SISD, vecCnt);
                                    cOut_SISD += costModel.getCastCost(Mode::SISD, Mode::SISD, vecCnt);
                                } else if (costChild_SIMD < costChild_SISD) {
                                    cOut_SIMD += costModel.getCastCost(Mode::SIMD, Mode::SIMD, vecCnt);
                                    cOut_SISD += costModel.getCastCost(Mode::SISD, Mode::SIMD, vecCnt);
                                } else {
                                    cOut_SIMD += costModel.getCastCost(Mode::SIMD, Mode::SIMD, vecCnt);
                                    cOut_SISD += costModel.getCastCost(Mode::SISD, Mode::SISD, vecCnt);
                                }
                            }

                            LLVM_DEBUG(llvm::dbgs() << "    [BU Eval] " << getOpName(op) << ":\n"
                                                    << "      SIMD | Self: " << cSelf_SIMD << " + OutCast: " << cOut_SIMD << " = " << (cSelf_SIMD + cOut_SIMD) << "\n"
                                                    << "      SISD | Self: " << cSelf_SISD << " + OutCast: " << cOut_SISD << " = " << (cSelf_SISD + cOut_SISD) << "\n");

                            double total_SIMD = cSelf_SIMD + cOut_SIMD;
                            double total_SISD = cSelf_SISD + cOut_SISD;

                            Mode newMode = nd.mode;
                            if (total_SIMD < total_SISD)
                                newMode = Mode::SIMD;
                            else if (total_SISD < total_SIMD)
                                newMode = Mode::SISD;

                            if (newMode != nd.mode) {
                                LLVM_DEBUG(llvm::dbgs() << "    [BU HIT] " << getOpName(op) << " -> " << (newMode == Mode::SIMD ? "SIMD" : "SISD") << "\n");
                                nd.mode = newMode;
                                updateNodeState(op);
                                changed = true;
                                if (nd.mode == Mode::SIMD)
                                    propagateSIMDUp(op);
                                else
                                    propagateSISDUp(op);
                                break;
                            }
                        }
                    }

                    IRRewriter rewriter(module.getContext());
                    llvm::DenseMap<Value, Value> rewriteMap;

                    for (Operation* op : topo) {
                        rewriteOperation(op, info[op], rewriteMap, rewriter);
                    }

                    for (Operation* op : topo) {
                        if (op->use_empty())
                            rewriter.eraseOp(op);
                        else {
                            for (int i = 0; i < op->getNumResults(); ++i) {
                                Value oldVal = op->getResult(i);
                                if (rewriteMap.count(oldVal)) {
                                    Value newVal = rewriteMap[oldVal];
                                    if (oldVal.getType() != newVal.getType()) {
                                        rewriter.setInsertionPoint(op);
                                        Value castBack = rewriter.create<UnrealizedConversionCastOp>(op->getLoc(), oldVal.getType(), newVal).getResult(0);
                                        oldVal.replaceAllUsesWith(castBack);
                                    } else {
                                        oldVal.replaceAllUsesWith(newVal);
                                    }
                                }
                            }
                            if (op->use_empty())
                                rewriter.eraseOp(op);
                        }
                    }

                    fixLoopCasts(func);
                    attachGlobalConfig(module);
                }
            }
        };

    } // namespace
} // namespace mlir::libra::mdsel