#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/IR/Verifier.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#include "ModeSelectPass.h"
#include "SIMDOps.h"
#include "SISDOps.h"
#include "SIMDCommon.h"

#include <algorithm>
#include <limits>
#include <queue>
#include <set>
#include <vector>
#include <string>
#include <memory>

// [新增] 定义 DEBUG_TYPE，用于命令行 --debug-only=mode-select 触发
#define DEBUG_TYPE "mode-select"

using namespace mlir;
using namespace mlir::libra;

static llvm::cl::opt<std::string> CostTablePath(
    "mode-select-cost-table",
    llvm::cl::desc("Path to the cost table JSON file for mode selection"),
    llvm::cl::value_desc("path"),
    llvm::cl::init(""));

namespace mlir::libra::mdsel {

#define GEN_PASS_DEF_CONVERTTOMDSELPASS
#include "ModeSelectPass.h.inc"

    namespace {

        // ============================================================
        // Constants
        // ============================================================
        enum class Mode { SIMD,
                          SISD };
        constexpr double INF_COST = 1e15;
        constexpr int MAX_SIMD_LEVEL = 31;
        constexpr int BOOT_LEVEL = 17;       // Bootstrapping 重置后的 Level
        constexpr int SLIM_BOOT_TRIGGER = 3; // 触发 Slim Boot 的阈值
        constexpr int MAX_ITERATIONS = 20;   // 最大迭代次数

        // ============================================================
        // Cost Model
        // ============================================================
        struct CostModel {
            llvm::StringMap<llvm::DenseMap<int64_t, double>> costTable;

            CostModel(StringRef jsonFilename) {
                // 简单的 JSON 加载逻辑
                auto fileOrErr = llvm::MemoryBuffer::getFile(jsonFilename);
                if (auto ec = fileOrErr.getError()) {
                    LLVM_DEBUG(llvm::dbgs() << "[ModeSel] Error opening file: " << ec.message() << "\n");
                    return;
                }
                llvm::Expected<llvm::json::Value> rootOrErr = llvm::json::parse(fileOrErr.get()->getBuffer());
                if (!rootOrErr) {
                    LLVM_DEBUG(llvm::dbgs() << "[ModeSel] JSON parse error\n");
                    return;
                }
                auto* obj = rootOrErr->getAsObject();
                if (!obj)
                    return;
                auto* costs = obj->getObject("costs");
                if (!costs)
                    return;

                // 解析逻辑 (假设 JSON 结构匹配)
                for (auto& modePair : *costs) {
                    StringRef mode = modePair.first;
                    auto* modeObj = modePair.second.getAsObject();
                    if (!modeObj)
                        continue;
                    for (auto& opPair : *modeObj) {
                        StringRef opname = opPair.first;
                        auto* opObj = opPair.second.getAsObject();
                        if (!opObj)
                            continue;
                        if (auto* lat = opObj->getObject("latency")) {
                            std::string key = (mode + "." + opname).str();
                            for (auto& kv : *lat) {
                                long long idx = 0;
                                if (!llvm::StringRef(kv.first).getAsInteger(10, idx) && kv.second.getAsNumber()) {
                                    costTable[key][idx] = *kv.second.getAsNumber();
                                }
                            }
                        }
                    }
                }
                LLVM_DEBUG(llvm::dbgs() << "[ModeSel] CostTable loaded successfully from " << jsonFilename << "\n");
            }

            double lookup(StringRef key, int64_t idx) const {
                auto it = costTable.find(key);
                if (it == costTable.end())
                    return INF_COST;
                auto jt = it->second.find(idx);
                if (jt == it->second.end()) {
                    if (!it->second.empty())
                        return it->second.begin()->second;
                    return INF_COST;
                }
                return jt->second;
            }

            std::string getOpKey(Operation* op, Mode m) const {
                std::string prefix = (m == Mode::SIMD) ? "simd." : "sisd.";
                if (isa<simd::SIMDAddOp, sisd::SISDAddOp>(op))
                    return prefix + "add";
                if (isa<simd::SIMDSubOp, sisd::SISDSubOp>(op))
                    return prefix + "sub";
                if (isa<simd::SIMDMultOp>(op))
                    return prefix + "mult";
                if (isa<simd::SIMDMinOp, sisd::SISDMinOp>(op))
                    return prefix + "min";
                if (isa<simd::SIMDEncryptOp, sisd::SISDEncryptOp>(op))
                    return prefix + "encrypt";
                if (isa<simd::SIMDDecryptOp, sisd::SISDDecryptOp>(op))
                    return prefix + "decrypt";
                return "";
            }

            double getOpCost(Operation* op, Mode m, int64_t param) const {
                if (isa<simd::SIMDMultOp>(op) && m == Mode::SISD)
                    return INF_COST; // Mult 必须 SIMD
                std::string key = getOpKey(op, m);
                if (key.empty())
                    return 0.0;
                return lookup(key, param);
            }

            double getBootCost() const { return lookup("simd.boot", BOOT_LEVEL); }

            double getCastCost(Mode from, Mode to, int64_t vecCnt) const {
                if (from == to)
                    return 0.0;
                std::string key = (from == Mode::SIMD) ? "simd.cast_to_sisd" : "sisd.cast_to_simd";
                return lookup(key, vecCnt);
            }
        };

        // ============================================================
        // Node Info State
        // ============================================================
        struct NodeInfo {
            Mode mode = Mode::SIMD;
            int finalLevel = MAX_SIMD_LEVEL;
            bool triggerBoot = false; // 是否在此算子前触发 Boot
            int64_t vectorCount = 8;
        };

        class ConvertToModeSelectIR : public impl::ConvertToMDSELPassBase<ConvertToModeSelectIR> {
        public:
            using impl::ConvertToMDSELPassBase<ConvertToModeSelectIR>::ConvertToMDSELPassBase;

            StringRef getOptionFilePath() {
                if (!CostTablePath.empty()) {
                    return CostTablePath;
                }
                return "cost_table.json";
            }

            // Helper for printing Op names concisely
            std::string getOpName(Operation* op) {
                return op->getName().getStringRef().str();
            }

            void runOnOperation() override {
                ModuleOp module = getOperation();
                CostModel costModel(getOptionFilePath());

                for (func::FuncOp func : module.getOps<func::FuncOp>()) {

                    if (func.getName() != "main") {
                        continue;
                    }

                    LLVM_DEBUG({
                        llvm::dbgs() << "\n=================================================================\n";
                        llvm::dbgs() << "[ModeSel] Processing Function: @" << func.getName() << "\n";
                        llvm::dbgs() << "=================================================================\n";
                    });

                    // ------------------------------------------------------------
                    // 1. Build Graph & Topological Sort
                    // ------------------------------------------------------------
                    SmallVector<Operation*, 64> ops;
                    func.walk([&](Operation* op) {
                        if (!op || !op->getDialect())
                            return;
                        StringRef ns = op->getDialect()->getNamespace();
                        if (ns == "simd" || ns == "sisd")
                            ops.push_back(op);
                    });
                    if (ops.empty()) {
                        LLVM_DEBUG(llvm::dbgs() << "  -> No SIMD/SISD ops found, skipping.\n");
                        continue;
                    }

                    llvm::DenseMap<Operation*, SmallVector<Operation*, 4>> preds, succs;
                    for (Operation* op : ops) {
                        for (Value v : op->getOperands()) {
                            if (auto* def = v.getDefiningOp()) {
                                if (std::find(ops.begin(), ops.end(), def) != ops.end()) {
                                    preds[op].push_back(def);
                                    succs[def].push_back(op);
                                }
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

                    LLVM_DEBUG({
                        llvm::dbgs() << "\n[Graph] Topological Sort Result (" << topo.size() << " nodes):\n";
                        for (Operation* op : topo) {
                            llvm::dbgs() << "  - " << getOpName(op) << " (Preds: " << preds[op].size() << ")\n";
                        }
                    });

                    // ------------------------------------------------------------
                    // 2. Initialize State
                    // ------------------------------------------------------------
                    llvm::DenseMap<Operation*, NodeInfo> info;
                    for (Operation* op : ops) {
                        info[op].mode = Mode::SIMD;
                        info[op].finalLevel = MAX_SIMD_LEVEL;
                        info[op].triggerBoot = false;

                        int64_t count = 8;
                        if (op->getNumResults() > 0) {
                            if (auto t = dyn_cast<simd::SIMDCipherType>(op->getResult(0).getType()))
                                count = t.getPlaintextCount();
                            else if (auto t = dyn_cast<sisd::SISDCipherType>(op->getResult(0).getType()))
                                count = t.getPlaintextCount();
                        }
                        info[op].vectorCount = count;
                    }

                    // ------------------------------------------------------------
                    // 3. Iterative Optimization Loop
                    // ------------------------------------------------------------
                    bool changed = true;
                    int iter = 0;

                    while (changed && iter < MAX_ITERATIONS) {
                        LLVM_DEBUG(llvm::dbgs() << "\n>>> --- Iteration " << iter << " Start --- <<<\n");
                        changed = false;
                        iter++;

                        // === Phase 1: Top-Down (Greedy Allocation) ===
                        LLVM_DEBUG(llvm::dbgs() << "  [Phase 1] Top-Down Greedy Allocation:\n");
                        for (Operation* op : topo) {
                            auto& nd = info[op];
                            int64_t vecCnt = nd.vectorCount;

                            // 1. Calculate Input Level (Bottleneck)
                            int l_in = MAX_SIMD_LEVEL;
                            if (!preds[op].empty()) {
                                for (auto* p : preds[op])
                                    l_in = std::min(l_in, info[p].finalLevel);
                            }

                            // 2. Check Boot Necessity (Slim Boot Trigger)
                            bool mustBoot = (l_in <= SLIM_BOOT_TRIGGER);

                            // 3. Calculate Costs
                            double cSIMD = costModel.getOpCost(op, Mode::SIMD, l_in);
                            if (mustBoot)
                                cSIMD += costModel.getBootCost();

                            double cSISD = costModel.getOpCost(op, Mode::SISD, vecCnt);

                            // 4. Input Cast Costs
                            double castIn_SIMD = 0.0;
                            double castIn_SISD = 0.0;
                            for (auto* p : preds[op]) {
                                castIn_SIMD += costModel.getCastCost(info[p].mode, Mode::SIMD, info[p].vectorCount);
                                castIn_SISD += costModel.getCastCost(info[p].mode, Mode::SISD, info[p].vectorCount);
                            }

                            double total_SIMD = cSIMD + castIn_SIMD;
                            double total_SISD = cSISD + castIn_SISD;

                            LLVM_DEBUG({
                                llvm::dbgs() << "    Eval: " << getOpName(op) << "\n"
                                             << "      L_in=" << l_in << ", mustBoot=" << (mustBoot ? "Yes" : "No") << "\n"
                                             << "      Cost -> SIMD: " << total_SIMD << " (Op:" << cSIMD << "+Cast:" << castIn_SIMD << ")\n"
                                             << "      Cost -> SISD: " << total_SISD << " (Op:" << cSISD << "+Cast:" << castIn_SISD << ")\n";
                            });

                            // 5. Greedy Decision
                            int opDepth = isa<simd::SIMDMultOp>(op) ? 1 : 0;

                            if (total_SISD <= total_SIMD) {
                                // Prefer SISD
                                if (nd.mode != Mode::SISD) {
                                    LLVM_DEBUG(llvm::dbgs() << "      => Decision: Switched to SISD (Cheaper)\n");
                                    nd.mode = Mode::SISD;
                                    changed = true;
                                } else {
                                    LLVM_DEBUG(llvm::dbgs() << "      => Decision: Kept SISD\n");
                                }
                                nd.finalLevel = l_in;
                                nd.triggerBoot = false;
                            } else {
                                // Prefer SIMD
                                if (nd.mode != Mode::SIMD) {
                                    LLVM_DEBUG(llvm::dbgs() << "      => Decision: Switched to SIMD\n");
                                    nd.mode = Mode::SIMD;
                                    changed = true;
                                } else {
                                    LLVM_DEBUG(llvm::dbgs() << "      => Decision: Kept SIMD\n");
                                }

                                // Update Boot Logic
                                if (!nd.triggerBoot && mustBoot) {
                                    LLVM_DEBUG(llvm::dbgs() << "      => [BOOT TRIGGERED] L_in(" << l_in << ") <= " << SLIM_BOOT_TRIGGER << "\n");
                                    nd.triggerBoot = true;
                                    changed = true;
                                }

                                int oldLvl = nd.finalLevel;
                                if (nd.triggerBoot) {
                                    nd.finalLevel = BOOT_LEVEL - opDepth;
                                } else {
                                    nd.finalLevel = l_in - opDepth;
                                }

                                if (oldLvl != nd.finalLevel)
                                    changed = true; // Level changes matter too
                            }
                        }

                        // === Phase 2: Bottom-Up (Pruning & Refinement) ===
                        LLVM_DEBUG(llvm::dbgs() << "  [Phase 2] Bottom-Up Pruning & Refinement:\n");
                        for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
                            Operation* op = *it;
                            auto& nd = info[op];

                            // 1. Calculate Demand
                            int l_req = 0;
                            bool anyChildSIMD = false;
                            bool anyChildSISD = false;

                            for (auto* child : succs[op]) {
                                int childInReq = 0;
                                if (info[child].mode == Mode::SIMD) {
                                    if (info[child].triggerBoot)
                                        childInReq = SLIM_BOOT_TRIGGER;
                                    else {
                                        int childDepth = isa<simd::SIMDMultOp>(child) ? 1 : 0;
                                        childInReq = info[child].finalLevel + childDepth;
                                    }
                                }
                                l_req = std::max(l_req, childInReq);

                                if (info[child].mode == Mode::SIMD)
                                    anyChildSIMD = true;
                                if (info[child].mode == Mode::SISD)
                                    anyChildSISD = true;
                            }

                            // 2. Slim Boot Pruning [CORE LOGIC]
                            if (nd.mode == Mode::SIMD && nd.triggerBoot) {
                                int opDepth = isa<simd::SIMDMultOp>(op) ? 1 : 0;
                                int needed = l_req + opDepth;

                                LLVM_DEBUG(llvm::dbgs() << "    Checking Prune: " << getOpName(op)
                                                        << " -> Available(" << SLIM_BOOT_TRIGGER << ") >= Needed(" << needed << ")?\n");

                                if (SLIM_BOOT_TRIGGER >= needed) {
                                    LLVM_DEBUG(llvm::dbgs() << "      [PRUNE SUCCESS] Boot Cancelled for " << getOpName(op) << "!\n");
                                    nd.triggerBoot = false; // Prune!
                                    changed = true;

                                    // Recalculate natural level
                                    int l_in = MAX_SIMD_LEVEL;
                                    for (auto* p : preds[op])
                                        l_in = std::min(l_in, info[p].finalLevel);
                                    nd.finalLevel = l_in - opDepth;
                                }
                            }

                            // 3. Mode Refinement (Broadcast Optimization)
                            if (nd.mode == Mode::SISD) {
                                double opCostSISD = costModel.getOpCost(op, Mode::SISD, nd.vectorCount);
                                double castOutSISD = anyChildSIMD ? costModel.getCastCost(Mode::SISD, Mode::SIMD, nd.vectorCount) : 0.0;
                                double costCurr = opCostSISD + castOutSISD;

                                // Hypothetical SIMD Cost
                                int l_in = MAX_SIMD_LEVEL;
                                for (auto* p : preds[op])
                                    l_in = std::min(l_in, info[p].finalLevel);

                                bool localNeedBoot = (l_in <= SLIM_BOOT_TRIGGER);
                                int opDepth = isa<simd::SIMDMultOp>(op) ? 1 : 0;

                                // Hypothetical Pruning logic
                                if (localNeedBoot && (SLIM_BOOT_TRIGGER >= l_req + opDepth))
                                    localNeedBoot = false;

                                double opCostSIMD = costModel.getOpCost(op, Mode::SIMD, l_in);
                                if (localNeedBoot)
                                    opCostSIMD += costModel.getBootCost();

                                double castOutSIMD = anyChildSISD ? costModel.getCastCost(Mode::SIMD, Mode::SISD, nd.vectorCount) : 0.0;
                                double costSwitch = opCostSIMD + castOutSIMD;

                                if (costSwitch < costCurr) {
                                    LLVM_DEBUG(llvm::dbgs() << "    [REFINE SUCCESS] " << getOpName(op) << ": SISD -> SIMD "
                                                            << "(Switch=" << costSwitch << " < Stay=" << costCurr << ")\n");
                                    nd.mode = Mode::SIMD;
                                    nd.triggerBoot = localNeedBoot;
                                    if (nd.triggerBoot)
                                        nd.finalLevel = BOOT_LEVEL - opDepth;
                                    else
                                        nd.finalLevel = l_in - opDepth;
                                    changed = true;
                                }
                            }
                        }
                    } // End While

                    LLVM_DEBUG({
                        llvm::dbgs() << "\n>>> --- Final Converged State (@" << func.getName() << ") --- <<<\n";
                        for (auto* op : topo) {
                            auto& nd = info[op];
                            llvm::dbgs() << "  " << getOpName(op) << ": "
                                         << (nd.mode == Mode::SIMD ? "SIMD" : "SISD")
                                         << ", Level=" << nd.finalLevel
                                         << ", Boot=" << (nd.triggerBoot ? "YES" : "NO") << "\n";
                        }
                    });

                    // ------------------------------------------------------------
                    // 4. Rewrite IR
                    // ------------------------------------------------------------
                    LLVM_DEBUG(llvm::dbgs() << "\n[Rewrite] Starting IR Materialization...\n");
                    IRRewriter rewriter(module.getContext());
                    llvm::DenseMap<Value, Value> rewriteMap;

                    for (Operation* op : topo) {
                        auto& nd = info[op];
                        rewriter.setInsertionPoint(op);
                        SmallVector<Value, 4> newOps;

                        LLVM_DEBUG(llvm::dbgs() << "  Rewriting: " << getOpName(op) << " -> "
                                                << (nd.mode == Mode::SIMD ? "SIMD" : "SISD") << "\n");

                        // === Step A: Prepare Operands (Cast & Boot) ===

                        // 确定当前算子期望的输入 Level
                        int reqInputLevel = nd.finalLevel;
                        if (nd.triggerBoot) {
                            reqInputLevel = SLIM_BOOT_TRIGGER;
                        } else if (isa<simd::SIMDMultOp>(op) && nd.mode == Mode::SIMD) {
                            reqInputLevel = nd.finalLevel + 1;
                        }

                        for (Value v : op->getOperands()) {
                            Value nv = rewriteMap.lookup(v);
                            if (!nv)
                                nv = v;

                            bool isDstSISD = (nd.mode == Mode::SISD);

                            // 只有当输入真的是密文时，才执行转换检查
                            if (isa<simd::SIMDCipherType, sisd::SISDCipherType>(nv.getType())) {
                                bool isSrcSISD = isa<sisd::SISDCipherType>(nv.getType());

                                // 1. 跨方案 Cast
                                if (isSrcSISD && !isDstSISD) {
                                    auto tt = cast<sisd::SISDCipherType>(nv.getType());
                                    auto dstTy = simd::SIMDCipherType::get(op->getContext(), reqInputLevel, tt.getPlaintextCount(), tt.getElementType());
                                    LLVM_DEBUG(llvm::dbgs() << "    - Inserting Cast: SISD -> SIMD\n");
                                    nv = rewriter.create<sisd::SISDCastSISDCipherToSIMDCipherOp>(op->getLoc(), dstTy, nv).getResult();
                                } else if (!isSrcSISD && isDstSISD) {
                                    auto st = cast<simd::SIMDCipherType>(nv.getType());
                                    auto dstTy = sisd::SISDCipherType::get(op->getContext(), st.getPlaintextCount(), st.getElementType());
                                    LLVM_DEBUG(llvm::dbgs() << "    - Inserting Cast: SIMD -> SISD\n");
                                    nv = rewriter.create<simd::SIMDCastSIMDCipherToSISDCipherOp>(op->getLoc(), dstTy, nv).getResult();
                                }

                                // 2. 处理 Bootstrapping
                                if (nd.triggerBoot && nd.mode == Mode::SIMD && isa<simd::SIMDCipherType>(nv.getType())) {
                                    auto oldTy = cast<simd::SIMDCipherType>(nv.getType());
                                    auto bootedTy = simd::SIMDCipherType::get(op->getContext(), BOOT_LEVEL, oldTy.getPlaintextCount(), oldTy.getElementType());
                                    LLVM_DEBUG(llvm::dbgs() << "    - Inserting Bootstrapping (Target Level " << BOOT_LEVEL << ")\n");
                                    nv = rewriter.create<simd::SIMDBootOp>(op->getLoc(), bootedTy, nv).getResult();
                                }
                            }

                            newOps.push_back(nv);
                        }

                        // === Step A.5: Level Alignment (关键修复) ===
                        if (nd.mode == Mode::SIMD && !newOps.empty()) {
                            bool needsAlign = isa<simd::SIMDAddOp, simd::SIMDSubOp, simd::SIMDMinOp,
                                                  simd::SIMDSelectOp, simd::SIMDCmpOp, simd::SIMDDivOp>(op);

                            if (needsAlign) {
                                int64_t minLevel = 10000;
                                bool hasSIMD = false;

                                // 1. 寻找最小 Level
                                for (Value v : newOps) {
                                    if (auto st = dyn_cast<simd::SIMDCipherType>(v.getType())) {
                                        minLevel = std::min(minLevel, st.getLevel());
                                        hasSIMD = true;
                                    }
                                }

                                // 2. 统一降级 (ModSwitch)
                                if (hasSIMD) {
                                    for (Value& v : newOps) {
                                        if (auto st = dyn_cast<simd::SIMDCipherType>(v.getType())) {
                                            if (st.getLevel() > minLevel) {
                                                LLVM_DEBUG(llvm::dbgs() << "    - Inserting ModSwitch (Aligning L" << st.getLevel() << " -> L" << minLevel << ")\n");
                                                auto targetTy = simd::SIMDCipherType::get(
                                                    op->getContext(), minLevel, st.getPlaintextCount(), st.getElementType());
                                                v = rewriter.create<simd::SIMDModSwitchOp>(op->getLoc(), targetTy, v).getResult();
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        // === Step B: Determine Result Type ===

                        Operation* newOp = nullptr;
                        int64_t vecCnt = nd.vectorCount;

                        if (isa<simd::SIMDDecryptOp, sisd::SISDDecryptOp>(op)) {
                            if (isa<sisd::SISDCipherType>(newOps[0].getType()))
                                newOp = rewriter.create<sisd::SISDDecryptOp>(op->getLoc(), op->getResultTypes(), newOps);
                            else
                                newOp = rewriter.create<simd::SIMDDecryptOp>(op->getLoc(), op->getResultTypes(), newOps);
                        } else {
                            Type resTy;
                            if (nd.mode == Mode::SISD) {
                                resTy = sisd::SISDCipherType::get(op->getContext(), vecCnt, rewriter.getI64Type());
                            } else {
                                // SIMD 模式
                                int resultLevel = nd.finalLevel;

                                // [修复] 再次校准结果 Level，确保它等于对齐后的输入 Level
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

                            // === Step C: Create Operation ===

                            if (isa<simd::SIMDAddOp, sisd::SISDAddOp>(op)) {
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

                            } else if (isa<simd::SIMDReduceAddOp>(op)) {
                                newOp = rewriter.create<simd::SIMDReduceAddOp>(op->getLoc(), resTy, newOps[0]);

                                // ==========================================
                                // 【修复点 1】：新增 LoadOp 的重写支持
                                // ==========================================
                            } else if (isa<simd::SIMDLoadOp, sisd::SISDLoadOp>(op)) {
                                if (nd.mode == Mode::SISD)
                                    newOp = rewriter.create<sisd::SISDLoadOp>(op->getLoc(), resTy, newOps[0], newOps[1]);
                                else
                                    newOp = rewriter.create<simd::SIMDLoadOp>(op->getLoc(), resTy, newOps[0], newOps[1]);

                                // ==========================================
                                // 【修复点 2】：完善 StoreOp (支持 SISD 并处理孤立返回)
                                // ==========================================
                            } else if (isa<simd::SIMDStoreOp, sisd::SISDStoreOp>(op)) {
                                if (nd.mode == Mode::SISD)
                                    rewriter.create<sisd::SISDStoreOp>(op->getLoc(), newOps[0], newOps[1]);
                                else
                                    rewriter.create<simd::SIMDStoreOp>(op->getLoc(), newOps[0], newOps[1]);

                                LLVM_DEBUG(llvm::dbgs() << "    - Created StoreOp (No Result)\n");
                                continue;

                            } else if (isa<simd::SIMDRescaleOp>(op)) {
                                auto inputCipher = newOps[0];
                                int inputLevel = 0;
                                if (auto st = dyn_cast<simd::SIMDCipherType>(inputCipher.getType())) {
                                    inputLevel = st.getLevel();
                                }
                                int targetLevel = nd.finalLevel;

                                if (targetLevel >= inputLevel) {
                                    LLVM_DEBUG(llvm::dbgs() << "    - Eliminated Redundant Rescale\n");
                                    rewriteMap[op->getResult(0)] = inputCipher;
                                    continue;
                                }
                                auto resTy = simd::SIMDCipherType::get(op->getContext(), targetLevel, vecCnt, rewriter.getI64Type());
                                newOp = rewriter.create<simd::SIMDRescaleOp>(op->getLoc(), resTy, inputCipher);
                            } else if (isa<simd::SIMDModSwitchOp>(op)) {
                                auto inputCipher = newOps[0];
                                int inputLevel = 0;
                                if (auto st = dyn_cast<simd::SIMDCipherType>(inputCipher.getType())) {
                                    inputLevel = st.getLevel();
                                }
                                int targetLevel = nd.finalLevel;

                                if (targetLevel >= inputLevel) {
                                    LLVM_DEBUG(llvm::dbgs() << "    - Eliminated Redundant ModSwitch\n");
                                    rewriteMap[op->getResult(0)] = inputCipher;
                                    continue;
                                }

                                auto resTy = simd::SIMDCipherType::get(op->getContext(), targetLevel, vecCnt, rewriter.getI64Type());
                                newOp = rewriter.create<simd::SIMDModSwitchOp>(op->getLoc(), resTy, inputCipher);
                            }
                        }

                        if (newOp) {
                            LLVM_DEBUG(llvm::dbgs() << "    - Successfully created replacement Op.\n");
                            rewriteMap[op->getResult(0)] = newOp->getResult(0);
                        }
                    }

                    // ------------------------------------------------------------
                    // 5. Final Replacement (Safe Deletion)
                    // ------------------------------------------------------------
                    LLVM_DEBUG(llvm::dbgs() << "\n[Cleanup] Performing Safe Deletions...\n");
                    for (Operation* op : topo) {
                        for (int i = 0; i < op->getNumResults(); ++i) {
                            Value oldVal = op->getResult(i);
                            if (rewriteMap.count(oldVal)) {
                                Value newVal = rewriteMap[oldVal];

                                // 【关键修复】：保持给外部消费者（如 scf.yield）的类型契约不变
                                if (oldVal.getType() != newVal.getType()) {
                                    // 在旧 op 所在的位置插入马甲（此时旧 op 还没被删除）
                                    rewriter.setInsertionPoint(op);
                                    Value castBack;

                                    // 根据类型差异，自动插入对应的互转 Cast 算子
                                    if (isa<simd::SIMDCipherType>(oldVal.getType()) && isa<sisd::SISDCipherType>(newVal.getType())) {
                                        castBack = rewriter.create<sisd::SISDCastSISDCipherToSIMDCipherOp>(
                                                               op->getLoc(), oldVal.getType(), newVal)
                                                       .getResult();
                                    } else if (isa<sisd::SISDCipherType>(oldVal.getType()) && isa<simd::SIMDCipherType>(newVal.getType())) {
                                        castBack = rewriter.create<simd::SIMDCastSIMDCipherToSISDCipherOp>(
                                                               op->getLoc(), oldVal.getType(), newVal)
                                                       .getResult();
                                    } else {
                                        // 保底的强制类型转换
                                        castBack = rewriter.create<UnrealizedConversionCastOp>(
                                                               op->getLoc(), oldVal.getType(), newVal)
                                                       .getResult(0);
                                    }

                                    // 让那些没被重写的外部指令，继续使用包装回原类型的值
                                    oldVal.replaceAllUsesWith(castBack);
                                } else {
                                    // 类型没变，安全地直接替换
                                    oldVal.replaceAllUsesWith(newVal);
                                }
                            }
                        }

                        // 如果旧的值已经被全部替换干净了，安全销毁旧算子
                        if (op->use_empty()) {
                            rewriter.eraseOp(op);
                        } else {
                            LLVM_DEBUG(llvm::dbgs() << "  [Warning] Op not erased due to remaining uses: " << getOpName(op) << "\n");
                        }
                    }

                    // ------------------------------------------------------------
                    // 6. Fix Loop Signatures (Post-Processing)
                    // ------------------------------------------------------------
                    LLVM_DEBUG(llvm::dbgs() << "\n[Post-Processing] Fixing SCF For Loops Signatures...\n");

                    // 辅助 Lambda：穿透讨厌的 Cast 面具，直达真实的底层类型
                    auto peelCast = [](Value v) -> Value {
                        if (auto castOp = v.getDefiningOp<sisd::SISDCastSISDCipherToSIMDCipherOp>())
                            return castOp.getOperand();
                        if (auto castOp = v.getDefiningOp<simd::SIMDCastSIMDCipherToSISDCipherOp>())
                            return castOp.getOperand();
                        if (auto castOp = v.getDefiningOp<UnrealizedConversionCastOp>())
                            return castOp.getOperand(0);
                        return v;
                    };

                    SmallVector<scf::ForOp> loopsToFix;
                    func.walk([&](scf::ForOp forOp) {
                        if (forOp.getNumResults() == 0)
                            return;
                        loopsToFix.push_back(forOp);
                    });

                    for (scf::ForOp forOp : loopsToFix) {
                        IRRewriter rewriter(forOp.getContext());
                        rewriter.setInsertionPoint(forOp);

                        SmallVector<Value> newInits;
                        bool needsFix = false;

                        // 穿透初始参数的 Cast
                        for (Value init : forOp.getInitArgs()) {
                            Value realInit = peelCast(init);
                            newInits.push_back(realInit);
                            if (realInit != init)
                                needsFix = true;
                        }

                        if (!needsFix)
                            continue;

                        // 创建完全基于 SISD（或真实类型）的新循环
                        auto newLoop = rewriter.create<scf::ForOp>(
                            forOp.getLoc(),
                            forOp.getLowerBound(),
                            forOp.getUpperBound(),
                            forOp.getStep(),
                            newInits);

                        Region& oldRegion = forOp.getRegion();
                        Region& newRegion = newLoop.getRegion();
                        Block* oldBody = &oldRegion.front();
                        Block* newBody = &newRegion.front();

                        // 更新循环体内参数的真实类型
                        newBody->getArgument(0).setType(oldBody->getArgument(0).getType());
                        for (size_t i = 0; i < newInits.size(); ++i) {
                            newBody->getArgument(i + 1).setType(newInits[i].getType());
                        }

                        rewriter.mergeBlocks(oldBody, newBody, newBody->getArguments());

                        // 【核心修复】：修复 scf.yield，让它也穿透 Cast 直接返回真实类型
                        auto yieldOp = cast<scf::YieldOp>(newBody->getTerminator());
                        rewriter.setInsertionPoint(yieldOp);
                        SmallVector<Value> newYields;
                        for (Value y : yieldOp.getOperands()) {
                            newYields.push_back(peelCast(y));
                        }
                        rewriter.replaceOpWithNewOp<scf::YieldOp>(yieldOp, newYields);

                        // 把旧循环的调用者连接到新循环
                        rewriter.setInsertionPointAfter(newLoop);
                        for (auto [oldRes, newRes] : llvm::zip(forOp.getResults(), newLoop.getResults())) {
                            // 为了桥接外部还没来得及清理的代码，我们反向贴一个 Cast。
                            // 别担心，MLIR 的 Canonicalizer 会把这个 Cast 和下游的 Cast 瞬间消消乐掉！
                            if (oldRes.getType() != newRes.getType()) {
                                Value castBack = rewriter.create<sisd::SISDCastSISDCipherToSIMDCipherOp>(
                                                             newLoop.getLoc(), oldRes.getType(), newRes)
                                                     .getResult();
                                oldRes.replaceAllUsesWith(castBack);
                            } else {
                                oldRes.replaceAllUsesWith(newRes);
                            }
                        }
                        rewriter.eraseOp(forOp);
                    }

                    // // ------------------------------------------------------------
                    // // 7. Clean up redundant casts (Post-Processing)
                    // // ------------------------------------------------------------
                    // LLVM_DEBUG(llvm::dbgs() << "\n[Post-Processing] Cleaning up redundant casts...\n");
                    // bool castChanged;
                    // do {
                    //     castChanged = false;
                    //     SmallVector<Operation*> castsToErase;

                    //     func.walk([&](Operation* op) {
                    //         if (isa<sisd::SISDCastSISDCipherToSIMDCipherOp, simd::SIMDCastSIMDCipherToSISDCipherOp>(op)) {
                    //             Value input = op->getOperand(0);
                    //             Type inTy = input.getType();
                    //             Type outTy = op->getResult(0).getType();

                    //             // 情况 1：类型完全相同 (比如 sisd -> sisd)，这层面具毫无意义
                    //             if (inTy == outTy) {
                    //                 op->getResult(0).replaceAllUsesWith(input);
                    //                 castsToErase.push_back(op);
                    //                 return;
                    //             }

                    //             // 情况 2：背靠背的互相抵消 (比如 sisd -> simd -> sisd)
                    //             if (auto prevOp = input.getDefiningOp()) {
                    //                 if (isa<sisd::SISDCastSISDCipherToSIMDCipherOp, simd::SIMDCastSIMDCipherToSISDCipherOp>(prevOp)) {
                    //                     Value origInput = prevOp->getOperand(0);
                    //                     // 如果转了两手之后，类型又回到了原点，直接把源头短接给下游
                    //                     if (origInput.getType() == outTy) {
                    //                         op->getResult(0).replaceAllUsesWith(origInput);
                    //                         castsToErase.push_back(op);
                    //                         return;
                    //                     }
                    //                 }
                    //             }
                    //         }
                    //     });

                    //     // 安全删除这些已经没用的废 Cast
                    //     for (Operation* op : castsToErase) {
                    //         op->erase();
                    //         castChanged = true;
                    //     }
                    // } while (castChanged); // 循环直到再也找不到可以消除的 Cast 为止

                    // ====================================================================
                    // 8. 自动推断并生成全局配置属性，按需挂载到 Module 上
                    // ====================================================================
                    LLVM_DEBUG(llvm::dbgs() << "\n[Post-Processing] Attaching Global FlyHE Config to Module...\n");

                    bool hasSIMD = false;
                    bool hasSISD = false;
                    bool hasBoot = false;

                    int64_t maxCount = 1;
                    int32_t initial_level = 1; // 专门记录最初始的 Level

                    // 1. 遍历整个 Module 扫描内部算子
                    module.walk([&](Operation* op) {
                        if (!op || !op->getDialect())
                            return;
                        StringRef ns = op->getDialect()->getNamespace();

                        if (ns == "simd")
                            hasSIMD = true;
                        if (ns == "sisd")
                            hasSISD = true;
                        if (isa<simd::SIMDBootOp>(op))
                            hasBoot = true;

                        // 提取密文类型中的 Count (全局扫描以寻找最大的 batch size)
                        auto checkCipherCount = [&](Type ty) {
                            if (auto simdTy = dyn_cast<simd::SIMDCipherType>(ty)) {
                                maxCount = std::max(maxCount, simdTy.getPlaintextCount());
                            } else if (auto sisdTy = dyn_cast<sisd::SISDCipherType>(ty)) {
                                maxCount = std::max(maxCount, sisdTy.getPlaintextCount());
                            }
                        };

                        for (Type ty : op->getOperandTypes())
                            checkCipherCount(ty);
                        for (Type ty : op->getResultTypes())
                            checkCipherCount(ty);

                        // 【关键修复】：只从计算的“源头”提取初始 Level
                        // 源头包括：加密算子、以及从 SISD 转换到 SIMD 的入口算子
                        if (isa<simd::SIMDEncryptOp, sisd::SISDCastSISDCipherToSIMDCipherOp>(op)) {
                            if (auto simdTy = dyn_cast<simd::SIMDCipherType>(op->getResult(0).getType())) {
                                initial_level = std::max<int32_t>(initial_level, simdTy.getLevel());
                            }
                        }
                    });

                    // 2. 检查函数的输入参数（它们也是计算源头）
                    module.walk([&](func::FuncOp func) {
                        for (Type ty : func.getArgumentTypes()) {
                            if (auto simdTy = dyn_cast<simd::SIMDCipherType>(ty)) {
                                initial_level = std::max<int32_t>(initial_level, simdTy.getLevel());
                            }
                        }
                    });

                    // 3. 推断运行模式
                    std::string modeStr = "SIMD";
                    if (hasSIMD && hasSISD) {
                        modeStr = "CROSS";
                    } else if (hasSISD) {
                        modeStr = "SISD";
                    }

                    // 4. 计算 logn (满足 maxCount 的最小 2 次幂指数)
                    int64_t logn = std::max<int64_t>(1, std::ceil(std::log2(maxCount)));

                    // 5. 确定 remaining_levels (最小初始 Level)
                    int32_t remaining_levels = initial_level;

                    // 使用 OpBuilder 构造 Attributes
                    OpBuilder builder(module.getContext());
                    SmallVector<NamedAttribute, 5> configAttrs;

                    // 无论什么模式，mode 参数是必须的
                    configAttrs.push_back(builder.getNamedAttr("mode", builder.getStringAttr(modeStr)));

                    // 只有 SIMD 和 CROSS 模式才需要挂载 CKKS 相关的参数
                    if (modeStr == "SIMD" || modeStr == "CROSS") {
                        configAttrs.push_back(builder.getNamedAttr("logN", builder.getI64IntegerAttr(16))); // 强制固定为 16
                        configAttrs.push_back(builder.getNamedAttr("logn", builder.getI64IntegerAttr(logn)));
                        configAttrs.push_back(builder.getNamedAttr("remaining_levels", builder.getI32IntegerAttr(remaining_levels)));
                        configAttrs.push_back(builder.getNamedAttr("bootstrapping_enabled", builder.getBoolAttr(hasBoot)));
                    }

                    // 构造 DictionaryAttr 并挂载到 Module
                    DictionaryAttr dictAttr = builder.getDictionaryAttr(configAttrs);
                    module->setAttr("he.config", dictAttr);

                    LLVM_DEBUG(llvm::dbgs() << "  -> Mode: " << modeStr << "\n");
                    LLVM_DEBUG(llvm::dbgs() << "  -> Computed logn: " << logn << " (from max plaintext count: " << maxCount << ")\n");
                    LLVM_DEBUG(llvm::dbgs() << "  -> Initial remaining_levels: " << remaining_levels << "\n");
                    LLVM_DEBUG(llvm::dbgs() << "  -> Bootstrapping Enabled: " << (hasBoot ? "True" : "False") << "\n");
                    LLVM_DEBUG(llvm::dbgs() << "[ModeSel] Pass Finished Successfully.\n");
                }
            }
        };

    } // namespace
} // namespace mlir::libra::mdsel