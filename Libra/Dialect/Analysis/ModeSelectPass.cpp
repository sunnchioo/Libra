//===----------------------------------------------------------------------===//
// ModeSelectPass.cpp
// Optimized Iterative Mode Selection with Slim Boot Pruning & Broadcast Costs
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/IR/Verifier.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/CommandLine.h"

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
                    llvm::errs() << "[ModeSel] Error opening file: " << ec.message() << "\n";
                    return;
                }
                llvm::Expected<llvm::json::Value> rootOrErr = llvm::json::parse(fileOrErr.get()->getBuffer());
                if (!rootOrErr) {
                    llvm::errs() << "[ModeSel] JSON parse error\n";
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

            // ... (Includes and other structs remain the same) ...

            // Helper for printing Op names concisely
            std::string getOpName(Operation* op) {
                return op->getName().getStringRef().str();
            }

            void runOnOperation() override {
                ModuleOp module = getOperation();
                CostModel costModel(getOptionFilePath());

                // [DEBUG] 开关
                bool debugMode = true;

                for (func::FuncOp func : module.getOps<func::FuncOp>()) {
                    if (debugMode)
                        llvm::errs() << "\n[ModeSel] Processing Function: " << func.getName() << "\n";

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
                    if (ops.empty())
                        continue;

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
                        if (debugMode)
                            llvm::errs() << "\n--- Iteration " << iter << " Start ---\n";
                        changed = false;
                        iter++;

                        // === Phase 1: Top-Down (Greedy Allocation) ===
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

                            // [DEBUG] Phase 1 Details
                            // llvm::errs() << "  [P1] " << getOpName(op) << " L_in=" << l_in
                            //              << " MustBoot=" << mustBoot
                            //              << " Cost(SIMD=" << total_SIMD << ", SISD=" << total_SISD << ")\n";

                            // 5. Greedy Decision
                            int opDepth = isa<simd::SIMDMultOp>(op) ? 1 : 0;

                            if (total_SISD < total_SIMD) {
                                // Prefer SISD
                                if (nd.mode != Mode::SISD) {
                                    if (debugMode)
                                        llvm::errs() << "    -> " << getOpName(op) << " switched to SISD (Cheaper)\n";
                                    nd.mode = Mode::SISD;
                                    changed = true;
                                }
                                nd.finalLevel = l_in;
                                nd.triggerBoot = false;
                            } else {
                                // Prefer SIMD
                                if (nd.mode != Mode::SIMD) {
                                    if (debugMode)
                                        llvm::errs() << "    -> " << getOpName(op) << " switched to SIMD\n";
                                    nd.mode = Mode::SIMD;
                                    changed = true;
                                }

                                // Update Boot Logic
                                if (!nd.triggerBoot && mustBoot) {
                                    if (debugMode)
                                        llvm::errs() << "    -> " << getOpName(op) << " Trigger Boot set to TRUE (Input Level " << l_in << " <= " << SLIM_BOOT_TRIGGER << ")\n";
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

                                // [DEBUG] Check Pruning
                                // llvm::errs() << "  [P2] Checking Prune " << getOpName(op)
                                //              << ": Available(3) >= Needed(" << needed << ")?\n";

                                if (SLIM_BOOT_TRIGGER >= needed) {
                                    if (debugMode)
                                        llvm::errs() << "    [PRUNE] " << getOpName(op) << ": Boot Cancelled! "
                                                     << "Available(" << SLIM_BOOT_TRIGGER << ") >= Needed(" << needed << ")\n";
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
                                    if (debugMode)
                                        llvm::errs() << "    [REFINE] " << getOpName(op) << ": SISD -> SIMD "
                                                     << "(Switch=" << costSwitch << " < Stay=" << costCurr << ")\n";
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

                    if (debugMode)
                        llvm::errs() << "--- Final Converged State ---\n";
                    if (debugMode) {
                        for (auto* op : topo) {
                            auto& nd = info[op];
                            llvm::errs() << getOpName(op) << ": "
                                         << (nd.mode == Mode::SIMD ? "SIMD" : "SISD")
                                         << ", Level=" << nd.finalLevel
                                         << ", Boot=" << (nd.triggerBoot ? "YES" : "NO") << "\n";
                        }
                    }

                    // // ------------------------------------------------------------
                    // // 4. Rewrite IR
                    // // ------------------------------------------------------------
                    // // ... (Rewrite logic remains the same) ...
                    // IRRewriter rewriter(module.getContext());
                    // llvm::DenseMap<Value, Value> rewriteMap;

                    // // ... (Paste original Rewrite Loop here) ...
                    // // Since the logging request is mainly for algorithm analysis,
                    // // I'll keep the rewrite part concise in this snippet.
                    // // Make sure to include the full rewrite loop from the previous answer.

                    // for (Operation* op : topo) {
                    //     auto& nd = info[op];
                    //     rewriter.setInsertionPoint(op);
                    //     SmallVector<Value, 4> newOps;

                    //     // [Same rewrite logic as before...]
                    //     int reqInputLevel = nd.finalLevel;
                    //     if (isa<simd::SIMDMultOp>(op) && nd.mode == Mode::SIMD) {
                    //         reqInputLevel = nd.finalLevel + 1;
                    //     } else if (nd.triggerBoot) {
                    //         reqInputLevel = SLIM_BOOT_TRIGGER;
                    //     }

                    //     for (Value v : op->getOperands()) {
                    //         Value nv = rewriteMap.lookup(v);
                    //         if (!nv)
                    //             nv = v;

                    //         bool isSrcSISD = isa<sisd::SISDCipherType>(nv.getType());
                    //         bool isDstSISD = (nd.mode == Mode::SISD);
                    //         bool isDecrypt = isa<simd::SIMDDecryptOp, sisd::SISDDecryptOp>(op);

                    //         if (!isDecrypt) {
                    //             if (isSrcSISD && !isDstSISD) {
                    //                 auto tt = cast<sisd::SISDCipherType>(nv.getType());
                    //                 auto dstTy = simd::SIMDCipherType::get(op->getContext(), reqInputLevel, tt.getPlaintextCount(), tt.getElementType());
                    //                 nv = rewriter.create<sisd::SISDCastSISDCipherToSIMDCipherOp>(op->getLoc(), dstTy, nv);
                    //             } else if (!isSrcSISD && isDstSISD) {
                    //                 auto st = cast<simd::SIMDCipherType>(nv.getType());
                    //                 auto dstTy = sisd::SISDCipherType::get(op->getContext(), st.getPlaintextCount(), st.getElementType());
                    //                 nv = rewriter.create<simd::SIMDCastSIMDCipherToSISDCipherOp>(op->getLoc(), dstTy, nv);
                    //             }
                    //         }
                    //         newOps.push_back(nv);
                    //     }

                    //     Operation* newOp = nullptr;
                    //     int64_t vecCnt = nd.vectorCount;
                    //     // ... Create Ops Logic ...
                    //     if (isa<simd::SIMDDecryptOp, sisd::SISDDecryptOp>(op)) {
                    //         if (isa<sisd::SISDCipherType>(newOps[0].getType()))
                    //             newOp = rewriter.create<sisd::SISDDecryptOp>(op->getLoc(), op->getResultTypes(), newOps);
                    //         else
                    //             newOp = rewriter.create<simd::SIMDDecryptOp>(op->getLoc(), op->getResultTypes(), newOps);
                    //     } else if (nd.mode == Mode::SISD) {
                    //         auto resTy = sisd::SISDCipherType::get(op->getContext(), vecCnt, rewriter.getI64Type());
                    //         if (isa<simd::SIMDAddOp, sisd::SISDAddOp>(op))
                    //             newOp = rewriter.create<sisd::SISDAddOp>(op->getLoc(), resTy, newOps);
                    //         else if (isa<simd::SIMDSubOp, sisd::SISDSubOp>(op))
                    //             newOp = rewriter.create<sisd::SISDSubOp>(op->getLoc(), resTy, newOps);
                    //         else if (isa<simd::SIMDMinOp, sisd::SISDMinOp>(op))
                    //             newOp = rewriter.create<sisd::SISDMinOp>(op->getLoc(), resTy, newOps);
                    //         else if (isa<simd::SIMDEncryptOp, sisd::SISDEncryptOp>(op))
                    //             newOp = rewriter.create<sisd::SISDEncryptOp>(op->getLoc(), resTy, newOps);
                    //     } else {
                    //         auto resTy = simd::SIMDCipherType::get(op->getContext(), nd.finalLevel, vecCnt, rewriter.getI64Type());
                    //         if (isa<simd::SIMDAddOp, sisd::SISDAddOp>(op))
                    //             newOp = rewriter.create<simd::SIMDAddOp>(op->getLoc(), resTy, newOps);
                    //         else if (isa<simd::SIMDSubOp, sisd::SISDSubOp>(op))
                    //             newOp = rewriter.create<simd::SIMDSubOp>(op->getLoc(), resTy, newOps);
                    //         else if (isa<simd::SIMDMultOp>(op))
                    //             newOp = rewriter.create<simd::SIMDMultOp>(op->getLoc(), resTy, newOps);
                    //         else if (isa<simd::SIMDMinOp, sisd::SISDMinOp>(op))
                    //             newOp = rewriter.create<simd::SIMDMinOp>(op->getLoc(), resTy, newOps);
                    //         else if (isa<simd::SIMDEncryptOp, sisd::SISDEncryptOp>(op))
                    //             newOp = rewriter.create<simd::SIMDEncryptOp>(op->getLoc(), resTy, newOps);
                    //     }
                    //     if (newOp)
                    //         rewriteMap[op->getResult(0)] = newOp->getResult(0);
                    // }

                    // for (Operation* op : topo) {
                    //     if (op->getNumResults() > 0 && rewriteMap.count(op->getResult(0)))
                    //         rewriter.replaceOp(op, rewriteMap[op->getResult(0)]);
                    // }

                    // ------------------------------------------------------------
                    // 4. Rewrite IR
                    // ------------------------------------------------------------
                    // IRRewriter rewriter(module.getContext());
                    // llvm::DenseMap<Value, Value> rewriteMap;

                    // for (Operation* op : topo) {
                    //     auto& nd = info[op];
                    //     rewriter.setInsertionPoint(op);
                    //     SmallVector<Value, 4> newOps;

                    //     // --- Step A: Prepare Operands (Cast & Boot) ---

                    //     // 确定当前算子期望的输入 Level
                    //     // 如果触发 Boot，则期望输入 Level 只需要达到 Trigger 即可
                    //     // 否则期望输入 Level 就是计算出的 finalLevel (对于 Mult 可能是 finalLevel + 1)
                    //     int reqInputLevel = nd.finalLevel;
                    //     if (nd.triggerBoot) {
                    //         reqInputLevel = SLIM_BOOT_TRIGGER;
                    //     } else if (isa<simd::SIMDMultOp>(op) && nd.mode == Mode::SIMD) {
                    //         // Mult 消耗一层，所以输入比输出高一层
                    //         reqInputLevel = nd.finalLevel + 1;
                    //     }

                    //     for (Value v : op->getOperands()) {
                    //         // 获取最新映射的值
                    //         Value nv = rewriteMap.lookup(v);
                    //         if (!nv)
                    //             nv = v;

                    //         bool isSrcSISD = isa<sisd::SISDCipherType>(nv.getType());
                    //         bool isDstSISD = (nd.mode == Mode::SISD);

                    //         // Decrypt 操作特殊处理：它接受 Cipher 输出 Plain，不需要类型转换逻辑
                    //         bool isDecrypt = isa<simd::SIMDDecryptOp, sisd::SISDDecryptOp>(op);

                    //         if (!isDecrypt) {
                    //             // 处理 SISD <-> SIMD 转换
                    //             if (isSrcSISD && !isDstSISD) {
                    //                 // SISD -> SIMD (需要指定目标 Level)
                    //                 auto tt = cast<sisd::SISDCipherType>(nv.getType());
                    //                 // 这里使用 reqInputLevel 作为转换后的初始 Level
                    //                 auto dstTy = simd::SIMDCipherType::get(op->getContext(), reqInputLevel, tt.getPlaintextCount(), tt.getElementType());
                    //                 nv = rewriter.create<sisd::SISDCastSISDCipherToSIMDCipherOp>(op->getLoc(), dstTy, nv).getResult();
                    //             } else if (!isSrcSISD && isDstSISD) {
                    //                 // SIMD -> SISD
                    //                 auto st = cast<simd::SIMDCipherType>(nv.getType());
                    //                 auto dstTy = sisd::SISDCipherType::get(op->getContext(), st.getPlaintextCount(), st.getElementType());
                    //                 nv = rewriter.create<simd::SIMDCastSIMDCipherToSISDCipherOp>(op->getLoc(), dstTy, nv).getResult();
                    //             }
                    //         }

                    //         // 处理 Bootstrapping
                    //         // 如果此节点标记为 triggerBoot，且当前是 SIMD 模式，且输入是 SIMD Cipher
                    //         if (nd.triggerBoot && nd.mode == Mode::SIMD && isa<simd::SIMDCipherType>(nv.getType())) {
                    //             // 插入 Boot Op 刷新噪声
                    //             auto oldTy = cast<simd::SIMDCipherType>(nv.getType());
                    //             // Boot 后的 Level 是系统设定的 BOOT_LEVEL (通常是最高 Level 或刷新后的 Level)
                    //             auto bootedTy = simd::SIMDCipherType::get(op->getContext(), BOOT_LEVEL, oldTy.getPlaintextCount(), oldTy.getElementType());
                    //             nv = rewriter.create<simd::SIMDBootOp>(op->getLoc(), bootedTy, nv).getResult();
                    //         }

                    //         newOps.push_back(nv);
                    //     }

                    //     // --- Step B: Determine Result Type ---

                    //     Operation* newOp = nullptr;
                    //     int64_t vecCnt = nd.vectorCount;

                    //     // 特殊处理 Decrypt (结果不是 Cipher)
                    //     if (isa<simd::SIMDDecryptOp, sisd::SISDDecryptOp>(op)) {
                    //         // Decrypt 的结果类型保持原样 (比如 vector<f64> 或 f64)
                    //         // 我们只需要决定创建 simd.decrypt 还是 sisd.decrypt
                    //         if (isa<sisd::SISDCipherType>(newOps[0].getType()))
                    //             newOp = rewriter.create<sisd::SISDDecryptOp>(op->getLoc(), op->getResultTypes(), newOps);
                    //         else
                    //             newOp = rewriter.create<simd::SIMDDecryptOp>(op->getLoc(), op->getResultTypes(), newOps);
                    //     } else {
                    //         // 计算 Cipher Result Type
                    //         Type resTy;
                    //         if (nd.mode == Mode::SISD) {
                    //             resTy = sisd::SISDCipherType::get(op->getContext(), vecCnt, rewriter.getI64Type());
                    //         } else {
                    //             // SIMD 模式
                    //             int resultLevel = nd.finalLevel;

                    //             // [关键修复]: 强制 Add/Sub/Min/Select 的结果 Level 等于输入 Level
                    //             // 以满足 SameOperandsAndResultType 约束
                    //             if (isa<simd::SIMDAddOp, simd::SIMDSubOp, simd::SIMDMinOp, simd::SIMDSelectOp>(op)) {
                    //                 // 检查第一个操作数
                    //                 if (!newOps.empty()) {
                    //                     if (auto st = dyn_cast<simd::SIMDCipherType>(newOps[0].getType())) {
                    //                         resultLevel = st.getLevel();
                    //                     }
                    //                     // 注意：Select 的第0个操作数是 Condition，第1个是 TrueVal
                    //                     // 对于 Select，我们应该看 TrueVal 的 Level
                    //                     if (isa<simd::SIMDSelectOp>(op) && newOps.size() > 1) {
                    //                         if (auto st = dyn_cast<simd::SIMDCipherType>(newOps[1].getType())) {
                    //                             resultLevel = st.getLevel();
                    //                         }
                    //                     }
                    //                 }
                    //             }

                    //             resTy = simd::SIMDCipherType::get(op->getContext(), resultLevel, vecCnt, rewriter.getI64Type());
                    //         }

                    //         // --- Step C: Create Operation ---

                    //         if (isa<simd::SIMDAddOp, sisd::SISDAddOp>(op)) {
                    //             if (nd.mode == Mode::SISD)
                    //                 newOp = rewriter.create<sisd::SISDAddOp>(op->getLoc(), resTy, newOps);
                    //             else
                    //                 newOp = rewriter.create<simd::SIMDAddOp>(op->getLoc(), resTy, newOps);
                    //         } else if (isa<simd::SIMDSubOp, sisd::SISDSubOp>(op)) {
                    //             if (nd.mode == Mode::SISD)
                    //                 newOp = rewriter.create<sisd::SISDSubOp>(op->getLoc(), resTy, newOps);
                    //             else
                    //                 newOp = rewriter.create<simd::SIMDSubOp>(op->getLoc(), resTy, newOps);
                    //         } else if (isa<simd::SIMDMultOp>(op)) { // Mult 只有 SIMD
                    //             newOp = rewriter.create<simd::SIMDMultOp>(op->getLoc(), resTy, newOps);
                    //         } else if (isa<simd::SIMDMinOp, sisd::SISDMinOp>(op)) {
                    //             if (nd.mode == Mode::SISD)
                    //                 newOp = rewriter.create<sisd::SISDMinOp>(op->getLoc(), resTy, newOps);
                    //             else
                    //                 newOp = rewriter.create<simd::SIMDMinOp>(op->getLoc(), resTy, newOps);
                    //         } else if (isa<simd::SIMDEncryptOp, sisd::SISDEncryptOp>(op)) {
                    //             if (nd.mode == Mode::SISD)
                    //                 newOp = rewriter.create<sisd::SISDEncryptOp>(op->getLoc(), resTy, newOps);
                    //             else
                    //                 newOp = rewriter.create<simd::SIMDEncryptOp>(op->getLoc(), resTy, newOps);
                    //         } else if (isa<simd::SIMDSelectOp>(op)) { // Select 只有 SIMD (或需要自定义 SISDSelect)
                    //             // 这里假设 Select 保持 SIMD，或者如果有 SISD Select 再加逻辑
                    //             newOp = rewriter.create<simd::SIMDSelectOp>(op->getLoc(), resTy, newOps[0], newOps[1], newOps[2]);
                    //         } else if (isa<simd::SIMDCmpOp>(op)) {
                    //             // Cmp 结果通常是 Mask，Level 跟随输入或保持
                    //             // 这里简单处理，直接创建
                    //             auto pred = cast<simd::SIMDCmpOp>(op).getPredicate();
                    //             newOp = rewriter.create<simd::SIMDCmpOp>(op->getLoc(), resTy, newOps[0], newOps[1], pred);
                    //         } else if (isa<simd::SIMDDivOp>(op)) {
                    //             newOp = rewriter.create<simd::SIMDDivOp>(op->getLoc(), resTy, newOps);
                    //         } else if (isa<simd::SIMDReduceAddOp>(op)) {
                    //             newOp = rewriter.create<simd::SIMDReduceAddOp>(op->getLoc(), resTy, newOps[0]);
                    //         } else if (isa<simd::SIMDStoreOp>(op)) {
                    //             // Store 没有返回值
                    //             rewriter.create<simd::SIMDStoreOp>(op->getLoc(), newOps[0], newOps[1]);
                    //             // 记得删除旧 op，但不需要更新 rewriteMap
                    //             // rewriter.eraseOp(op);
                    //             continue;
                    //         }
                    //     }

                    //     if (newOp) {
                    //         // 记录映射：旧结果 -> 新结果
                    //         rewriteMap[op->getResult(0)] = newOp->getResult(0);
                    //         // 删除旧 Op (稍后统一清理也可以，这里立即删除需小心 Use-Def 链，但因为是拓扑序重写，且有 Map，通常安全)
                    //         // 为了安全起见，我们可以在最后统一 erase，或者依靠 rewriteMap 覆盖
                    //     }
                    // }

                    // // 5. Final Replacement (清理残留引用)
                    // // 由于我们没有直接 eraseOp (除了 Store)，旧 Op 还留在 IR 里。
                    // // 我们需要遍历所有 Op，如果它是被重写过的，就用新值替换它的所有 User，然后删除它。
                    // // 但由于我们在 Loop 里是 `create` 新 Op，没有 `replaceOp`。

                    // // 正确的做法是：
                    // for (Operation* op : topo) {
                    //     if (op->getNumResults() > 0) {
                    //         Value oldVal = op->getResult(0);
                    //         if (rewriteMap.count(oldVal)) {
                    //             Value newVal = rewriteMap[oldVal];
                    //             oldVal.replaceAllUsesWith(newVal);
                    //         }
                    //     }
                    //     // 现在可以安全删除了，因为 Use 已经被替换
                    //     op->erase();
                    // }

                    // ------------------------------------------------------------
                    // 4. Rewrite IR
                    // ------------------------------------------------------------
                    IRRewriter rewriter(module.getContext());
                    llvm::DenseMap<Value, Value> rewriteMap;

                    for (Operation* op : topo) {
                        auto& nd = info[op];
                        rewriter.setInsertionPoint(op);
                        SmallVector<Value, 4> newOps;

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

                            bool isSrcSISD = isa<sisd::SISDCipherType>(nv.getType());
                            bool isDstSISD = (nd.mode == Mode::SISD);
                            bool isDecrypt = isa<simd::SIMDDecryptOp, sisd::SISDDecryptOp>(op);

                            if (!isDecrypt) {
                                // 处理 SISD <-> SIMD 转换
                                if (isSrcSISD && !isDstSISD) {
                                    auto tt = cast<sisd::SISDCipherType>(nv.getType());
                                    auto dstTy = simd::SIMDCipherType::get(op->getContext(), reqInputLevel, tt.getPlaintextCount(), tt.getElementType());
                                    nv = rewriter.create<sisd::SISDCastSISDCipherToSIMDCipherOp>(op->getLoc(), dstTy, nv).getResult();
                                } else if (!isSrcSISD && isDstSISD) {
                                    auto st = cast<simd::SIMDCipherType>(nv.getType());
                                    auto dstTy = sisd::SISDCipherType::get(op->getContext(), st.getPlaintextCount(), st.getElementType());
                                    nv = rewriter.create<simd::SIMDCastSIMDCipherToSISDCipherOp>(op->getLoc(), dstTy, nv).getResult();
                                }
                            }

                            // 处理 Bootstrapping
                            if (nd.triggerBoot && nd.mode == Mode::SIMD && isa<simd::SIMDCipherType>(nv.getType())) {
                                auto oldTy = cast<simd::SIMDCipherType>(nv.getType());
                                auto bootedTy = simd::SIMDCipherType::get(op->getContext(), BOOT_LEVEL, oldTy.getPlaintextCount(), oldTy.getElementType());
                                nv = rewriter.create<simd::SIMDBootOp>(op->getLoc(), bootedTy, nv).getResult();
                            }

                            newOps.push_back(nv);
                        }

                        // === Step A.5: Level Alignment (关键修复) ===
                        // 如果是 Add/Sub/Min/Select/Cmp/Div，必须保证所有 SIMD 操作数 Level 一致
                        // 我们找到最小的 Level，将其他操作数 ModSwitch 到该 Level
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
                                                // 创建 ModSwitchOp 进行降级
                                                auto targetTy = simd::SIMDCipherType::get(
                                                    op->getContext(), minLevel, st.getPlaintextCount(), st.getElementType());

                                                // 注意：如果没有定义 SIMDModSwitchOp，你可以暂时用 SIMDRescaleOp 替代，
                                                // 但语义上 ModSwitch 更准确 (不除以缩放因子)
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
                                    // 此时 newOps 已经对齐，取第一个 SIMD 操作数的 Level 即可
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
                                // Select 通常三个操作数都需要对齐 (Condition, True, False)
                                newOp = rewriter.create<simd::SIMDSelectOp>(op->getLoc(), resTy, newOps[0], newOps[1], newOps[2]);
                            } else if (isa<simd::SIMDCmpOp>(op)) {
                                auto pred = cast<simd::SIMDCmpOp>(op).getPredicate();
                                newOp = rewriter.create<simd::SIMDCmpOp>(op->getLoc(), resTy, newOps[0], newOps[1], pred);
                            } else if (isa<simd::SIMDDivOp>(op)) {
                                newOp = rewriter.create<simd::SIMDDivOp>(op->getLoc(), resTy, newOps);
                            } else if (isa<simd::SIMDReduceAddOp>(op)) {
                                newOp = rewriter.create<simd::SIMDReduceAddOp>(op->getLoc(), resTy, newOps[0]);
                            } else if (isa<simd::SIMDStoreOp>(op)) {
                                rewriter.create<simd::SIMDStoreOp>(op->getLoc(), newOps[0], newOps[1]);
                                // [重要] 不要在这里 eraseOp，留给最后一步
                                continue;
                            } else if (isa<simd::SIMDRescaleOp>(op)) {
                                // Rescale 的逻辑：输入 Level -> 输出 Level
                                // 如果 nd.mode 是 SISD，这可能不合法（除非有 SISDRescale）
                                // 这里假设保持 SIMD

                                // 重新计算 Rescale 的输出 Level
                                // 通常 Rescale 降低一层。
                                // 但 ModeSelect 可能改变了决策。
                                // 简单起见，我们根据 nd.finalLevel 创建一个新的 Rescale

                                auto inputCipher = newOps[0];
                                int inputLevel = 0;
                                if (auto st = dyn_cast<simd::SIMDCipherType>(inputCipher.getType())) {
                                    inputLevel = st.getLevel();
                                }

                                // 目标 Level
                                int targetLevel = nd.finalLevel;

                                // 如果目标 Level >= 输入 Level，说明不需要 Rescale (或者该 Rescale 是多余的)
                                if (targetLevel >= inputLevel) {
                                    // 消除 Rescale：直接将结果映射为输入
                                    // 注意：这里没有创建 newOp，我们需要特殊处理 rewriteMap
                                    rewriteMap[op->getResult(0)] = inputCipher;
                                    continue;
                                }

                                // 否则，创建新的 Rescale
                                auto resTy = simd::SIMDCipherType::get(op->getContext(), targetLevel, vecCnt, rewriter.getI64Type());
                                newOp = rewriter.create<simd::SIMDRescaleOp>(op->getLoc(), resTy, inputCipher);
                            } else if (isa<simd::SIMDModSwitchOp>(op)) {
                                // 同 Rescale 逻辑
                                auto inputCipher = newOps[0];
                                int inputLevel = 0;
                                if (auto st = dyn_cast<simd::SIMDCipherType>(inputCipher.getType())) {
                                    inputLevel = st.getLevel();
                                }
                                int targetLevel = nd.finalLevel;

                                if (targetLevel >= inputLevel) {
                                    rewriteMap[op->getResult(0)] = inputCipher;
                                    continue;
                                }

                                auto resTy = simd::SIMDCipherType::get(op->getContext(), targetLevel, vecCnt, rewriter.getI64Type());
                                newOp = rewriter.create<simd::SIMDModSwitchOp>(op->getLoc(), resTy, inputCipher);
                            }
                        }

                        if (newOp) {
                            rewriteMap[op->getResult(0)] = newOp->getResult(0);
                        }
                    }

                    // 5. Final Replacement (Safe Deletion)
                    // for (Operation* op : topo) {
                    //     if (op->getNumResults() > 0) {
                    //         Value oldVal = op->getResult(0);
                    //         if (rewriteMap.count(oldVal)) {
                    //             Value newVal = rewriteMap[oldVal];
                    //             oldVal.replaceAllUsesWith(newVal);
                    //         }
                    //     }
                    //     op->erase();
                    // }

                    // ------------------------------------------------------------
                    // 5. Final Replacement (Safe Deletion)
                    // ------------------------------------------------------------
                    for (Operation* op : topo) {
                        // A. 替换结果引用
                        // 遍历该 Op 的所有结果 (Result)
                        for (int i = 0; i < op->getNumResults(); ++i) {
                            Value oldVal = op->getResult(i);

                            // 检查是否有对应的重写映射
                            if (rewriteMap.count(oldVal)) {
                                Value newVal = rewriteMap[oldVal];
                                // 核心操作：将所有使用 oldVal 的地方替换为 newVal
                                oldVal.replaceAllUsesWith(newVal);
                            }
                        }

                        // B. 安全删除
                        // 只有当 Op 没有任何 User 时才删除。
                        // 如果 rewriteMap 漏掉了某种情况，use_empty() 会返回 false。
                        // 此时我们跳过删除，虽然会留下死代码，但能防止编译器 Crash (Assertion failed)。
                        if (op->use_empty()) {
                            // 使用 rewriter 删除可以通知监听器（如果有）
                            rewriter.eraseOp(op);
                        } else {
                            // [可选] 调试信息：打印出哪些 Op 没被清理干净
                            // llvm::errs() << "[Warning] Op not erased due to remaining uses: " << getOpName(op) << "\n";
                        }
                    }

                    // ------------------------------------------------------------
                    // 6. Fix Loop Signatures (Post-Processing) [关键修复]
                    // ------------------------------------------------------------
                    // ModeSelect 可能降低了循环内部的 Level (例如 Yield L30)，
                    // 但 Loop Op 本身还保留着旧的签名 (L31)。我们需要更新 Loop。

                    // 我们收集需要更新的 Loop，避免在遍历时修改
                    SmallVector<affine::AffineForOp> loopsToFix;
                    func.walk([&](affine::AffineForOp forOp) {
                        if (forOp.getNumResults() == 0)
                            return;
                        loopsToFix.push_back(forOp);
                    });

                    for (auto forOp : loopsToFix) {
                        Block* body = forOp.getBody();
                        auto yieldOp = cast<affine::AffineYieldOp>(body->getTerminator());

                        bool needsUpdate = false;
                        SmallVector<Type> newTypes;
                        SmallVector<Value> newInits;

                        rewriter.setInsertionPoint(forOp); // 确保在 Loop 之前插入 Cast/ModSwitch

                        // 检查 Yield vs Init/Arg 的类型一致性
                        for (auto it : llvm::zip(forOp.getInits(), forOp.getRegionIterArgs(), yieldOp.getOperands())) {
                            Value init = std::get<0>(it);
                            Value yielded = std::get<2>(it);

                            Type yieldTy = yielded.getType();
                            Type initTy = init.getType();

                            if (yieldTy != initTy) {
                                needsUpdate = true;

                                // 我们需要将 'init' (可能是 L31) 降级为 'yieldTy' (L30)
                                Value newInit = init;

                                // 尝试使用 SIMD ModSwitch 进行降级
                                if (auto simdDst = dyn_cast<simd::SIMDCipherType>(yieldTy)) {
                                    if (auto simdSrc = dyn_cast<simd::SIMDCipherType>(initTy)) {
                                        if (simdSrc.getLevel() > simdDst.getLevel()) {
                                            //
                                            newInit = rewriter.create<simd::SIMDModSwitchOp>(forOp.getLoc(), yieldTy, init).getResult();
                                        }
                                    }
                                }

                                // 如果 ModSwitch 不适用（类型完全不同），使用通用 Cast 兜底
                                if (newInit.getType() != yieldTy) {
                                    newInit = rewriter.create<UnrealizedConversionCastOp>(forOp.getLoc(), yieldTy, newInit).getResult(0);
                                }

                                newInits.push_back(newInit);
                                newTypes.push_back(yieldTy);
                            } else {
                                newInits.push_back(init);
                                newTypes.push_back(initTy);
                            }
                        }

                        if (!needsUpdate)
                            continue;

                        // === 创建新的 Loop ===
                        auto newLoop = rewriter.create<affine::AffineForOp>(
                            forOp.getLoc(),
                            forOp.getLowerBoundOperands(), forOp.getLowerBoundMap(),
                            forOp.getUpperBoundOperands(), forOp.getUpperBoundMap(),
                            forOp.getStep().getSExtValue(),
                            newInits,
                            [&](OpBuilder&, Location, Value, ValueRange) {} // 空 Body
                        );

                        // === 迁移 Body ===
                        Block* newBody = newLoop.getBody();
                        newBody->clear(); // 清除默认 terminator

                        // 建立参数映射: Old IV/Args -> New IV/Args
                        SmallVector<Value> mapArgs;
                        mapArgs.push_back(newLoop.getInductionVar());
                        for (auto arg : newLoop.getRegionIterArgs())
                            mapArgs.push_back(arg);

                        rewriter.mergeBlocks(body, newBody, mapArgs);

                        // 强制更新 Block Argument 类型 (mergeBlocks 可能保留了旧类型引用)
                        for (size_t i = 0; i < newLoop.getRegionIterArgs().size(); ++i) {
                            newLoop.getRegionIterArgs()[i].setType(newTypes[i]);
                        }

                        // === 替换旧 Loop ===
                        // 注意：新 Loop 返回 L30，旧 Loop 的 User 可能期待 L31。
                        // 为了通过验证，我们需要 Cast Back，或者如果 User 不介意（如 Store），则直接替换。
                        // 最安全的方法是插入 CastBack，让 Canonicalizer 去消除它。
                        for (size_t i = 0; i < forOp.getNumResults(); ++i) {
                            Value oldRes = forOp.getResult(i);
                            Value newRes = newLoop.getResult(i);

                            if (oldRes.getType() != newRes.getType()) {
                                Value castBack = rewriter.create<UnrealizedConversionCastOp>(
                                                             forOp.getLoc(), oldRes.getType(), newRes)
                                                     .getResult(0);
                                oldRes.replaceAllUsesWith(castBack);
                            } else {
                                oldRes.replaceAllUsesWith(newRes);
                            }
                        }

                        // 安全删除旧 Loop
                        forOp.erase();
                    }
                }
            }
        };

    } // namespace
} // namespace mlir::libra::mdsel