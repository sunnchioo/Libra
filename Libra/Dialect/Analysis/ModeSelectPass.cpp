#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/MemoryBuffer.h"

#include "ModeSelectPass.h"
#include "SIMDOps.h"
#include "SISDOps.h"
#include "SIMDCommon.h"

#include <algorithm>
#include <limits>
#include <vector>
#include <memory>
#include <cmath>

using namespace mlir;
using namespace mlir::libra;

namespace mlir::libra::mdsel {

#define GEN_PASS_DEF_CONVERTTOMDSELPASS
#include "ModeSelectPass.h.inc"

    namespace {

        // --- Constants ---
        constexpr int MAX_SIMD_LEVEL = simd::DEFAULT_LEVEL;
        constexpr int STATE_SISD = MAX_SIMD_LEVEL + 1;
        constexpr int NUM_STATES = STATE_SISD + 1;
        constexpr double INF_COST = 1e15;

        // --- Enums ---
        enum class Mode { SIMD,
                          SISD };

        // --- Cost Model ---
        struct CostModel {
            llvm::StringMap<llvm::DenseMap<int64_t, double>> costTable;

            CostModel(StringRef jsonFilename) {
                llvm::errs() << "[CostModel] Loading from: " << jsonFilename << "\n";
                auto fileOrErr = llvm::MemoryBuffer::getFile(jsonFilename);
                if (auto ec = fileOrErr.getError()) {
                    llvm::errs() << "[Error] Cannot open cost model: " << jsonFilename << "\n";
                    return;
                }
                std::unique_ptr<llvm::MemoryBuffer> fileBuffer = std::move(*fileOrErr);
                llvm::Expected<llvm::json::Value> rootOrErr = llvm::json::parse(fileBuffer->getBuffer());
                if (!rootOrErr)
                    return;
                auto* obj = rootOrErr->getAsObject();
                if (!obj)
                    return;
                auto* costs = obj->getObject("costs");
                if (!costs)
                    return;

                int entryCount = 0;
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
                                    entryCount++;
                                }
                            }
                        }
                    }
                }
                llvm::errs() << "[CostModel] Loaded " << entryCount << " entries.\n";
            }

            double lookup(StringRef key, int64_t idx) const {
                auto it = costTable.find(key);
                if (it == costTable.end())
                    return INF_COST;
                auto jt = it->second.find(idx);
                if (jt == it->second.end())
                    return INF_COST;
                return jt->second;
            }

            double getBootCost(int64_t targetLevel) const {
                return lookup("simd.boot", targetLevel);
            }

            double getCastCost(bool fromSimd, bool toSimd, int64_t dim) const {
                if (fromSimd == toSimd)
                    return 0.0;
                std::string key = fromSimd ? "simd.cast_to_sisd" : "sisd.cast_to_simd";
                return lookup(key, dim);
            }
        };

        // --- Graph Structures ---
        struct Edge {
            Operation* producer;
            Operation* consumer;
            Value val;
        };

        struct Node {
            Operation* op;
            SmallVector<Edge*, 2> inEdges;
            SmallVector<Edge*, 2> outEdges;
            int vectorCount = -1;
            std::vector<double> dpCost;
            int chosenOutputState = -1;

            Node() : dpCost(NUM_STATES, INF_COST) {}
        };

        // --- Main Pass ---
        class ConvertToModeSelectIR : public impl::ConvertToMDSELPassBase<ConvertToModeSelectIR> {
        public:
            using impl::ConvertToMDSELPassBase<ConvertToModeSelectIR>::ConvertToMDSELPassBase;

            StringRef getOptionFilePath() {
                return "/mnt/data0/home/syt/Libra/Libra/Dialect/Analysis/cost_table.json";
            }

            int getLevelConsumption(Operation* op) {
                if (isa<simd::SIMDMultOp>(op))
                    return 1;
                if (isa<simd::SIMDMinOp, sisd::SISDMinOp>(op))
                    return 3;
                return 0;
            }

            std::string getOpKeyName(Operation* op, bool isSimd) {
                if (isSimd) {
                    if (isa<simd::SIMDAddOp>(op))
                        return "simd.add";
                    if (isa<simd::SIMDSubOp>(op))
                        return "simd.sub";
                    if (isa<simd::SIMDMultOp>(op))
                        return "simd.mult";
                    if (isa<simd::SIMDMinOp>(op))
                        return "simd.min";
                } else {
                    if (isa<simd::SIMDAddOp, sisd::SISDAddOp>(op))
                        return "sisd.add";
                    if (isa<simd::SIMDSubOp, sisd::SISDSubOp>(op))
                        return "sisd.sub";
                    if (isa<simd::SIMDMinOp, sisd::SISDMinOp>(op))
                        return "sisd.min";
                }
                return "";
            }

            // ----------------------------------------------------------------
            // Phase 1: Shape Inference
            // ----------------------------------------------------------------
            void runShapeInference(const std::vector<Node*>& topo,
                                   const llvm::DenseMap<Operation*, std::unique_ptr<Node>>& nodeMap) {
                llvm::errs() << "\n=== [Phase 1] Shape Inference ===\n";
                for (Node* n : topo) {
                    // [修改 1] 初始化为 -1，表示尚未确定
                    int64_t currentCount = -1;

                    // [步骤 A] 尝试从父节点继承 (Propagation)
                    if (!n->inEdges.empty()) {
                        Operation* parentOp = n->inEdges[0]->producer;
                        auto it = nodeMap.find(parentOp);
                        if (it != nodeMap.end()) {
                            currentCount = it->second->vectorCount;
                        }
                    }

                    // [步骤 B] 如果无法继承 (源节点 或 父节点不在图中)，从类型提取 (Type Analysis)
                    if (currentCount == -1) {
                        Type resTy = n->op->getResult(0).getType();

                        // 检查是否为 SIMD 类型
                        if (auto t = dyn_cast<simd::SIMDCipherType>(resTy)) {
                            currentCount = t.getPlaintextCount();
                        }
                        // [新增] 检查是否为 SISD 类型 (防止源节点是 SISD)
                        else if (auto t = dyn_cast<sisd::SISDCipherType>(resTy)) {
                            currentCount = t.getPlaintextCount();
                        }
                    }

                    // [步骤 C] 兜底处理 (Safety Fallback)
                    // 如果既没有父节点，类型里也读不到 (即 currentCount 还是 -1)，则设为默认值 8
                    if (currentCount == -1) {
                        llvm::errs() << "[Warning] Shape inference failed for " << n->op->getName() << ", defaulting to 8.\n";
                        currentCount = 8;
                    }

                    // [步骤 D] 算子特殊逻辑 (Op Specific Logic)
                    if (isa<simd::SIMDMinOp, sisd::SISDMinOp>(n->op)) {
                        n->vectorCount = 1;
                    } else {
                        n->vectorCount = currentCount;
                    }

                    llvm::errs() << "Node: " << n->op->getName() << " | VectorCount: " << n->vectorCount << "\n";
                }
            }

            // ----------------------------------------------------------------
            // Main Run
            // ----------------------------------------------------------------
            void runOnOperation() override {
                ModuleOp module = getOperation();
                CostModel costModel(getOptionFilePath());
                // double bootCost = costModel.getBootCost();

                for (func::FuncOp func : module.getOps<func::FuncOp>()) {
                    llvm::errs() << "\nProcessing Function: " << func.getName() << "\n";
                    llvm::DenseMap<Operation*, std::unique_ptr<Node>> nodeMap;
                    std::vector<Node*> topo;
                    std::vector<std::unique_ptr<Edge>> allEdges;

                    // 1. Build Nodes
                    func.walk([&](Operation* op) {
                        if (isa<simd::SIMDSubOp, simd::SIMDMultOp, simd::SIMDMinOp,
                                simd::SIMDEncryptOp, simd::SIMDDecryptOp, simd::SIMDAddOp,
                                sisd::SISDSubOp, sisd::SISDMinOp, sisd::SISDAddOp,
                                sisd::SISDEncryptOp, sisd::SISDDecryptOp>(op)) {
                            auto n = std::make_unique<Node>();
                            n->op = op;
                            nodeMap[op] = std::move(n);
                        }
                    });

                    // 2. Build Edges
                    for (auto& p : nodeMap) {
                        Operation* op = p.first;
                        Node* node = p.second.get();
                        for (Value v : op->getOperands()) {
                            if (Operation* def = v.getDefiningOp()) {
                                if (nodeMap.count(def)) {
                                    allEdges.push_back(std::make_unique<Edge>(Edge{def, op, v}));
                                    Edge* e = allEdges.back().get();
                                    node->inEdges.push_back(e);
                                    nodeMap[def]->outEdges.push_back(e);
                                }
                            }
                        }
                    }

                    // 3. Topo Sort
                    llvm::DenseMap<Operation*, int> indeg;
                    SmallVector<Operation*, 64> stack;
                    for (auto& p : nodeMap) {
                        indeg[p.first] = p.second->inEdges.size();
                        if (p.second->inEdges.empty())
                            stack.push_back(p.first);
                    }
                    while (!stack.empty()) {
                        Operation* op = stack.pop_back_val();
                        Node* n = nodeMap[op].get();
                        topo.push_back(n);
                        for (Edge* e : n->outEdges) {
                            indeg[e->consumer]--;
                            if (indeg[e->consumer] == 0)
                                stack.push_back(e->consumer);
                        }
                    }

                    llvm::errs() << "Topo Sort Order:\n";
                    for (auto* n : topo) llvm::errs() << "  -> " << n->op->getName() << "\n";

                    // 4. Shape Inference
                    runShapeInference(topo, nodeMap);

                    // ------------------------------------------------------------
                    // 5. DP: Bottom-Up
                    // ------------------------------------------------------------
                    llvm::errs() << "\n=== [Phase 2] DP Bottom-Up ===\n";
                    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
                        Node* u = *it;
                        Operation* op = u->op;
                        int consume = getLevelConsumption(op);
                        int64_t vecCnt = u->vectorCount;

                        llvm::errs() << "[DP] Processing " << op->getName() << " (Consume=" << consume << ")\n";

                        // --- A. SISD State ---
                        bool supportsSISD = true;
                        if (isa<simd::SIMDMultOp>(op))
                            supportsSISD = false;

                        if (supportsSISD) {
                            double childrenSum = 0;
                            bool possible = true;
                            for (Edge* e : u->outEdges) {
                                Node* v = nodeMap[e->consumer].get();
                                double minV = INF_COST;
                                // Child uses SISD
                                minV = std::min(minV, v->dpCost[STATE_SISD]);
                                // Child uses SIMD (Cast SISD->SIMD)
                                double castC = costModel.getCastCost(false, true, vecCnt);
                                for (int k = 0; k <= MAX_SIMD_LEVEL; ++k) {
                                    minV = std::min(minV, v->dpCost[k] + castC);
                                }
                                if (minV >= INF_COST) {
                                    possible = false;
                                    break;
                                }
                                childrenSum += minV;
                            }

                            if (possible) {
                                double localC = 0.0;
                                if (isa<simd::SIMDEncryptOp, sisd::SISDEncryptOp,
                                        simd::SIMDDecryptOp, sisd::SISDDecryptOp>(op)) {
                                    localC = 0.0;
                                } else {
                                    std::string key = getOpKeyName(op, false);
                                    localC = costModel.lookup(key, vecCnt);
                                }

                                if (localC < INF_COST) {
                                    u->dpCost[STATE_SISD] = localC + childrenSum;
                                    llvm::errs() << "  -> SISD Cost: " << u->dpCost[STATE_SISD] << " (Local=" << localC << " Children=" << childrenSum << ")\n";
                                }
                            }
                        }

                        // --- B. SIMD States ---
                        // for (int L = 0; L <= MAX_SIMD_LEVEL; ++L) {
                        //     double childrenSum = 0;
                        //     bool possible = true;
                        //     for (Edge* e : u->outEdges) {
                        //         Node* v = nodeMap[e->consumer].get();
                        //         double minV = INF_COST;
                        //         // Child uses SISD
                        //         double castToSisd = costModel.getCastCost(true, false, vecCnt);
                        //         minV = std::min(minV, v->dpCost[STATE_SISD] + castToSisd);
                        //         // Child uses SIMD
                        //         for (int K = 0; K <= MAX_SIMD_LEVEL; ++K) {
                        //             int reqIn = K + getLevelConsumption(v->op);
                        //             if (L >= reqIn)
                        //                 minV = std::min(minV, v->dpCost[K]);
                        //         }
                        //         if (minV >= INF_COST) {
                        //             possible = false;
                        //             break;
                        //         }
                        //         childrenSum += minV;
                        //     }
                        //     if (!possible)
                        //         continue;

                        //     if (isa<simd::SIMDEncryptOp, simd::SIMDDecryptOp>(op)) {
                        //         u->dpCost[L] = std::min(u->dpCost[L], 0.0 + childrenSum);
                        //     } else {
                        //         std::string key = getOpKeyName(op, true);
                        //         int reqInput = L + consume;

                        //         // Path 1: Normal
                        //         if (reqInput <= MAX_SIMD_LEVEL) {
                        //             double opC = costModel.lookup(key, reqInput);
                        //             if (opC < INF_COST) {
                        //                 double val = opC + childrenSum;
                        //                 if (val < u->dpCost[L])
                        //                     u->dpCost[L] = val;
                        //             }
                        //         }

                        //         // Path 2: Bootstrapping
                        //         if (L <= MAX_SIMD_LEVEL - consume) {
                        //             double opC = costModel.lookup(key, MAX_SIMD_LEVEL);
                        //             double total = bootCost + opC + childrenSum;
                        //             if (total < u->dpCost[L]) {
                        //                 u->dpCost[L] = total;
                        //                 // 仅在 debug 开启时打印，避免刷屏，这里默认打印关键层
                        //             }
                        //         }
                        //     }
                        //     // Log significant updates
                        //     if (u->dpCost[L] < INF_COST) {
                        //         // llvm::errs() << "  -> SIMD L=" << L << " Cost=" << u->dpCost[L] << "\n";
                        //     }
                        // }
                        // --- B. SIMD States ---
                        for (int L = 0; L <= MAX_SIMD_LEVEL; ++L) {
                            double childrenSum = 0;
                            bool possible = true;
                            // ... (Children sum calculation 保持不变，复制原代码即可) ...
                            for (Edge* e : u->outEdges) {
                                Node* v = nodeMap[e->consumer].get();
                                double minV = INF_COST;
                                double castToSisd = costModel.getCastCost(true, false, vecCnt);
                                minV = std::min(minV, v->dpCost[STATE_SISD] + castToSisd);
                                for (int K = 0; K <= MAX_SIMD_LEVEL; ++K) {
                                    int reqIn = K + getLevelConsumption(v->op);
                                    if (L >= reqIn)
                                        minV = std::min(minV, v->dpCost[K]);
                                }
                                if (minV >= INF_COST) {
                                    possible = false;
                                    break;
                                }
                                childrenSum += minV;
                            }
                            if (!possible)
                                continue;

                            if (isa<simd::SIMDEncryptOp, simd::SIMDDecryptOp>(op)) {
                                u->dpCost[L] = std::min(u->dpCost[L], 0.0 + childrenSum);
                            } else {
                                std::string key = getOpKeyName(op, true);
                                int reqInput = L + consume;

                                // [关键修改] 只保留 Normal Path。
                                // 如果 reqInput > MAX_SIMD_LEVEL，dpCost[L] 保持 INF。
                                // Boot 的决策移交给 Phase 3 的边转换逻辑。
                                if (reqInput <= MAX_SIMD_LEVEL) {
                                    double opC = costModel.lookup(key, reqInput);
                                    if (opC < INF_COST) {
                                        double val = opC + childrenSum;
                                        if (val < u->dpCost[L])
                                            u->dpCost[L] = val;
                                    }
                                }
                            }
                        }

                        // Print summary of valid SIMD levels
                        llvm::errs() << "  -> Valid SIMD Levels: ";
                        for (int k = 0; k <= MAX_SIMD_LEVEL; ++k)
                            if (u->dpCost[k] < INF_COST)
                                llvm::errs() << k << " ";
                        llvm::errs() << "\n";
                    }

                    // ------------------------------------------------------------
                    // 6. Selection & Rewrite
                    // ------------------------------------------------------------
                    // llvm::errs() << "\n=== [Phase 3] Selection & Rewrite ===\n";
                    // llvm::DenseMap<Value, Value> rewriteMap;
                    // IRRewriter rewriter(&getContext());

                    // for (Node* n : topo) {
                    //     Operation* op = n->op;

                    //     // 6a. Choose Best State (保持不变)
                    //     double bestVal = INF_COST;
                    //     int bestS = -1;

                    //     for (int s = 0; s < NUM_STATES; ++s) {
                    //         if (n->dpCost[s] >= INF_COST)
                    //             continue;

                    //         double transitionCost = 0;
                    //         bool possible = true;

                    //         for (Edge* e : n->inEdges) {
                    //             Node* parent = nodeMap[e->producer].get();
                    //             int pState = parent->chosenOutputState;
                    //             double edgeC = 0;

                    //             if (pState == STATE_SISD) {
                    //                 if (s != STATE_SISD)
                    //                     edgeC = costModel.getCastCost(false, true, n->vectorCount);
                    //             } else {
                    //                 if (s == STATE_SISD) {
                    //                     edgeC = costModel.getCastCost(true, false, n->vectorCount);
                    //                 } else {
                    //                     int req = s + getLevelConsumption(op);
                    //                     if (req > MAX_SIMD_LEVEL) {
                    //                         edgeC = 0;
                    //                     } else {
                    //                         if (pState < req)
                    //                             possible = false;
                    //                     }
                    //                 }
                    //             }
                    //             transitionCost += edgeC;
                    //         }

                    //         if (possible) {
                    //             if (transitionCost + n->dpCost[s] < bestVal) {
                    //                 bestVal = transitionCost + n->dpCost[s];
                    //                 bestS = s;
                    //             }
                    //         }
                    //     }

                    //     if (bestS == -1) {
                    //         llvm::errs() << "Fatal: No valid state found for " << op->getName() << "\n";
                    //         return;
                    //     }
                    //     n->chosenOutputState = bestS;
                    //     llvm::errs() << "Node " << op->getName() << " Chosen State: " << (bestS == STATE_SISD ? "SISD" : ("SIMD Level " + std::to_string(bestS))) << " (TotalCost=" << bestVal << ")\n";

                    //     // 6b. Generate IR [核心修复部分]
                    //     rewriter.setInsertionPoint(op);
                    //     SmallVector<Value, 2> newOperands;

                    //     // [修复] 遍历原 Op 的操作数，而不是 Edge
                    //     for (Value oldOperand : op->getOperands()) {
                    //         Operation* defOp = oldOperand.getDefiningOp();

                    //         // 检查这个操作数是否来自我们管理的节点 (Graph Internal Edge)
                    //         if (defOp && nodeMap.count(defOp)) {
                    //             Node* parent = nodeMap[defOp].get();
                    //             int pState = parent->chosenOutputState;

                    //             Value incoming = rewriteMap.lookup(oldOperand);
                    //             if (!incoming)
                    //                 incoming = oldOperand;  // Should catch by rewriteMap ideally

                    //             bool needSISD = (bestS == STATE_SISD);
                    //             Value processed = incoming;

                    //             // Handle Casts
                    //             if (pState == STATE_SISD && !needSISD) {
                    //                 llvm::errs() << "  [Insert] Cast SISD -> SIMD\n";
                    //                 auto newTy = simd::SIMDCipherType::get(op->getContext(), MAX_SIMD_LEVEL, n->vectorCount, rewriter.getI64Type());
                    //                 processed = rewriter.create<sisd::SISDCastSISDCipherToSIMDCipherOp>(op->getLoc(), newTy, processed);
                    //             } else if (pState != STATE_SISD && needSISD) {
                    //                 llvm::errs() << "  [Insert] Cast SIMD -> SISD\n";
                    //                 auto newTy = sisd::SISDCipherType::get(op->getContext(), n->vectorCount, rewriter.getI64Type());
                    //                 processed = rewriter.create<simd::SIMDCastSIMDCipherToSISDCipherOp>(op->getLoc(), newTy, processed);
                    //             }

                    //             // Handle Bootstrapping
                    //             if (!needSISD && pState != STATE_SISD) {
                    //                 int req = bestS + getLevelConsumption(op);
                    //                 if (req > MAX_SIMD_LEVEL) {
                    //                     llvm::errs() << "  [Insert] Bootstrapping\n";
                    //                     auto bootTy = simd::SIMDCipherType::get(op->getContext(), MAX_SIMD_LEVEL, n->vectorCount, rewriter.getI64Type());
                    //                     // processed = rewriter.create<simd::SIMDBootOp>(op->getLoc(), bootTy, processed);
                    //                 }
                    //             }
                    //             newOperands.push_back(processed);
                    //         } else {
                    //             // [修复] 外部输入（如 Encrypt 的明文），直接透传
                    //             newOperands.push_back(oldOperand);
                    //         }
                    //     }

                    //     // Create New Op (保持不变)
                    //     Operation* newOp = nullptr;

                    //     if (bestS == STATE_SISD) {
                    //         auto resTy = sisd::SISDCipherType::get(op->getContext(), n->vectorCount, rewriter.getI64Type());

                    //         if (isa<simd::SIMDAddOp, sisd::SISDAddOp>(op)) {
                    //             newOp = rewriter.create<sisd::SISDAddOp>(op->getLoc(), resTy, newOperands[0], newOperands[1]);
                    //         } else if (isa<simd::SIMDSubOp, sisd::SISDSubOp>(op)) {
                    //             newOp = rewriter.create<sisd::SISDSubOp>(op->getLoc(), resTy, newOperands[0], newOperands[1]);
                    //         } else if (isa<simd::SIMDMinOp, sisd::SISDMinOp>(op)) {
                    //             newOp = rewriter.create<sisd::SISDMinOp>(op->getLoc(), resTy, newOperands[0]);
                    //         } else if (isa<simd::SIMDEncryptOp, sisd::SISDEncryptOp>(op)) {
                    //             newOp = rewriter.create<sisd::SISDEncryptOp>(op->getLoc(), resTy, newOperands[0]);
                    //         } else if (isa<simd::SIMDDecryptOp, sisd::SISDDecryptOp>(op)) {
                    //             newOp = rewriter.create<sisd::SISDDecryptOp>(op->getLoc(), op->getResultTypes(), newOperands[0]);
                    //         }
                    //     } else {
                    //         // SIMD
                    //         int outLvl = bestS;
                    //         auto resTy = simd::SIMDCipherType::get(op->getContext(), outLvl, n->vectorCount, rewriter.getI64Type());

                    //         if (isa<simd::SIMDAddOp, sisd::SISDAddOp>(op)) {
                    //             newOp = rewriter.create<simd::SIMDAddOp>(op->getLoc(), resTy, newOperands[0], newOperands[1]);
                    //         } else if (isa<simd::SIMDSubOp, sisd::SISDSubOp>(op)) {
                    //             newOp = rewriter.create<simd::SIMDSubOp>(op->getLoc(), resTy, newOperands[0], newOperands[1]);
                    //         } else if (isa<simd::SIMDMultOp>(op)) {
                    //             newOp = rewriter.create<simd::SIMDMultOp>(op->getLoc(), resTy, newOperands[0], newOperands[1]);
                    //         } else if (isa<simd::SIMDMinOp, sisd::SISDMinOp>(op)) {
                    //             newOp = rewriter.create<simd::SIMDMinOp>(op->getLoc(), resTy, newOperands[0]);
                    //         } else if (isa<simd::SIMDEncryptOp, sisd::SISDEncryptOp>(op)) {
                    //             newOp = rewriter.create<simd::SIMDEncryptOp>(op->getLoc(), resTy, newOperands[0]);
                    //         } else if (isa<simd::SIMDDecryptOp, sisd::SISDDecryptOp>(op)) {
                    //             newOp = rewriter.create<simd::SIMDDecryptOp>(op->getLoc(), op->getResultTypes(), newOperands[0]);
                    //         }
                    //     }

                    //     if (newOp) {
                    //         rewriteMap[op->getResult(0)] = newOp->getResult(0);
                    //     }
                    // }

                    // ------------------------------------------------------------
                    // 6. Selection & Rewrite
                    // ------------------------------------------------------------
                    llvm::errs() << "\n=== [Phase 3] Selection & Rewrite ===\n";
                    llvm::DenseMap<Value, Value> rewriteMap;
                    IRRewriter rewriter(&getContext());

                    for (Node* n : topo) {
                        Operation* op = n->op;

                        // --------------------------------------------------------
                        // 6a. Choose Best State
                        // --------------------------------------------------------
                        double bestVal = INF_COST;
                        int bestS = -1;

                        for (int s = 0; s < NUM_STATES; ++s) {
                            if (n->dpCost[s] >= INF_COST)
                                continue;

                            double transitionCost = 0;
                            bool possible = true;

                            for (Edge* e : n->inEdges) {
                                Node* parent = nodeMap[e->producer].get();
                                int pState = parent->chosenOutputState;
                                double edgeC = 0;

                                if (pState == STATE_SISD) {
                                    if (s != STATE_SISD)  // SISD -> SIMD Cast
                                        edgeC = costModel.getCastCost(false, true, n->vectorCount);
                                } else {
                                    if (s == STATE_SISD) {  // SIMD -> SISD Cast
                                        edgeC = costModel.getCastCost(true, false, n->vectorCount);
                                    } else {
                                        // SIMD -> SIMD
                                        // 计算当前 Op 在状态 s 下所需的输入 Level
                                        int req = s + getLevelConsumption(op);

                                        if (req > MAX_SIMD_LEVEL) {
                                            // 物理上不可能的操作 (需要超过 29 层的输入)
                                            possible = false;
                                        } else {
                                            // [逻辑修改] 检查是否需要 Boot
                                            if (pState < req) {
                                                // 父节点提供的 Level 不够，必须 Boot。
                                                // 成本 = Boot 到目标 req Level 的成本
                                                edgeC = costModel.getBootCost(req);
                                            } else {
                                                // 父节点 Level 足够，无额外成本
                                                edgeC = 0;
                                            }
                                        }
                                    }
                                }
                                transitionCost += edgeC;
                            }

                            if (possible) {
                                if (transitionCost + n->dpCost[s] < bestVal) {
                                    bestVal = transitionCost + n->dpCost[s];
                                    bestS = s;
                                }
                            }
                        }

                        if (bestS == -1) {
                            llvm::errs() << "Fatal: No valid state found for " << op->getName() << "\n";
                            return;
                        }
                        n->chosenOutputState = bestS;
                        llvm::errs() << "Node " << op->getName() << " Chosen State: "
                                     << (bestS == STATE_SISD ? "SISD" : ("SIMD Level " + std::to_string(bestS)))
                                     << " (TotalCost=" << bestVal << ")\n";

                        // --------------------------------------------------------
                        // 6b. Generate IR
                        // --------------------------------------------------------
                        rewriter.setInsertionPoint(op);
                        SmallVector<Value, 2> newOperands;

                        for (Value oldOperand : op->getOperands()) {
                            Operation* defOp = oldOperand.getDefiningOp();

                            // 只处理图内部的边
                            if (defOp && nodeMap.count(defOp)) {
                                Node* parent = nodeMap[defOp].get();
                                int pState = parent->chosenOutputState;

                                Value incoming = rewriteMap.lookup(oldOperand);
                                if (!incoming)
                                    incoming = oldOperand;

                                bool needSISD = (bestS == STATE_SISD);
                                Value processed = incoming;

                                // 1. Handle Casts
                                if (pState == STATE_SISD && !needSISD) {
                                    llvm::errs() << "  [Insert] Cast SISD -> SIMD\n";
                                    // Cast 后通常假设恢复到 Max Level 或者特定 Level，
                                    // 如果还需要降级/Boot，由下面的逻辑继续处理
                                    auto newTy = simd::SIMDCipherType::get(op->getContext(), MAX_SIMD_LEVEL, n->vectorCount, rewriter.getI64Type());
                                    processed = rewriter.create<sisd::SISDCastSISDCipherToSIMDCipherOp>(op->getLoc(), newTy, processed);
                                    pState = MAX_SIMD_LEVEL;  // 更新状态为 Cast 后的状态
                                } else if (pState != STATE_SISD && needSISD) {
                                    llvm::errs() << "  [Insert] Cast SIMD -> SISD\n";
                                    auto newTy = sisd::SISDCipherType::get(op->getContext(), n->vectorCount, rewriter.getI64Type());
                                    processed = rewriter.create<simd::SIMDCastSIMDCipherToSISDCipherOp>(op->getLoc(), newTy, processed);
                                }

                                // 2. Handle Bootstrapping
                                if (!needSISD && pState != STATE_SISD) {
                                    int req = bestS + getLevelConsumption(op);

                                    // [使用 SIMDBootOp]
                                    if (pState < req) {
                                        llvm::errs() << "  [Insert] Bootstrapping to Level " << req << "\n";

                                        // 构造目标类型：Level = req
                                        auto bootTy = simd::SIMDCipherType::get(op->getContext(), req, n->vectorCount, rewriter.getI64Type());

                                        // 使用您定义的 SIMDBootOp
                                        processed = rewriter.create<simd::SIMDBootOp>(op->getLoc(), bootTy, processed);
                                    }
                                }
                                newOperands.push_back(processed);
                            } else {
                                // 外部输入直接透传
                                newOperands.push_back(oldOperand);
                            }
                        }

                        // 3. Create New Op
                        Operation* newOp = nullptr;

                        if (bestS == STATE_SISD) {
                            auto resTy = sisd::SISDCipherType::get(op->getContext(), n->vectorCount, rewriter.getI64Type());
                            if (isa<simd::SIMDAddOp, sisd::SISDAddOp>(op)) {
                                newOp = rewriter.create<sisd::SISDAddOp>(op->getLoc(), resTy, newOperands[0], newOperands[1]);
                            } else if (isa<simd::SIMDSubOp, sisd::SISDSubOp>(op)) {
                                newOp = rewriter.create<sisd::SISDSubOp>(op->getLoc(), resTy, newOperands[0], newOperands[1]);
                            } else if (isa<simd::SIMDMinOp, sisd::SISDMinOp>(op)) {
                                newOp = rewriter.create<sisd::SISDMinOp>(op->getLoc(), resTy, newOperands[0]);
                            } else if (isa<simd::SIMDEncryptOp, sisd::SISDEncryptOp>(op)) {
                                newOp = rewriter.create<sisd::SISDEncryptOp>(op->getLoc(), resTy, newOperands[0]);
                            } else if (isa<simd::SIMDDecryptOp, sisd::SISDDecryptOp>(op)) {
                                newOp = rewriter.create<sisd::SISDDecryptOp>(op->getLoc(), op->getResultTypes(), newOperands[0]);
                            }
                        } else {
                            // SIMD
                            int outLvl = bestS;
                            auto resTy = simd::SIMDCipherType::get(op->getContext(), outLvl, n->vectorCount, rewriter.getI64Type());

                            if (isa<simd::SIMDAddOp, sisd::SISDAddOp>(op)) {
                                newOp = rewriter.create<simd::SIMDAddOp>(op->getLoc(), resTy, newOperands[0], newOperands[1]);
                            } else if (isa<simd::SIMDSubOp, sisd::SISDSubOp>(op)) {
                                newOp = rewriter.create<simd::SIMDSubOp>(op->getLoc(), resTy, newOperands[0], newOperands[1]);
                            } else if (isa<simd::SIMDMultOp>(op)) {
                                newOp = rewriter.create<simd::SIMDMultOp>(op->getLoc(), resTy, newOperands[0], newOperands[1]);
                            } else if (isa<simd::SIMDMinOp, sisd::SISDMinOp>(op)) {
                                newOp = rewriter.create<simd::SIMDMinOp>(op->getLoc(), resTy, newOperands[0]);
                            } else if (isa<simd::SIMDEncryptOp, sisd::SISDEncryptOp>(op)) {
                                newOp = rewriter.create<simd::SIMDEncryptOp>(op->getLoc(), resTy, newOperands[0]);
                            } else if (isa<simd::SIMDDecryptOp, sisd::SISDDecryptOp>(op)) {
                                newOp = rewriter.create<simd::SIMDDecryptOp>(op->getLoc(), op->getResultTypes(), newOperands[0]);
                            }
                        }

                        if (newOp) {
                            rewriteMap[op->getResult(0)] = newOp->getResult(0);
                        }
                    }

                    // Cleanup Old Ops
                    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
                        Operation* oldOp = (*it)->op;
                        if (rewriteMap.count(oldOp->getResult(0))) {
                            rewriter.replaceOp(oldOp, rewriteMap[oldOp->getResult(0)]);
                        }
                    }
                }
            }
        };
    }
}