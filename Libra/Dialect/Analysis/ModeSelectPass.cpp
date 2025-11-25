// #include "mlir/Dialect/Func/IR/FuncOps.h"
// #include "mlir/IR/Builders.h"
// #include "mlir/IR/PatternMatch.h"
// #include "mlir/Pass/Pass.h"
// #include "mlir/Transforms/GreedyPatternRewriteDriver.h"
// #include "mlir/Dialect/MemRef/IR/MemRef.h"
// #include "mlir/Dialect/Arith/IR/Arith.h"

// #include "llvm/ADT/DenseMap.h"
// #include "llvm/ADT/StringMap.h"
// #include "llvm/Support/JSON.h"
// #include "llvm/Support/raw_ostream.h"
// #include "llvm/Support/MemoryBuffer.h"

// #include "ModeSelectPass.h"
// #include "SIMDOps.h"
// #include "SISDOps.h"
// #include "SIMDCommon.h"

// #include <algorithm>
// #include <limits>
// #include <vector>
// #include <memory>
// #include <cmath>
// #include <tuple>
// #include <string>

// using namespace mlir;
// using namespace mlir::libra;

// namespace mlir::libra::mdsel {

// #define GEN_PASS_DEF_CONVERTTOMDSELPASS
// #include "ModeSelectPass.h.inc"

//     namespace {

// // ============================================================
// // [Debug Switch]
// // ============================================================
// #define MDSEL_DEBUG 1

// #if MDSEL_DEBUG
// #define DBG(msg) llvm::errs() << "[ModeSel] " << msg
// #else
// #define DBG(msg)
// #endif

//         // --- Constants ---
//         constexpr int MAX_SIMD_LEVEL = 29;
//         constexpr double INF_COST = 1e15;

//         // --- NodeState ---
//         struct NodeState {
//             enum class Mode { SIMD,
//                               SISD };
//             Mode mode;
//             int level;
//             int scale;
//             int basis;

//             static NodeState createSISD() { return {Mode::SISD, 0, 0, 0}; }
//             static NodeState createSIMD(int l, int s, int b) { return {Mode::SIMD, l, s, b}; }

//             bool isSISD() const { return mode == Mode::SISD; }
//             bool isSIMD() const { return mode == Mode::SIMD; }

//             static constexpr int SIMD_STATE_COUNT = (MAX_SIMD_LEVEL + 1) * 4;
//             static constexpr int TOTAL_STATES = SIMD_STATE_COUNT + 1;

//             int toIndex() const {
//                 if (mode == Mode::SISD)
//                     return SIMD_STATE_COUNT;
//                 return level * 4 + (scale - 1) * 2 + (basis - 2);
//             }

//             static NodeState fromIndex(int idx) {
//                 if (idx == SIMD_STATE_COUNT)
//                     return createSISD();
//                 int b = (idx % 2) + 2;
//                 int s = ((idx / 2) % 2) + 1;
//                 int l = idx / 4;
//                 return createSIMD(l, s, b);
//             }

//             std::string toString() const {
//                 if (isSISD())
//                     return "SISD";
//                 return "SIMD(L=" + std::to_string(level) +
//                        ", S=" + (scale == 1 ? "Clean" : "Dirty") +
//                        ", B=" + (basis == 2 ? "Std" : "Ext") + ")";
//             }
//         };

//         // --- Cost Model ---
//         struct CostModel {
//             llvm::StringMap<llvm::DenseMap<int64_t, double>> costTable;

//             CostModel(StringRef jsonFilename) {
//                 DBG("Loading CostModel from: " << jsonFilename << "\n");
//                 auto fileOrErr = llvm::MemoryBuffer::getFile(jsonFilename);
//                 if (auto ec = fileOrErr.getError())
//                     return;
//                 std::unique_ptr<llvm::MemoryBuffer> fileBuffer = std::move(*fileOrErr);
//                 llvm::Expected<llvm::json::Value> rootOrErr = llvm::json::parse(fileBuffer->getBuffer());
//                 if (!rootOrErr)
//                     return;
//                 auto* obj = rootOrErr->getAsObject();
//                 if (!obj)
//                     return;
//                 auto* costs = obj->getObject("costs");
//                 if (!costs)
//                     return;

//                 for (auto& modePair : *costs) {
//                     StringRef mode = modePair.first;
//                     auto* modeObj = modePair.second.getAsObject();
//                     if (!modeObj)
//                         continue;
//                     for (auto& opPair : *modeObj) {
//                         StringRef opname = opPair.first;
//                         auto* opObj = opPair.second.getAsObject();
//                         if (!opObj)
//                             continue;
//                         if (auto* lat = opObj->getObject("latency")) {
//                             std::string key = (mode + "." + opname).str();
//                             for (auto& kv : *lat) {
//                                 long long idx = 0;
//                                 if (!llvm::StringRef(kv.first).getAsInteger(10, idx) && kv.second.getAsNumber()) {
//                                     costTable[key][idx] = *kv.second.getAsNumber();
//                                 }
//                             }
//                         }
//                     }
//                 }
//             }

//             double lookup(StringRef key, int64_t idx) const {
//                 auto it = costTable.find(key);
//                 if (it == costTable.end())
//                     return INF_COST;
//                 auto jt = it->second.find(idx);
//                 if (jt == it->second.end())
//                     return INF_COST;
//                 return jt->second;
//             }

//             double getBootCost(int64_t targetLevel) const { return lookup("simd.boot", targetLevel); }
//             double getRescaleCost(int64_t level) const { return lookup("simd.rescale", level); }
//             double getRelinCost(int64_t level) const { return lookup("simd.relinearize", level); }
//             double getCastCost(bool fromSimd, bool toSimd, int64_t dim) const {
//                 if (fromSimd == toSimd)
//                     return 0.0;
//                 std::string key = fromSimd ? "simd.cast_to_sisd" : "sisd.cast_to_simd";
//                 return lookup(key, dim);
//             }
//         };

//         struct Edge {
//             Operation* producer;
//             Operation* consumer;
//             Value val;
//         };

//         struct Node {
//             Operation* op;
//             SmallVector<Edge*, 2> inEdges;
//             SmallVector<Edge*, 2> outEdges;
//             int vectorCount = -1;
//             std::vector<double> dpCost;
//             int chosenOutputState = -1;
//             Node() : dpCost(NodeState::TOTAL_STATES, INF_COST) {}
//         };

//         class ConvertToModeSelectIR : public impl::ConvertToMDSELPassBase<ConvertToModeSelectIR> {
//         public:
//             using impl::ConvertToMDSELPassBase<ConvertToModeSelectIR>::ConvertToMDSELPassBase;

//             StringRef getOptionFilePath() { return "/mnt/data0/home/syt/Libra/Libra/Dialect/Analysis/cost_table.json"; }

//             bool isSimdMult(Operation* op) { return isa<simd::SIMDMultOp>(op); }
//             bool isSimdAddSub(Operation* op) { return isa<simd::SIMDAddOp, simd::SIMDSubOp>(op); }

//             std::string getOpKeyName(Operation* op, bool isSimd) {
//                 if (isSimd) {
//                     if (isa<simd::SIMDAddOp>(op))
//                         return "simd.add";
//                     if (isa<simd::SIMDSubOp>(op))
//                         return "simd.sub";
//                     if (isa<simd::SIMDMultOp>(op))
//                         return "simd.mult";
//                     if (isa<simd::SIMDMinOp>(op))
//                         return "simd.min";
//                 } else {
//                     if (isa<simd::SIMDAddOp, sisd::SISDAddOp>(op))
//                         return "sisd.add";
//                     if (isa<simd::SIMDSubOp, sisd::SISDSubOp>(op))
//                         return "sisd.sub";
//                     if (isa<simd::SIMDMinOp, sisd::SISDMinOp>(op))
//                         return "sisd.min";
//                 }
//                 return "";
//             }

//             // 状态反向映射：从子节点的输出状态推导输入需求
//             std::pair<bool, NodeState> mapConsumerStateToInputReq(Operation* op, const NodeState& outputState) {
//                 if (outputState.isSISD())
//                     return {true, outputState};

//                 int L = outputState.level;
//                 int S = outputState.scale;
//                 int B = outputState.basis;

//                 if (isa<simd::SIMDEncryptOp>(op)) {
//                     // Encrypt 是源，它不消费 FHE 输入。返回 outputState 仅作占位。
//                     return {true, outputState};
//                 } else if (isSimdMult(op)) {
//                     // Mult 输出 Dirty(2)/Extended(3)，但输入要求 Clean(1)/Standard(2)
//                     if (S == 2 && B == 3) {
//                         return {true, NodeState::createSIMD(L, 1, 2)};
//                     }
//                     return {false, NodeState()};
//                 } else if (isSimdAddSub(op)) {
//                     return {true, outputState};
//                 } else if (isa<simd::SIMDMinOp>(op)) {
//                     // Min 消耗 Level
//                     int reqInL = L + 3;
//                     if (reqInL > MAX_SIMD_LEVEL)
//                         return {false, NodeState()};
//                     return {true, NodeState::createSIMD(reqInL, S, B)};
//                 } else if (isa<simd::SIMDDecryptOp>(op)) {
//                     // Decrypt 接受任意状态 (包括 Dirty/Extended)，直接透传
//                     return {true, outputState};
//                 }
//                 return {true, outputState};
//             }

//             double getTransitionCost(const NodeState& p, const NodeState& req, int64_t vecCnt, const CostModel& cm) {
//                 if (p.isSISD() && req.isSISD())
//                     return 0;

//                 if (p.isSISD() && req.isSIMD()) {
//                     if (req.scale != 1 || req.basis != 2 || req.level > 20)
//                         return INF_COST;
//                     return cm.getCastCost(false, true, vecCnt);
//                 }

//                 if (p.isSIMD() && req.isSISD()) {
//                     double cost = 0;
//                     int currLvl = p.level;
//                     if (p.basis == 3)
//                         cost += cm.getRelinCost(currLvl);
//                     if (p.scale == 2) {
//                         if (currLvl <= 0)
//                             return INF_COST;
//                         cost += cm.getRescaleCost(currLvl);
//                     }
//                     return cost + cm.getCastCost(true, false, vecCnt);
//                 }

//                 // SIMD -> SIMD
//                 double cost = 0;
//                 int currLvl = p.level;
//                 int currScale = p.scale;
//                 int currBasis = p.basis;

//                 if (currBasis == 3 && req.basis == 2) {
//                     cost += cm.getRelinCost(currLvl);
//                     currBasis = 2;
//                 } else if (currBasis == 2 && req.basis == 3)
//                     return INF_COST;

//                 if (currScale == 2 && req.scale == 1) {
//                     if (currLvl <= 0)
//                         return INF_COST;
//                     cost += cm.getRescaleCost(currLvl);
//                     currLvl--;
//                     currScale = 1;
//                 } else if (currScale == 1 && req.scale == 2)
//                     return INF_COST;

//                 if (currLvl > req.level) {
//                     if (currScale != 1 || currBasis != 2)
//                         return INF_COST;
//                 } else if (currLvl < req.level) {
//                     if (currScale != 1 || currBasis != 2)
//                         return INF_COST;
//                     cost += cm.getBootCost(req.level);
//                 }
//                 return cost;
//             }

//             void runShapeInference(const std::vector<Node*>& topo, const llvm::DenseMap<Operation*, std::unique_ptr<Node>>& nodeMap) {
//                 DBG("\n=== [Phase 1] Shape Inference ===\n");
//                 for (Node* n : topo) {
//                     int64_t currentCount = -1;
//                     if (!n->inEdges.empty()) {
//                         Operation* parentOp = n->inEdges[0]->producer;
//                         auto it = nodeMap.find(parentOp);
//                         if (it != nodeMap.end())
//                             currentCount = it->second->vectorCount;
//                     }
//                     if (currentCount == -1) {
//                         Type resTy = n->op->getResult(0).getType();
//                         if (auto t = dyn_cast<simd::SIMDCipherType>(resTy))
//                             currentCount = t.getPlaintextCount();
//                         else if (auto t = dyn_cast<sisd::SISDCipherType>(resTy))
//                             currentCount = t.getPlaintextCount();
//                     }
//                     if (currentCount == -1)
//                         currentCount = 8;
//                     if (isa<simd::SIMDMinOp, sisd::SISDMinOp>(n->op))
//                         n->vectorCount = 1;
//                     else
//                         n->vectorCount = currentCount;
//                     DBG("Node: " << n->op->getName() << " -> Count=" << n->vectorCount << "\n");
//                 }
//             }

//             void runOnOperation() override {
//                 ModuleOp module = getOperation();
//                 CostModel costModel(getOptionFilePath());

//                 for (func::FuncOp func : module.getOps<func::FuncOp>()) {
//                     DBG("\nProcessing Function: " << func.getName() << "\n");
//                     llvm::DenseMap<Operation*, std::unique_ptr<Node>> nodeMap;
//                     std::vector<Node*> topo;

//                     func.walk([&](Operation* op) {
//                         if (isa<simd::SIMDSubOp, simd::SIMDMultOp, simd::SIMDMinOp,
//                                 simd::SIMDEncryptOp, simd::SIMDDecryptOp, simd::SIMDAddOp,
//                                 sisd::SISDSubOp, sisd::SISDMinOp, sisd::SISDAddOp,
//                                 sisd::SISDEncryptOp, sisd::SISDDecryptOp>(op)) {
//                             auto n = std::make_unique<Node>();
//                             n->op = op;
//                             nodeMap[op] = std::move(n);
//                         }
//                     });

//                     // Build Edges
//                     std::vector<std::unique_ptr<Edge>> allEdges;
//                     for (auto& p : nodeMap) {
//                         for (Value v : p.first->getOperands()) {
//                             if (Operation* def = v.getDefiningOp()) {
//                                 if (nodeMap.count(def)) {
//                                     allEdges.push_back(std::make_unique<Edge>(Edge{def, p.first, v}));
//                                     Edge* e = allEdges.back().get();
//                                     p.second->inEdges.push_back(e);
//                                     nodeMap[def]->outEdges.push_back(e);
//                                 }
//                             }
//                         }
//                     }

//                     // Topo Sort
//                     llvm::DenseMap<Operation*, int> indeg;
//                     SmallVector<Operation*, 64> stack;
//                     for (auto& p : nodeMap) {
//                         indeg[p.first] = p.second->inEdges.size();
//                         if (p.second->inEdges.empty())
//                             stack.push_back(p.first);
//                     }
//                     while (!stack.empty()) {
//                         Operation* op = stack.pop_back_val();
//                         Node* n = nodeMap[op].get();
//                         topo.push_back(n);
//                         for (Edge* e : n->outEdges) {
//                             indeg[e->consumer]--;
//                             if (indeg[e->consumer] == 0)
//                                 stack.push_back(e->consumer);
//                         }
//                     }

//                     runShapeInference(topo, nodeMap);

//                     // Phase 2: DP Bottom-Up
//                     DBG("\n=== [Phase 2] DP Bottom-Up ===\n");
//                     for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
//                         Node* u = *it;
//                         Operation* op = u->op;
//                         int64_t vecCnt = u->vectorCount;

//                         DBG("[DP] Processing " << op->getName() << "\n");
//                         int validStates = 0;

//                         for (int sIdx = 0; sIdx < NodeState::TOTAL_STATES; ++sIdx) {
//                             NodeState currState = NodeState::fromIndex(sIdx);
//                             double localCost = INF_COST;
//                             bool isValidOp = false;

//                             if (currState.isSISD()) {
//                                 if (!isSimdMult(op)) {
//                                     isValidOp = true;
//                                     if (isa<simd::SIMDEncryptOp, sisd::SISDEncryptOp, simd::SIMDDecryptOp, sisd::SISDDecryptOp>(op))
//                                         localCost = 0;
//                                     else
//                                         localCost = costModel.lookup(getOpKeyName(op, false), vecCnt);
//                                 }
//                             } else {  // SIMD
//                                 int L = currState.level;
//                                 int S = currState.scale;
//                                 int B = currState.basis;
//                                 std::string opKey = getOpKeyName(op, true);

//                                 if (isa<simd::SIMDEncryptOp>(op)) {
//                                     // [修正] Encrypt 可以加密到任意 Level，只要 Clean & Standard
//                                     if (S == 1 && B == 2) {
//                                         isValidOp = true;
//                                         localCost = 0;
//                                     }
//                                 } else if (isSimdMult(op)) {
//                                     if (S == 2 && B == 3) {
//                                         isValidOp = true;
//                                         localCost = costModel.lookup(opKey, L);
//                                     }
//                                 } else if (isSimdAddSub(op)) {
//                                     isValidOp = true;
//                                     localCost = costModel.lookup(opKey, L);
//                                 } else if (isa<simd::SIMDMinOp>(op)) {
//                                     isValidOp = true;
//                                     int reqIn = L + 3;
//                                     if (reqIn <= MAX_SIMD_LEVEL)
//                                         localCost = costModel.lookup(opKey, reqIn);
//                                     else
//                                         isValidOp = false;
//                                 } else if (isa<simd::SIMDDecryptOp>(op)) {
//                                     // [修正] Decrypt 可以接收任意 Level/Scale/Basis
//                                     isValidOp = true;
//                                     localCost = 0;
//                                 }
//                             }

//                             if (!isValidOp || localCost >= INF_COST)
//                                 continue;

//                             double childrenSum = 0;
//                             bool possible = true;

//                             for (Edge* e : u->outEdges) {
//                                 Node* v = nodeMap[e->consumer].get();
//                                 double minChildCost = INF_COST;
//                                 for (int k = 0; k < NodeState::TOTAL_STATES; ++k) {
//                                     if (v->dpCost[k] >= INF_COST)
//                                         continue;
//                                     NodeState childOutputState = NodeState::fromIndex(k);
//                                     auto [isValidMap, childInputReq] = mapConsumerStateToInputReq(v->op, childOutputState);
//                                     if (!isValidMap)
//                                         continue;
//                                     double trans = getTransitionCost(currState, childInputReq, vecCnt, costModel);
//                                     if (trans < INF_COST)
//                                         minChildCost = std::min(minChildCost, v->dpCost[k] + trans);
//                                 }
//                                 if (minChildCost >= INF_COST) {
//                                     possible = false;
//                                     break;
//                                 }
//                                 childrenSum += minChildCost;
//                             }

//                             if (possible) {
//                                 u->dpCost[sIdx] = localCost + childrenSum;
//                                 validStates++;
//                                 DBG("  Valid: " << currState.toString() << " | Cost=" << u->dpCost[sIdx] << "\n");
//                             }
//                         }
//                         DBG("Summary: " << op->getName() << " has " << validStates << " valid states.\n");
//                     }

//                     // Phase 3: Selection & Rewrite
//                     // ------------------------------------------------------------
//                     DBG("\n=== [Phase 3] Selection & Rewrite ===\n");
//                     llvm::DenseMap<Value, Value> rewriteMap;
//                     IRRewriter rewriter(&getContext());
//                     double globalRunningCost = 0.0;  // [新增] 全局累积成本

//                     for (Node* n : topo) {
//                         Operation* op = n->op;
//                         double bestVal = INF_COST;
//                         int bestS = -1;

//                         // Choose Best State
//                         for (int s = 0; s < NodeState::TOTAL_STATES; ++s) {
//                             if (n->dpCost[s] >= INF_COST)
//                                 continue;

//                             NodeState myOutputTarget = NodeState::fromIndex(s);
//                             auto [isValidReq, myInputReq] = mapConsumerStateToInputReq(op, myOutputTarget);
//                             if (!isValidReq)
//                                 continue;

//                             double transSum = 0;
//                             bool possible = true;
//                             for (Edge* e : n->inEdges) {
//                                 Node* parent = nodeMap[e->producer].get();
//                                 NodeState p = NodeState::fromIndex(parent->chosenOutputState);
//                                 double edgeC = getTransitionCost(p, myInputReq, n->vectorCount, costModel);
//                                 if (edgeC >= INF_COST) {
//                                     possible = false;
//                                     break;
//                                 }
//                                 transSum += edgeC;
//                             }
//                             if (possible && (transSum + n->dpCost[s] < bestVal)) {
//                                 bestVal = transSum + n->dpCost[s];
//                                 bestS = s;
//                             }
//                         }

//                         if (bestS == -1) {
//                             llvm::errs() << "Fatal: No valid state found for " << op->getName() << "\n";
//                             return;
//                         }
//                         n->chosenOutputState = bestS;
//                         NodeState bestState = NodeState::fromIndex(bestS);

//                         // [新增] 计算并打印本地累积成本
//                         double currentLocalCost = 0.0;
//                         if (bestState.isSISD()) {
//                             if (isa<simd::SIMDEncryptOp, sisd::SISDEncryptOp>(op))
//                                 currentLocalCost = 0;
//                             else
//                                 currentLocalCost = costModel.lookup(getOpKeyName(op, false), n->vectorCount);
//                         } else {
//                             int L = bestState.level;
//                             if (isa<simd::SIMDEncryptOp, simd::SIMDDecryptOp>(op))
//                                 currentLocalCost = 0;
//                             else if (isSimdMult(op) || isSimdAddSub(op))
//                                 currentLocalCost = costModel.lookup(getOpKeyName(op, true), L);
//                             else if (isa<simd::SIMDMinOp>(op))
//                                 currentLocalCost = costModel.lookup(getOpKeyName(op, true), L + 3);
//                         }
//                         globalRunningCost += currentLocalCost;
//                         DBG("Node " << op->getName() << " | Selected: " << bestState.toString() << " | LocalCost: " << currentLocalCost << " | GlobalAccum: " << globalRunningCost << "\n");

//                         // Rewrite IR
//                         rewriter.setInsertionPoint(op);
//                         SmallVector<Value, 2> newOperands;
//                         auto [_, targetForEdge] = mapConsumerStateToInputReq(op, bestState);

//                         for (Value oldOperand : op->getOperands()) {
//                             Operation* defOp = oldOperand.getDefiningOp();
//                             if (defOp && nodeMap.count(defOp)) {
//                                 Node* parent = nodeMap[defOp].get();
//                                 NodeState pState = NodeState::fromIndex(parent->chosenOutputState);
//                                 Value incoming = rewriteMap.lookup(oldOperand);
//                                 if (!incoming)
//                                     incoming = oldOperand;
//                                 int64_t edgeVecCnt = parent->vectorCount;

//                                 // Explicit Transitions
//                                 if (pState.isSISD() && targetForEdge.isSIMD()) {
//                                     double c = costModel.getCastCost(false, true, edgeVecCnt);
//                                     globalRunningCost += c;
//                                     DBG("  [Insert] Cast SISD->SIMD (Cost=" << c << ")\n");
//                                     auto newTy = simd::SIMDCipherType::get(op->getContext(), targetForEdge.level, edgeVecCnt, rewriter.getI64Type(), 1, 2);
//                                     incoming = rewriter.create<sisd::SISDCastSISDCipherToSIMDCipherOp>(op->getLoc(), newTy, incoming);
//                                 } else if (pState.isSIMD() && targetForEdge.isSISD()) {
//                                     if (pState.basis == 3) {
//                                         double c = costModel.getRelinCost(pState.level);
//                                         globalRunningCost += c;
//                                         DBG("  [Insert] Relin (Cost=" << c << ")\n");
//                                         auto ty = simd::SIMDCipherType::get(op->getContext(), pState.level, edgeVecCnt, rewriter.getI64Type(), pState.scale, 2);
//                                         incoming = rewriter.create<simd::SIMDRelinOp>(op->getLoc(), ty, incoming);
//                                         pState.basis = 2;
//                                     }
//                                     if (pState.scale == 2) {
//                                         double c = costModel.getRescaleCost(pState.level);
//                                         globalRunningCost += c;
//                                         DBG("  [Insert] Rescale (Cost=" << c << ")\n");
//                                         auto ty = simd::SIMDCipherType::get(op->getContext(), pState.level - 1, edgeVecCnt, rewriter.getI64Type(), 1, pState.basis);
//                                         incoming = rewriter.create<simd::SIMDRescaleOp>(op->getLoc(), ty, incoming);
//                                         pState.level--;
//                                         pState.scale = 1;
//                                     }
//                                     double c = costModel.getCastCost(true, false, edgeVecCnt);
//                                     globalRunningCost += c;
//                                     DBG("  [Insert] Cast SIMD->SISD (Cost=" << c << ")\n");
//                                     auto newTy = sisd::SISDCipherType::get(op->getContext(), edgeVecCnt, rewriter.getI64Type(), 1);
//                                     incoming = rewriter.create<simd::SIMDCastSIMDCipherToSISDCipherOp>(op->getLoc(), newTy, incoming);
//                                 } else if (pState.isSIMD() && targetForEdge.isSIMD()) {
//                                     if (pState.basis == 3 && targetForEdge.basis == 2) {
//                                         double c = costModel.getRelinCost(pState.level);
//                                         globalRunningCost += c;
//                                         DBG("  [Insert] Relin (Cost=" << c << ")\n");
//                                         auto ty = simd::SIMDCipherType::get(op->getContext(), pState.level, edgeVecCnt, rewriter.getI64Type(), pState.scale, 2);
//                                         incoming = rewriter.create<simd::SIMDRelinOp>(op->getLoc(), ty, incoming);
//                                         pState.basis = 2;
//                                     }
//                                     if (pState.scale == 2 && targetForEdge.scale == 1) {
//                                         double c = costModel.getRescaleCost(pState.level);
//                                         globalRunningCost += c;
//                                         DBG("  [Insert] Rescale (Cost=" << c << ")\n");
//                                         auto ty = simd::SIMDCipherType::get(op->getContext(), pState.level - 1, edgeVecCnt, rewriter.getI64Type(), 1, pState.basis);
//                                         incoming = rewriter.create<simd::SIMDRescaleOp>(op->getLoc(), ty, incoming);
//                                         pState.level--;
//                                         pState.scale = 1;
//                                     }
//                                     if (pState.level < targetForEdge.level) {
//                                         double c = costModel.getBootCost(targetForEdge.level);
//                                         globalRunningCost += c;
//                                         DBG("  [Insert] Boot to " << targetForEdge.level << " (Cost=" << c << ")\n");
//                                         auto ty = simd::SIMDCipherType::get(op->getContext(), targetForEdge.level, edgeVecCnt, rewriter.getI64Type(), 1, 2);
//                                         incoming = rewriter.create<simd::SIMDBootOp>(op->getLoc(), ty, incoming);
//                                     } else if (pState.level > targetForEdge.level) {
//                                         DBG("  [Insert] ModSwitch (Cost=0)\n");
//                                         auto ty = simd::SIMDCipherType::get(op->getContext(), targetForEdge.level, edgeVecCnt, rewriter.getI64Type(), pState.scale, pState.basis);
//                                         incoming = rewriter.create<simd::SIMDModSwitchOp>(op->getLoc(), ty, incoming);
//                                     }
//                                 }
//                                 newOperands.push_back(incoming);
//                             } else {
//                                 newOperands.push_back(oldOperand);
//                             }
//                         }

//                         // Create New Op
//                         // 注意：这里创建 Op 本身时，结果类型使用 n->vectorCount 是正确的 (Min 输出 1)
//                         Operation* newOp = nullptr;
//                         if (bestState.isSISD()) {
//                             auto resTy = sisd::SISDCipherType::get(op->getContext(), n->vectorCount, rewriter.getI64Type(), 1);
//                             if (isa<simd::SIMDAddOp, sisd::SISDAddOp>(op))
//                                 newOp = rewriter.create<sisd::SISDAddOp>(op->getLoc(), resTy, newOperands[0], newOperands[1]);
//                             else if (isa<simd::SIMDSubOp, sisd::SISDSubOp>(op))
//                                 newOp = rewriter.create<sisd::SISDSubOp>(op->getLoc(), resTy, newOperands[0], newOperands[1]);
//                             else if (isa<simd::SIMDMinOp, sisd::SISDMinOp>(op))
//                                 newOp = rewriter.create<sisd::SISDMinOp>(op->getLoc(), resTy, newOperands[0]);
//                             else if (isa<simd::SIMDEncryptOp, sisd::SISDEncryptOp>(op))
//                                 newOp = rewriter.create<sisd::SISDEncryptOp>(op->getLoc(), resTy, newOperands[0]);
//                             else if (isa<simd::SIMDDecryptOp, sisd::SISDDecryptOp>(op))
//                                 newOp = rewriter.create<sisd::SISDDecryptOp>(op->getLoc(), op->getResultTypes(), newOperands[0]);
//                         } else {
//                             auto resTy = simd::SIMDCipherType::get(op->getContext(), bestState.level, n->vectorCount, rewriter.getI64Type(), bestState.scale, bestState.basis);
//                             if (isa<simd::SIMDAddOp, sisd::SISDAddOp>(op))
//                                 newOp = rewriter.create<simd::SIMDAddOp>(op->getLoc(), resTy, newOperands[0], newOperands[1]);
//                             else if (isa<simd::SIMDSubOp, sisd::SISDSubOp>(op))
//                                 newOp = rewriter.create<simd::SIMDSubOp>(op->getLoc(), resTy, newOperands[0], newOperands[1]);
//                             else if (isa<simd::SIMDMultOp>(op))
//                                 newOp = rewriter.create<simd::SIMDMultOp>(op->getLoc(), resTy, newOperands[0], newOperands[1]);
//                             else if (isa<simd::SIMDMinOp, sisd::SISDMinOp>(op))
//                                 newOp = rewriter.create<simd::SIMDMinOp>(op->getLoc(), resTy, newOperands[0]);
//                             else if (isa<simd::SIMDEncryptOp, sisd::SISDEncryptOp>(op))
//                                 newOp = rewriter.create<simd::SIMDEncryptOp>(op->getLoc(), resTy, newOperands[0]);
//                             else if (isa<simd::SIMDDecryptOp, sisd::SISDDecryptOp>(op))
//                                 newOp = rewriter.create<simd::SIMDDecryptOp>(op->getLoc(), op->getResultTypes(), newOperands[0]);
//                         }

//                         if (newOp)
//                             rewriteMap[op->getResult(0)] = newOp->getResult(0);
//                     }

//                     for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
//                         Operation* oldOp = (*it)->op;
//                         if (rewriteMap.count(oldOp->getResult(0))) {
//                             rewriter.replaceOp(oldOp, rewriteMap[oldOp->getResult(0)]);
//                         }
//                     }
//                 }
//             }
//         };
//     }
// }

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
#include <tuple>
#include <string>

using namespace mlir;
using namespace mlir::libra;

namespace mlir::libra::mdsel {

#define GEN_PASS_DEF_CONVERTTOMDSELPASS
#include "ModeSelectPass.h.inc"

    namespace {

// ============================================================
// [Debug Switch] 1 = Enable Logs, 0 = Disable Logs
// ============================================================
#define MDSEL_DEBUG 1

#if MDSEL_DEBUG
#define DBG(msg) llvm::errs() << "[ModeSel] " << msg
#else
#define DBG(msg)
#endif

        // --- Constants ---
        constexpr int MAX_SIMD_LEVEL = 29;
        constexpr double INF_COST = 1e15;

        // --- NodeState ---
        struct NodeState {
            enum class Mode { SIMD,
                              SISD };
            Mode mode;
            int level;
            int scale;
            int basis;

            static NodeState createSISD() { return {Mode::SISD, 0, 0, 0}; }
            static NodeState createSIMD(int l, int s, int b) { return {Mode::SIMD, l, s, b}; }

            bool isSISD() const { return mode == Mode::SISD; }
            bool isSIMD() const { return mode == Mode::SIMD; }

            static constexpr int SIMD_STATE_COUNT = (MAX_SIMD_LEVEL + 1) * 4;
            static constexpr int TOTAL_STATES = SIMD_STATE_COUNT + 1;

            int toIndex() const {
                if (mode == Mode::SISD)
                    return SIMD_STATE_COUNT;
                return level * 4 + (scale - 1) * 2 + (basis - 2);
            }

            static NodeState fromIndex(int idx) {
                if (idx == SIMD_STATE_COUNT)
                    return createSISD();
                int b = (idx % 2) + 2;
                int s = ((idx / 2) % 2) + 1;
                int l = idx / 4;
                return createSIMD(l, s, b);
            }

            std::string toString() const {
                if (isSISD())
                    return "SISD";
                return "SIMD(L=" + std::to_string(level) +
                       ", S=" + (scale == 1 ? "Clean" : "Dirty") +
                       ", B=" + (basis == 2 ? "Std" : "Ext") + ")";
            }
        };

        // --- Cost Model ---
        struct CostModel {
            llvm::StringMap<llvm::DenseMap<int64_t, double>> costTable;

            CostModel(StringRef jsonFilename) {
                DBG("Loading CostModel from: " << jsonFilename << "\n");
                auto fileOrErr = llvm::MemoryBuffer::getFile(jsonFilename);
                if (auto ec = fileOrErr.getError())
                    return;
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
                if (jt == it->second.end())
                    return INF_COST;
                return jt->second;
            }

            double getBootCost(int64_t targetLevel) const { return lookup("simd.boot", targetLevel); }
            double getRescaleCost(int64_t level) const { return lookup("simd.rescale", level); }
            double getRelinCost(int64_t level) const { return lookup("simd.relinearize", level); }
            double getCastCost(bool fromSimd, bool toSimd, int64_t dim) const {
                if (fromSimd == toSimd)
                    return 0.0;
                std::string key = fromSimd ? "simd.cast_to_sisd" : "sisd.cast_to_simd";
                return lookup(key, dim);
            }
        };

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
            // [Optimization] Cache local costs to avoid re-lookup in Phase 3
            std::vector<double> localCostCache;

            int chosenOutputState = -1;
            Node() : dpCost(NodeState::TOTAL_STATES, INF_COST),
                     localCostCache(NodeState::TOTAL_STATES, INF_COST) {}
        };

        class ConvertToModeSelectIR : public impl::ConvertToMDSELPassBase<ConvertToModeSelectIR> {
        public:
            using impl::ConvertToMDSELPassBase<ConvertToModeSelectIR>::ConvertToMDSELPassBase;

            StringRef getOptionFilePath() { return "/mnt/data0/home/syt/Libra/Libra/Dialect/Analysis/cost_table.json"; }

            bool isSimdMult(Operation* op) { return isa<simd::SIMDMultOp>(op); }
            bool isSimdAddSub(Operation* op) { return isa<simd::SIMDAddOp, simd::SIMDSubOp>(op); }

            // Helper to classify multiplication type
            enum class MultType { Cipher,
                                  Plain,
                                  Integer };
            MultType classifyMultOp(Operation* op) {
                if (!isSimdMult(op))
                    return MultType::Cipher;    // Should not happen
                Value rhs = op->getOperand(1);  // Assume RHS is the constant/plain
                Operation* defOp = rhs.getDefiningOp();
                if (!defOp)
                    return MultType::Cipher;

                if (auto constOp = dyn_cast<arith::ConstantOp>(defOp)) {
                    Attribute attr = constOp.getValue();

                    // Check for Float (Scalar or Tensor/Vector elements)
                    if (isa<FloatAttr>(attr)) {
                        return MultType::Plain;
                    }
                    if (auto dense = dyn_cast<DenseElementsAttr>(attr)) {
                        if (isa<FloatType>(dense.getElementType())) {
                            return MultType::Plain;
                        }
                    }

                    // Check for Integer (Scalar or Tensor/Vector elements)
                    if (isa<IntegerAttr>(attr)) {
                        return MultType::Integer;
                    }
                    if (auto dense = dyn_cast<DenseElementsAttr>(attr)) {
                        if (isa<IntegerType>(dense.getElementType())) {
                            return MultType::Integer;
                        }
                    }
                }
                return MultType::Cipher;
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

            std::pair<bool, NodeState> mapConsumerStateToInputReq(Operation* op, const NodeState& outputState) {
                if (outputState.isSISD())
                    return {true, outputState};

                int L = outputState.level;
                int S = outputState.scale;
                int B = outputState.basis;

                if (isa<simd::SIMDEncryptOp>(op)) {
                    return {true, outputState};
                } else if (isSimdMult(op)) {
                    MultType mType = classifyMultOp(op);
                    if (mType == MultType::Cipher) {
                        // Cipher Mult: Out(Dirty, Ext) -> In(Clean, Std)
                        if (S == 2 && B == 3)
                            return {true, NodeState::createSIMD(L, 1, 2)};
                    } else if (mType == MultType::Plain) {
                        // Plain Mult: Out(Dirty, Std) -> In(Clean, Std)
                        // Optimization: No Relin needed (Basis remains 2)
                        if (S == 2 && B == 2)
                            return {true, NodeState::createSIMD(L, 1, 2)};
                    } else if (mType == MultType::Integer) {
                        // Integer Mult: Transparent (State Preserved)
                        return {true, outputState};
                    }
                    return {false, NodeState()};
                } else if (isSimdAddSub(op)) {
                    return {true, outputState};
                } else if (isa<simd::SIMDMinOp>(op)) {
                    int reqInL = L + 3;
                    if (reqInL > MAX_SIMD_LEVEL)
                        return {false, NodeState()};
                    return {true, NodeState::createSIMD(reqInL, S, B)};
                } else if (isa<simd::SIMDDecryptOp>(op)) {
                    return {true, outputState};
                }
                return {true, outputState};
            }

            double getTransitionCost(const NodeState& p, const NodeState& req, int64_t vecCnt, const CostModel& cm) {
                if (p.isSISD() && req.isSISD())
                    return 0;

                if (p.isSISD() && req.isSIMD()) {
                    if (req.scale != 1 || req.basis != 2 || req.level > 20)
                        return INF_COST;
                    return cm.getCastCost(false, true, vecCnt);
                }

                if (p.isSIMD() && req.isSISD()) {
                    double cost = 0;
                    int currLvl = p.level;
                    if (p.basis == 3)
                        cost += cm.getRelinCost(currLvl);
                    if (p.scale == 2) {
                        if (currLvl <= 0)
                            return INF_COST;
                        cost += cm.getRescaleCost(currLvl);
                    }
                    return cost + cm.getCastCost(true, false, vecCnt);
                }

                // SIMD -> SIMD
                double cost = 0;
                int currLvl = p.level;
                int currScale = p.scale;
                int currBasis = p.basis;

                if (currBasis == 3 && req.basis == 2) {
                    cost += cm.getRelinCost(currLvl);
                    currBasis = 2;
                } else if (currBasis == 2 && req.basis == 3)
                    return INF_COST;

                if (currScale == 2 && req.scale == 1) {
                    if (currLvl <= 0)
                        return INF_COST;
                    cost += cm.getRescaleCost(currLvl);
                    currLvl--;
                    currScale = 1;
                } else if (currScale == 1 && req.scale == 2)
                    return INF_COST;

                if (currLvl > req.level) {
                    if (currScale != 1 || currBasis != 2)
                        return INF_COST;
                } else if (currLvl < req.level) {
                    if (currScale != 1 || currBasis != 2)
                        return INF_COST;
                    cost += cm.getBootCost(req.level);
                }
                return cost;
            }

            void runShapeInference(const std::vector<Node*>& topo, const llvm::DenseMap<Operation*, std::unique_ptr<Node>>& nodeMap) {
                DBG("\n=== [Phase 1] Shape Inference ===\n");
                for (Node* n : topo) {
                    int64_t currentCount = -1;
                    if (!n->inEdges.empty()) {
                        Operation* parentOp = n->inEdges[0]->producer;
                        auto it = nodeMap.find(parentOp);
                        if (it != nodeMap.end())
                            currentCount = it->second->vectorCount;
                    }
                    if (currentCount == -1) {
                        Type resTy = n->op->getResult(0).getType();
                        if (auto t = dyn_cast<simd::SIMDCipherType>(resTy))
                            currentCount = t.getPlaintextCount();
                        else if (auto t = dyn_cast<sisd::SISDCipherType>(resTy))
                            currentCount = t.getPlaintextCount();
                    }
                    if (currentCount == -1)
                        currentCount = 8;
                    if (isa<simd::SIMDMinOp, sisd::SISDMinOp>(n->op))
                        n->vectorCount = 1;
                    else
                        n->vectorCount = currentCount;
                    DBG("Node: " << n->op->getName() << " -> Count=" << n->vectorCount << "\n");
                }
            }

            void runOnOperation() override {
                ModuleOp module = getOperation();
                CostModel costModel(getOptionFilePath());

                for (func::FuncOp func : module.getOps<func::FuncOp>()) {
                    DBG("\nProcessing Function: " << func.getName() << "\n");
                    llvm::DenseMap<Operation*, std::unique_ptr<Node>> nodeMap;
                    std::vector<Node*> topo;

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

                    std::vector<std::unique_ptr<Edge>> allEdges;
                    for (auto& p : nodeMap) {
                        for (Value v : p.first->getOperands()) {
                            if (Operation* def = v.getDefiningOp()) {
                                if (nodeMap.count(def)) {
                                    allEdges.push_back(std::make_unique<Edge>(Edge{def, p.first, v}));
                                    Edge* e = allEdges.back().get();
                                    p.second->inEdges.push_back(e);
                                    nodeMap[def]->outEdges.push_back(e);
                                }
                            }
                        }
                    }

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

                    runShapeInference(topo, nodeMap);

                    // Phase 2: DP Bottom-Up
                    DBG("\n=== [Phase 2] DP Bottom-Up ===\n");
                    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
                        Node* u = *it;
                        Operation* op = u->op;
                        int64_t vecCnt = u->vectorCount;

                        DBG("[DP] Processing " << op->getName() << "\n");
                        int validStates = 0;

                        for (int sIdx = 0; sIdx < NodeState::TOTAL_STATES; ++sIdx) {
                            NodeState currState = NodeState::fromIndex(sIdx);
                            double localCost = INF_COST;
                            bool isValidOp = false;

                            if (currState.isSISD()) {
                                if (!isSimdMult(op)) {
                                    isValidOp = true;
                                    if (isa<simd::SIMDEncryptOp, sisd::SISDEncryptOp, simd::SIMDDecryptOp, sisd::SISDDecryptOp>(op))
                                        localCost = 0;
                                    else
                                        localCost = costModel.lookup(getOpKeyName(op, false), vecCnt);
                                }
                            } else {  // SIMD
                                int L = currState.level;
                                int S = currState.scale;
                                int B = currState.basis;
                                std::string opKey = getOpKeyName(op, true);

                                if (isa<simd::SIMDEncryptOp>(op)) {
                                    if (S == 1 && B == 2) {
                                        isValidOp = true;
                                        localCost = 0;
                                    }
                                } else if (isSimdMult(op)) {
                                    MultType mType = classifyMultOp(op);
                                    if (mType == MultType::Cipher) {
                                        // Cipher * Cipher -> Dirty(2) & Extended(3)
                                        if (S == 2 && B == 3) {
                                            isValidOp = true;
                                            localCost = costModel.lookup(opKey, L);
                                        }
                                    } else if (mType == MultType::Plain) {
                                        // Cipher * Plain -> Dirty(2) & Standard(2)
                                        if (S == 2 && B == 2) {
                                            isValidOp = true;
                                            localCost = costModel.lookup("simd.mult_plain", L);
                                        }
                                    } else if (mType == MultType::Integer) {
                                        // Integer -> Transparent
                                        isValidOp = true;
                                        localCost = costModel.lookup(opKey, L);  // Assume cheap cost
                                    }
                                } else if (isSimdAddSub(op)) {
                                    isValidOp = true;
                                    localCost = costModel.lookup(opKey, L);
                                } else if (isa<simd::SIMDMinOp>(op)) {
                                    isValidOp = true;
                                    int reqIn = L + 3;
                                    if (reqIn <= MAX_SIMD_LEVEL)
                                        localCost = costModel.lookup(opKey, reqIn);
                                    else
                                        isValidOp = false;
                                } else if (isa<simd::SIMDDecryptOp>(op)) {
                                    isValidOp = true;
                                    localCost = 0;
                                }
                            }

                            u->localCostCache[sIdx] = localCost;

                            if (!isValidOp || localCost >= INF_COST)
                                continue;

                            double childrenSum = 0;
                            bool possible = true;

                            for (Edge* e : u->outEdges) {
                                Node* v = nodeMap[e->consumer].get();
                                double minChildCost = INF_COST;
                                for (int k = 0; k < NodeState::TOTAL_STATES; ++k) {
                                    if (v->dpCost[k] >= INF_COST)
                                        continue;
                                    NodeState childOutputState = NodeState::fromIndex(k);
                                    auto [isValidMap, childInputReq] = mapConsumerStateToInputReq(v->op, childOutputState);
                                    if (!isValidMap)
                                        continue;
                                    double trans = getTransitionCost(currState, childInputReq, vecCnt, costModel);
                                    if (trans < INF_COST)
                                        minChildCost = std::min(minChildCost, v->dpCost[k] + trans);
                                }
                                if (minChildCost >= INF_COST) {
                                    possible = false;
                                    break;
                                }
                                childrenSum += minChildCost;
                            }

                            if (possible) {
                                u->dpCost[sIdx] = localCost + childrenSum;
                                validStates++;
                                DBG("  Valid: " << currState.toString() << " | Cost=" << u->dpCost[sIdx] << "\n");
                            }
                        }
                        DBG("Summary: " << op->getName() << " has " << validStates << " valid states.\n");
                    }

                    // Phase 3: Selection & Rewrite
                    DBG("\n=== [Phase 3] Selection & Rewrite ===\n");
                    llvm::DenseMap<Value, Value> rewriteMap;
                    IRRewriter rewriter(&getContext());
                    double globalRunningCost = 0.0;

                    for (Node* n : topo) {
                        Operation* op = n->op;
                        double bestVal = INF_COST;
                        int bestS = -1;

                        for (int s = 0; s < NodeState::TOTAL_STATES; ++s) {
                            if (n->dpCost[s] >= INF_COST)
                                continue;

                            NodeState myOutputTarget = NodeState::fromIndex(s);
                            auto [isValidReq, myInputReq] = mapConsumerStateToInputReq(op, myOutputTarget);
                            if (!isValidReq)
                                continue;

                            double transSum = 0;
                            bool possible = true;
                            for (Edge* e : n->inEdges) {
                                Node* parent = nodeMap[e->producer].get();
                                NodeState p = NodeState::fromIndex(parent->chosenOutputState);
                                double edgeC = getTransitionCost(p, myInputReq, n->vectorCount, costModel);
                                if (edgeC >= INF_COST) {
                                    possible = false;
                                    break;
                                }
                                transSum += edgeC;
                            }
                            if (possible && (transSum + n->dpCost[s] < bestVal)) {
                                bestVal = transSum + n->dpCost[s];
                                bestS = s;
                            }
                        }

                        if (bestS == -1) {
                            llvm::errs() << "Fatal: No valid state found for " << op->getName() << "\n";
                            return;
                        }
                        n->chosenOutputState = bestS;
                        NodeState bestState = NodeState::fromIndex(bestS);

                        double currentLocalCost = n->localCostCache[bestS];
                        globalRunningCost += currentLocalCost;

                        DBG("Node " << op->getName() << " | Selected: " << bestState.toString()
                                    << " | LocalCost: " << currentLocalCost
                                    << " | GlobalAccum: " << globalRunningCost << "\n");

                        rewriter.setInsertionPoint(op);
                        SmallVector<Value, 2> newOperands;
                        auto [_, targetForEdge] = mapConsumerStateToInputReq(op, bestState);

                        for (Value oldOperand : op->getOperands()) {
                            Operation* defOp = oldOperand.getDefiningOp();
                            if (defOp && nodeMap.count(defOp)) {
                                Node* parent = nodeMap[defOp].get();
                                NodeState pState = NodeState::fromIndex(parent->chosenOutputState);
                                Value incoming = rewriteMap.lookup(oldOperand);
                                if (!incoming)
                                    incoming = oldOperand;
                                int64_t edgeVecCnt = parent->vectorCount;

                                // Explicit Transitions with Logging
                                if (pState.isSISD() && targetForEdge.isSIMD()) {
                                    double c = costModel.getCastCost(false, true, edgeVecCnt);
                                    globalRunningCost += c;
                                    DBG("  [Insert] Cast SISD->SIMD (Cost=" << c << ")\n");
                                    auto newTy = simd::SIMDCipherType::get(op->getContext(), targetForEdge.level, edgeVecCnt, rewriter.getI64Type(), 1, 2);
                                    incoming = rewriter.create<sisd::SISDCastSISDCipherToSIMDCipherOp>(op->getLoc(), newTy, incoming);
                                } else if (pState.isSIMD() && targetForEdge.isSISD()) {
                                    if (pState.basis == 3) {
                                        double c = costModel.getRelinCost(pState.level);
                                        globalRunningCost += c;
                                        DBG("  [Insert] Relin (Cost=" << c << ")\n");
                                        auto ty = simd::SIMDCipherType::get(op->getContext(), pState.level, edgeVecCnt, rewriter.getI64Type(), pState.scale, 2);
                                        incoming = rewriter.create<simd::SIMDRelinOp>(op->getLoc(), ty, incoming);
                                        pState.basis = 2;
                                    }
                                    if (pState.scale == 2) {
                                        double c = costModel.getRescaleCost(pState.level);
                                        globalRunningCost += c;
                                        DBG("  [Insert] Rescale (Cost=" << c << ")\n");
                                        auto ty = simd::SIMDCipherType::get(op->getContext(), pState.level - 1, edgeVecCnt, rewriter.getI64Type(), 1, pState.basis);
                                        incoming = rewriter.create<simd::SIMDRescaleOp>(op->getLoc(), ty, incoming);
                                        pState.level--;
                                        pState.scale = 1;
                                    }
                                    double c = costModel.getCastCost(true, false, edgeVecCnt);
                                    globalRunningCost += c;
                                    DBG("  [Insert] Cast SIMD->SISD (Cost=" << c << ")\n");
                                    auto newTy = sisd::SISDCipherType::get(op->getContext(), edgeVecCnt, rewriter.getI64Type(), 1);
                                    incoming = rewriter.create<simd::SIMDCastSIMDCipherToSISDCipherOp>(op->getLoc(), newTy, incoming);
                                } else if (pState.isSIMD() && targetForEdge.isSIMD()) {
                                    if (pState.basis == 3 && targetForEdge.basis == 2) {
                                        double c = costModel.getRelinCost(pState.level);
                                        globalRunningCost += c;
                                        DBG("  [Insert] Relin (Cost=" << c << ")\n");
                                        auto ty = simd::SIMDCipherType::get(op->getContext(), pState.level, edgeVecCnt, rewriter.getI64Type(), pState.scale, 2);
                                        incoming = rewriter.create<simd::SIMDRelinOp>(op->getLoc(), ty, incoming);
                                        pState.basis = 2;
                                    }
                                    if (pState.scale == 2 && targetForEdge.scale == 1) {
                                        double c = costModel.getRescaleCost(pState.level);
                                        globalRunningCost += c;
                                        DBG("  [Insert] Rescale (Cost=" << c << ")\n");
                                        auto ty = simd::SIMDCipherType::get(op->getContext(), pState.level - 1, edgeVecCnt, rewriter.getI64Type(), 1, pState.basis);
                                        incoming = rewriter.create<simd::SIMDRescaleOp>(op->getLoc(), ty, incoming);
                                        pState.level--;
                                        pState.scale = 1;
                                    }
                                    if (pState.level < targetForEdge.level) {
                                        double c = costModel.getBootCost(targetForEdge.level);
                                        globalRunningCost += c;
                                        DBG("  [Insert] Boot to " << targetForEdge.level << " (Cost=" << c << ")\n");
                                        auto ty = simd::SIMDCipherType::get(op->getContext(), targetForEdge.level, edgeVecCnt, rewriter.getI64Type(), 1, 2);
                                        incoming = rewriter.create<simd::SIMDBootOp>(op->getLoc(), ty, incoming);
                                    } else if (pState.level > targetForEdge.level) {
                                        DBG("  [Insert] ModSwitch (Cost=0)\n");
                                        auto ty = simd::SIMDCipherType::get(op->getContext(), targetForEdge.level, edgeVecCnt, rewriter.getI64Type(), pState.scale, pState.basis);
                                        incoming = rewriter.create<simd::SIMDModSwitchOp>(op->getLoc(), ty, incoming);
                                    }
                                }
                                newOperands.push_back(incoming);
                            } else {
                                newOperands.push_back(oldOperand);
                            }
                        }

                        // Create Op
                        Operation* newOp = nullptr;
                        if (bestState.isSISD()) {
                            auto resTy = sisd::SISDCipherType::get(op->getContext(), n->vectorCount, rewriter.getI64Type(), 1);
                            if (isa<simd::SIMDAddOp, sisd::SISDAddOp>(op))
                                newOp = rewriter.create<sisd::SISDAddOp>(op->getLoc(), resTy, newOperands[0], newOperands[1]);
                            else if (isa<simd::SIMDSubOp, sisd::SISDSubOp>(op))
                                newOp = rewriter.create<sisd::SISDSubOp>(op->getLoc(), resTy, newOperands[0], newOperands[1]);
                            else if (isa<simd::SIMDMinOp, sisd::SISDMinOp>(op))
                                newOp = rewriter.create<sisd::SISDMinOp>(op->getLoc(), resTy, newOperands[0]);
                            else if (isa<simd::SIMDEncryptOp, sisd::SISDEncryptOp>(op))
                                newOp = rewriter.create<sisd::SISDEncryptOp>(op->getLoc(), resTy, newOperands[0]);
                            else if (isa<simd::SIMDDecryptOp, sisd::SISDDecryptOp>(op))
                                newOp = rewriter.create<sisd::SISDDecryptOp>(op->getLoc(), op->getResultTypes(), newOperands[0]);
                        } else {
                            auto resTy = simd::SIMDCipherType::get(op->getContext(), bestState.level, n->vectorCount, rewriter.getI64Type(), bestState.scale, bestState.basis);
                            if (isa<simd::SIMDAddOp, sisd::SISDAddOp>(op))
                                newOp = rewriter.create<simd::SIMDAddOp>(op->getLoc(), resTy, newOperands[0], newOperands[1]);
                            else if (isa<simd::SIMDSubOp, sisd::SISDSubOp>(op))
                                newOp = rewriter.create<simd::SIMDSubOp>(op->getLoc(), resTy, newOperands[0], newOperands[1]);
                            else if (isa<simd::SIMDMultOp>(op))
                                newOp = rewriter.create<simd::SIMDMultOp>(op->getLoc(), resTy, newOperands[0], newOperands[1]);
                            else if (isa<simd::SIMDMinOp, sisd::SISDMinOp>(op))
                                newOp = rewriter.create<simd::SIMDMinOp>(op->getLoc(), resTy, newOperands[0]);
                            else if (isa<simd::SIMDEncryptOp, sisd::SISDEncryptOp>(op))
                                newOp = rewriter.create<simd::SIMDEncryptOp>(op->getLoc(), resTy, newOperands[0]);
                            else if (isa<simd::SIMDDecryptOp, sisd::SISDDecryptOp>(op))
                                newOp = rewriter.create<simd::SIMDDecryptOp>(op->getLoc(), op->getResultTypes(), newOperands[0]);
                        }

                        if (newOp)
                            rewriteMap[op->getResult(0)] = newOp->getResult(0);
                    }

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