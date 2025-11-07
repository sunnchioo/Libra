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

using namespace mlir;
using namespace mlir::libra;

namespace mlir::libra::mdsel {

#define GEN_PASS_DEF_CONVERTTOMDSELPASS
#include "ModeSelectPass.h.inc"
    namespace {

        enum class Mode { SIMD,
                          SISD };

        ///------------------------------ COST MODEL ------------------------------
        struct CostModel {
            llvm::StringMap<llvm::DenseMap<int64_t, double>> costTable;

            CostModel(StringRef jsonFilename) {
                auto fileOrErr = llvm::MemoryBuffer::getFile(jsonFilename);
                if (auto ec = fileOrErr.getError()) {
                    llvm::errs() << "Error: cannot open cost model: " << jsonFilename
                                 << ": " << ec.message() << "\n";
                    return;
                }
                std::unique_ptr<llvm::MemoryBuffer> fileBuffer = std::move(*fileOrErr);
                llvm::Expected<llvm::json::Value> rootOrErr =
                    llvm::json::parse(fileBuffer->getBuffer());
                if (!rootOrErr) {
                    llvm::errs() << "Error parsing JSON: "
                                 << llvm::toString(rootOrErr.takeError()) << "\n";
                    return;
                }
                auto* obj = rootOrErr->getAsObject();
                if (!obj)
                    return;
                auto* costs = obj->getObject("costs");
                if (!costs)
                    return;
                for (auto& modePair : *costs) {
                    llvm::StringRef mode = modePair.first;
                    auto* modeObj = modePair.second.getAsObject();
                    if (!modeObj)
                        continue;
                    for (auto& opPair : *modeObj) {
                        llvm::StringRef opname = opPair.first;
                        auto* opObj = opPair.second.getAsObject();
                        if (!opObj)
                            continue;
                        if (auto* lat = opObj->getObject("latency")) {
                            std::string key = (mode + "." + opname).str();
                            auto& tbl = costTable[key];
                            for (auto& kv : *lat) {
                                llvm::StringRef keyStr = kv.first;
                                long long idx = 0;
                                if (keyStr.getAsInteger(10, idx))  // 返回 true 表示失败
                                    continue;
                                if (auto num = kv.second.getAsNumber())
                                    tbl[idx] = *num;
                            }
                        }
                    }
                }
            }

            double lookup(StringRef key, int64_t idx) const {
                auto it = costTable.find(key);
                if (it == costTable.end())
                    return std::numeric_limits<double>::infinity();
                auto jt = it->second.find(idx);
                if (jt == it->second.end())
                    return std::numeric_limits<double>::infinity();
                return jt->second;
            }

            double costOp(Operation* op, Mode mode) const {
                if (!isa<simd::SIMDSubOp, simd::SIMDMultOp, simd::SIMDMinOp,
                         sisd::SISDSubOp, sisd::SISDMinOp>(op))
                    return 0.0;

                StringRef modeName = (mode == Mode::SIMD ? "simd" : "sisd");
                std::string opName;
                if (isa<simd::SIMDSubOp, sisd::SISDSubOp>(op))
                    opName = "sub";
                else if (isa<simd::SIMDMultOp>(op))
                    opName = "mult";
                else if (isa<simd::SIMDMinOp, sisd::SISDMinOp>(op))
                    opName = "min";
                else
                    return 0.0;

                int64_t key = 1;
                if (op->getNumOperands() == 0) {
                    llvm::errs()
                        << "[COSTOP] Warning: Op " << op->getName()
                        << " has 0 operands. Using default key=1.\n";
                } else {
                    Type inTy = op->getOperand(0).getType();

                    if (mode == Mode::SIMD) {
                        if (auto t = dyn_cast<simd::SIMDCipherType>(inTy)) {
                            key = t.getLevel();
                        }
                    } else {
                        if (auto t = dyn_cast<simd::SIMDCipherType>(inTy)) {
                            key = t.getPlaintextCount();
                        } else if (auto t = dyn_cast<sisd::SISDCipherType>(inTy)) {
                            key = t.getPlaintextCount();
                        }
                    }
                }

                double c = lookup((modeName + "." + opName).str(), key);

                llvm::errs() << "[COSTOP] op=" << op->getName()
                             << " mode=" << modeName
                             << " opName=" << opName
                             << " key=" << key
                             << " cost=" << c << "\n";

                return c;
            }

            double costCast(Mode from, Mode to, Value v) const {
                if (from == to)
                    return 0.0;

                std::string key = (from == Mode::SIMD ? "simd.cast_to_sisd"
                                                      : "sisd.cast_to_simd");
                int64_t lvl = 1;
                Type vTy = v.getType();  // 获取类型

                if (from == Mode::SIMD) {
                    if (auto t = dyn_cast<simd::SIMDCipherType>(vTy))
                        lvl = t.getPlaintextCount();  // <-- 修复: 按照规则使用 plaintextCount
                } else {
                    if (auto t = dyn_cast<sisd::SISDCipherType>(vTy))
                        lvl = t.getPlaintextCount();  // <-- 这部分(SISD->SIMD)原本就是正确的
                }

                double c = lookup(key, lvl);
                llvm::errs() << "[COSTCAST] from=" << (from == Mode::SIMD ? "SIMD" : "SISD")
                             << " to=" << (to == Mode::SIMD ? "SIMD" : "SISD")
                             << " key=" << key << " lvl=" << lvl
                             << " cost=" << c << "\n";
                if (!std::isfinite(c))
                    return 10000.0;  // 返回一个高成本
                return c;
            }

            bool feasible(Operation* op, Mode mode) const {
                // 修复 #2: 确保 SIMD-only ops 在 SISD 模式下不可行
                if (mode == Mode::SISD) {
                    if (isa<simd::SIMDMultOp,
                            simd::SIMDEncryptOp,
                            simd::SIMDDecryptOp>(op))
                        return false;
                }
                return true;
            }
        };

        ///------------------------------- DAG ------------------------------------
        struct Edge {
            Operation* producer;
            Operation* consumer;
            Value val;
        };

        struct Node {
            Operation* op;
            SmallVector<Edge*, 4> inEdges;
            SmallVector<Edge*, 4> outEdges;
            double costSIMD = std::numeric_limits<double>::infinity();
            double costSISD = std::numeric_limits<double>::infinity();
            Mode chosenMode = Mode::SIMD;  // 默认
        };

        ///------------------------------ MAIN PASS -------------------------------
        class ConvertToModeSelectIR
            : public impl::ConvertToMDSELPassBase<ConvertToModeSelectIR> {
        public:
            using impl::ConvertToMDSELPassBase<ConvertToModeSelectIR>::ConvertToMDSELPassBase;

            StringRef getOptionFilePath() {
                // change to your cost file
                return "/mnt/data0/home/syt/Libra/Libra/Dialect/Analysis/cost_table.json";
            }

            void runOnOperation() override {
                ModuleOp module = getOperation();
                CostModel costModel(getOptionFilePath());

                for (func::FuncOp func : module.getOps<func::FuncOp>()) {
                    llvm::DenseMap<Operation*, std::unique_ptr<Node>> nodes;
                    SmallVector<Node*, 64> topo;
                    std::vector<std::unique_ptr<Edge>> allEdges;  // 用于管理 Edge 内存

                    // Build nodes
                    func.walk([&](Operation* op) {
                        // 只为我们关心的 dialect ops 创建节点
                        if (isa<simd::SIMDSubOp, simd::SIMDMultOp, simd::SIMDMinOp,
                                simd::SIMDEncryptOp, simd::SIMDDecryptOp,
                                sisd::SISDSubOp, sisd::SISDMinOp>(op)) {
                            auto n = std::make_unique<Node>();
                            n->op = op;
                            nodes[op] = std::move(n);
                        }
                    });

                    // Build edges
                    for (auto& p : nodes) {
                        Operation* op = p.first;
                        Node* node = p.second.get();
                        for (Value v : op->getOperands()) {
                            if (Operation* def = v.getDefiningOp()) {
                                if (nodes.count(def)) {  // 只连接在 graph 中的节点
                                    allEdges.push_back(std::make_unique<Edge>(Edge{def, op, v}));
                                    Edge* e = allEdges.back().get();
                                    node->inEdges.push_back(e);
                                    nodes[def]->outEdges.push_back(e);
                                }
                            }
                        }
                    }

                    // Topo sort (Kahn)
                    llvm::DenseMap<Operation*, int> indeg;
                    SmallVector<Operation*, 64> q;

                    for (auto& p : nodes) {
                        Operation* op = p.first;
                        int cnt = p.second->inEdges.size();  // 使用我们构建的边
                        indeg[op] = cnt;
                        if (cnt == 0)
                            q.push_back(op);
                    }

                    while (!q.empty()) {
                        Operation* op = q.pop_back_val();
                        if (!nodes.count(op))
                            continue;  // 安全检查
                        topo.push_back(nodes[op].get());
                        for (Edge* e : nodes[op]->outEdges) {
                            indeg[e->consumer]--;
                            if (indeg[e->consumer] == 0)
                                q.push_back(e->consumer);
                        }
                    }

                    // DP forward
                    for (Node* n : topo) {
                        Operation* op = n->op;
                        llvm::errs() << "[DP-FWD] Evaluating node: " << op->getName() << "\n";

                        for (Mode m : {Mode::SIMD, Mode::SISD}) {
                            if (!costModel.feasible(op, m)) {
                                llvm::errs() << "  Mode " << (m == Mode::SIMD ? "SIMD" : "SISD")
                                             << " not feasible for " << op->getName() << "\n";
                                continue;
                            }

                            double inCost = 0.0;
                            for (Edge* e : n->inEdges) {
                                Node* pnode = nodes[e->producer].get();

                                double costFromSIMD = pnode->costSIMD +
                                                      costModel.costCast(Mode::SIMD, m, e->val);
                                double costFromSISD = pnode->costSISD +
                                                      costModel.costCast(Mode::SISD, m, e->val);

                                double bestPredCost = std::min(costFromSIMD, costFromSISD);

                                llvm::errs() << "    Edge from " << pnode->op->getName()
                                             << " to " << op->getName()
                                             << " [pathSIMD=" << costFromSIMD
                                             << " pathSISD=" << costFromSISD
                                             << " best=" << bestPredCost << "]\n";

                                // 修复 #3: 关键路径应该是最长的前驱路径
                                inCost = std::max(inCost, bestPredCost);
                            }

                            double nodeC = costModel.costOp(op, m);
                            double total = inCost + nodeC;

                            llvm::errs() << "  Mode " << (m == Mode::SIMD ? "SIMD" : "SISD")
                                         << " -> inCost=" << inCost
                                         << ", opCost=" << nodeC
                                         << ", total=" << total << "\n";

                            if (m == Mode::SIMD)
                                n->costSIMD = total;
                            else
                                n->costSISD = total;
                        }

                        llvm::errs() << "  => Final for " << op->getName()
                                     << ": SIMD=" << n->costSIMD
                                     << ", SISD=" << n->costSISD << "\n\n";
                    }

                    // Backtrack
                    llvm::DenseMap<Operation*, Mode> finalMode;
                    // 从拓扑排序的末端（sinks）开始
                    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
                        Node* n = *it;
                        Operation* op = n->op;

                        if (!finalMode.count(op)) {
                            // 如果是 sink (没有出边在 graph 中), 基于自身成本决定
                            if (n->outEdges.empty()) {
                                finalMode[op] = (n->costSIMD <= n->costSISD ? Mode::SIMD : Mode::SISD);
                            } else {
                                // 否则, 它已经被后继节点设置过了
                                // 如果图断开了, 这里做个保险
                                if (!finalMode.count(op)) {
                                    finalMode[op] = (n->costSIMD <= n->costSISD ? Mode::SIMD : Mode::SISD);
                                }
                            }
                        }

                        Mode cm = finalMode[op];
                        n->chosenMode = cm;

                        // 为前驱节点选择模式
                        for (Edge* e : n->inEdges) {
                            Node* pnode = nodes[e->producer].get();
                            double cSIMD = pnode->costSIMD +
                                           costModel.costCast(Mode::SIMD, cm, e->val);
                            double cSISD = pnode->costSISD +
                                           costModel.costCast(Mode::SISD, cm, e->val);
                            Mode nm = (cSIMD <= cSISD ? Mode::SIMD : Mode::SISD);

                            // 在回溯时设置前驱节点的模式
                            finalMode[pnode->op] = nm;
                        }
                    }

                    llvm::errs() << "=== Mode selection for " << func.getName() << " ===\n";
                    for (Node* n : topo) {
                        llvm::errs() << "Op " << n->op->getName()
                                     << " SIMD=" << n->costSIMD
                                     << " SISD=" << n->costSISD
                                     << " chosen="
                                     << (n->chosenMode == Mode::SIMD ? "SIMD" : "SISD")
                                     << "\n";
                    }

                    // Rewrite IR
                    IRRewriter rewriter(&getContext());
                    // 按拓扑排序的 *正常* 顺序（从 source 到 sink）重写
                    // 这样在处理当前 op 时，它的输入已经被重写或 cast
                    for (Node* n : topo) {
                        Operation* op = n->op;
                        Mode m = n->chosenMode;

                        // 1. 为操作数插入 Casts
                        // 必须先插入 cast，这样后续的 op 重写才能使用它们
                        rewriter.setInsertionPoint(op);
                        for (OpOperand& use : op->getOpOperands()) {
                            if (Operation* p = use.get().getDefiningOp()) {
                                if (nodes.count(p)) {  // 确保前驱是我们追踪的节点
                                    Mode pm = nodes[p]->chosenMode;
                                    if (pm != m) {
                                        Value v = use.get();
                                        Value casted;
                                        if (pm == Mode::SIMD && m == Mode::SISD) {
                                            auto vTy = dyn_cast<simd::SIMDCipherType>(v.getType());
                                            if (!vTy)
                                                continue;
                                            auto targetTy = sisd::SISDCipherType::get(
                                                op->getContext(), vTy.getPlaintextCount(), vTy.getElementType());
                                            casted = rewriter.create<simd::SIMDCastSIMDCipherToSISDCipherOp>(
                                                op->getLoc(), targetTy, v);
                                        } else if (pm == Mode::SISD && m == Mode::SIMD) {
                                            auto vTy = dyn_cast<sisd::SISDCipherType>(v.getType());
                                            if (!vTy)
                                                continue;

                                            // 推断 SIMD level
                                            // 默认 1，但如果 op 结果是 !simd.simdcipher, 用它的 level
                                            int64_t targetLevel = simd::DEFAULT_LEVEL;
                                            if (auto resTy = dyn_cast<simd::SIMDCipherType>(op->getResult(0).getType())) {
                                                targetLevel = resTy.getLevel();
                                            }

                                            auto targetTy = simd::SIMDCipherType::get(
                                                op->getContext(), targetLevel, vTy.getPlaintextCount(), vTy.getElementType());
                                            casted = rewriter.create<sisd::SISDCastSISDCipherToSIMDCipherOp>(
                                                op->getLoc(), targetTy, v);
                                        }
                                        if (casted)
                                            use.set(casted);
                                    }
                                }
                            }
                        }

                        // 2. 重写 Op 自身 (例如 SIMDMinOp -> SISDMinOp)
                        // if (isa<simd::SIMDMinOp>(op) && m == Mode::SISD) {
                        //     rewriter.setInsertionPoint(op);
                        //     auto simdOp = cast<simd::SIMDMinOp>(op);
                        //     Value in = simdOp.getOperand();  // 这个操作数现在应该已经被 cast 了

                        //     auto sTy = cast<simd::SIMDCipherType>(op->getResult(0).getType());
                        //     Type elemTy = sTy.getElementType();

                        //     // 结果类型也需要是 SISD
                        //     auto sisdResTy = sisd::SISDCipherType::get(op->getContext(),
                        //                                                sTy.getPlaintextCount(), elemTy);

                        //     Value newMin = rewriter.create<sisd::SISDMinOp>(
                        //         op->getLoc(), sisdResTy, in);

                        //     // 因为后续的 op (decrypt) 期望 SIMD，所以需要 cast回去
                        //     Value back = rewriter.create<sisd::SISDCastSISDCipherToSIMDCipherOp>(
                        //         op->getLoc(), sTy, newMin);

                        //     rewriter.replaceOp(op, back);
                        // }

                        // 2. 重写 Op 自身 (例如 SIMDMinOp -> SISDMinOp)
                        rewriter.setInsertionPoint(op);  // 设置插入点

                        // --- Case 1: SIMD Op -> SISD Op ---

                        if (isa<simd::SIMDMinOp>(op) && m == Mode::SISD) {
                            auto simdOp = cast<simd::SIMDMinOp>(op);
                            Value in = simdOp.getOperand();  // 假设 MinOp 是一元操作
                            auto sTy = cast<simd::SIMDCipherType>(op->getResult(0).getType());

                            auto sisdResTy = sisd::SISDCipherType::get(op->getContext(),
                                                                       sTy.getPlaintextCount(), sTy.getElementType());

                            Value newMin = rewriter.create<sisd::SISDMinOp>(
                                op->getLoc(), sisdResTy, in);

                            // Cast back，因为原op的user期望一个SIMD类型
                            Value back = rewriter.create<sisd::SISDCastSISDCipherToSIMDCipherOp>(
                                op->getLoc(), sTy, newMin);
                            rewriter.replaceOp(op, back);
                        } else if (isa<simd::SIMDSubOp>(op) && m == Mode::SISD) {
                            // 假设 SubOp 是二元操作
                            Value lhs = op->getOperand(0);  // 应该已被 Step 1 cast
                            Value rhs = op->getOperand(1);  // 应该已被 Step 1 cast

                            auto sTy = cast<simd::SIMDCipherType>(op->getResult(0).getType());
                            auto sisdResTy = sisd::SISDCipherType::get(op->getContext(),
                                                                       sTy.getPlaintextCount(), sTy.getElementType());

                            Value newSub = rewriter.create<sisd::SISDSubOp>(
                                op->getLoc(), sisdResTy, lhs, rhs);

                            // Cast back
                            Value back = rewriter.create<sisd::SISDCastSISDCipherToSIMDCipherOp>(
                                op->getLoc(), sTy, newSub);
                            rewriter.replaceOp(op, back);
                        }

                        // --- Case 2: SISD Op -> SIMD Op ---

                        else if (isa<sisd::SISDMinOp>(op) && m == Mode::SIMD) {
                            auto sisdOp = cast<sisd::SISDMinOp>(op);
                            Value in = sisdOp.getOperand();  // 假设 MinOp 是一元操作
                            auto sisdTy = cast<sisd::SISDCipherType>(op->getResult(0).getType());

                            // 推断 target level (同 Step 1 cast 逻辑)
                            int64_t targetLevel = simd::DEFAULT_LEVEL;
                            if (auto inTy = dyn_cast<simd::SIMDCipherType>(in.getType())) {
                                targetLevel = inTy.getLevel();
                            }

                            auto simdResTy = simd::SIMDCipherType::get(op->getContext(),
                                                                       targetLevel,
                                                                       sisdTy.getPlaintextCount(),
                                                                       sisdTy.getElementType());

                            Value newMin = rewriter.create<simd::SIMDMinOp>(
                                op->getLoc(), simdResTy, in);

                            // Cast back
                            Value back = rewriter.create<simd::SIMDCastSIMDCipherToSISDCipherOp>(
                                op->getLoc(), sisdTy, newMin);
                            rewriter.replaceOp(op, back);
                        } else if (isa<sisd::SISDSubOp>(op) && m == Mode::SIMD) {
                            // 假设 SubOp 是二元操作
                            Value lhs = op->getOperand(0);  // 应该已被 Step 1 cast
                            Value rhs = op->getOperand(1);  // 应该已被 Step 1 cast

                            auto sisdTy = cast<sisd::SISDCipherType>(op->getResult(0).getType());

                            // 推断 target level
                            int64_t targetLevel = simd::DEFAULT_LEVEL;
                            if (auto lhsTy = dyn_cast<simd::SIMDCipherType>(lhs.getType())) {
                                targetLevel = lhsTy.getLevel();
                            }

                            auto simdResTy = simd::SIMDCipherType::get(op->getContext(),
                                                                       targetLevel,
                                                                       sisdTy.getPlaintextCount(),
                                                                       sisdTy.getElementType());

                            Value newSub = rewriter.create<simd::SIMDSubOp>(
                                op->getLoc(), simdResTy, lhs, rhs);

                            // Cast back
                            Value back = rewriter.create<simd::SIMDCastSIMDCipherToSISDCipherOp>(
                                op->getLoc(), sisdTy, newSub);
                            rewriter.replaceOp(op, back);
                        }

                        // 注意: SIMDMultOp, SIMDEncryptOp, SIMDDecryptOp 不需要替换
                        // 因为它们在 SISD 模式下是 'not feasible' 的
                    }
                    // allEdges 会在 func 循环结束时自动释放内存
                }
            }
        };
    }  // namespace

}  // namespace mlir::libra::mdsel