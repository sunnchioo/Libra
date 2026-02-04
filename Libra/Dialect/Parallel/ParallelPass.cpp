//===----------------------------------------------------------------------===//
// ParallelPass.cpp - Parallelization & Synchronization Pass
//===----------------------------------------------------------------------===//

#include "ParallelPass.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include <algorithm>
#include <map>
#include <vector>

using namespace mlir;
using namespace mlir::libra;

namespace mlir::libra::parallel {

#define GEN_PASS_DEF_ADDBOUNDEDSTREAMID
#include "ParallelPass.h.inc"

    namespace {

        // 将 Pass 改为 ModulePass，以便安全地插入全局函数声明
        class AddBoundedStreamId : public impl::AddBoundedStreamIdBase<AddBoundedStreamId> {
            using impl::AddBoundedStreamIdBase<AddBoundedStreamId>::AddBoundedStreamIdBase;

            void runOnOperation() override {
                ModuleOp module = getOperation();

                // 1. 确保同步函数声明存在
                ensureSyncFuncDeclarations(module);

                // 2. 遍历所有 Function 进行处理
                // 由于我们现在是 ModulePass，需要手动遍历函数
                module.walk([&](func::FuncOp func) {
                    // 跳过刚才插入的声明函数（它们没有 Body）
                    if (func.isExternal())
                        return;

                    // 2.1 Stream Assignment
                    func.walk([&](Block* block) {
                        runAssignmentOnBlock(block);
                    });

                    // 2.2 Synchronization Insertion
                    IRRewriter rewriter(&getContext());
                    func.walk([&](Block* block) {
                        insertSyncOnBlock(block, rewriter);
                    });
                });
            }

            // --- Part 0: Ensure Declarations ---
            void ensureSyncFuncDeclarations(ModuleOp module) {
                OpBuilder builder(module.getBodyRegion());
                // 确保插入点在 Module 最前面
                if (!module.getBody()->empty()) {
                    builder.setInsertionPoint(&module.getBody()->front());
                }

                MLIRContext* ctx = &getContext();
                Type indexType = builder.getIndexType();
                Type noneType = builder.getNoneType();  // 或者 void

                // 声明 @parallel.signal: () -> index
                if (!module.lookupSymbol("parallel.signal")) {
                    auto funcType = builder.getFunctionType({}, {indexType});
                    auto funcOp = builder.create<func::FuncOp>(
                        module.getLoc(), "parallel.signal", funcType);
                    funcOp.setPrivate();  // 设置为私有，避免导出
                }

                // 声明 @parallel.await: (index) -> ()
                if (!module.lookupSymbol("parallel.await")) {
                    auto funcType = builder.getFunctionType({indexType}, {});
                    auto funcOp = builder.create<func::FuncOp>(
                        module.getLoc(), "parallel.await", funcType);
                    funcOp.setPrivate();
                }
            }

            // --- Part 1: Stream Assignment (不变) ---
            void runAssignmentOnBlock(Block* block) {
                llvm::DenseMap<Operation*, int> opLevels;
                llvm::DenseMap<Operation*, int> opStreamIds;

                llvm::SmallVector<Operation*, 16> ops;
                for (Operation& op : *block) {
                    StringRef ns = op.getName().getDialectNamespace();
                    bool isTarget = (ns == "simd" || ns == "sisd");
                    if (!isTarget && ns == "vector") {
                        if (op.getName().getStringRef().contains("transfer_read"))
                            isTarget = true;
                    }
                    if (isTarget)
                        ops.push_back(&op);
                }
                if (ops.empty())
                    return;

                bool changed = true;
                while (changed) {
                    changed = false;
                    for (Operation* op : ops) {
                        int maxPredLevel = -1;
                        bool hasDependency = false;
                        for (Value operand : op->getOperands()) {
                            if (Operation* def = operand.getDefiningOp()) {
                                if (opLevels.count(def)) {
                                    maxPredLevel = std::max(maxPredLevel, opLevels[def]);
                                    hasDependency = true;
                                }
                            }
                        }
                        int newLevel = hasDependency ? (maxPredLevel + 1) : 0;
                        if (opLevels.find(op) == opLevels.end() || opLevels[op] != newLevel) {
                            opLevels[op] = newLevel;
                            changed = true;
                        }
                    }
                }

                std::map<int, std::vector<Operation*>> levels;
                int maxLevelFound = 0;
                for (const auto& pair : opLevels) {
                    levels[pair.second].push_back(pair.first);
                    maxLevelFound = std::max(maxLevelFound, pair.second);
                }

                std::vector<int> streamLoadInLevel(maxStreams, 0);
                for (int l = 0; l <= maxLevelFound; ++l) {
                    auto& currentOps = levels[l];
                    std::fill(streamLoadInLevel.begin(), streamLoadInLevel.end(), 0);

                    for (Operation* op : currentOps) {
                        int assignedId = -1;
                        for (Value operand : op->getOperands()) {
                            if (Operation* def = operand.getDefiningOp()) {
                                if (opStreamIds.count(def)) {
                                    int predStream = opStreamIds[def];
                                    if (streamLoadInLevel[predStream] == 0) {
                                        assignedId = predStream;
                                        break;
                                    }
                                }
                            }
                        }
                        if (assignedId == -1) {
                            for (int id = 0; id < maxStreams; ++id) {
                                if (streamLoadInLevel[id] == 0) {
                                    assignedId = id;
                                    break;
                                }
                            }
                        }
                        if (assignedId == -1) {
                            auto minIt = std::min_element(streamLoadInLevel.begin(), streamLoadInLevel.end());
                            assignedId = std::distance(streamLoadInLevel.begin(), minIt);
                        }

                        opStreamIds[op] = assignedId;
                        streamLoadInLevel[assignedId]++;

                        Builder builder(op->getContext());
                        op->setAttr("parallel.stream_id", builder.getI32IntegerAttr(assignedId));
                    }
                }
            }

            // --- Part 2: Synchronization Insertion (不变) ---
            int getStreamId(Operation* op) {
                if (auto attr = op->getAttrOfType<IntegerAttr>("parallel.stream_id")) {
                    return attr.getInt();
                }
                return 0;
            }

            void insertSyncOnBlock(Block* block, IRRewriter& rewriter) {
                llvm::DenseMap<Operation*, Value> opSignals;

                for (auto& op : llvm::make_early_inc_range(*block)) {
                    if (!op.hasAttr("parallel.stream_id"))
                        continue;

                    int consumerStream = getStreamId(&op);

                    for (Value operand : op.getOperands()) {
                        Operation* producer = operand.getDefiningOp();
                        if (!producer)
                            continue;
                        if (!producer->hasAttr("parallel.stream_id"))
                            continue;

                        int producerStream = getStreamId(producer);

                        if (producerStream != consumerStream) {
                            if (opSignals.find(producer) == opSignals.end()) {
                                rewriter.setInsertionPointAfter(producer);
                                auto eventType = rewriter.getIndexType();
                                auto signalOp = rewriter.create<func::CallOp>(
                                    producer->getLoc(),
                                    "parallel.signal",
                                    TypeRange{eventType},
                                    ValueRange{});
                                signalOp->setAttr("parallel.stream_id", rewriter.getI32IntegerAttr(producerStream));
                                opSignals[producer] = signalOp.getResult(0);
                            }

                            Value eventHandle = opSignals[producer];
                            rewriter.setInsertionPoint(&op);
                            auto awaitOp = rewriter.create<func::CallOp>(
                                op.getLoc(),
                                "parallel.await",
                                TypeRange{},
                                ValueRange{eventHandle});
                            awaitOp->setAttr("parallel.stream_id", rewriter.getI32IntegerAttr(consumerStream));
                        }
                    }
                }
            }
        };

    }  // namespace

    // std::unique_ptr<Pass> createAddBoundedStreamIdPass() {
    //     return std::make_unique<AddBoundedStreamId>();
    // }

}  // namespace mlir::libra::parallel