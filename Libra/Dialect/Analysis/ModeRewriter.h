#ifndef LIBRA_MDSEL_MODEREWRITER_H
#define LIBRA_MDSEL_MODEREWRITER_H

#include "CostModel.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir::libra::mdsel {

    struct NodeInfo {
        Mode mode = Mode::SIMD;
        int finalLevel = MAX_SIMD_LEVEL;
        bool triggerBoot = false;
        int64_t vectorCount = 8;
    };

    // 执行单个操作的重写（包括类型转换、对齐和新 Op 创建）
    void rewriteOperation(Operation* op,
                          const NodeInfo& nd,
                          llvm::DenseMap<Value, Value>& rewriteMap,
                          IRRewriter& rewriter);

    // 修复循环参数和结果的 Cast
    void fixLoopCasts(func::FuncOp func);

    // 添加全局配置属性
    void attachGlobalConfig(ModuleOp module);

} // namespace mlir::libra::mdsel

#endif // LIBRA_MDSEL_MODEREWRITER_H