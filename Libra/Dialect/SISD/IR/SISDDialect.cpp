// #include "mlir/IR/Builders.h"
// #include "mlir/IR/DialectImplementation.h"
// #include "llvm/ADT/TypeSwitch.h"

// #include "SISDTypes.h"
// #include "SISDDialect.h"

// using namespace mlir;
// using namespace mlir::libra::sisd;

// #define GET_TYPEDEF_CLASSES
// #include "SISDTypes.cpp.inc"

// #define GET_OP_CLASSES
// #include "SISDOps.cpp.inc"

// #include "SISDDialect.cpp.inc"

// void SISDDialect::initialize() {
//     // llvm::outs() << "=== SISDDialect::initialize() running ===\n";

//     addTypes<
// #define GET_TYPEDEF_LIST
// #include "SISDTypes.cpp.inc"
//         >();

//     addOperations<
// #define GET_OP_LIST
// #include "SISDOps.cpp.inc"
//         >();
// }

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/PatternMatch.h" // 【新增】用于模式匹配和重写

#include "llvm/ADT/TypeSwitch.h"

#include "SISDTypes.h"
#include "SISDDialect.h"
#include "SISDOps.h" // 【新增】引入 SISD 算子定义
#include "SIMDOps.h" // 【新增】引入 SIMD 算子定义，以便识别 SIMD->SISD 的 Cast

using namespace mlir;
using namespace mlir::libra::sisd;

// =============================================================================
// Canonicalization Patterns (自动消除冗余 Cast)
// =============================================================================

// 规则：消除冗余的 SISD -> SIMD Cast
struct FoldSISDCastToSIMD : public OpRewritePattern<SISDCastSISDCipherToSIMDCipherOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(SISDCastSISDCipherToSIMDCipherOp op,
                                  PatternRewriter& rewriter) const override {
        Value input = op.getInput();
        Type inTy = input.getType();
        Type outTy = op.getResult().getType();

        // 【新增】：情况 1 - 如果输入和输出的类型一模一样，直接抵消
        if (inTy == outTy) {
            rewriter.replaceOp(op, input);
            return success();
        }

        // 情况 2：背靠背互相抵消 (simd.cast_to_sisd 之后立刻 sisd.cast_to_simd)
        if (auto prevCast = input.getDefiningOp<libra::simd::SIMDCastSIMDCipherToSISDCipherOp>()) {
            Value originalInput = prevCast.getInput();
            if (originalInput.getType() == outTy) {
                rewriter.replaceOp(op, originalInput);
                return success();
            }
        }
        return failure();
    }
};

// 注册重写规则，供 MLIR 的 Canonicalizer 调用
void SISDCastSISDCipherToSIMDCipherOp::getCanonicalizationPatterns(
    RewritePatternSet& results, MLIRContext* context) {
    results.add<FoldSISDCastToSIMD>(context);
}

// =============================================================================

#define GET_TYPEDEF_CLASSES
#include "SISDTypes.cpp.inc"

#define GET_OP_CLASSES
#include "SISDOps.cpp.inc"

#include "SISDDialect.cpp.inc"

void SISDDialect::initialize() {
    // llvm::outs() << "=== SISDDialect::initialize() running ===\n";

    addTypes<
#define GET_TYPEDEF_LIST
#include "SISDTypes.cpp.inc"
        >();

    addOperations<
#define GET_OP_LIST
#include "SISDOps.cpp.inc"
        >();
}