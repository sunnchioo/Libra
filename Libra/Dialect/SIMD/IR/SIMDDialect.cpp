#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h" // 【新增】用于匹配和重写

#include "llvm/ADT/TypeSwitch.h"

#include "SIMDDialect.h"
#include "SIMDTypes.h"
#include "SIMDOps.h" // 【新增】引入 SIMD 算子定义
#include "SISDOps.h" // 【新增】引入 SISD 算子定义，以便识别 SISD->SIMD 的 Cast

using namespace mlir;
using namespace mlir::libra::simd;

static ParseResult parseCipherDim(AsmParser& parser, int64_t& dim) {
    if (succeeded(parser.parseOptionalQuestion())) {
        dim = ShapedType::kDynamic;
        return success();
    }
    return parser.parseInteger(dim);
}

static void printCipherDim(AsmPrinter& printer, int64_t dim) {
    if (dim == ShapedType::kDynamic) {
        printer << "?";
    } else {
        printer << dim;
    }
}

// =============================================================================
// Canonicalization Patterns (自动消除冗余 Cast)
// =============================================================================

// 规则：消除冗余的 SIMD -> SISD Cast
struct FoldSIMDCastToSISD : public OpRewritePattern<SIMDCastSIMDCipherToSISDCipherOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(SIMDCastSIMDCipherToSISDCipherOp op,
                                  PatternRewriter& rewriter) const override {
        Value input = op.getInput();
        Type inTy = input.getType();
        Type outTy = op.getResult().getType();

        // 【新增】：情况 1 - 如果输入和输出的类型一模一样，直接扒掉面具
        if (inTy == outTy) {
            rewriter.replaceOp(op, input);
            return success();
        }

        // 情况 2：背靠背互相抵消 (sisd.cast_to_simd 之后立刻 simd.cast_to_sisd)
        if (auto prevCast = input.getDefiningOp<libra::sisd::SISDCastSISDCipherToSIMDCipherOp>()) {
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
void SIMDCastSIMDCipherToSISDCipherOp::getCanonicalizationPatterns(
    RewritePatternSet& results, MLIRContext* context) {
    results.add<FoldSIMDCastToSISD>(context);
}

// =============================================================================

#define GET_TYPEDEF_CLASSES
#include "SIMDTypes.cpp.inc"

#define GET_OP_CLASSES
#include "SIMDOps.cpp.inc"

#include "SIMDDialect.cpp.inc"

void SIMDDialect::initialize() {
    addTypes<
#define GET_TYPEDEF_LIST
#include "SIMDTypes.cpp.inc"
        >();

    addOperations<
#define GET_OP_LIST
#include "SIMDOps.cpp.inc"
        >();
}