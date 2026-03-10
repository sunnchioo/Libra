#include "SCFHEDialect.h"
#include "SCFHETypes.h"
#include "SCFHEOps.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h" // 用于调试打印

#include <limits>

using namespace mlir;
using namespace mlir::libra::scfhe;

// =============================================================================
// 1. Helper 函数
// =============================================================================

static ParseResult parseCipherDim(AsmParser& parser, int64_t& dim) {
    if (succeeded(parser.parseOptionalQuestion())) {
        dim = ShapedType::kDynamic;
        return success();
    }
    return parser.parseInteger(dim);
}

static void printCipherDim(AsmPrinter& printer, int64_t dim) {
    if (dim == ShapedType::kDynamic || dim == std::numeric_limits<int64_t>::min()) {
        printer << "?";
    } else {
        printer << dim;
    }
}

// =============================================================================
// 2. Canonicalization Patterns (优化规则)
// =============================================================================

// Pattern A: 优化 Encrypt (从 memref.cast 恢复静态形状)
struct RefineEncryptShape : public OpRewritePattern<SCFHEEncryptOp> {
    using OpRewritePattern<SCFHEEncryptOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(SCFHEEncryptOp op, PatternRewriter& rewriter) const override {
        // 1. 获取当前输出类型 (这是动态的 <?>)
        auto currentType = llvm::cast<SCFHECipherType>(op.getType());

        // 2. 获取输入并穿透 Cast
        Value originalInput = op.getOperand(); // 或 op.getOperand(0);
        Value staticInput = originalInput;

        if (auto castOp = originalInput.getDefiningOp<memref::CastOp>()) {
            staticInput = castOp.getSource();
        }

        if (!llvm::isa<MemRefType>(staticInput.getType())) {
            return failure();
        }

        // 3. 检查源头是否为静态 (现在由于有了上面的 isa 检查，这里的 cast 绝对安全了)
        auto inputMemRefType = llvm::cast<MemRefType>(staticInput.getType());
        if (!inputMemRefType.hasStaticShape()) {
            return failure();
        }

        int64_t staticSize = inputMemRefType.getDimSize(0);

        // 4. 防死循环检查
        // 注意：如果已经是静态且大小匹配，且输入也没变，才退出。
        // 如果输入变了（比如去掉了 memref.cast），即使类型是静态也要处理，但这里主要还是防 <10> -> <10>
        if (!currentType.isDynamic() && currentType.getPlaintextCount() == staticSize) {
            return failure();
        }

        // 5. 构建新的静态类型 (例如 <10>)
        auto newStaticType = SCFHECipherType::get(op.getContext(), staticSize, currentType.getElementType());

        llvm::errs() << "[DEBUG-Encrypt] Optimizing Encrypt shape to " << staticSize << " (with Cast wrapper).\n";

        // ==========================================
        // 【核心修改】不直接 ReplaceOpWithNewOp
        // ==========================================

        // A. 创建新的静态 EncryptOp
        // 它的输入是 staticInput (去掉了 memref.cast)
        // 它的输出是 newStaticType (<10>)
        auto newEncryptOp = rewriter.create<SCFHEEncryptOp>(op.getLoc(), newStaticType, staticInput);

        // B. 创建适配器 (CastOp)
        // 将 <10> 转回原本的 <?>，以满足外部使用者 (如 func.call) 的需求
        auto castOp = rewriter.create<SCFHECastOp>(
            op.getLoc(),
            currentType,             // 目标类型：原来的 <?>
            newEncryptOp.getResult() // 源数据：新的 <10>
        );

        // C. 用 Cast 的结果 (<?>) 替换掉老 Op 的结果
        // 这样外部看起来类型没变，就不会报错了
        rewriter.replaceOp(op, castOp.getResult());

        return success();
    }
};

// Pattern B: 优化二元操作 (Add, Sub, Mult, Div)
template <typename BinaryOp>
struct RefineBinaryOpShape : public OpRewritePattern<BinaryOp> {
    using OpRewritePattern<BinaryOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(BinaryOp op, PatternRewriter& rewriter) const override {
        // 1. 获取当前结果类型
        auto currentResultType = llvm::cast<SCFHECipherType>(op.getType());

        // 2. 剥离 Cast 的辅助 Lambda
        auto peelCast = [](Value v) -> Value {
            if (auto castOp = v.getDefiningOp<SCFHECastOp>()) {
                if (auto srcType = llvm::dyn_cast<SCFHECipherType>(castOp.getSource().getType())) {
                    if (srcType.hasStaticShape()) {
                        return castOp.getSource();
                    }
                }
            }
            return v;
        };

        // 3. 获取处理后的输入
        Value lhs = op->getOperand(0);
        Value rhs = op->getOperand(1);

        Value newLhs = peelCast(lhs);
        Value newRhs = peelCast(rhs);

        // 4. 类型检查 (使用 dyn_cast 进行安全检查)
        // 【关键修复】：这里必须用 dyn_cast，因为 newRhs 可能是 f64 (明文)
        auto newLhsType = llvm::dyn_cast<SCFHECipherType>(newLhs.getType());
        auto newRhsType = llvm::dyn_cast<SCFHECipherType>(newRhs.getType());

        // LHS 必须是密文，否则无法推导形状
        if (!newLhsType) {
            return failure();
        }

        // 确定新的结果类型 (newResultType)
        SCFHECipherType newResultType;

        if (newRhsType) {
            // === Case A: 两个都是密文 (Add, Sub, Mult, 或 Div密文) ===
            // 要求两者都是静态形状且形状匹配
            bool inputsAreStatic = newLhsType.hasStaticShape() && newRhsType.hasStaticShape();
            bool shapesMatch = newLhsType == newRhsType;

            if (!inputsAreStatic || !shapesMatch) {
                return failure();
            }
            newResultType = newLhsType;
        } else {
            // === Case B: RHS 是明文 (例如 Div f64) ===
            // 这种情况下，形状完全由 LHS 决定
            // 我们只需要确保 LHS 是静态形状即可进行优化
            if (!newLhsType.hasStaticShape()) {
                return failure();
            }
            newResultType = newLhsType;
        }

        // =======================================================
        // 5. 防死循环检查 (Guard Clause)
        // =======================================================

        // 检查 A: 结果形状是否已经是最优的静态形状
        bool resultIsAlreadyStatic = currentResultType.hasStaticShape() &&
                                     (currentResultType == newResultType);

        // 检查 B: 输入操作数是否发生了变化 (是否成功剥离了 Cast)
        bool inputsDidNotChange = (lhs == newLhs) && (rhs == newRhs);

        // 结论: 如果结果已经是理想形状，且输入没有变得更好，则无需重写
        if (resultIsAlreadyStatic && inputsDidNotChange) {
            return failure();
        }

        // =======================================================
        // 6. 执行重写
        // =======================================================

        llvm::errs() << "[DEBUG-Binary] Optimizing BinaryOp shape to "
                     << newResultType.getPlaintextCount() << "\n";

        rewriter.replaceOpWithNewOp<BinaryOp>(op, newResultType, newLhs, newRhs);

        return success();
    }
};

// Pattern C: 优化 Decrypt (从 CipherType 推导 MemRefShape)
struct RefineDecryptShape : public OpRewritePattern<SCFHEDecryptOp> {
    using OpRewritePattern<SCFHEDecryptOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(SCFHEDecryptOp op, PatternRewriter& rewriter) const override {
        Value input = op.getCipher();
        Value staticInput = input;

        // 尝试剥离 Cast
        if (auto castOp = input.getDefiningOp<SCFHECastOp>()) {
            staticInput = castOp.getSource();
        }

        auto inputType = llvm::cast<SCFHECipherType>(staticInput.getType());

        // 1. 基础检查：Input 必须声明为 Static
        if (!inputType.hasStaticShape()) {
            return failure();
        }

        int64_t size = inputType.getPlaintextCount();

        // =======================================================
        // 【关键修复 1】安全检查：防止获取到无效的 Dynamic Size
        // =======================================================
        // 如果底层类型实现有问题，导致 hasStaticShape=true 但 size 却是 kDynamic，
        // 我们必须立刻停止，否则会创建出 memref<?xf64> 导致死循环。
        if (size == ShapedType::kDynamic) {
            return failure();
        }

        auto currentMemRefType = llvm::cast<MemRefType>(op.getType());

        // 构造预期的目标类型
        auto newMemRefType = MemRefType::get({size}, currentMemRefType.getElementType());

        // =======================================================
        // 【关键修复 2】防死循环：如果类型完全没有变化，必须退出！
        // =======================================================
        if (newMemRefType == currentMemRefType) {
            // 如果我们也剥离了 Cast，那么 input != staticInput，我们可能还需要继续。
            // 但如果连 input 也没变，那就是纯粹的原地踏步。
            if (input == staticInput) {
                return failure();
            }
        }

        // =======================================================
        // 【关键修复 3】逻辑检查：如果当前已经是正确的静态形状
        // =======================================================
        if (currentMemRefType.hasStaticShape() &&
            currentMemRefType.getDimSize(0) == size &&
            staticInput == input) {
            return failure();
        }

        llvm::errs() << "[DEBUG-Decrypt] Optimizing Decrypt to static memref<" << size << ">.\n";

        rewriter.replaceOpWithNewOp<SCFHEDecryptOp>(op, newMemRefType, staticInput);

        return success();
    }
};

struct ReduceAddLoadFold : public OpRewritePattern<SCFHEReduceAddOp> {
    using OpRewritePattern::OpRewritePattern;
    LogicalResult matchAndRewrite(SCFHEReduceAddOp op, PatternRewriter& rewriter) const override {
        Value operand = op.getOperand();
        if (auto castOp = operand.getDefiningOp<SCFHECastOp>()) {
            rewriter.modifyOpInPlace(op, [&]() {
                op.setOperand(castOp.getOperand());
            });
            return success();
        }
        return failure();
    }
};

// =============================================================================
// 3. 注册 Canonicalization Patterns
// =============================================================================

// 为 Encrypt 注册
void SCFHEEncryptOp::getCanonicalizationPatterns(RewritePatternSet& results, MLIRContext* context) {
    results.add<RefineEncryptShape>(context);
}

// 为 Add 注册 (注意：必须在 .td 中有 hasCanonicalizer = 1)
void SCFHEAddOp::getCanonicalizationPatterns(RewritePatternSet& results, MLIRContext* context) {
    results.add<RefineBinaryOpShape<SCFHEAddOp>>(context);
}

// 为 Sub 注册
void SCFHESubOp::getCanonicalizationPatterns(RewritePatternSet& results, MLIRContext* context) {
    results.add<RefineBinaryOpShape<SCFHESubOp>>(context);
}

// 为 Mult 注册
void SCFHEMultOp::getCanonicalizationPatterns(RewritePatternSet& results, MLIRContext* context) {
    results.add<RefineBinaryOpShape<SCFHEMultOp>>(context);
}

// 为 Div 注册
void SCFHEDivOp::getCanonicalizationPatterns(RewritePatternSet& results, MLIRContext* context) {
    results.add<RefineBinaryOpShape<SCFHEDivOp>>(context);
}

// 为 Decrypt 注册
void SCFHEDecryptOp::getCanonicalizationPatterns(RewritePatternSet& results, MLIRContext* context) {
    results.add<RefineDecryptShape>(context);
}

// 为 ReduceAdd 注册
void SCFHEReduceAddOp::getCanonicalizationPatterns(RewritePatternSet& results, MLIRContext* context) {
    results.add<ReduceAddLoadFold>(context);
}

// =============================================================================
// 4. Shape Inference Implementation
// =============================================================================

static LogicalResult inferBinaryCipherShape(MLIRContext* context, Value lhs, Value rhs, SmallVectorImpl<Type>& inferredReturnTypes) {
    auto lhsType = llvm::dyn_cast<SCFHECipherType>(lhs.getType());
    auto rhsType = llvm::dyn_cast<SCFHECipherType>(rhs.getType());

    // 【修改点 1】LHS 必须是密文
    if (!lhsType)
        return failure();

    // 【修改点 2】RHS 可以是密文，也可以是浮点/整数标量 (针对 Div 优化)
    if (!rhsType && !llvm::isa<FloatType, IntegerType>(rhs.getType())) {
        return failure(); // RHS 既不是密文也不是支持的标量
    }

    // 确定结果的 Plaintext Count
    int64_t count = lhsType.getPlaintextCount();

    // 如果 RHS 也是密文，则需要处理动态形状传播逻辑
    if (rhsType) {
        if (lhsType.isDynamic() && !rhsType.isDynamic()) {
            count = rhsType.getPlaintextCount();
        } else if (!lhsType.isDynamic() && !rhsType.isDynamic()) {
            // 两个都是静态，通常取较大的（虽然一般应该相等）
            count = std::max(lhsType.getPlaintextCount(), rhsType.getPlaintextCount());
        }
    }
    // 如果 RHS 是明文，结果形状完全跟随 LHS，无需改变 count

    Type elemType = lhsType.getElementType();
    auto resultType = SCFHECipherType::get(context, count, elemType);
    inferredReturnTypes.push_back(resultType);
    return success();
}

LogicalResult SCFHEAddOp::inferReturnTypes(
    MLIRContext* context, std::optional<Location> location,
    ValueRange operands, DictionaryAttr attributes, OpaqueProperties properties,
    RegionRange regions, SmallVectorImpl<Type>& inferredReturnTypes) {
    return inferBinaryCipherShape(context, operands[0], operands[1], inferredReturnTypes);
}

LogicalResult SCFHESubOp::inferReturnTypes(
    MLIRContext* context, std::optional<Location> location,
    ValueRange operands, DictionaryAttr attributes, OpaqueProperties properties,
    RegionRange regions, SmallVectorImpl<Type>& inferredReturnTypes) {
    return inferBinaryCipherShape(context, operands[0], operands[1], inferredReturnTypes);
}

LogicalResult SCFHEMultOp::inferReturnTypes(
    MLIRContext* context, std::optional<Location> location,
    ValueRange operands, DictionaryAttr attributes, OpaqueProperties properties,
    RegionRange regions, SmallVectorImpl<Type>& inferredReturnTypes) {
    return inferBinaryCipherShape(context, operands[0], operands[1], inferredReturnTypes);
}

LogicalResult SCFHEDivOp::inferReturnTypes(
    MLIRContext* context, std::optional<Location> location,
    ValueRange operands, DictionaryAttr attributes, OpaqueProperties properties,
    RegionRange regions, SmallVectorImpl<Type>& inferredReturnTypes) {
    return inferBinaryCipherShape(context, operands[0], operands[1], inferredReturnTypes);
}

LogicalResult SCFHEDecryptOp::inferReturnTypes(
    MLIRContext* context, std::optional<Location> location,
    ValueRange operands, DictionaryAttr attributes, OpaqueProperties properties,
    RegionRange regions, SmallVectorImpl<Type>& inferredReturnTypes) {
    auto inputType = llvm::cast<SCFHECipherType>(operands[0].getType());
    if (inputType.isDynamic()) {
        inferredReturnTypes.push_back(MemRefType::get({ShapedType::kDynamic}, Float64Type::get(context)));
    } else {
        inferredReturnTypes.push_back(MemRefType::get({inputType.getPlaintextCount()}, Float64Type::get(context)));
    }
    return success();
}

// =============================================================================
// 5. Inliner & Init (保持不变)
// =============================================================================

struct SCFHEInlinerInterface : public DialectInlinerInterface {
    using DialectInlinerInterface::DialectInlinerInterface;
    bool isLegalToInline(Operation* call, Operation* callable, bool wouldBeCloned) const final { return true; }
    bool isLegalToInline(Operation* op, Region* dest, bool wouldBeCloned, IRMapping& valueMapping) const final { return true; }
    bool isLegalToInline(Region* dest, Region* src, bool wouldBeCloned, IRMapping& valueMapping) const final { return true; }
};

#define GET_TYPEDEF_CLASSES
#include "SCFHETypes.cpp.inc"
#define GET_OP_CLASSES
#include "SCFHEOps.cpp.inc"
#include "SCFHEDialect.cpp.inc"

void SCFHEDialect::initialize() {

    addTypes<
#define GET_TYPEDEF_LIST
#include "SCFHETypes.cpp.inc"
        >();
    addOperations<
#define GET_OP_LIST
#include "SCFHEOps.cpp.inc"
        >();

    addInterfaces<SCFHEInlinerInterface>();
}