// SCFHETypeConverter.cpp
#include "SCFHETypeConverter.h"
#include "SCFHETypes.h"
#include "SCFHEOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "scfhe-pass"

using namespace mlir;
using namespace mlir::libra::scfhe;

SCFHETypeConverter::SCFHETypeConverter(MLIRContext* ctx) {
    // 1. 默认转换：遇到不认识或无需加密的类型（比如 index），直接原样返回
    addConversion([](Type type) { return type; });

    // 2. 处理 MemRef -> Cipher
    addConversion([ctx](MemRefType type) -> std::optional<Type> {
        Type cipherElemType = IntegerType::get(ctx, 64);
        int64_t packedSize = type.hasStaticShape() ? type.getNumElements() : ShapedType::kDynamic;
        auto resultType = SCFHECipherType::get(ctx, packedSize, cipherElemType);

        LLVM_DEBUG(llvm::dbgs() << "[TypeConverter] Converted MemRefType: "
                                << type << " -> " << resultType << "\n");
        return resultType;
    });

    // 3. 处理浮点标量 f64 -> Cipher
    addConversion([ctx](Float64Type type) -> std::optional<Type> {
        Type cipherElemType = IntegerType::get(ctx, 64);
        auto resultType = SCFHECipherType::get(ctx, 1, cipherElemType);

        LLVM_DEBUG(llvm::dbgs() << "[TypeConverter] Converted Float64Type: "
                                << type << " -> " << resultType << "\n");
        return resultType;
    });

    // 4. 处理整数标量 (比如 i32, i64) -> Cipher
    addConversion([ctx](IntegerType type) -> std::optional<Type> {
        Type cipherElemType = IntegerType::get(ctx, 64);
        auto resultType = SCFHECipherType::get(ctx, 1, cipherElemType);

        LLVM_DEBUG(llvm::dbgs() << "[TypeConverter] Converted IntegerType: "
                                << type << " -> " << resultType << "\n");
        return resultType;
    });

    // 5. Source Materialization (反向修补：从密文回退到明文类型)
    addSourceMaterialization([](OpBuilder& b, Type t, ValueRange i, Location l) -> Value {
        LLVM_DEBUG(llvm::dbgs() << "[TypeConverter] SourceMaterialization: Wrapping with Cast to -> "
                                << t << "\n");
        return b.create<UnrealizedConversionCastOp>(l, t, i).getResult(0);
    });

    // 6. Target Materialization (正向修补：从明文强制套上密文马甲)
    addTargetMaterialization([](OpBuilder& b, Type t, ValueRange i, Location l) -> Value {
        LLVM_DEBUG(llvm::dbgs() << "[TypeConverter] TargetMaterialization: Wrapping with Cast to -> " << t << "\n");

        if (isa<SCFHECipherType>(t) && i.size() == 1) {
            Type inType = i[0].getType();
            if (isa<FloatType>(inType) || isa<IntegerType>(inType)) {
                LLVM_DEBUG(llvm::dbgs() << "  -> Auto-encrypting plaintext to cipher.\n");
                return b.create<scfhe::SCFHEEncryptOp>(l, t, i[0]).getResult();
            }
        }

        return b.create<UnrealizedConversionCastOp>(l, t, i).getResult(0);
    });
}