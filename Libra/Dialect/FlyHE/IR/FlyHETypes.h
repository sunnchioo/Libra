#ifndef LIBRA_DIALECT_FLYHE_IR_FLYHETYPES_H_
#define LIBRA_DIALECT_FLYHE_IR_FLYHETYPES_H_

#include "mlir/IR/OpDefinition.h" // from @llvm-project
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"

#include "FlyHEDialect.h"

// #define GET_TYPEDEF_DECLARATIONS
// #include "FlyHETypes.h.inc"

// 2. 包含自定义Dialect头文件（前向声明Dialect即可，避免循环依赖）
namespace mlir {
    namespace flyhe {
        class FlyHEDialect;
    } // namespace flyhe
} // namespace mlir

// 3. 引入TableGen生成的类型声明（通常放在自定义类型定义前）
// 注意：需确保FlyHETypes.h.inc由mlir_tablegen正确生成
#define GET_TYPEDEF_DECLARATIONS
#include "FlyHETypes.h.inc"

// 4. 定义类型的Storage存储类（保存类型的实际数据，如元素类型）
namespace mlir {
    namespace flyhe {
        struct FlyHE_SIMDCipherTypeStorage : public mlir::TypeStorage {
            // 存储元素类型（根据你的需求添加，如需要其他参数也在此定义）
            Type elementType;

            // 构造函数：初始化元素类型
            FlyHE_SIMDCipherTypeStorage(Type elementType) : elementType(elementType) {}

            // 用于类型唯一化的Key（MLIR通过Key缓存类型实例）
            using Key = Type;

            // 计算Key的哈希值（供MLIR缓存使用）
            static llvm::hash_code hashKey(const Key &key) {
                return llvm::hash_value(key);
            }

            // 比较两个Key是否相等（判断类型是否相同）
            bool operator==(const Key &key) const {
                return elementType == key;
            }

            // 创建Storage实例（供MLIR内部调用）
            static FlyHE_SIMDCipherTypeStorage *create(
                mlir::TypeStorageAllocator &allocator, const Key &key) {
                // 复制Key到allocator管理的内存（避免悬空引用）
                Type elementType = allocator.copyInto(key);
                return new (allocator.allocate<FlyHE_SIMDCipherTypeStorage>())
                    FlyHE_SIMDCipherTypeStorage(elementType);
            };
        } // namespace detail
    } // namespace flyhe
} // namespace mlir

// 5. 定义自定义类型（正确继承TypeBase，补充所有模板参数）
namespace mlir {
    namespace flyhe {
        class FlyHE_SIMDCipherType
            : public TypeBase<
                  FlyHE_SIMDCipherType,                // 1. 具体类型（自身）
                  Type,                                // 2. 基类（mlir::Type）
                  detail::FlyHE_SIMDCipherTypeStorage, // 3. Storage类型
                  TypeTrait::IsType> {                 // 4. 类型特征（可多个，用逗号分隔）
        public:
            // 继承父类的构造函数（必须）
            using Base::Base;

            // 6. 对外接口：创建类型实例
            static FlyHE_SIMDCipherType get(MLIRContext *ctx, Type elementType) {
                // 校验元素类型合法性（可选，根据你的需求实现）
                if (!isValidElementType(elementType)) {
                    return nullptr;
                }
                // 调用MLIR的get方法，传入Dialect命名空间和Key（元素类型）
                return Base::get(ctx, FlyHEDialect::getDialectNamespace(), elementType);
            }

            // 7. 对外接口：获取元素类型
            Type getElementType() {
                return getStorage()->elementType;
            }

            // 8. 元素类型合法性校验（示例：仅允许整数/浮点类型）
            static bool isValidElementType(Type elementType) {
                return elementType.isa<IntegerType>() || elementType.isa<FloatType>();
            }
        };
    } // namespace flyhe
} // namespace mlir

// 5. 引入 TableGen 生成的类型声明（由 mlir_tablegen 生成）
// #define GET_TYPEDEF_DECLARATIONS
// #include "FlyHETypes.h.inc"

#endif // LIBRA_DIALECT_FLYHE_IR_FLYHETYPES_H_