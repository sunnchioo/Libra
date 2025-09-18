#include "FlyHETypes.h"
#include "FlyHEDialect.h"
// 1. 关键：添加 LLVM 哈希头文件，解决 llvm::hash_value 未定义问题
#include "llvm/ADT/HashExtras.h"
#include "llvm/ADT/Hashing.h"

using namespace mlir;
using namespace mlir::flyhe;

// 2. 定义 TypeStorage（修复哈希函数，符合 MLIR 规范）
struct PhantomCipherTypeStorage : public TypeStorage {
    // 存储类型的核心参数（元素类型）
    Type elementType;

    // 构造函数：初始化元素类型
    PhantomCipherTypeStorage(Type elementType) : elementType(elementType) {}

    // 3. 哈希函数：参数为 Storage 实例（与 operator== 逻辑对齐）
    static llvm::hash_code hashValue(const PhantomCipherTypeStorage &storage) {
        // 哈希元素类型（使用 LLVM 的 hash_value，自动处理 MLIR Type）
        return llvm::hash_value(storage.elementType);
    }

    // 4. 比较函数：判断两个 Storage 是否相等（类型唯一性的核心）
    bool operator==(const PhantomCipherTypeStorage &other) const {
        return elementType == other.elementType;
    }

    // 5. 构造 Storage 实例（供 MLIR TypeBase 调用，参数需与 get 函数一致）
    static PhantomCipherTypeStorage *construct(TypeStorageAllocator &alloc,
                                               Type elementType) {
        // 使用 MLIR 的分配器分配内存（避免内存泄漏）
        return new (alloc.allocate<PhantomCipherTypeStorage>())
            PhantomCipherTypeStorage(elementType);
    }
};

// 6. 实现 FlyHE_SIMDCipherType 的外部接口：获取元素类型
Type FlyHE_SIMDCipherType::getElementType() {
    // getImpl() 是 TypeBase 提供的方法，返回存储实例指针
    return getImpl()->elementType;
}

// 7. 实现 FlyHE_SIMDCipherType 的外部接口：创建类型实例
FlyHE_SIMDCipherType FlyHE_SIMDCipherType::get(MLIRContext *ctx, Type elementType) {
    // 验证元素类型合法性（可选，增强鲁棒性）
    if (!isValidElementType(elementType)) {
        llvm::report_fatal_error("Invalid element type for FlyHE_SIMDCipherType");
    }
    // 调用 TypeBase::get，自动管理类型唯一性（依赖 Storage 的 hash/==）
    return Base::get(ctx, elementType);
}

// 8. （可选）实现类型合法性验证（例如：不允许空类型、仅支持整数/浮点类型）
bool FlyHE_SIMDCipherType::isValidElementType(Type elementType) {
    return elementType && (elementType.isIntOrFloat());
}

// 9. 引入 TableGen 生成的类型定义（需确保与 .td 文件中的 typedef 对应）
#define GET_TYPEDEF_DEFINITIONS
#include "FlyHETypes.h.inc"

// 10. 实现方言的类型注册函数（需在 FlyHEDialect.h 中声明）
void flyhe::FlyHEDialect::registerTypes() {
    // 添加所有自定义类型（由 TableGen 生成的 GET_TYPEDEF_LIST 自动枚举）
    addTypes<
#define GET_TYPEDEF_LIST
#include "FlyHETypes.h.inc"
        >();
}