#ifndef LIB_DIALECT_TFHERUST_IR_TFHERUSTDIALECT_H_
#define LIB_DIALECT_TFHERUST_IR_TFHERUSTDIALECT_H_

#include "mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/IR/Dialect.h"                // from @llvm-project
#include "mlir/IR/DialectImplementation.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"           // from @llvm-project
#include "mlir/IR/PatternMatch.h"           // from @llvm-project

// Generated headers (block clang-format from messing up order)
#include "FlyHEDialect.h.inc"
#include "mlir/IR/Types.h"  // from @llvm-project

namespace mlir {
    namespace flyhe {

        template <typename ConcreteType>
        class PassByReference
            : public TypeTrait::TraitBase<ConcreteType, PassByReference> {};

        template <typename ConcreteType>
        class EncryptedInteger
            : public TypeTrait::TraitBase<ConcreteType, EncryptedInteger> {};

    }  // namespace flyhe
}  // namespace mlir

#endif  // LIB_DIALECT_TFHERUST_IR_TFHERUSTDIALECT_H_
