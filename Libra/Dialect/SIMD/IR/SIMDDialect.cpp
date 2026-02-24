#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/TypeSwitch.h"

#include "SIMDDialect.h"
#include "SIMDTypes.h"

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

#define GET_TYPEDEF_CLASSES
#include "SIMDTypes.cpp.inc"

#define GET_OP_CLASSES
#include "SIMDOps.cpp.inc"

#include "SIMDDialect.cpp.inc"

void SIMDDialect::initialize() {
    // llvm::outs() << "=== SIMDDialect::initialize() running ===\n";

    addTypes<
#define GET_TYPEDEF_LIST
#include "SIMDTypes.cpp.inc"
        >();

    addOperations<
#define GET_OP_LIST
#include "SIMDOps.cpp.inc"
        >();
}
