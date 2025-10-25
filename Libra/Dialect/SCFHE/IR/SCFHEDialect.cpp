#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "SCFHETypes.h"
#include "SCFHEDialect.h"

using namespace mlir;
using namespace mlir::libra::scfhe;

#define GET_TYPEDEF_CLASSES
#include "SCFHETypes.cpp.inc"
#define GET_OP_CLASSES
#include "SCFHEOps.cpp.inc"

#include "SCFHEDialect.cpp.inc"

void SCFHEDialect::initialize() {
    // llvm::outs() << "=== SCFHEDialect::initialize() running ===\n";

    addTypes<
#define GET_TYPEDEF_LIST
#include "SCFHETypes.cpp.inc"
        >();

    addOperations<
#define GET_OP_LIST
#include "SCFHEOps.cpp.inc"
        >();
}
