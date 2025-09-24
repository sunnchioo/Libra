#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "FlyHETypes.h"
#include "FlyHEDialect.h"

using namespace mlir;
using namespace mlir::flyhe;

#define GET_TYPEDEF_CLASSES
#include "FlyHETypes.cpp.inc"
#define GET_OP_CLASSES
#include "FlyHEOps.cpp.inc"

#include "FlyHEDialect.cpp.inc"

void FlyHEDialect::initialize() {
    llvm::outs() << "=== FlyHEDialect::initialize() running ===\n";

    addTypes<
#define GET_TYPEDEF_LIST
#include "FlyHETypes.cpp.inc"
        >();

    addOperations<
#define GET_OP_LIST
#include "FlyHEOps.cpp.inc"
        >();
}
