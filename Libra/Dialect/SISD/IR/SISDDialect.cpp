#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "SISDTypes.h"
#include "SISDDialect.h"

using namespace mlir;
using namespace mlir::libra::sisd;

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
