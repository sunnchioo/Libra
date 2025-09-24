#ifndef IR_FlyHE_DIALECT_TD_
#define IR_FlyHE_DIALECT_TD_

include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"

def FlyHE_Dialect : Dialect {
    let name = "flyhe";

    let description = [{
        FlyHE Dialect.
    }];

    let cppNamespace = "::mlir::flyhe";

    let useDefaultTypePrinterParser = 1;
}

#endif // IR_FlyHE_DIALECT_TD_
