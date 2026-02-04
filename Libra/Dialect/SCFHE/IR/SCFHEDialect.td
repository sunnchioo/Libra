#ifndef IR_SCFHE_DIALECT_TD_
#define IR_SCFHE_DIALECT_TD_

include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"

def SCFHE_Dialect : Dialect {
    let name = "scfhe";

    let description = [{
        SCFHE Dialect.
    }];

    let cppNamespace = "::mlir::libra::scfhe";

    let useDefaultTypePrinterParser = 1;
}

#endif // IR_SCFHE_DIALECT_TD_
