#ifndef IR_SISD_DIALECT_TD_
#define IR_SISD_DIALECT_TD_

include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"

def SISD_Dialect : Dialect {
    let name = "sisd";

    let description = [{
        SISD Dialect.
    }];

    let cppNamespace = "::mlir::sisd";

    let useDefaultTypePrinterParser = 1;
}

#endif // IR_SISD_DIALECT_TD_
