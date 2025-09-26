#ifndef IR_SIMD_DIALECT_TD_
#define IR_SIMD_DIALECT_TD_

include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"

def SIMD_Dialect : Dialect {
    let name = "simd";

    let description = [{
        SIMD Dialect.
    }];

    let cppNamespace = "::mlir::simd";

    let useDefaultTypePrinterParser = 1;
}

#endif // IR_SIMD_DIALECT_TD_
