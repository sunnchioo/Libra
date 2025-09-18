#ifndef DIALECT_FLYHE
#define DIALECT_FLYHE

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

// PhantomLoad操作定义
// def PhantomLoadOp : Op<FlyHE_Dialect, "phantom_load",
//     [Pure, HasOperandsInterface, HasResultsInterface]> {
//   let summary = "Load from a FlyHE_SIMDCipher";
//   let description = "Load a value from a FlyHE_SIMDCipher type";

//   let arguments = (ins
//     FlyHE_PhantomCipherType:$memref,
//     Variadic<IndexType>:$indices
//   );

//   let results = (outs AnyType:$result);

//   let assemblyFormat = "$memref `[` $indices `]` attr-dict `:` type($memref)";
// }

#endif // DIALECT_FLYHE
