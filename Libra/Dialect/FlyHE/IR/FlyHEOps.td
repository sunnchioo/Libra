#ifndef LIBRA_FLYHE_IR_FLYHEOPS_TD_
#define LIBRA_FLYHE_IR_FLYHEOPS_TD_

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "FlyHETypes.td"

def FlyHE_Dialect : Dialect {
  let name = "flyhe";
  let cppNamespace = "::mlir::flyhe";
}

// PhantomLoad操作定义
def PhantomLoadOp : Op<FlyHE_Dialect, "phantom_load",
    [Pure, HasOperandsInterface, HasResultsInterface]> {
  let summary = "Load from a FlyHE_SIMDCipher";
  let description = "Load a value from a FlyHE_SIMDCipher type";

  let arguments = (ins
    FlyHE_PhantomCipherType:$memref,
    Variadic<IndexType>:$indices
  );

  let results = (outs AnyType:$result);

  let assemblyFormat = "$memref `[` $indices `]` attr-dict `:` type($memref)";
}

#endif // LIBRA_FLYHE_IR_FLYHEOPS_TD_
