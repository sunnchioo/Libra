#ifndef IR_FlyHETYPES_TD_
#define IR_FlyHETYPES_TD_

// include "mlir/IR/AttrTypeBase.td"
// include "mlir/IR/BuiltinTypeInterfaces.td"
// include "mlir/IR/CommonTypeConstraints.td"
// include "mlir/IR/DialectBase.td"
// include "mlir/IR/Traits.td"
// include "mlir/IR/OpBase.td"
// include "mlir/Interfaces/InferTypeOpInterface.td"

include "mlir/IR/DialectBase.td"
include "mlir/IR/Traits.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

include "FlyHEDialect.td"

class FlyHE_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<FlyHE_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def FlyHE_SIMDCipherType  : FlyHE_Type<"SIMDCipher", "simdcipher"> {
  let summary = "An encrypted FlyHE Ciphertext type.";
  let description = "SIMDCipher type in FlyHE dialect.";
  let parameters = (ins);
  let assemblyFormat = "";
}

#endif // IR_FlyHETYPES_TD_
