#ifndef IR_SIMDTYPES_TD_
#define IR_SIMDTYPES_TD_

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

include "SIMDDialect.td"

class SIMD_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<SIMD_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def SIMD_SIMDCipherType  : SIMD_Type<"SIMDCipher", "simdcipher"> {
  let summary = "An encrypted SIMD Ciphertext type.";
  let description = "SIMDCipher type in SIMD dialect.";
  let parameters = (ins);
  let assemblyFormat = "";
}

#endif // IR_SIMDTYPES_TD_
