#ifndef IR_SISDTYPES_TD_
#define IR_SISDTYPES_TD_

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

include "SISDDialect.td"

class SISD_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<SISD_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def SISD_SISDCipherType  : SISD_Type<"SISDCipher", "sisdcipher"> {
  let summary = "An encrypted SISD Ciphertext type.";
  let description = "SISDCipher type in SISD dialect.";
  let parameters = (ins);
  let assemblyFormat = "";
}

#endif // IR_SISDTYPES_TD_
