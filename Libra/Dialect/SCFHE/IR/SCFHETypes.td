#ifndef IR_SCFHETYPES_TD_
#define IR_SCFHETYPES_TD_

include "mlir/IR/DialectBase.td"
include "mlir/IR/Traits.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

include "SCFHEDialect.td"

class SCFHE_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<SCFHE_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def SCFHE_SCFHECipherType  : SCFHE_Type<"SCFHECipher", "scfhecipher"> {
  let summary = "An encrypted SCFHE Ciphertext type.";
  let description = "SCFHECipher type in SCFHE dialect.";
  let parameters = (ins);
  let assemblyFormat = "";
}

#endif // IR_SCFHETYPES_TD_
