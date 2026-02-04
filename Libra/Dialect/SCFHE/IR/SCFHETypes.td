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

def SCFHE_SCFHECipherType : SCFHE_Type<"SCFHECipher", "scfhecipher"> {
  let summary = "Ciphertext type with element type and plaintext count";
  let description = [{
    Represents a ciphertext encoding a fixed number of plaintexts.

    Syntax:
      !scfhe.scfhecipher<Nxetype>

    Example:
      !scfhe.scfhecipher<8xi64>
      !scfhe.scfhecipher<1xf64>
  }];

  let parameters = (ins
    "int64_t":$plaintextCount,
    "Type":$elementType
  );

  let assemblyFormat = "`<`$plaintextCount`x`$elementType`>`";
}



#endif // IR_SCFHETYPES_TD_
