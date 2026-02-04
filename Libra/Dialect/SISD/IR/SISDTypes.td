#ifndef IR_SISDTYPES_TD_
#define IR_SISDTYPES_TD_

include "mlir/IR/DialectBase.td"
include "mlir/IR/Traits.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

include "SISDDialect.td"

class SISD_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<SISD_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def SISD_SISDCipherType : SISD_Type<"SISDCipher", "sisdcipher"> {
  let summary = "Ciphertext type with element type and plaintext count";
  let description = [{
    Represents a ciphertext encoding a fixed number of plaintexts.

    Syntax:
      !sisd.sisdcipher<Nxetype>

    Example:
      !sisd.sisdcipher<8xi64>
      !sisd.sisdcipher<1xf64>
  }];

  let parameters = (ins
    "int64_t":$plaintextCount,
    "Type":$elementType
  );

  let assemblyFormat = "`<`$plaintextCount`x`$elementType`>`";
}

#endif // IR_SISDTYPES_TD_