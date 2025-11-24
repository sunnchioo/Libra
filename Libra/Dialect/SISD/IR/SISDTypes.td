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
  let summary = "Ciphertext type with element type, plaintext count and scale";
  let description = [{
    Represents a TFHE/LWE ciphertext.

    Syntax:
      !sisd.sisdcipher<N x elementType, scale>

    Example:
      !sisd.sisdcipher<8xi64, 1> // 8 LWE ciphertexts, Scale 1 (Clean)
  }];

  let parameters = (ins
    "int64_t":$plaintextCount,
    "Type":$elementType,
    "int64_t":$scale // Usually 1, can be used to track noise level conceptually
  );

  let assemblyFormat = "`<` $plaintextCount `x` $elementType `,` $scale `>`";
}

#endif // IR_SISDTYPES_TD_