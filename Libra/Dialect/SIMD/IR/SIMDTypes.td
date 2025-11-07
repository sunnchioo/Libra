#ifndef IR_SIMDTYPES_TD_
#define IR_SIMDTYPES_TD_

include "mlir/IR/DialectBase.td"
include "mlir/IR/Traits.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

include "SIMDDialect.td"

class SIMD_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<SIMD_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def SIMD_SIMDCipherType : SIMD_Type<"SIMDCipher", "simdcipher"> {
  let summary = "Ciphertext type with level, element type and plaintext count";
  let description = [{
    Represents a ciphertext encoding a fixed number of plaintexts and a level.

    Syntax:
      !simd.simdcipher<levelxN x elementType>

    Example:
      !simd.simdcipher<12x8xi64>
      !simd.simdcipher<1x1xf64>
  }];

  let parameters = (ins
    "int64_t":$level,
    "int64_t":$plaintextCount,
    "Type":$elementType
  );

  let assemblyFormat = "`<`$level`x`$plaintextCount`x`$elementType`>`";
}


#endif // IR_SIMDTYPES_TD_
