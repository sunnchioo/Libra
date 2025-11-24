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
  let summary = "Ciphertext type with level, element type, scale state and basis size";
  let description = [{
    Represents a CKKS ciphertext encoding.
    
    Parameters:
    - Level: Current RNS level (0..29).
    - PlaintextCount: Number of packed slots.
    - ElementType: Underlying data type (e.g., i64, f64).
    - Scale: 1 = Clean (Delta), 2 = Dirty (Delta^2).
    - Basis: 2 = Standard (2 polys), 3 = Extended (3 polys after mult).

    Syntax:
      !simd.simdcipher<level x N x elementType, scale, basis>

    Example:
      !simd.simdcipher<12x8xi64, 1, 2>  // Level 12, Clean, Standard
      !simd.simdcipher<11x8xi64, 2, 3>  // Level 11, Dirty, Extended
  }];

  let parameters = (ins
    "int64_t":$level,
    "int64_t":$plaintextCount,
    "Type":$elementType,
    "int64_t":$scale, // 1 or 2
    "int64_t":$basis  // 2 or 3
  );

  // 使用逗号分隔状态参数，使其更易读
  let assemblyFormat = "`<` $level `x` $plaintextCount `x` $elementType `,` $scale `,` $basis `>`";
}

#endif // IR_SIMDTYPES_TD_