#ifndef LIB_DIALECT_FlyHE_IR_TFHERUSTTYPES_TD_
#define LIB_DIALECT_FlyHE_IR_TFHERUSTTYPES_TD_

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

include "FlyHEDialect.td"


class FlyHE_Type<string name, list<Trait> traits = []>
    : TypeDef<FlyHE_Dialect, name, traits> { 
  let mnemonic = typeMnemonic;
}

def PassByReference : NativeTypeTrait<"PassByReference"> {
  let cppNamespace = "::mlir::flyhe";
}

def EncryptedInteger : NativeTypeTrait<"EncryptedInteger"> {
  let cppNamespace = "::mlir::flyhe";
}

class FlyHE_SIMDCipher
  : FlyHE_Type<"EncryptedUInt", [PassByReference, EncryptedInteger, MemRefElementTypeInterface]> {
  let summary = "An encrypted unsigned integer corresponding to tfhe-rs's FHEUint" # width # " type";
}

def FlyHE_SIMDCipher  : FlyHE_SIMDCipher;

def FlyHE_CiphertextType :
  AnyTypeOf<[
    FlyHE_SIMDCipher,
  ]>;


def FlyHE_CiphertextLikeType : TypeOrValueSemanticsContainer<FlyHE_CiphertextType, "tfhe-ciphertext-like">;

#endif  // LIB_DIALECT_FlyHE_IR_TFHERUSTTYPES_TD_
