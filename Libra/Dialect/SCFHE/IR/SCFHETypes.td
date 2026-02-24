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

def SCFHE_SCFHECipherType : SCFHE_Type<"SCFHECipher", "scfhecipher", [MemRefElementTypeInterface]> {
  let summary = "Ciphertext type with element type and plaintext count";
  let description = [{
    Represents a ciphertext encoding a fixed number of plaintexts.
    Syntax:
      !scfhe.scfhecipher<NxType>
      !scfhe.scfhecipher<?xType>
  }];

  let parameters = (ins
    "int64_t":$plaintextCount,
    "Type":$elementType
  );

  let assemblyFormat = "`<` custom<CipherDim>($plaintextCount) `x` $elementType `>`";

  let builders = [
    TypeBuilder<(ins "Type":$elementType), [{
      return Base::get(context, -1, elementType); 
    }]>
  ];

  let extraClassDeclaration = [{
    static constexpr int64_t kDynamicSize = -1;
    bool isDynamic() const { return getPlaintextCount() == kDynamicSize; }
    bool hasStaticShape() const { return !isDynamic(); }
  }];
}

#endif // IR_SCFHETYPES_TD_