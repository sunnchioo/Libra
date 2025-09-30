#ifndef FLYHEOPS_TD_
#define FLYHEOPS_TD_

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "FlyHETypes.td"
include "FlyHEDialect.td"

class FlyHE_Op<string mnemonic, list<Trait> traits = []> :
        Op<FlyHE_Dialect, mnemonic, traits> {
    let summary = "FlyHE dialect operations base class";
}

def SIMDAddOp : FlyHE_Op<"simdadd", [Pure, SameOperandsAndResultType]> {
    let summary = "Define an add operation.";
    let description = [{
        Define an add operation.
    }];

    let arguments = (ins 
        FlyHE_SIMDCipherType:$input0,
        FlyHE_SIMDCipherType:$input1
    );
    let results = (outs FlyHE_SIMDCipherType:$res);

    let assemblyFormat = [{
        $input0 `+` $input1 attr-dict `:` type($input0)
    }];
}

def SIMDMultOp : FlyHE_Op<"simdmult", [Pure, SameOperandsAndResultType]> {
  let summary = "Define a multiply operation.";
  let description = [{
    Define a multiply operation between two SIMD cipher values.
  }];

  let arguments = (ins 
    FlyHE_SIMDCipherType:$input0,
    FlyHE_SIMDCipherType:$input1
  );
  let results = (outs FlyHE_SIMDCipherType:$result);

  let assemblyFormat = [{
    $input0 `*` $input1 attr-dict `:` type($input0)
  }];
}

def SIMDLoadOp : FlyHE_Op<"simd_load"> {
  let summary = "Load a SIMD cipher value";
  let arguments = (ins AnyType:$src);
  let results   = (outs FlyHE_SIMDCipherType:$result);
  let assemblyFormat = [{
    $src `:` type($src) attr-dict `:` type($result)
  }];
}

def SIMDStoreOp : FlyHE_Op<"simd_store"> {
  let summary = "Store a SIMD cipher value into a SIMD slot (or memref placeholder)";
  let description = [{
    Stores a value of type `!flyhe.simd_cipher` into a destination which may be
    a memref (before type conversion) or a !flyhe.simd_cipher (after).
  }];

  let arguments = (ins
    FlyHE_SIMDCipherType:$value,
    AnyType:$dest
  );

  // ensure the parser can infer the type of $dest
  let assemblyFormat = "$value `,` $dest `:` type($dest) attr-dict";
}



#endif // FLYHEOPS_TD_
