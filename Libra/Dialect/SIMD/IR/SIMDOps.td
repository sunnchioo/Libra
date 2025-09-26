#ifndef SIMDOPS_TD_
#define SIMDOPS_TD_

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "SIMDTypes.td"
include "SIMDDialect.td"

class SIMD_Op<string mnemonic, list<Trait> traits = []> :
        Op<SIMD_Dialect, mnemonic, traits> {
    let summary = "SIMD dialect operations base class";
}

def SIMDAddOp : SIMD_Op<"simdadd", [Pure, SameOperandsAndResultType]> {
    let summary = "Define an add operation.";
    let description = [{
        Define an add operation.
    }];

    let arguments = (ins 
        SIMD_SIMDCipherType:$input0,
        SIMD_SIMDCipherType:$input1
    );
    let results = (outs SIMD_SIMDCipherType:$res);

    let assemblyFormat = [{
        $input0 `+` $input1 attr-dict `:` type($input0)
    }];
}

// def SIMDLoadOp : SIMD_Op<"simd_load", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
//   let summary = "Load data from a SIMDCipherType buffer";
//   let description = "Load a scalar value from a SIMD cipher buffer (replaces affine.load for SIMDCipherType)";

//   let arguments = (ins
//     SIMD_SIMDCipherType:$buffer,       // 内存 buffer
//     Variadic<Index>:$indices            // 可选索引（保留与 affine.load 兼容）
//   );

//   let results = (outs F64:$result);     // 输出一个 F64

//   let assemblyFormat = "$buffer `[` $indices `]` attr-dict `:` type($buffer)";

//   let extraClassDefinition = [{
//     void getEffects(
//       mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<
//         mlir::MemoryEffects::Effect>> &effects);
//   }];
// }

def SIMDLoadOp : SIMD_Op<"simd_load"> {
  let summary = "Load a SIMD cipher value";
  let arguments = (ins SIMD_SIMDCipherType:$src);
  let results   = (outs SIMD_SIMDCipherType:$result);
  let assemblyFormat = [{
    $src `:` type($src) attr-dict `:` type($result)
  }];

  // let extraClassDefinition = [{
  //   bool isSIMDLoad() { return true; }
  // }];
}



#endif // SIMDOPS_TD_
