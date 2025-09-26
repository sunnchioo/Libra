#ifndef SISDOPS_TD_
#define SISDOPS_TD_

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "SISDTypes.td"
include "SISDDialect.td"

class SISD_Op<string mnemonic, list<Trait> traits = []> :
        Op<SISD_Dialect, mnemonic, traits> {
    let summary = "SISD dialect operations base class";
}

def SISDAddOp : SISD_Op<"sisdadd", [Pure, SameOperandsAndResultType]> {
    let summary = "Define an add operation.";
    let description = [{
        Define an add operation.
    }];

    let arguments = (ins 
        SISD_SISDCipherType:$input0,
        SISD_SISDCipherType:$input1
    );
    let results = (outs SISD_SISDCipherType:$res);

    let assemblyFormat = [{
        $input0 `+` $input1 attr-dict `:` type($input0)
    }];
}

// def SISDLoadOp : SISD_Op<"sisd_load", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
//   let summary = "Load data from a SISDCipherType buffer";
//   let description = "Load a scalar value from a SISD cipher buffer (replaces affine.load for SISDCipherType)";

//   let arguments = (ins
//     SISD_SISDCipherType:$buffer,       // 内存 buffer
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

def SISDLoadOp : SISD_Op<"sisd_load"> {
  let summary = "Load a SISD cipher value";
  let arguments = (ins SISD_SISDCipherType:$src);
  let results   = (outs SISD_SISDCipherType:$result);
  let assemblyFormat = [{
    $src `:` type($src) attr-dict `:` type($result)
  }];

  // let extraClassDefinition = [{
  //   bool isSISDLoad() { return true; }
  // }];
}



#endif // SISDOPS_TD_
