#ifndef FLYHEOPS_TD_
#define FLYHEOPS_TD_

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "FlyHETypes.td"
include "FlyHEDialect.td"

class FlyHE_Op<string mnemonic, list<Trait> traits = []> :
        Op<FlyHE_Dialect, mnemonic, traits>;

def SIMDADDOp : FlyHE_Op<"smidadd", [Pure, SameOperandsAndResultType]> {
    let summary = "Define an add operation.";
    let description = [{
        Define an add operation.
    }];

    let arguments = (ins 
        UI64:$input0,
        UI64:$input1
    );
    let results = (outs UI64:$res);

    let assemblyFormat = [{
        $input0 `+` $input1 attr-dict `:` type($input0)
    }];
}

#endif // FLYHEOPS_TD_
