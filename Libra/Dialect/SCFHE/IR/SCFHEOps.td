#ifndef SCFHEOPS_TD_
#define SCFHEOPS_TD_

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "SCFHETypes.td"
include "SCFHEDialect.td"

// === Base Class Definition ===

class SCFHE_Op<string mnemonic, list<Trait> traits = []> : Op<SCFHE_Dialect, mnemonic, traits> {
    let summary = "SCFHE dialect operations base class";
}

// =============================================================================
// 1. Arithmetic Operations (基础算术运算)
// =============================================================================

def SCFHEAddOp : SCFHE_Op<"add", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
    let summary = "Define an add operation.";
    let arguments = (ins AnyType:$input0, AnyType:$input1);
    let results = (outs AnyType:$res);
    let assemblyFormat = [{
        $input0 `+` $input1 attr-dict `:` type($input0) `,` type($input1)
    }];
    let hasCanonicalizer = 1;
}

def SCFHESubOp : SCFHE_Op<"sub", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
    let summary = "Define an sub operation.";
    let arguments = (ins AnyType:$input0, AnyType:$input1);
    let results = (outs AnyType:$res);
    let assemblyFormat = [{
        $input0 `_` $input1 attr-dict `:` type($input0) `,` type($input1)
    }];
    let hasCanonicalizer = 1;
}

def SCFHEMultOp : SCFHE_Op<"mult", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
    let summary = "Define a multiply operation.";
    let arguments = (ins AnyType:$input0, AnyType:$input1);
    let results = (outs AnyType:$result);
    let assemblyFormat = [{
        $input0 `*` $input1 attr-dict `:` type($input0) `,` type($input1)
    }];
    let hasCanonicalizer = 1;
}

def SCFHEDivOp : SCFHE_Op<"div", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
    let summary = "Define a division operation.";
    let description = [{ 
        Define a division operation between two SCFHE cipher values.
        Typically used for computing averages (Mean = Sum / Count).
    }];
    let arguments = (ins AnyType:$lhs, AnyType:$rhs);
    let results = (outs AnyType:$res);
    let assemblyFormat = [{
        $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs)
    }];
    let hasCanonicalizer = 1;
}

def SCFHEExpOp : SCFHE_Op<"exp", [Pure]> {
    let summary = "Homomorphic Exponential operation";
    let arguments = (ins AnyType:$input);
    let results = (outs AnyType:$output);
    let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
}

// =============================================================================
// 2. Comparison & Selection Operations (比较与选择运算)
// =============================================================================

def SCFHECmpOp : SCFHE_Op<"cmp", [Pure]> {
    let summary = "Homomorphic comparison operation";
    let arguments = (ins AnyType:$lhs, AnyType:$rhs, I64Attr:$predicate);
    let results = (outs AnyType:$result);
    let assemblyFormat = [{ $lhs `,` $rhs `,` $predicate attr-dict `:` type($lhs) `,` type($rhs) `->` type($result) }];
}

def SCFHESelectOp : SCFHE_Op<"select", [Pure]> {
    let summary = "Homomorphic select operation (mux)";
    let arguments = (ins AnyType:$condition, AnyType:$true_value, AnyType:$false_value);
    let results = (outs AnyType:$result);
    let assemblyFormat = [{ $condition `,` $true_value `,` $false_value attr-dict `:` type($condition) `,` type($true_value) `,` type($false_value) `->` type($result) }];
}

def SCFHEThresholdCountOp : SCFHE_Op<"threshold_count", [Pure]> {
    let summary = "Count elements in a vector greater/less than a threshold";
    let arguments = (ins AnyType:$input, AnyType:$threshold, I64Attr:$predicate);
    let results = (outs AnyType:$result);
    let assemblyFormat = [{ $input `,` $threshold `,` $predicate attr-dict `:` type($input) `,` type($threshold) `->` type($result) }];
}

// =============================================================================
// 3. Reduction & Aggregation Operations (归约与聚合运算)
// =============================================================================

def SCFHEMinOp : SCFHE_Op<"min", [Pure]> {
    let summary = "Finds the minimum value in a ciphertext vector";
    let arguments = (ins AnyType:$operand);
    let results = (outs AnyType:$result);
    let assemblyFormat = "$operand attr-dict `:` type($operand) `->` type($result)";
}

def SCFHEArgMinOp : SCFHE_Op<"argmin", [Pure]> {
    let summary = "Computes the minimum value and its index from a ciphertext";
    let arguments = (ins AnyType:$operand);
    let results = (outs AnyType:$min_val, AnyType:$min_idx);
    let assemblyFormat = "$operand attr-dict `:` type($operand) `->` type($min_val) `,` type($min_idx)";
}

def SCFHEReduceAddOp : SCFHE_Op<"reduce_add", [Pure]> {
    let summary = "Reduces a cipher vector to a single scalar cipher via addition";
    let arguments = (ins AnyType:$input);
    let results = (outs AnyType:$result);
    let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
    let hasCanonicalizer = 1;
}

// =============================================================================
// 4. Memory & Type Operations (内存与类型操作)
// =============================================================================

def SCFHEAllocaOp : SCFHE_Op<"alloca"> {
    let summary = "Allocate encrypted memory for a single SCFHE cipher";
    let arguments = (ins);
    let results = (outs AnyType:$result);
    let assemblyFormat = "`:` type($result) attr-dict";
}

def SCFHELoadOp : SCFHE_Op<"load", [Pure]> {
    let summary = "Extract a scalar cipher from a cipher array at a specific index";
    let arguments = (ins SCFHE_SCFHECipherType:$array, Index:$index);
    let results = (outs SCFHE_SCFHECipherType:$result);
    let assemblyFormat = "$array `[` $index `]` attr-dict `:` type($array) `->` type($result)";
}

def SCFHEStoreOp : SCFHE_Op<"store"> {
    let summary = "Store a SCFHE cipher value";
    let arguments = (ins AnyType:$value, AnyType:$dest);
    let assemblyFormat = "$value `,` $dest `:` type($value) `,` type($dest) attr-dict";
}

def SCFHECastOp : SCFHE_Op<"cast", [Pure]> {
    let summary = "Cast between static and dynamic SCFHE cipher types";
    let arguments = (ins SCFHE_SCFHECipherType:$source);
    let results = (outs SCFHE_SCFHECipherType:$dest);
    let assemblyFormat = "$source attr-dict `:` type($source) `to` type($dest)";
    // let hasCanonicalizer = 1;
}

// =============================================================================
// 5. Cryptography Operations (加解密操作)
// =============================================================================

def SCFHEEncryptOp : SCFHE_Op<"encrypt", [Pure]> {
    let summary = "Encrypt a plaintext floating-point value into a SCFHE cipher.";
    let arguments = (ins AnyType:$input);
    let results = (outs AnyType:$cipher);
    let assemblyFormat = [{ $input attr-dict `:` type($input) `->` type($cipher) }];
    let hasCanonicalizer = 1; 
}

def SCFHEDecryptOp : SCFHE_Op<"decrypt", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
    let summary = "Decrypt a SCFHE ciphertext back into a plaintext value or vector.";
    let arguments = (ins AnyType:$cipher);
    let results = (outs AnyType:$output);
    let assemblyFormat = [{ $cipher attr-dict `:` type($cipher) `:` type($output) }];
    let hasCanonicalizer = 1;
}

// =============================================================================
// 6. Debugging Operations (调试操作)
// =============================================================================

def SCFHEPrintOp : SCFHE_Op<"print", [Pure]> {
    let summary = "Print one or more SCFHE ciphertexts (for debug or output).";
    let arguments = (ins Variadic<SCFHE_SCFHECipherType>:$inputs);
    let results = (outs);
    let assemblyFormat = [{ $inputs attr-dict `:` type($inputs) }];
}

#endif  // SCFHEOPS_TD_