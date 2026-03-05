#ifndef SIMDOPS_TD_
#define SIMDOPS_TD_

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "SIMDTypes.td"
include "SIMDDialect.td"

class SIMD_Op<string mnemonic, list<Trait> traits = []> : Op<SIMD_Dialect, mnemonic, traits> {
    let summary = "SIMD dialect operations base class";
}

// ============================================================================
// Group 1: Cryptographic I/O & Memory Management
// ============================================================================

def SIMDEncryptOp : SIMD_Op<"encrypt", [Pure]> {
    let summary = "Encrypt a plaintext vector into a SIMD cipher.";
    let arguments = (ins AnyType:$input);
    let results = (outs AnyType:$cipher);
    let assemblyFormat = "$input attr-dict `:` type($input) `->` type($cipher)";
}

def SIMDDecryptOp : SIMD_Op<"decrypt", [Pure]> {
    let summary = "Decrypt a SIMD ciphertext back into a plaintext vector.";
    let arguments = (ins AnyType:$cipher);
    let results = (outs AnyType:$output);
    let assemblyFormat = "$cipher attr-dict `:` type($cipher) `->` type($output)";
}

def SIMDLoadOp : SIMD_Op<"load", [Pure]> {
    let summary = "Extract a scalar from a SIMD array at a specific index.";
    let arguments = (ins AnyType:$array, Index:$index);
    let results = (outs AnyType:$result);
    let assemblyFormat = "$array `[` $index `]` attr-dict `:` type($array) `->` type($result)";
}

def SIMDStoreOp : SIMD_Op<"store"> {
    let summary = "Store a SIMD cipher value into a destination.";
    let arguments = (ins AnyType:$value, AnyType:$dest);
    let assemblyFormat = "$value `,` $dest attr-dict `:` type($value) `,` type($dest)";
}

// ============================================================================
// Group 2: Linear & Non-Linear Arithmetic
// ============================================================================

// 密文-密文加法 (Add levels if mismatched)
def SIMDAddOp : SIMD_Op<"add", [Pure]> {
    let summary = "Ciphertext-Ciphertext SIMD addition.";
    let arguments = (ins AnyType:$lhs, AnyType:$rhs);
    let results = (outs AnyType:$res);
    let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($res)";
}

def SIMDAddPlainOp : SIMD_Op<"add_plain", [Pure]> {
    let summary = "Ciphertext-Plaintext SIMD addition.";
    let arguments = (ins AnyType:$cipher, AnyType:$plain);
    let results = (outs AnyType:$res);
    let assemblyFormat = "$cipher `,` $plain attr-dict `:` type($cipher) `,` type($plain) `->` type($res)";
}

def SIMDSubOp : SIMD_Op<"sub", [Pure]> {
    let summary = "Ciphertext-Ciphertext SIMD subtraction.";
    let arguments = (ins AnyType:$lhs, AnyType:$rhs);
    let results = (outs AnyType:$res);
    let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($res)";
}

def SIMDSublainOp : SIMD_Op<"sub_plain", [Pure]> {
    let summary = "Ciphertext-Plaintext SIMD subtraction.";
    let arguments = (ins AnyType:$cipher, AnyType:$plain);
    let results = (outs AnyType:$res);
    let assemblyFormat = "$cipher `,` $plain attr-dict `:` type($cipher) `,` type($plain) `->` type($res)";
}

// 密文-密文乘法 (产生 dirty scale，需要后续 Relin + Rescale)
def SIMDMultOp : SIMD_Op<"mult", [Pure]> {
    let summary = "Ciphertext-Ciphertext SIMD multiplication.";
    let arguments = (ins AnyType:$lhs, AnyType:$rhs);
    let results = (outs AnyType:$res);
    let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($res)";
}

// 密文-明文乘法 (比密文乘法快得多，不需要 Relinearize)
def SIMDMultPlainOp : SIMD_Op<"mult_plain", [Pure]> {
    let summary = "Ciphertext-Plaintext SIMD multiplication.";
    let arguments = (ins AnyType:$cipher, AnyType:$plain);
    let results = (outs AnyType:$res);
    let assemblyFormat = "$cipher `,` $plain attr-dict `:` type($cipher) `,` type($plain) `->` type($res)";
}

// ============================================================================
// Group 3: SIMD Specific Operations (Rotations)
// ============================================================================

def SIMDRotateOp : SIMD_Op<"rotate", [Pure]> {
    let summary = "Cyclically rotate the slots of a SIMD ciphertext.";
    let description = [{
        Moves elements across slots. Does not consume levels, 
        but is computationally expensive due to key-switching.
    }];
    let arguments = (ins AnyType:$input, I64Attr:$offset);
    let results = (outs AnyType:$res);
    let assemblyFormat = "$input `,` $offset attr-dict `:` type($input) `->` type($res)";
}

def SIMDReduceAddOp : SIMD_Op<"reduce_add", [Pure]> {
    let summary = "Sum all slots within one ciphertext (using log2(N) rotations).";
    let arguments = (ins AnyType:$input);
    let results = (outs AnyType:$res);
    let assemblyFormat = "$input attr-dict `:` type($input) `->` type($res)";
}

// ============================================================================
// Group 4: CKKS Management (Crucial for Level/Scale Tracking)
// ============================================================================

def SIMDRelinOp : SIMD_Op<"relinearize", [Pure]> {
    let summary = "Reduce ciphertext size from 3 to 2 polynomials after mult.";
    let arguments = (ins AnyType:$input);
    let results = (outs AnyType:$res);
    let assemblyFormat = "$input attr-dict `:` type($input) `->` type($res)";
}

def SIMDRescaleOp : SIMD_Op<"rescale", [Pure]> {
    let summary = "Divide by modulus to manage scale and consume one level.";
    let arguments = (ins AnyType:$input);
    let results = (outs AnyType:$res);
    let assemblyFormat = "$input attr-dict `:` type($input) `->` type($res)";
}

def SIMDModSwitchOp : SIMD_Op<"mod_switch", [Pure]> {
    let summary = "Drop a level without rescaling (for level alignment).";
    let arguments = (ins AnyType:$input);
    let results = (outs AnyType:$res);
    let assemblyFormat = "$input attr-dict `:` type($input) `->` type($res)";
}

def SIMDBootOp : SIMD_Op<"boot", [Pure]> {
    let summary = "CKKS Bootstrapping to refresh levels.";
    let arguments = (ins AnyType:$input);
    let results = (outs AnyType:$res);
    let assemblyFormat = "$input attr-dict `:` type($input) `->` type($res)";
}

// ============================================================================
// Group 5: Polynomial Approximation (Non-linear Function Approximation)
// ============================================================================
// Note: These ops are NOT native to CKKS. They will be lowered into a DAG 
// of additions and multiplications representing a polynomial (e.g., Chebyshev).

def SIMDSelectOp : SIMD_Op<"select", [Pure]> {
    let summary = "SIMD Homomorphic select (mux) operation";
    let description = [{
        Selects values between true_value and false_value based on the condition bit.
        Result = condition ? true_value : false_value.
        
        The condition is typically a ciphertext resulting from a comparison (0 or 1).
    }];
    // 根据错误信息推断，它接受三个操作数：条件, 真值, 假值
    let arguments = (ins AnyType:$condition, AnyType:$true_value, AnyType:$false_value);
    let results = (outs AnyType:$result);
    
    // 定义汇编格式
    let assemblyFormat = [{
        $condition `,` $true_value `,` $false_value attr-dict 
        `:` type($condition) `,` type($true_value) `,` type($false_value) `->` type($result)
    }];
}

def SIMDCmpOp : SIMD_Op<"cmp", [Pure]> {
    let summary = "Compare two ciphertexts using polynomial approximation.";
    let description = [{
        Approximates the signum function. 
        'degree' represents the polynomial degree. If -1, the compiler 
        chooses a degree based on the current noise budget/precision heuristic.
    }];

    let arguments = (ins 
        AnyType:$lhs, 
        AnyType:$rhs, 
        I64Attr:$predicate,
        DefaultValuedAttr<I64Attr, "-1">:$degree
    ); 

    let results = (outs AnyType:$result);
    let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def SIMDMinOp : SIMD_Op<"min", [Pure]> {
    let summary = "Find the minimum of two ciphertexts.";
    let arguments = (ins 
        AnyType:$lhs, 
        AnyType:$rhs,
        DefaultValuedAttr<I64Attr, "-1">:$degree
    );
    let results = (outs AnyType:$result);
    let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def SIMDMaxOp : SIMD_Op<"max", [Pure]> {
    let summary = "Find the maximum of two ciphertexts.";
    let arguments = (ins 
        AnyType:$lhs, 
        AnyType:$rhs,
        DefaultValuedAttr<I64Attr, "-1">:$degree
    );
    let results = (outs AnyType:$result);
    let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def SIMDExpOp : SIMD_Op<"exp", [Pure]> {
    let summary = "Calculate exponential (e^x) of a ciphertext.";
    let arguments = (ins 
        AnyType:$input,
        DefaultValuedAttr<I64Attr, "-1">:$degree
    );
    let results = (outs AnyType:$result);
    let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
}

def SIMDInvOp : SIMD_Op<"inv", [Pure]> {
    let summary = "Calculate the reciprocal (1/x) of a ciphertext.";
    let description = [{
        Approximates the inverse of the input. 
        In CKKS, this is typically implemented using:
        1. Newton-Raphson iterations: x_{n+1} = x_n * (2 - input * x_n)
        2. Or a direct Chebyshev polynomial over a specific range [a, b].
        
        'degree' here can represent the number of iterations or the polynomial degree.
    }];

    let arguments = (ins 
        AnyType:$input,
        DefaultValuedAttr<I64Attr, "-1">:$degree
    );
    let results = (outs AnyType:$result);

    let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
}

def SIMDDivOp : SIMD_Op<"div", [Pure]> {
    let summary = "Divide two ciphertexts (lhs / rhs).";
    let description = [{
        Approximates division. Most backends will lower this to:
        %inv_rhs = simd.inv %rhs
        %res = simd.mult %lhs, %inv_rhs
        
        'degree' controls the precision of the underlying reciprocal approximation.
    }];

    let arguments = (ins 
        AnyType:$lhs, 
        AnyType:$rhs,
        DefaultValuedAttr<I64Attr, "-1">:$degree
    );
    let results = (outs AnyType:$result);

    let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

// ============================================================================
// Group 6: Cipher Conversion
// ============================================================================

def SIMDCastSIMDCipherToSISDCipherOp : SIMD_Op<"cast_to_sisd", [Pure]> {
    let summary = "Cast SIMD ciphertext to SISD (drop batch dimension).";
    let arguments = (ins AnyType:$input);
    let results = (outs AnyType:$result);
    let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";

    let hasCanonicalizer = 1;
}

#endif // SIMDOPS_TD_