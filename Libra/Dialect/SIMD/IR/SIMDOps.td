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

def SIMDAddOp : SIMD_Op<"add", [Pure]> {
    let summary = "Define an add operation.";
    let arguments = (ins AnyType:$input0, AnyType:$input1);
    let results = (outs AnyType:$res);
    // 允许不同类型
    let assemblyFormat = [{
        $input0 `+` $input1 attr-dict `:` type($input0) `,` type($input1) `->` type($res)
    }];
}

def SIMDSubOp : SIMD_Op<"sub", [Pure]> {
    let summary = "Define a sub operation.";
    let arguments = (ins AnyType:$input0, AnyType:$input1);
    let results = (outs AnyType:$res);
    // 允许不同类型
    let assemblyFormat = [{
        $input0 `_` $input1 attr-dict `:` type($input0) `,` type($input1) `->` type($res)
    }];
}

def SIMDMultOp : SIMD_Op<"mult", [Pure]> {
    let summary = "Define a multiply operation.";
    let description = [{
        Define a multiply operation between two SIMD cipher values.
    }];

    let arguments = (ins 
        AnyType:$input0,
        AnyType:$input1);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $input0 `*` $input1 attr-dict `:` type($input0) `,` type($input1) `->` type($result)
    }];
}

def SIMDMinOp : SIMD_Op<"min", [Pure]> {
  let summary = "Finds the minimum value in a ciphertext vector, returning a scalar ciphertext.";
  
  let arguments = (ins AnyType:$operand);
  let results = (outs AnyType:$result);

  let assemblyFormat = "$operand attr-dict `:` type($operand) `->` type($result)";
}

def SIMDCastSIMDCipherToSISDCipherOp : SIMD_Op<"cast_to_sisd", [Pure]> {
  let summary = "Cast a !simd.simdcipher<NxMxT> to a !sisd.sisdcipher<MxT> by dropping the leading SIMD dimension";

  let description = [{
    This operation performs a logical reinterpretation between SIMD and SISD dialect
    ciphertext representations, allowing the removal of one or more leading shape dimensions
    while preserving the underlying element type.

    Example:
      %12 = simd.cast_to_sisd %11 :
        !simd.simdcipher<12 x 8 x i64> -> !sisd.sisdcipher<8 x i64>
  }];

  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);

  // Assembly matches the given example format exactly
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";

  let hasCanonicalizer = 1;
}

def SIMDBootOp : SIMD_Op<"boot", [Pure]> {
    let summary = "Performs FHE bootstrapping to refresh the noise budget to a specific level.";
    
    let description = [{
        This operation takes a ciphertext and returns a new ciphertext with a refreshed 
        noise budget (level).
        
        Unlike traditional bootstrapping which might always restore to the maximum level,
        this operation's output level is determined by the compiler (e.g., via dynamic 
        programming) to meet the minimum level requirements of downstream consumers 
        while minimizing total latency.

        Example:
          // Bootstrapping to Level 15 because downstream only needs Level 15
          %new = simd.boot %old : !simd.simdcipher<0 x 8 x i64> -> !simd.simdcipher<15 x 8 x i64>
    }];

    let arguments = (ins AnyType:$input);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $input attr-dict `:` type($input) `->` type($result)
    }];
}

def SIMDRescaleOp : SIMD_Op<"rescale", [Pure]> {
  let summary = "Rescale a ciphertext, consuming one level to reduce the scaling factor.";
  let description = [{
    Operation that divides the ciphertext message by the last modulus in the RNS chain.
    This is used to reduce the 'Scale' from Delta^2 (Dirty) back to Delta (Clean).
    
    Side Effect: Reduces the level by 1.
    Input: !simd.simdcipher<L x Dirty x T>
    Output: !simd.simdcipher<L-1 x Clean x T>
  }];

  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);

  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
}

def SIMDRelinOp : SIMD_Op<"relinearize", [Pure]> {
  let summary = "Relinearize a ciphertext to reduce its size (basis).";
  let description = [{
    Reduces the number of polynomials in the ciphertext (e.g., from 3 to 2) after multiplication.
    This operation is expensive but necessary to prevent ciphertext expansion.
    
    Input: !simd.simdcipher<L x S x Extended>
    Output: !simd.simdcipher<L x S x Standard>
  }];

  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);

  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
}

def SIMDModSwitchOp : SIMD_Op<"mod_switch", [Pure]> {
  let summary = "Explicitly drop levels without rescaling.";
  let description = [{
    Drops the modulus chain to a lower level without changing the scale.
    Used for aligning levels between two operands of an Add/Sub operation.
    Cost is usually negligible.
  }];

  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);

  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
}

// def SIMDLoadOp : SIMD_Op<"load"> {
//     let summary = "Load a SIMD cipher value";
//     let arguments = (ins AnyType:$src);
//     let results = (outs AnyType:$result);
//     let assemblyFormat = [{
//         $src `:` type($src) attr-dict `:` type($result)
//     }];
// }

def SIMDLoadOp : SIMD_Op<"load", [Pure]> {
    let summary = "Extract a scalar cipher from a SIMD cipher array at a specific index";
    let arguments = (ins AnyType:$array, Index:$index);
    let results = (outs AnyType:$result);
    let assemblyFormat = "$array `[` $index `]` attr-dict `:` type($array) `->` type($result)";
}

def SIMDStoreOp : SIMD_Op<"store"> {
    let summary = "Store a SIMD cipher value into a SIMD slot (or memref placeholder)";
    let description = [{
        Stores a value of type `!flyhe.cipher` into a destination which may be
        a memref (before type conversion) or a !flyhe.cipher (after).
    }];

    let arguments = (ins 
        AnyType:$value,
        AnyType:$dest);

    let assemblyFormat = "$value `,` $dest `:` type($value) `,` type($dest) attr-dict";
}

// def SIMDStoreOp : SIMD_Op<"store"> {
//     let summary = "Store a SIMD cipher value.";
    
//     // 添加 Variadic<Index>:$indices 以支持带索引的存储
//     let arguments = (ins 
//         AnyType:$value,
//         AnyType:$dest,
//         Variadic<Index>:$indices); 

//     // 修改 assemblyFormat 以支持索引列表 [idx1, idx2]
//     let assemblyFormat = [{
//         $value `,` $dest (`[` $indices^ `]`)? attr-dict `:` type($value) `,` type($dest)
//     }];
// }

def VectorOfFloatTypeConstraint : Type<
  CPred<"llvm::isa<::mlir::VectorType>($_self) && "
        "llvm::isa<::mlir::FloatType>(llvm::cast<::mlir::VectorType>($_self).getElementType())">,
  "vector of float type">;

def SIMDEncryptOp : SIMD_Op<"encrypt", [Pure]> {
    let summary = "Encrypt a plaintext floating-point value into a SIMD cipher.";
    let description = [{
        Takes a plaintext value (e.g., f64) and produces an encrypted ciphertext
        of type !simd.cipher. This operation models encoding or encryption
        under homomorphic encryption context.
    }];

    let arguments = (ins AnyType:$input);

    let results = (outs AnyType:$cipher);

    let assemblyFormat = [{
        $input attr-dict `:` type($input) `->` type($cipher)
    }];
}

def SIMDDecryptOp : SIMD_Op<"decrypt", [Pure]> {
    let summary = "Decrypt a SIMD ciphertext back into a plaintext value or vector.";
    let description = [{
        Converts an encrypted !simd.cipher value back into a plaintext float or vector,
        simulating the decryption stage of FHE computation.

        This operation can produce either a scalar float (e.g. f64)
        or a vector of floats (e.g. vector<8xf64>), depending on the context.
        It is typically inserted before vector.transfer_write to convert
        ciphertext vectors back into plaintext data.
    }];

    let arguments = (ins AnyType:$cipher);

    let results = (outs AnyType:$output);

    let assemblyFormat = [{
        $cipher attr-dict `:` type($cipher) `:` type($output)
    }];
}

def SIMDAllocaOp : SIMD_Op<"alloca"> {
  let summary = "Allocate encrypted memory for a single SIMD cipher";
  let description = [{
    Allocate a 0-D memref whose element type must be the SIMD cipher type.
    Example assembly:
      %c = simd.alloca : memref<!simd.sisdcipher>
  }];

  // The op has no operands and one result (an AnyType so parser can accept memref types).
  let arguments = (ins);
  let results = (outs AnyType:$result);

  // Allow parser to print the result type
  let assemblyFormat = "`:` type($result) attr-dict";
}

def SIMDPrintOp : SIMD_Op<"print", [Pure]> {
    let summary = "Print one or more SIMD ciphertexts (for debug or output).";
    let description = [{
        This operation takes one or more values of type `!simd.cipher`
        and conceptually prints their decrypted plaintext representation.
        It has no result and is considered a side-effecting debug op (though
        technically marked NoSideEffect for testing convenience).
    }];

    let arguments = (ins Variadic<SIMD_SIMDCipherType>:$inputs);

    let results = (outs);

    let assemblyFormat = [{
        $inputs attr-dict `:` type($inputs)
    }];
}

// === 补充缺失的算术/逻辑操作 ===
def SIMDDivOp : SIMD_Op<"div", [Pure]> {
    let summary = "Define a division operation.";
    let description = [{
        Define a division operation between two SIMD cipher values.
        Consumes levels (noise budget).
    }];

    let arguments = (ins 
        AnyType:$lhs,
        AnyType:$rhs);
    let results = (outs AnyType:$res);

    let assemblyFormat = [{
        $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($res)
    }];
}

def SIMDSelectOp : SIMD_Op<"select", [Pure]> {
    let summary = "Select operation based on an encrypted condition (mask).";
    let description = [{
        Return input1 if condition is true (1), else input2.
        Condition must be a ciphertext acting as a mask (usually 0 or 1).
    }];

    let arguments = (ins 
        AnyType:$condition,
        AnyType:$true_value,
        AnyType:$false_value);
        
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $condition `,` $true_value `,` $false_value attr-dict 
        `:` type($condition) `,` type($true_value) `,` type($false_value) `->` type($result)
    }];
}

def SIMDCmpOp : SIMD_Op<"cmp", [Pure]> {
    let summary = "Compare two ciphertexts.";
    let description = [{
        Compares two ciphertexts based on a predicate (e.g. OLT, OEQ).
        Returns a ciphertext representing the boolean result (mask).
        
        The predicate is stored as a 64-bit integer attribute (matching arith.cmpf).
    }];

    let arguments = (ins 
        AnyType:$lhs,
        AnyType:$rhs,
        I64Attr:$predicate); // Predicate value (e.g., from arith::CmpFPredicate)

    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)
    }];
}

def SIMDReduceAddOp : SIMD_Op<"reduce_add", [Pure]> {
    let summary = "Sum all elements in a SIMD vector cipher to a scalar cipher.";
    
    let arguments = (ins AnyType:$input);
    let results = (outs AnyType:$result);

    let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
}

def SIMDExpOp : SIMD_Op<"exp", [Pure]> {
    let summary = "Calculate exponential of a ciphertext.";
    let description = [{
        Calculates the exponential (e^x) of the input ciphertext.
        Depending on the backend, this might be lowered to a polynomial approximation 
        (e.g., Taylor series) or a programmable bootstrap lookup.
    }];

    let arguments = (ins AnyType:$input);
    let results = (outs AnyType:$result);

    let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
}


#endif  // SIMDOPS_TD_
