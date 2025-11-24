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

def SIMDAddOp : SIMD_Op<"add", [Pure, SameOperandsAndResultType]> {
    let summary = "Define an add operation.";
    let description = [{
        Define an add operation.
    }];

    let arguments = (ins 
        AnyType:$input0,
        AnyType:$input1);
    let results = (outs AnyType:$res);

    let assemblyFormat = [{
        $input0 `+` $input1 attr-dict `:` type($input0)
    }];
}

def SIMDSubOp : SIMD_Op<"sub", [Pure, SameOperandsAndResultType]> {
    let summary = "Define an sub operation.";
    let description = [{
        Define an sub operation.
    }];

    let arguments = (ins 
        AnyType:$input0,
        AnyType:$input1);
    let results = (outs AnyType:$res);

    let assemblyFormat = [{
        $input0 `_` $input1 attr-dict `:` type($input0)
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

def SIMDLoadOp : SIMD_Op<"load"> {
    let summary = "Load a SIMD cipher value";
    let arguments = (ins AnyType:$src);
    let results = (outs AnyType:$result);
    let assemblyFormat = [{
        $src `:` type($src) attr-dict `:` type($result)
    }];
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

    let arguments = (ins AnyTypeOf<[AnyFloat, VectorOfFloatTypeConstraint]>:$input);
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

    let results = (outs AnyTypeOf<[AnyFloat, VectorOfFloatTypeConstraint]>:$output);

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

#endif  // SIMDOPS_TD_
