#ifndef SISDOPS_TD_
#define SISDOPS_TD_

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "SISDTypes.td"
include "SISDDialect.td"

class SISD_Op<string mnemonic, list<Trait> traits = []> : Op<SISD_Dialect, mnemonic, traits> {
    let summary = "SISD dialect operations base class";
}

// ============================================================================
// Group 1: Cryptographic I/O & Utility Operations
// ============================================================================

def SISDEncryptOp : SISD_Op<"encrypt", [Pure]> {
    let summary = "Encrypt a plaintext value into a SISD cipher.";
    let description = [{
        Takes a plaintext value and produces an encrypted ciphertext.
        This models the encryption stage at the boundary of HE computation.
    }];
    let arguments = (ins AnyType:$input);
    let results = (outs AnyType:$cipher);
    let assemblyFormat = "$input attr-dict `:` type($input) `->` type($cipher)";
}

def SISDDecryptOp : SISD_Op<"decrypt", [Pure]> {
    let summary = "Decrypt a SISD ciphertext back into a plaintext value.";
    let description = [{
        Converts an encrypted value back into a plaintext float or integer,
        simulating the decryption stage of FHE computation.
    }];
    let arguments = (ins AnyType:$cipher);
    let results = (outs AnyType:$output);
    let assemblyFormat = "$cipher attr-dict `:` type($cipher) `->` type($output)";
}

def SISDPrintOp : SISD_Op<"print"> {
    let summary = "Print one or more SISD ciphertexts (for debug or output).";
    let arguments = (ins Variadic<AnyType>:$inputs);
    let results = (outs);
    let assemblyFormat = "$inputs attr-dict `:` type($inputs)";
}

// ============================================================================
// Group 2: Linear Arithmetic Operations (Level-Consuming)
// ============================================================================
// Note: In TFHE, Cipher-Cipher ops increase noise. Cipher-Plain ops do not.

def SISDAddOp : SISD_Op<"add", [Pure]> {
    let summary = "Ciphertext + Ciphertext addition.";
    let arguments = (ins AnyType:$lhs, AnyType:$rhs);
    let results = (outs AnyType:$result);
    let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def SISDAddPlainOp : SISD_Op<"add_plain", [Pure]> {
    let summary = "Ciphertext + Plaintext addition (Zero noise growth).";
    let arguments = (ins AnyType:$cipher, AnyType:$plain);
    let results = (outs AnyType:$result);
    let assemblyFormat = "$cipher `,` $plain attr-dict `:` type($cipher) `,` type($plain) `->` type($result)";
}

def SISDSubOp : SISD_Op<"sub", [Pure]> {
    let summary = "Ciphertext - Ciphertext subtraction.";
    let arguments = (ins AnyType:$lhs, AnyType:$rhs);
    let results = (outs AnyType:$result);
    let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def SISDMultPlainOp : SISD_Op<"mult_plain", [Pure]> {
    let summary = "Ciphertext * Plaintext multiplication (Cheap, linear noise growth).";
    let arguments = (ins AnyType:$cipher, AnyType:$plain);
    let results = (outs AnyType:$result);
    let assemblyFormat = "$cipher `,` $plain attr-dict `:` type($cipher) `,` type($plain) `->` type($result)";
}

// TFHE is not natively supported, but is reserved.
def SISDMultOp : SISD_Op<"mult", [Pure]> {
    let summary = "Ciphertext * Ciphertext multiplication (Warning: High cost in TFHE).";
    let arguments = (ins AnyType:$lhs, AnyType:$rhs);
    let results = (outs AnyType:$result);
    let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

// TFHE is not natively supported, but is reserved.
def SISDDivOp : SISD_Op<"div", [Pure]> {
    let summary = "Ciphertext / Ciphertext division.";
    let description = [{
        Integer division between two encrypted values.
        Note: This is a very expensive operation in FHE.
    }];
    let arguments = (ins AnyType:$lhs, AnyType:$rhs);
    let results = (outs AnyType:$result);
    let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

// ============================================================================
// Group 3: Non-Linear & Bootstrapping Operations
// ============================================================================

def SISDfbsOp : SISD_Op<"fbs", [Pure]> {
    let summary = "Evaluate a Look-Up Table using Programmable Bootstrapping.";
    let description = [{
        This is the core of modern TFHE. It evaluates a function over the ciphertext
        by encoding the function's truth table into the `fbs` attribute.
        It inherently reduces the noise of the ciphertext (Bootstrapping).
    }];
    let arguments = (ins AnyType:$operand, ElementsAttr:$fbs);
    let results = (outs AnyType:$result);
    let assemblyFormat = "$operand `,` $fbs attr-dict `:` type($operand) `->` type($result)";
}

def SISDRefreshOp : SISD_Op<"refresh", [Pure]> {
    let summary = "Refresh the noise budget of a SISD ciphertext.";
    let description = [{
        Performs Programmable Bootstrapping with an Identity function (f(x) = x)
        purely to reduce noise, without altering the underlying plaintext value.
    }];
    let arguments = (ins AnyType:$input);
    let results = (outs AnyType:$result);
    let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
}

def SISDCmpOp : SISD_Op<"cmp", [Pure]> {
    let summary = "Compare two ciphertexts, returning an encrypted boolean (0 or 1).";
    let description = [{
        High-level operation. In TFHE, this will typically be lowered to a `sub` 
        followed by a `fbs` that extracts the sign bit.
    }];
    let arguments = (ins AnyType:$lhs, AnyType:$rhs);
    let results = (outs AnyType:$result);
    let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def SISDMinOp : SISD_Op<"min", [Pure]> {
    let summary = "Finds the minimum value (High-level Op).";
    let description = [{
        High-level operation. Can be lowered to:
        %cmp = sisd.cmp %a, %b
        // Then algebraic combination or MUX using the %cmp result.
    }];
    let arguments = (ins AnyType:$lhs, AnyType:$rhs);
    let results = (outs AnyType:$result);
    let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

// ============================================================================
// Group 4: Cipher Conversion
// ============================================================================

def SISDCastSISDCipherToSIMDCipherOp : SISD_Op<"cast_to_simd", [Pure]> {
    let summary = "Cast a !sisd.sisdcipher to a !simd.simdcipher by adding leading dimensions";
    let arguments = (ins AnyType:$input);
    let results = (outs AnyType:$result);
    let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
    
    let hasCanonicalizer = 1;
}

// ============================================================================
// Group 5: Memory & Data Movement
// ============================================================================

def SISDAllocaOp : SISD_Op<"alloca"> {
    let summary = "Allocate encrypted memory for a single SISD cipher";
    let arguments = (ins);
    let results = (outs AnyType:$result);
    let assemblyFormat = "`:` type($result) attr-dict";
}

def SISDLoadOp : SISD_Op<"load", [Pure]> {
    let summary = "Extract a scalar cipher from a SISD cipher array at a specific index";
    let arguments = (ins AnyType:$array, Index:$index);
    let results = (outs AnyType:$result);
    let assemblyFormat = "$array `[` $index `]` attr-dict `:` type($array) `->` type($result)";
}

def SISDStoreOp : SISD_Op<"store"> {
    let summary = "Store a SISD cipher value into a SISD slot (or memref placeholder)";
    let arguments = (ins AnyType:$value, AnyType:$dest);
    let assemblyFormat = "$value `,` $dest attr-dict `:` type($value) `,` type($dest)";
}

#endif  // SISDOPS_TD_
