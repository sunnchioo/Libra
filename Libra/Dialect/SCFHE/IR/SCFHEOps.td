#ifndef FLYHEOPS_TD_
#define FLYHEOPS_TD_

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "SCFHETypes.td"
include "SCFHEDialect.td"

class SCFHE_Op<string mnemonic, list<Trait> traits = []> : Op<SCFHE_Dialect, mnemonic, traits> {
    let summary = "SCFHE dialect operations base class";
}

def SCFHEAddOp : SCFHE_Op<"scfhe_add", [Pure, SameOperandsAndResultType]> {
    let summary = "Define an add operation.";
    let description = [{
        Define an add operation.
    }];

    let arguments = (ins SCFHE_SCFHECipherType:$input0,
        SCFHE_SCFHECipherType:$input1);
    let results = (outs SCFHE_SCFHECipherType:$res);

    let assemblyFormat = [{
        $input0 `+` $input1 attr-dict `:` type($input0)
    }];
}

def SCFHEMultOp : SCFHE_Op<"scfhe_mult", [Pure, SameOperandsAndResultType]> {
    let summary = "Define a multiply operation.";
    let description = [{
        Define a multiply operation between two SCFHE cipher values.
    }];

    let arguments = (ins SCFHE_SCFHECipherType:$input0,
        SCFHE_SCFHECipherType:$input1);
    let results = (outs SCFHE_SCFHECipherType:$result);

    let assemblyFormat = [{
        $input0 `*` $input1 attr-dict `:` type($input0)
    }];
}

def SCFHELoadOp : SCFHE_Op<"scfhe_load"> {
    let summary = "Load a SCFHE cipher value";
    let arguments = (ins AnyType:$src);
    let results = (outs SCFHE_SCFHECipherType:$result);
    let assemblyFormat = [{
        $src `:` type($src) attr-dict `:` type($result)
    }];
}

def SCFHEStoreOp : SCFHE_Op<"scfhe_store"> {
    let summary = "Store a SCFHE cipher value into a SCFHE slot (or memref placeholder)";
    let description = [{
        Stores a value of type `!flyhe.scfhe_cipher` into a destination which may be
        a memref (before type conversion) or a !flyhe.scfhe_cipher (after).
    }];

    let arguments = (ins SCFHE_SCFHECipherType:$value,
        AnyType:$dest);

    // ensure the parser can infer the type of $dest
    let assemblyFormat = "$value `,` $dest `:` type($dest) attr-dict";
}

def SCFHEEncryptOp : SCFHE_Op<"scfhe_encrypt", [Pure]> {
    let summary = "Encrypt a plaintext floating-point value into a SCFHE cipher.";
    let description = [{
        Takes a plaintext value (e.g., f64) and produces an encrypted ciphertext
        of type !scfhe.scfhe_cipher. This operation models encoding or encryption
        under homomorphic encryption context.
    }];

    let arguments = (ins AnyFloat:$input);
    let results = (outs SCFHE_SCFHECipherType:$cipher);

    let assemblyFormat = [{
        $input attr-dict `:` type($input) `:` type($cipher)
    }];
}

def SCFHEDecryptOp : SCFHE_Op<"scfhe_decrypt", [Pure]> {
    let summary = "Decrypt a SCFHE ciphertext back into a plaintext value.";
    let description = [{
        Converts an encrypted !scfhe.scfhe_cipher value back into a plaintext float,
        simulating the decryption stage of FHE computation.
    }];

    let arguments = (ins SCFHE_SCFHECipherType:$cipher);
    let results = (outs AnyFloat:$output);

    let assemblyFormat = [{
        $cipher attr-dict `:` type($cipher) `:` type($output)
    }];
}

#endif  // FLYHEOPS_TD_
