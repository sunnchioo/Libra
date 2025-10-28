#ifndef SCFHEOPS_TD_
#define SCFHEOPS_TD_

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "SCFHETypes.td"
include "SCFHEDialect.td"

class SCFHE_Op<string mnemonic, list<Trait> traits = []> : Op<SCFHE_Dialect, mnemonic, traits> {
    let summary = "SCFHE dialect operations base class";
}

def SCFHEAddOp : SCFHE_Op<"add", [Pure, SameOperandsAndResultType]> {
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

def SCFHESubOp : SCFHE_Op<"sub", [Pure, SameOperandsAndResultType]> {
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

def SCFHEMultOp : SCFHE_Op<"mult", [Pure, SameOperandsAndResultType]> {
    let summary = "Define a multiply operation.";
    let description = [{
        Define a multiply operation between two SCFHE cipher values.
    }];

    let arguments = (ins 
        AnyType:$input0,
        AnyType:$input1);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $input0 `*` $input1 attr-dict `:` type($input0)
    }];
}

def SCFHELoadOp : SCFHE_Op<"load"> {
    let summary = "Load a SCFHE cipher value";
    let arguments = (ins AnyType:$src);
    let results = (outs AnyType:$result);
    let assemblyFormat = [{
        $src `:` type($src) attr-dict `:` type($result)
    }];
}

def SCFHEStoreOp : SCFHE_Op<"store"> {
    let summary = "Store a SCFHE cipher value into a SCFHE slot (or memref placeholder)";
    let description = [{
        Stores a value of type `!flyhe.cipher` into a destination which may be
        a memref (before type conversion) or a !flyhe.cipher (after).
    }];

    let arguments = (ins 
        AnyType:$value,
        AnyType:$dest);

    // 修正：同时提供了 $value 和 $dest 的类型解析规则
    let assemblyFormat = "$value `,` $dest `:` type($value) `,` type($dest) attr-dict";
}

def VectorOfFloatTypeConstraint : Type<
  CPred<"llvm::isa<::mlir::VectorType>($_self) && "
        "llvm::isa<::mlir::FloatType>(llvm::cast<::mlir::VectorType>($_self).getElementType())">,
  "vector of float type">;

def SCFHEEncryptOp : SCFHE_Op<"encrypt", [Pure]> {
    let summary = "Encrypt a plaintext floating-point value into a SCFHE cipher.";
    let description = [{
        Takes a plaintext value (e.g., f64) and produces an encrypted ciphertext
        of type !scfhe.cipher. This operation models encoding or encryption
        under homomorphic encryption context.
    }];

    let arguments = (ins AnyTypeOf<[AnyFloat, VectorOfFloatTypeConstraint]>:$input);
    let results = (outs AnyType:$cipher);

    let assemblyFormat = [{
        $input attr-dict `:` type($input) `->` type($cipher)
    }];
}

def SCFHEDecryptOp : SCFHE_Op<"decrypt", [Pure]> {
    let summary = "Decrypt a SCFHE ciphertext back into a plaintext value or vector.";
    let description = [{
        Converts an encrypted !scfhe.cipher value back into a plaintext float or vector,
        simulating the decryption stage of FHE computation.

        This operation can produce either a scalar float (e.g. f64)
        or a vector of floats (e.g. vector<8xf64>), depending on the context.
        It is typically inserted before vector.transfer_write to convert
        ciphertext vectors back into plaintext data.
    }];

    let arguments = (ins AnyType:$cipher);

    // 支持标量浮点或向量类型输出
    let results = (outs AnyTypeOf<[AnyFloat, VectorOfFloatTypeConstraint]>:$output);

    let assemblyFormat = [{
        $cipher attr-dict `:` type($cipher) `:` type($output)
    }];
}


def SCFHEAllocaOp : SCFHE_Op<"alloca"> {
  let summary = "Allocate encrypted memory for a single SCFHE cipher";
  let description = [{
    Allocate a 0-D memref whose element type must be the SCFHE cipher type.
    Example assembly:
      %c = scfhe.alloca : memref<!scfhe.scfhecipher>
  }];

  // The op has no operands and one result (an AnyType so parser can accept memref types).
  let arguments = (ins);
  let results = (outs AnyType:$result);

  // Allow parser to print the result type
  let assemblyFormat = "`:` type($result) attr-dict";
}

def SCFHEPrintOp : SCFHE_Op<"print", [Pure]> {
    let summary = "Print one or more SCFHE ciphertexts (for debug or output).";
    let description = [{
        This operation takes one or more values of type `!scfhe.cipher`
        and conceptually prints their decrypted plaintext representation.
        It has no result and is considered a side-effecting debug op (though
        technically marked NoSideEffect for testing convenience).
    }];

    let arguments = (ins Variadic<SCFHE_SCFHECipherType>:$inputs);

    let results = (outs);

    let assemblyFormat = [{
        $inputs attr-dict `:` type($inputs)
    }];
}

def SCFHEMinOp : SCFHE_Op<"min", [Pure]> {
  let summary = "Finds the minimum value in a ciphertext vector, returning a scalar ciphertext.";
  
  let arguments = (ins AnyType:$operand);
  let results = (outs AnyType:$result);

  let assemblyFormat = "$operand attr-dict `:` type($operand) `->` type($result)";
}


#endif  // SCFHEOPS_TD_
