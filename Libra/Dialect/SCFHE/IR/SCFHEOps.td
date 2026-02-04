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

    let arguments = (ins AnyType:$input);

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

    let results = (outs AnyType:$output);

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

def SCFHEArgMinOp : SCFHE_Op<"argmin", [Pure]> {
    let summary = "Computes the minimum value and its index from a ciphertext";
    let description = [{
        Input: A ciphertext vector.
        Output: 
          - result 0: The minimum value (ciphertext).
          - result 1: The index of the minimum value (ciphertext).
    }];

    // 输入: 一个密文向量
    let arguments = (ins AnyType:$operand);

    // 输出: (最小值, 最小索引)
    let results = (outs AnyType:$min_val, AnyType:$min_idx);

    let assemblyFormat = "$operand attr-dict `:` type($operand) `->` type($min_val) `,` type($min_idx)";
}

// 定义比较操作
// def SCFHECmpOp : SCFHE_Op<"cmp", [Pure]> {
//     let summary = "Homomorphic comparison operation";
//     let arguments = (ins AnyType:$lhs, AnyType:$rhs, I64Attr:$predicate);
//     let results = (outs AnyType:$result);
//     // 这里 predicate 直接传 int64，或者你可以定义 EnumAttr
// }

// // 定义选择操作 (Multiplexer)
// def SCFHESelectOp : SCFHE_Op<"select", [Pure]> {
//     let summary = "Homomorphic select operation (mux)";
//     // condition 是加密的 mask (0 或 1)
//     let arguments = (ins AnyType:$condition, AnyType:$true_value, AnyType:$false_value);
//     let results = (outs AnyType:$result);
// }
// 定义比较操作
def SCFHECmpOp : SCFHE_Op<"cmp", [Pure]> {
    let summary = "Homomorphic comparison operation";
    let arguments = (ins AnyType:$lhs, AnyType:$rhs, I64Attr:$predicate);
    let results = (outs AnyType:$result);
    
    // === 建议添加 ===
    // 格式示例: %res = scfhe.cmp %lhs, %rhs, 4 : !scfhe.cipher, !scfhe.cipher -> !scfhe.cipher
    let assemblyFormat = [{
        $lhs `,` $rhs `,` $predicate attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)
    }];
}

// 定义选择操作 (Multiplexer)
def SCFHESelectOp : SCFHE_Op<"select", [Pure]> {
    let summary = "Homomorphic select operation (mux)";
    // condition 是加密的 mask (0 或 1)
    let arguments = (ins AnyType:$condition, AnyType:$true_value, AnyType:$false_value);
    let results = (outs AnyType:$result);

    // === 建议添加 ===
    // 格式示例: %res = scfhe.select %cond, %true, %false : !scfhe.cipher, !scfhe.cipher, !scfhe.cipher -> !scfhe.cipher
    let assemblyFormat = [{
        $condition `,` $true_value `,` $false_value attr-dict `:` type($condition) `,` type($true_value) `,` type($false_value) `->` type($result)
    }];
}

def SCFHEThresholdCountOp : SCFHE_Op<"threshold_count", [Pure]> {
    let summary = "Count elements in a vector greater/less than a threshold";
    let description = [{
        High-level operation that replaces a loop of compare+accumulate.
        Input: 
          - input: Source data (MemRef or Ciphertext Vector)
          - threshold: Comparison value (Scalar Ciphertext or Plaintext)
        Output:
          - result: Scalar ciphertext containing the count.
    }];

    // input: 待统计的数据 (支持 MemRef 或 Cipher)
    // threshold: 阈值
    // predicate: 比较条件 (gt, lt, eq...)
    let arguments = (ins AnyType:$input, AnyType:$threshold, I64Attr:$predicate);
    
    // result: 统计结果 (加密的标量)
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $input `,` $threshold `,` $predicate attr-dict `:` type($input) `,` type($threshold) `->` type($result)
    }];
}

def SCFHEReduceAddOp : SCFHE_Op<"reduce_add", [Pure]> {
    let summary = "Reduces a cipher vector to a single scalar cipher via addition";
    let description = [{
        This operation sums all slots in the ciphertext vector and returns 
        a ciphertext representing the scalar sum (size 1).
        
        Example:
        %sum = scfhe.reduce_add %vec_cipher : !scfhe.cipher<32768xf64> -> !scfhe.cipher<1xf64>
    }];

    let arguments = (ins AnyType:$input);
    let results = (outs AnyType:$result);

    let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
    
    // 验证器：确保输入维度 > 1，输出维度 == 1，且元素类型一致
    // let hasVerifier = 1;
}

def SCFHEDivOp : SCFHE_Op<"div", [Pure, SameOperandsAndResultType]> {
    let summary = "Define a division operation.";
    let description = [{
        Define a division operation between two SCFHE cipher values.
        Typically used for computing averages (Mean = Sum / Count).
    }];

    let arguments = (ins 
        AnyType:$lhs,
        AnyType:$rhs);
    let results = (outs AnyType:$res);

   let assemblyFormat = [{
        $lhs `,` $rhs attr-dict `:` type($lhs)
    }];
}

def SCFHEExpOp : SCFHE_Op<"exp", [Pure]> {
    let summary = "Homomorphic Exponential operation";
    let arguments = (ins AnyType:$input);
    let results = (outs AnyType:$output);
    let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
}

#endif  // SCFHEOPS_TD_
