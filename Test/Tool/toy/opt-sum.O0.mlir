module {
  llvm.mlir.global internal constant @str0("sum: %f\0A\00") {addr_space = 0 : i32}
  llvm.func @printf(!llvm.ptr, ...) -> i32
  func.func @_Z3sumRdS_S_(%arg0: !flyhe.simdcipher, %arg1: !flyhe.simdcipher, %arg2: !flyhe.simdcipher) {
    %0 = flyhe.simd_load %arg0 : !flyhe.simdcipher : !flyhe.simdcipher
    %1 = flyhe.simd_load %arg1 : !flyhe.simdcipher : !flyhe.simdcipher
    %2 = flyhe.simdadd %0 + %1 : !flyhe.simdcipher
    flyhe.simd_store %2, %arg2 : !flyhe.simdcipher
    return
  }
  func.func @_Z4multRdS_S_(%arg0: !flyhe.simdcipher, %arg1: !flyhe.simdcipher, %arg2: !flyhe.simdcipher) {
    %0 = flyhe.simd_load %arg0 : !flyhe.simdcipher : !flyhe.simdcipher
    %1 = flyhe.simd_load %arg1 : !flyhe.simdcipher : !flyhe.simdcipher
    %2 = flyhe.simdmult %0 * %1 : !flyhe.simdcipher
    flyhe.simd_store %2, %arg2 : !flyhe.simdcipher
    return
  }
}

