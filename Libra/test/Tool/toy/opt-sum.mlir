module {
  func.func @_Z3sumRdS_(%arg0: !flyhe.simdcipher, %arg1: !flyhe.simdcipher) -> !flyhe.simdcipher attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = flyhe.simd_load %arg0 : !flyhe.simdcipher : !flyhe.simdcipher
    %1 = flyhe.simd_load %arg1 : !flyhe.simdcipher : !flyhe.simdcipher
    %2 = flyhe.smidadd %0 + %1 : !flyhe.simdcipher
    return %2 : !flyhe.simdcipher
  }
}

