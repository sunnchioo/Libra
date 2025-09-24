module {
  func.func @_Z3sumRdS_(%arg0: memref<?xf64>, %arg1: memref<?xf64>) -> f64 attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = affine.load %arg0[0] : memref<?xf64>
    %1 = affine.load %arg1[0] : memref<?xf64>
    %2 = arith.addf %0, %1 : f64
    return %2 : f64
  }
}
