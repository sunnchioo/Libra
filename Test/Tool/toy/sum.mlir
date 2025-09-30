module {
  llvm.mlir.global internal constant @str0("sum: %f\0A\00") {addr_space = 0 : i32}
  llvm.func @printf(!llvm.ptr, ...) -> i32
  func.func @_Z3sumRdS_S_(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>) {
    %0 = affine.load %arg0[0] : memref<?xf64>
    %1 = affine.load %arg1[0] : memref<?xf64>
    %2 = arith.addf %0, %1 : f64
    affine.store %2, %arg2[0] : memref<?xf64>
    return
  }
  func.func @_Z4multRdS_S_(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>) {
    %0 = affine.load %arg0[0] : memref<?xf64>
    %1 = affine.load %arg1[0] : memref<?xf64>
    %2 = arith.mulf %0, %1 : f64
    affine.store %2, %arg2[0] : memref<?xf64>
    return
  }
}
