module attributes {he.config = {bootstrapping_enabled = false, logN = 16 : i64, logn = 15 : i64, mode = "SIMD", remaining_levels = 31 : i32}} {
  llvm.mlir.global internal constant @str0("%f + %f = %f\0A\00") {addr_space = 0 : i32}
  llvm.func @printf(!llvm.ptr, ...) -> i32
  func.func @random_real(%arg0: memref<?xf64>, %arg1: i32) {
    %cst = arith.constant 4.000000e+00 : f64
    %cst_0 = arith.constant 0x41DFFFFFFFC00000 : f64
    %cst_1 = arith.constant -2.000000e+00 : f64
    %0 = arith.index_cast %arg1 : i32 to index
    affine.for %arg2 = 0 to %0 {
      %1 = func.call @rand() : () -> i32
      %2 = arith.sitofp %1 : i32 to f64
      %3 = arith.divf %2, %cst_0 : f64
      %4 = arith.mulf %3, %cst : f64
      %5 = arith.addf %4, %cst_1 : f64
      affine.store %5, %arg0[%arg2] : memref<?xf64>
    }
    return
  }
  func.func private @rand() -> i32
  func.func @main() -> i32 {
    %cst = arith.constant -2.000000e+00 : f64
    %cst_0 = arith.constant 0x41DFFFFFFFC00000 : f64
    %cst_1 = arith.constant 4.000000e+00 : f64
    %0 = llvm.mlir.addressof @str0 : !llvm.ptr
    %c0_i32 = arith.constant 0 : i32
    %alloc = memref.alloc() : memref<32768xf64>
    %alloc_2 = memref.alloc() : memref<32768xf64>
    affine.for %arg0 = 0 to 32768 {
      %6 = func.call @rand() : () -> i32
      %7 = arith.sitofp %6 : i32 to f64
      %8 = arith.divf %7, %cst_0 : f64
      %9 = arith.mulf %8, %cst_1 : f64
      %10 = arith.addf %9, %cst : f64
      affine.store %10, %alloc[%arg0] : memref<32768xf64>
    }
    affine.for %arg0 = 0 to 32768 {
      %6 = func.call @rand() : () -> i32
      %7 = arith.sitofp %6 : i32 to f64
      %8 = arith.divf %7, %cst_0 : f64
      %9 = arith.mulf %8, %cst_1 : f64
      %10 = arith.addf %9, %cst : f64
      affine.store %10, %alloc_2[%arg0] : memref<32768xf64>
    }
    %1 = simd.encrypt %alloc : memref<32768xf64> -> !simd.simdcipher<31 x 32768 x i64>
    %2 = simd.encrypt %alloc_2 : memref<32768xf64> -> !simd.simdcipher<31 x 32768 x i64>
    %3 = simd.add %1, %2 : !simd.simdcipher<31 x 32768 x i64>, !simd.simdcipher<31 x 32768 x i64> -> !simd.simdcipher<31 x 32768 x i64>
    %4 = simd.decrypt %3 : !simd.simdcipher<31 x 32768 x i64> -> memref<32768xf64>
    %5 = llvm.getelementptr %0[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<14 x i8>
    affine.for %arg0 = 0 to 32768 {
      %6 = affine.load %alloc[%arg0] : memref<32768xf64>
      %7 = affine.load %alloc_2[%arg0] : memref<32768xf64>
      %8 = affine.load %4[%arg0] : memref<32768xf64>
      %9 = llvm.call @printf(%5, %6, %7, %8) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64, f64, f64) -> i32
    }
    memref.dealloc %alloc : memref<32768xf64>
    memref.dealloc %alloc_2 : memref<32768xf64>
    memref.dealloc %4 : memref<32768xf64>
    return %c0_i32 : i32
  }
}

