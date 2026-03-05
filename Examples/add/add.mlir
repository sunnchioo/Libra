module {
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
  func.func @add(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: i64) -> i32 {
    %c0_i32 = arith.constant 0 : i32
    %0 = arith.index_cast %arg3 : i64 to index
    affine.for %arg4 = 0 to %0 {
      %1 = affine.load %arg1[%arg4] : memref<?xf64>
      %2 = affine.load %arg2[%arg4] : memref<?xf64>
      %3 = arith.addf %1, %2 : f64
      affine.store %3, %arg0[%arg4] : memref<?xf64>
    }
    return %c0_i32 : i32
  }
  func.func @main() -> i32 {
    %c0_i32 = arith.constant 0 : i32
    %c256_i64 = arith.constant 256 : i64
    %c256_i32 = arith.constant 256 : i32
    %alloc = memref.alloc() : memref<256xf64>
    %cast = memref.cast %alloc : memref<256xf64> to memref<?xf64>
    %alloc_0 = memref.alloc() : memref<256xf64>
    %cast_1 = memref.cast %alloc_0 : memref<256xf64> to memref<?xf64>
    %alloc_2 = memref.alloc() : memref<256xf64>
    %cast_3 = memref.cast %alloc_2 : memref<256xf64> to memref<?xf64>
    call @random_real(%cast, %c256_i32) : (memref<?xf64>, i32) -> ()
    call @random_real(%cast_1, %c256_i32) : (memref<?xf64>, i32) -> ()
    %0 = call @add(%cast_3, %cast, %cast_1, %c256_i64) : (memref<?xf64>, memref<?xf64>, memref<?xf64>, i64) -> i32
    %1 = llvm.mlir.addressof @str0 : !llvm.ptr
    %2 = llvm.getelementptr %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<14 x i8>
    affine.for %arg0 = 0 to 256 {
      %3 = affine.load %alloc[%arg0] : memref<256xf64>
      %4 = affine.load %alloc_0[%arg0] : memref<256xf64>
      %5 = affine.load %alloc_2[%arg0] : memref<256xf64>
      %6 = llvm.call @printf(%2, %3, %4, %5) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64, f64, f64) -> i32
    }
    memref.dealloc %alloc : memref<256xf64>
    memref.dealloc %alloc_0 : memref<256xf64>
    memref.dealloc %alloc_2 : memref<256xf64>
    return %c0_i32 : i32
  }
}

