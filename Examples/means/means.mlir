module {
  llvm.mlir.global internal constant @str3("Average: %.2f\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str2("\0A--- result ---\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str1("Element[%zu]: %.2f\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str0("--- data ---\0A\00") {addr_space = 0 : i32}
  llvm.func @printf(!llvm.ptr, ...) -> i32
  func.func @random_real(%arg0: memref<?xf64>, %arg1: i32) {
    %cst = arith.constant 1.000000e+02 : f64
    %cst_0 = arith.constant 0x41DFFFFFFFC00000 : f64
    %cst_1 = arith.constant 0.000000e+00 : f64
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
  func.func @average_array(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: i64) -> i32 {
    %cst = arith.constant 0.000000e+00 : f64
    %c0_i32 = arith.constant 0 : i32
    %0 = arith.index_cast %arg2 : i64 to index
    %1 = affine.for %arg3 = 0 to %0 iter_args(%arg4 = %cst) -> (f64) {
      %4 = affine.load %arg1[%arg3] : memref<?xf64>
      %5 = arith.addf %arg4, %4 : f64
      affine.yield %5 : f64
    }
    %2 = arith.sitofp %arg2 : i64 to f64
    %3 = arith.divf %1, %2 : f64
    affine.store %3, %arg0[0] : memref<?xf64>
    return %c0_i32 : i32
  }
  func.func @main() -> i32 {
    %c10_i64 = arith.constant 10 : i64
    %c0_i32 = arith.constant 0 : i32
    %c10_i32 = arith.constant 10 : i32
    %alloca = memref.alloca() : memref<1xf64>
    %0 = llvm.mlir.undef : f64
    affine.store %0, %alloca[0] : memref<1xf64>
    %alloc = memref.alloc() : memref<10xf64>
    %cast = memref.cast %alloc : memref<10xf64> to memref<?xf64>
    call @random_real(%cast, %c10_i32) : (memref<?xf64>, i32) -> ()
    %cast_0 = memref.cast %alloca : memref<1xf64> to memref<?xf64>
    %1 = call @average_array(%cast_0, %cast, %c10_i64) : (memref<?xf64>, memref<?xf64>, i64) -> i32
    %2 = llvm.mlir.addressof @str0 : !llvm.ptr
    %3 = llvm.getelementptr %2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<14 x i8>
    %4 = llvm.call @printf(%3) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
    %5 = llvm.mlir.addressof @str1 : !llvm.ptr
    %6 = llvm.getelementptr %5[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<20 x i8>
    affine.for %arg0 = 0 to 10 {
      %14 = arith.index_cast %arg0 : index to i64
      %15 = affine.load %alloc[%arg0] : memref<10xf64>
      %16 = llvm.call @printf(%6, %14, %15) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i64, f64) -> i32
    }
    %7 = llvm.mlir.addressof @str2 : !llvm.ptr
    %8 = llvm.getelementptr %7[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<17 x i8>
    %9 = llvm.call @printf(%8) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
    %10 = llvm.mlir.addressof @str3 : !llvm.ptr
    %11 = llvm.getelementptr %10[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<15 x i8>
    %12 = affine.load %alloca[0] : memref<1xf64>
    %13 = llvm.call @printf(%11, %12) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64) -> i32
    memref.dealloc %alloc : memref<10xf64>
    return %c0_i32 : i32
  }
}

