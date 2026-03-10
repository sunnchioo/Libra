module attributes {he.config = {mode = "SISD"}} {
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
  func.func @main() -> i32 {
    %cst = arith.constant 1.600000e+01 : f64
    %cst_0 = arith.constant 0.000000e+00 : f64
    %cst_1 = arith.constant 0x41DFFFFFFFC00000 : f64
    %cst_2 = arith.constant 1.000000e+02 : f64
    %0 = llvm.mlir.addressof @str3 : !llvm.ptr
    %1 = llvm.mlir.addressof @str2 : !llvm.ptr
    %2 = llvm.mlir.addressof @str1 : !llvm.ptr
    %3 = llvm.mlir.addressof @str0 : !llvm.ptr
    %4 = llvm.mlir.undef : f64
    %c0_i32 = arith.constant 0 : i32
    %alloca = memref.alloca() : memref<1xf64>
    affine.store %4, %alloca[0] : memref<1xf64>
    %alloc = memref.alloc() : memref<16xf64>
    affine.for %arg0 = 0 to 16 {
      %19 = func.call @rand() : () -> i32
      %20 = arith.sitofp %19 : i32 to f64
      %21 = arith.divf %20, %cst_1 : f64
      %22 = arith.mulf %21, %cst_2 : f64
      %23 = arith.addf %22, %cst_0 : f64
      affine.store %23, %alloc[%arg0] : memref<16xf64>
    }
    %5 = sisd.encrypt %alloc : memref<16xf64> -> !sisd.sisdcipher<16 x i64>
    %6 = sisd.encrypt %cst_0 : f64 -> !sisd.sisdcipher<1 x i64>
    %7 = sisd.reduce_add %5 : !sisd.sisdcipher<16 x i64> -> !sisd.sisdcipher<1 x i64>
    %8 = sisd.add %7, %6 : !sisd.sisdcipher<1 x i64>, !sisd.sisdcipher<1 x i64> -> !sisd.sisdcipher<1 x i64>
    %9 = sisd.div %8, %cst : !sisd.sisdcipher<1 x i64>, f64 -> !sisd.sisdcipher<1 x i64>
    %10 = sisd.decrypt %9 : !sisd.sisdcipher<1 x i64> -> memref<1xf64>
    memref.copy %10, %alloca : memref<1xf64> to memref<1xf64>
    %11 = llvm.getelementptr %3[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<14 x i8>
    %12 = llvm.call @printf(%11) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
    %13 = llvm.getelementptr %2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<20 x i8>
    affine.for %arg0 = 0 to 16 {
      %19 = arith.index_cast %arg0 : index to i64
      %20 = affine.load %alloc[%arg0] : memref<16xf64>
      %21 = llvm.call @printf(%13, %19, %20) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i64, f64) -> i32
    }
    %14 = llvm.getelementptr %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<17 x i8>
    %15 = llvm.call @printf(%14) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
    %16 = llvm.getelementptr %0[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<15 x i8>
    %17 = affine.load %alloca[0] : memref<1xf64>
    %18 = llvm.call @printf(%16, %17) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64) -> i32
    memref.dealloc %alloc : memref<16xf64>
    return %c0_i32 : i32
  }
}

