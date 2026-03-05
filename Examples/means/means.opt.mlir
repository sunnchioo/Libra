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
    %c15 = arith.constant 15 : index
    %c14 = arith.constant 14 : index
    %c13 = arith.constant 13 : index
    %c12 = arith.constant 12 : index
    %c11 = arith.constant 11 : index
    %c10 = arith.constant 10 : index
    %c9 = arith.constant 9 : index
    %c8 = arith.constant 8 : index
    %c7 = arith.constant 7 : index
    %c6 = arith.constant 6 : index
    %c5 = arith.constant 5 : index
    %c4 = arith.constant 4 : index
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
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
      %49 = func.call @rand() : () -> i32
      %50 = arith.sitofp %49 : i32 to f64
      %51 = arith.divf %50, %cst_1 : f64
      %52 = arith.mulf %51, %cst_2 : f64
      %53 = arith.addf %52, %cst_0 : f64
      affine.store %53, %alloc[%arg0] : memref<16xf64>
    }
    %5 = sisd.encrypt %alloc : memref<16xf64> -> !sisd.sisdcipher<16 x i64>
    %6 = sisd.encrypt %cst_0 : f64 -> !sisd.sisdcipher<1 x i64>
    %7 = sisd.load %5[%c0] : !sisd.sisdcipher<16 x i64> -> !sisd.sisdcipher<1 x i64>
    %8 = sisd.add %6, %7 : !sisd.sisdcipher<1 x i64>, !sisd.sisdcipher<1 x i64> -> !sisd.sisdcipher<1 x i64>
    %9 = sisd.load %5[%c1] : !sisd.sisdcipher<16 x i64> -> !sisd.sisdcipher<1 x i64>
    %10 = sisd.add %8, %9 : !sisd.sisdcipher<1 x i64>, !sisd.sisdcipher<1 x i64> -> !sisd.sisdcipher<1 x i64>
    %11 = sisd.load %5[%c2] : !sisd.sisdcipher<16 x i64> -> !sisd.sisdcipher<1 x i64>
    %12 = sisd.add %10, %11 : !sisd.sisdcipher<1 x i64>, !sisd.sisdcipher<1 x i64> -> !sisd.sisdcipher<1 x i64>
    %13 = sisd.load %5[%c3] : !sisd.sisdcipher<16 x i64> -> !sisd.sisdcipher<1 x i64>
    %14 = sisd.add %12, %13 : !sisd.sisdcipher<1 x i64>, !sisd.sisdcipher<1 x i64> -> !sisd.sisdcipher<1 x i64>
    %15 = sisd.load %5[%c4] : !sisd.sisdcipher<16 x i64> -> !sisd.sisdcipher<1 x i64>
    %16 = sisd.add %14, %15 : !sisd.sisdcipher<1 x i64>, !sisd.sisdcipher<1 x i64> -> !sisd.sisdcipher<1 x i64>
    %17 = sisd.load %5[%c5] : !sisd.sisdcipher<16 x i64> -> !sisd.sisdcipher<1 x i64>
    %18 = sisd.add %16, %17 : !sisd.sisdcipher<1 x i64>, !sisd.sisdcipher<1 x i64> -> !sisd.sisdcipher<1 x i64>
    %19 = sisd.load %5[%c6] : !sisd.sisdcipher<16 x i64> -> !sisd.sisdcipher<1 x i64>
    %20 = sisd.add %18, %19 : !sisd.sisdcipher<1 x i64>, !sisd.sisdcipher<1 x i64> -> !sisd.sisdcipher<1 x i64>
    %21 = sisd.load %5[%c7] : !sisd.sisdcipher<16 x i64> -> !sisd.sisdcipher<1 x i64>
    %22 = sisd.add %20, %21 : !sisd.sisdcipher<1 x i64>, !sisd.sisdcipher<1 x i64> -> !sisd.sisdcipher<1 x i64>
    %23 = sisd.load %5[%c8] : !sisd.sisdcipher<16 x i64> -> !sisd.sisdcipher<1 x i64>
    %24 = sisd.add %22, %23 : !sisd.sisdcipher<1 x i64>, !sisd.sisdcipher<1 x i64> -> !sisd.sisdcipher<1 x i64>
    %25 = sisd.load %5[%c9] : !sisd.sisdcipher<16 x i64> -> !sisd.sisdcipher<1 x i64>
    %26 = sisd.add %24, %25 : !sisd.sisdcipher<1 x i64>, !sisd.sisdcipher<1 x i64> -> !sisd.sisdcipher<1 x i64>
    %27 = sisd.load %5[%c10] : !sisd.sisdcipher<16 x i64> -> !sisd.sisdcipher<1 x i64>
    %28 = sisd.add %26, %27 : !sisd.sisdcipher<1 x i64>, !sisd.sisdcipher<1 x i64> -> !sisd.sisdcipher<1 x i64>
    %29 = sisd.load %5[%c11] : !sisd.sisdcipher<16 x i64> -> !sisd.sisdcipher<1 x i64>
    %30 = sisd.add %28, %29 : !sisd.sisdcipher<1 x i64>, !sisd.sisdcipher<1 x i64> -> !sisd.sisdcipher<1 x i64>
    %31 = sisd.load %5[%c12] : !sisd.sisdcipher<16 x i64> -> !sisd.sisdcipher<1 x i64>
    %32 = sisd.add %30, %31 : !sisd.sisdcipher<1 x i64>, !sisd.sisdcipher<1 x i64> -> !sisd.sisdcipher<1 x i64>
    %33 = sisd.load %5[%c13] : !sisd.sisdcipher<16 x i64> -> !sisd.sisdcipher<1 x i64>
    %34 = sisd.add %32, %33 : !sisd.sisdcipher<1 x i64>, !sisd.sisdcipher<1 x i64> -> !sisd.sisdcipher<1 x i64>
    %35 = sisd.load %5[%c14] : !sisd.sisdcipher<16 x i64> -> !sisd.sisdcipher<1 x i64>
    %36 = sisd.add %34, %35 : !sisd.sisdcipher<1 x i64>, !sisd.sisdcipher<1 x i64> -> !sisd.sisdcipher<1 x i64>
    %37 = sisd.load %5[%c15] : !sisd.sisdcipher<16 x i64> -> !sisd.sisdcipher<1 x i64>
    %38 = sisd.add %36, %37 : !sisd.sisdcipher<1 x i64>, !sisd.sisdcipher<1 x i64> -> !sisd.sisdcipher<1 x i64>
    %39 = sisd.div %38, %cst : !sisd.sisdcipher<1 x i64>, f64 -> !sisd.sisdcipher<1 x i64>
    %40 = sisd.decrypt %39 : !sisd.sisdcipher<1 x i64> -> memref<1xf64>
    memref.copy %40, %alloca : memref<1xf64> to memref<1xf64>
    %41 = llvm.getelementptr %3[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<14 x i8>
    %42 = llvm.call @printf(%41) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
    %43 = llvm.getelementptr %2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<20 x i8>
    affine.for %arg0 = 0 to 16 {
      %49 = arith.index_cast %arg0 : index to i64
      %50 = affine.load %alloc[%arg0] : memref<16xf64>
      %51 = llvm.call @printf(%43, %49, %50) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i64, f64) -> i32
    }
    %44 = llvm.getelementptr %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<17 x i8>
    %45 = llvm.call @printf(%44) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
    %46 = llvm.getelementptr %0[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<15 x i8>
    %47 = affine.load %alloca[0] : memref<1xf64>
    %48 = llvm.call @printf(%46, %47) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64) -> i32
    memref.dealloc %alloc : memref<16xf64>
    return %c0_i32 : i32
  }
}

