module {
  llvm.mlir.global internal constant @str1("result c[%d]: %f\0A\00") {addr_space = 0 : i32}
  llvm.func @printf(!llvm.ptr, ...) -> i32
  llvm.mlir.global internal constant @str0("%lf %lf\00") {addr_space = 0 : i32}
  llvm.func @__isoc99_scanf(!llvm.ptr, ...) -> i32
  func.func @main() -> i32 {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c0_i32 = arith.constant 0 : i32
    %alloca = memref.alloca() : memref<8xf64>
    %alloca_0 = memref.alloca() : memref<8xf64>
    %alloca_1 = memref.alloca() : memref<8xf64>
    %0 = llvm.mlir.addressof @str0 : !llvm.ptr
    %1 = llvm.getelementptr %0[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<8 x i8>
    %2 = "polygeist.memref2pointer"(%alloca_1) : (memref<8xf64>) -> !llvm.ptr
    %3 = "polygeist.memref2pointer"(%alloca_0) : (memref<8xf64>) -> !llvm.ptr
    affine.for %arg0 = 0 to 8 {
      %10 = arith.muli %arg0, %c8 : index
      %11 = arith.index_cast %10 : index to i64
      %12 = llvm.getelementptr %2[%11] : (!llvm.ptr, i64) -> !llvm.ptr, i8
      %13 = llvm.getelementptr %3[%11] : (!llvm.ptr, i64) -> !llvm.ptr, i8
      %14 = llvm.call @__isoc99_scanf(%1, %12, %13) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
    }
    %cst = arith.constant 0.000000e+00 : f64
    %4 = vector.transfer_read %alloca_1[%c0], %cst : memref<8xf64>, vector<8xf64>
    %cst_2 = arith.constant 0.000000e+00 : f64
    %5 = vector.transfer_read %alloca_0[%c0], %cst_2 : memref<8xf64>, vector<8xf64>
    %6 = arith.addf %4, %5 : vector<8xf64>
    %7 = arith.mulf %6, %6 : vector<8xf64>
    vector.transfer_write %7, %alloca[%c0] : vector<8xf64>, memref<8xf64>
    %8 = llvm.mlir.addressof @str1 : !llvm.ptr
    %9 = llvm.getelementptr %8[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<18 x i8>
    affine.for %arg0 = 0 to 8 {
      %10 = arith.index_cast %arg0 : index to i32
      %11 = affine.load %alloca[%arg0] : memref<8xf64>
      %12 = llvm.call @printf(%9, %10, %11) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32, f64) -> i32
    }
    return %c0_i32 : i32
  }
}

