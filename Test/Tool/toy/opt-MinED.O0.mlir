module {
  llvm.mlir.global internal constant @str1("min: %f\0A\00") {addr_space = 0 : i32}
  llvm.func @printf(!llvm.ptr, ...) -> i32
  llvm.mlir.global internal constant @str0("%lf %lf\00") {addr_space = 0 : i32}
  llvm.func @__isoc99_scanf(!llvm.ptr, ...) -> i32
  func.func @main() -> i32 {
    %0 = llvm.mlir.addressof @str1 : !llvm.ptr
    %1 = llvm.mlir.addressof @str0 : !llvm.ptr
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 0.000000e+00 : f64
    %alloca = memref.alloca() : memref<8xf64>
    %alloca_0 = memref.alloca() : memref<8xf64>
    %2 = llvm.getelementptr %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<8 x i8>
    %3 = "polygeist.memref2pointer"(%alloca_0) : (memref<8xf64>) -> !llvm.ptr
    %4 = "polygeist.memref2pointer"(%alloca) : (memref<8xf64>) -> !llvm.ptr
    affine.for %arg0 = 0 to 8 {
      %16 = arith.muli %arg0, %c8 : index
      %17 = arith.index_cast %16 : index to i64
      %18 = llvm.getelementptr %3[%17] : (!llvm.ptr, i64) -> !llvm.ptr, i8
      %19 = llvm.getelementptr %4[%17] : (!llvm.ptr, i64) -> !llvm.ptr, i8
      %20 = llvm.call @__isoc99_scanf(%2, %18, %19) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
    }
    %5 = vector.transfer_read %alloca_0[%c0], %cst {in_bounds = [true]} : memref<8xf64>, vector<8xf64>
    %6 = scfhe.encrypt %5 : vector<8xf64> -> !scfhe.scfhecipher<8 x i64>
    %7 = vector.transfer_read %alloca[%c0], %cst {in_bounds = [true]} : memref<8xf64>, vector<8xf64>
    %8 = scfhe.encrypt %7 : vector<8xf64> -> !scfhe.scfhecipher<8 x i64>
    %9 = scfhe.sub %6 _ %8 : !scfhe.scfhecipher<8 x i64>
    %10 = scfhe.mult %9 * %9 : !scfhe.scfhecipher<8 x i64>
    %11 = scfhe.alloca : !scfhe.scfhecipher<8 x i64>
    scfhe.store %10, %11 : !scfhe.scfhecipher<8 x i64>, !scfhe.scfhecipher<8 x i64>
    %12 = scfhe.min %11 : !scfhe.scfhecipher<8 x i64> -> !scfhe.scfhecipher<1 x i64>
    %13 = llvm.getelementptr %0[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<9 x i8>
    %14 = scfhe.decrypt %12 : !scfhe.scfhecipher<1 x i64> : f64
    %15 = llvm.call @printf(%13, %14) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64) -> i32
    return %c0_i32 : i32
  }
}

