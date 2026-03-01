// === Auto-Generated FlyHE CUDA C++ Program ===
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cstdio>
#include <algorithm>

#include "FlyHEContext.h"

int main() {
  // --- 1. FHE Context Initialization ---
  auto config = FlyHEConfig::CreateCROSS(16, 4, 31, false);
  FlyHEContext<> hectx(config);
  auto evaluator = hectx.tfhe_evaluator;

  // --- 2. Program Execution ---
  double v0 = 1.000000e+01;
  int v1 = 10;
  int v2 = 0;
  int v3 = 1;
  double v4 = 0.000000e+00;
  double v5 = 2.147484e+09;
  double v6 = 1.000000e+02;
  const char* v7 = "%f + %f = %f\n";
  const char* v8 = "%f + %f = %f\n";
  const char* v9 = "%f + %f = %f\n";
  const char* v10 = "%f + %f = %f\n";
  auto v11 = 0; // llvm.mlir.undef
  int v12 = 0;
  std::vector<double> v13(1);
  v13[0] = v11;
  std::vector<double> v14(10);
  for (int v15 = 0; v15 < 10; ++v15) {
  int v16 = std::rand();
  double v17 = (double)v16;
  auto v18 = v17 / v5;
  auto v19 = v18 * v6;
  auto v20 = v19 + v4;
  v14[v15] = v20;
  }
  Pointer<cuTLWE<lwe_enc_lvl>> v21(10);
  FlyHE_SISDEncrypt(hectx, v21, v14, 10);
  Pointer<cuTLWE<lwe_enc_lvl>> v22(1);
  FlyHE_SISDEncrypt(hectx, v22, v4, 1);
  auto v24 = v22;
  for (int v23 = v2; v23 < v1; v23 += v3) {
  auto v25 = v21[v23];
  Pointer<cuTLWE<lwe_enc_lvl>> v26(1);
  FlyHE_SISDAdd(hectx, v26, v24, v25, 1);
    v24 = v26;
  }
  Pointer<cuTLWE<lwe_enc_lvl>> v27(1);
  FlyHE_SISDEncrypt(hectx, v27, v0, 1);
  Pointer<cuTLWE<lwe_enc_lvl>> v28(1);
  FlyHE_SISDDiv(hectx, v28, v24, v27, 1);
  std::vector<double> v29(1);
  FlyHE_SISDDecrypt(hectx, v29, v28, 1);
  std::copy(v29.begin(), v29.end(), v13.begin());
  std::printf(v10);
  for (int v30 = 0; v30 < 10; ++v30) {
  int v31 = (int)v30;
  auto v32 = v14[v30];
  std::printf(v9, v31, v32);
  }
  std::printf(v8);
  auto v33 = v13[0];
  std::printf(v7, v33);
  return v12;
}
