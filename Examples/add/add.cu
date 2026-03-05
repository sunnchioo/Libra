// === Auto-Generated FlyHE CUDA C++ Program ===
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cstdio>
#include <algorithm>

#include "FlyHEContext.h"

int main() {
  // --- 1. FHE Context Initialization ---
  auto config = FlyHEConfig::CreateSISD();
  FlyHEContext<> hectx(config);
  auto evaluator = hectx.tfhe_evaluator;

  // --- 2. Program Execution ---
  double v0 = -2.000000e+00;
  double v1 = 2.147484e+09;
  double v2 = 4.000000e+00;
  const char* v3 = "%f + %f = %f\n";
  int v4 = 0;
  std::vector<double> v5(256);
  std::vector<double> v6(256);
  for (int v7 = 0; v7 < 256; ++v7) {
  int v8 = std::rand();
  double v9 = (double)v8;
  auto v10 = v9 / v1;
  auto v11 = v10 * v2;
  auto v12 = v11 + v0;
  v5[v7] = v12;
  }
  for (int v13 = 0; v13 < 256; ++v13) {
  int v14 = std::rand();
  double v15 = (double)v14;
  auto v16 = v15 / v1;
  auto v17 = v16 * v2;
  auto v18 = v17 + v0;
  v6[v13] = v18;
  }
  Pointer<cuTLWE<lwe_enc_lvl>> v19(256);
  FlyHE_SISDEncrypt(hectx, v19, v5, 256);
  Pointer<cuTLWE<lwe_enc_lvl>> v20(256);
  FlyHE_SISDEncrypt(hectx, v20, v6, 256);
  Pointer<cuTLWE<lwe_enc_lvl>> v21(256);
  FlyHE_SISDAdd(hectx, v21, v19, v20, 256);
  std::vector<double> v22(256);
  FlyHE_SISDDecrypt(hectx, v22, v21, 256);
  for (int v23 = 0; v23 < 256; ++v23) {
  auto v24 = v5[v23];
  auto v25 = v6[v23];
  auto v26 = v22[v23];
  std::printf(v3, v24, v25, v26);
  }
  return v4;
}
