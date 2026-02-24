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
  std::vector<double> v5(10);
  std::vector<double> v6(10);
  std::vector<double> v7(10);
  for (int v8 = 0; v8 < 10; ++v8) {
  int v9 = std::rand();
  double v10 = (double)v9;
  auto v11 = v10 / v1;
  auto v12 = v11 * v2;
  auto v13 = v12 + v0;
  v5[v8] = v13;
  }
  for (int v14 = 0; v14 < 10; ++v14) {
  int v15 = std::rand();
  double v16 = (double)v15;
  auto v17 = v16 / v1;
  auto v18 = v17 * v2;
  auto v19 = v18 + v0;
  v6[v14] = v19;
  }
  auto v20 = FlyHE_SISDEncrypt(hectx, v5, 10);
  auto v21 = FlyHE_SISDEncrypt(hectx, v6, 10);
  auto v22 = FlyHE_SISDAdd(hectx, v20, v21, 10);
  auto v23 = FlyHE_SISDDecrypt(hectx, v22, 10);
  std::copy(v23.begin(), v23.end(), v7.begin());
  for (int v25 = 0; v25 < 10; ++v25) {
  auto v26 = v5[v25];
  auto v27 = v6[v25];
  auto v28 = v7[v25];
  std::printf(v3, v26, v27, v28);
  }
  return 0;
}
