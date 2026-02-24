#include <iostream>
#include <vector>
#include <cstdlib>
#include <cstdio>
#include <algorithm>

#include "FlyHEContext.h" // 包含了你更新后的 tlwevaluator

int main() {
    // ---------------------------------------------------
    // 1. MLIR 常量与内存分配
    // ---------------------------------------------------
    const double cst = -2.0;
    const double cst_0 = 2147483647.0; // 0x41DFFFFFFFC00000
    const double cst_1 = 4.0;
    const int c0_i32 = 0;
    const char* str0 = "%f + %f = %f\n";
    const int batch_size = 10;

    std::vector<double> alloc(batch_size);
    std::vector<double> alloc_2(batch_size);
    std::vector<double> alloc_3(batch_size);

    // ---------------------------------------------------
    // 2. 随机数生成循环 (affine.for)
    // ---------------------------------------------------
    for (int arg0 = 0; arg0 < batch_size; ++arg0) {
        int v6 = std::rand();
        double v7 = static_cast<double>(v6);
        alloc[arg0] = ((v7 / cst_0) * cst_1) + cst;
    }

    for (int arg0 = 0; arg0 < batch_size; ++arg0) {
        int v6 = std::rand();
        double v7 = static_cast<double>(v6);
        alloc_2[arg0] = ((v7 / cst_0) * cst_1) + cst;
    }

    // ---------------------------------------------------
    // 3. FHE 上下文初始化
    // ---------------------------------------------------
    auto config = FlyHEConfig::CreateSISD();
    FlyHEContext<> hectx(config);
    // ... 假设此处完成 hectx 内部的 keys 和 evaluator 初始化 ...

    auto evaluator = hectx.tfhe_evaluator; // 拿到你的 tlwevaluator 实例

    // 分配 GPU 密文空间
    Pointer<cuTLWE<lwe_enc_lvl>> d_cipher_1(batch_size);
    Pointer<cuTLWE<lwe_enc_lvl>> d_cipher_2(batch_size);
    Pointer<cuTLWE<lwe_enc_lvl>> d_cipher_add_res(batch_size);

    auto val_1 = d_cipher_1->template get<lwe_enc_lvl>();
    auto val_2 = d_cipher_2->template get<lwe_enc_lvl>();
    auto val_3 = d_cipher_add_res->template get<lwe_enc_lvl>();

    // ---------------------------------------------------
    // 4. 执行 FHE 算子 (完美对应 sisd 方言)
    // ---------------------------------------------------
    // %1 = sisd.encrypt %alloc
    evaluator->encrypt(val_1, alloc.data(), batch_size);

    // %2 = sisd.encrypt %alloc_2
    evaluator->encrypt(val_2, alloc_2.data(), batch_size);

    // %3 = sisd.add %1 + %2
    evaluator->add(val_3, val_1, val_2, batch_size);
    cudaDeviceSynchronize(); // 确保核函数执行完毕

    // %4 = sisd.decrypt %3 -> memref.copy %4, %alloc_3
    evaluator->decrypt(alloc_3.data(), val_3, batch_size);

    // ---------------------------------------------------
    // 5. MLIR 打印逻辑 (affine.for)
    // ---------------------------------------------------
    for (int arg0 = 0; arg0 < batch_size; ++arg0) {
        std::printf(str0, alloc[arg0], alloc_2[arg0], alloc_3[arg0]);
    }

    return c0_i32;
}