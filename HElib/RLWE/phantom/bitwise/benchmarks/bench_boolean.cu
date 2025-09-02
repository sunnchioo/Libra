#include <benchmark/benchmark.h>

#include "openfhe.h"
#include "binfhecontext.cuh"

using namespace lbcrypto;
using namespace phantom::bitwise;

static void bench_cpu_acc(benchmark::State& state) {
    // Perform setup here

    auto set = static_cast<BINFHE_PARAMSET>(state.range(0));
    auto method = static_cast<BINFHE_METHOD>(state.range(1));

    auto cc = BinFHEContext();
    cc.GenerateBinFHEContext(set, method);
    auto sk = cc.KeyGen();
    cc.BTKeyGen(sk);
    auto ct1 = cc.Encrypt(sk, 0);
    auto ct2 = cc.Encrypt(sk, 1);

    for (auto _: state) {
        // This code gets timed
        auto ct = cc.EvalBinGate(AND, ct1, ct2);
    }
}

// Register the function as a benchmark
BENCHMARK(bench_cpu_acc)
        ->Args({STD128, AP})
        ->Args({STD128, GINX})
        ->Args({STD128Q_3, AP})
        ->Args({STD128Q_3, GINX});


static void bench_gpu_acc(benchmark::State& state) {
    // Perform setup here

    auto set = static_cast<BINFHE_PARAMSET>(state.range(0));
    auto method = static_cast<BINFHE_METHOD>(state.range(1));

    auto cc = BinFHEContext();
    cc.GenerateBinFHEContext(set, method);
    GPUBinFHEContext gpu_cc(cc);
    auto sk = cc.KeyGen();
    gpu_cc.GPUBTKeyGen(sk);

    auto ct1 = cc.Encrypt(sk, 0);
    auto ct2 = cc.Encrypt(sk, 1);

    for (auto _: state) {
        // This code gets timed
        auto ct = gpu_cc.GPUEvalBinGate(AND, ct1, ct2);
    }
}

// Register the function as a benchmark
BENCHMARK(bench_gpu_acc)
        ->Args({STD128, AP})
        ->Args({STD128, GINX})
        ->Args({STD128_Binary, GINX})
        ->Args({STD128Q_3, AP})
        ->Args({STD128Q_3, GINX})
        ->Args({STD128Q_3_Binary, GINX})
        ->Args({T_1024_36, AP})
        ->Args({T_1024_36, GINX})
        ->Args({T_1024_36_Binary, GINX})
        ->Args({T_2048_50, AP})
        ->Args({T_2048_50, GINX})
        ->Args({T_2048_50_Binary, GINX});


static void bench_gpu_acc_batch(benchmark::State& state) {
    // Perform setup here

    auto set = static_cast<BINFHE_PARAMSET>(state.range(0));
    auto method = static_cast<BINFHE_METHOD>(state.range(1));
    size_t batch_size = state.range(2);

    auto cc = BinFHEContext();
    cc.GenerateBinFHEContext(set, method);
    GPUBinFHEContext gpu_cc(cc);
    auto sk = cc.KeyGen();
    gpu_cc.GPUBTKeyGen(sk);

    auto ct1 = cc.Encrypt(sk, 0);
    auto ct2 = cc.Encrypt(sk, 1);

    std::vector<LWECiphertext> v_ct1;
    std::vector<LWECiphertext> v_ct2;
    for (size_t j = 0; j < batch_size; j++) {
        v_ct1.push_back(cc.Encrypt(sk, 0));
        v_ct2.push_back(cc.Encrypt(sk, 1));
    }

    for (auto _: state) {
        // This code gets timed
        auto v_ct = gpu_cc.BatchGPUEvalBinGate(AND, v_ct1, v_ct2);
    }
}

// Register the function as a benchmark
BENCHMARK(bench_gpu_acc_batch)
        ->Args({STD128, AP, 1024})
        ->Args({STD128, GINX, 1024})
        ->Args({STD128_Binary, GINX, 1024})
        ->Args({STD128Q_3, AP, 1024})
        ->Args({STD128Q_3, GINX, 1024})
        ->Args({STD128Q_3_Binary, GINX, 1024})
        ->Args({T_1024_36, AP, 1024})
        ->Args({T_1024_36, GINX, 1024})
        ->Args({T_1024_36_Binary, GINX, 1024})
        ->Args({T_2048_50, AP, 1024})
        ->Args({T_2048_50, GINX, 1024})
        ->Args({T_2048_50_Binary, GINX, 1024});


// Run the benchmark
BENCHMARK_MAIN();
