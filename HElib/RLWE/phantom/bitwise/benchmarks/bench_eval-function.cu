#include <benchmark/benchmark.h>

#include "openfhe.h"
#include "binfhecontext.cuh"

using namespace lbcrypto;
using namespace phantom::bitwise;

static void bench_cpu_acc(benchmark::State &state) {
    // Perform setup here

    auto set = static_cast<BINFHE_PARAMSET>(state.range(0));
    auto method = static_cast<BINFHE_METHOD>(state.range(1));

    auto cc = BinFHEContext();
    cc.GenerateBinFHEContext(set, true, 11, 0, method);

    auto sk = cc.KeyGen();
    cc.BTKeyGen(sk);

    // Sample Program: Step 3: Create the to-be-evaluated function and obtain its corresponding LUT
    int p = cc.GetMaxPlaintextSpace().ConvertToInt();  // Obtain the maximum plaintext space

    // Initialize Function f(x) = x^3 % p
    auto fp = [](NativeInteger m, NativeInteger p1) -> NativeInteger {
        if (m < p1)
            return (m * m * m) % p1;
        else
            return ((m - p1 / 2) * (m - p1 / 2) * (m - p1 / 2)) % p1;
    };

    // Generate LUT from function f(x)
    auto lut = cc.GenerateLUTviaFunction(fp, p);

    auto ct1 = cc.Encrypt(sk, 0 % p, FRESH, p);

    for (auto _: state) {
        // This code gets timed
        auto ct_cube = cc.EvalFunc(ct1, lut);
    }
}

// Register the function as a benchmark
BENCHMARK(bench_cpu_acc)
        ->Args({STD128, AP})
        ->Args({STD128, GINX});


static void bench_gpu_acc(benchmark::State &state) {
    // Perform setup here

    auto set = static_cast<BINFHE_PARAMSET>(state.range(0));
    auto method = static_cast<BINFHE_METHOD>(state.range(1));

    auto cc = BinFHEContext();
    cc.GenerateBinFHEContext(set, true, 11, 0, method);
    auto gpu_cc = GPUBinFHEContext(cc);

    auto sk = cc.KeyGen();
    gpu_cc.GPUBTKeyGen(sk);

    // Sample Program: Step 3: Create the to-be-evaluated function and obtain its corresponding LUT
    int p = cc.GetMaxPlaintextSpace().ConvertToInt();  // Obtain the maximum plaintext space

    // Initialize Function f(x) = x^3 % p
    auto fp = [](NativeInteger m, NativeInteger p1) -> NativeInteger {
        if (m < p1)
            return (m * m * m) % p1;
        else
            return ((m - p1 / 2) * (m - p1 / 2) * (m - p1 / 2)) % p1;
    };

    // Generate LUT from function f(x)
    auto lut = cc.GenerateLUTviaFunction(fp, p);

    auto ct1 = cc.Encrypt(sk, 0 % p, FRESH, p);

    for (auto _: state) {
        // This code gets timed
        auto ct_cube = gpu_cc.GPUEvalFunc(ct1, lut);
    }
}

// Register the function as a benchmark
BENCHMARK(bench_gpu_acc)
        ->Args({STD128, AP})
        ->Args({STD128, GINX})
        ->Args({STD128_Binary, GINX});


static void bench_gpu_acc_batch(benchmark::State &state) {
    // Perform setup here

    auto set = static_cast<BINFHE_PARAMSET>(state.range(0));
    auto method = static_cast<BINFHE_METHOD>(state.range(1));
    size_t batch_size = state.range(2);

    auto cc = BinFHEContext();
    cc.GenerateBinFHEContext(set, true, 11, 0, method);
    auto gpu_cc = GPUBinFHEContext(cc);

    auto sk = cc.KeyGen();
    gpu_cc.GPUBTKeyGen(sk);

    // Sample Program: Step 3: Create the to-be-evaluated function and obtain its corresponding LUT
    int p = cc.GetMaxPlaintextSpace().ConvertToInt();  // Obtain the maximum plaintext space

    // Initialize Function f(x) = x^3 % p
    auto fp = [](NativeInteger m, NativeInteger p1) -> NativeInteger {
        if (m < p1)
            return (m * m * m) % p1;
        else
            return ((m - p1 / 2) * (m - p1 / 2) * (m - p1 / 2)) % p1;
    };

    // Generate LUT from function f(x)
    auto lut = cc.GenerateLUTviaFunction(fp, p);

    std::vector<LWECiphertext> v_ct;
    for (size_t j = 0; j < batch_size; j++) {
        v_ct.push_back(cc.Encrypt(sk, 0 % p, FRESH, p));
    }

    for (auto _: state) {
        // This code gets timed
        auto v_ct_cube = gpu_cc.BatchGPUEvalFunc(v_ct, lut);
    }
}

// Register the function as a benchmark
BENCHMARK(bench_gpu_acc_batch)
        ->Args({STD128, AP, 1024})
        ->Args({STD128, GINX, 1024})
        ->Args({STD128_Binary, GINX, 1024});


// Run the benchmark
BENCHMARK_MAIN();
