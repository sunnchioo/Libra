// batch tfhe mult
/**
 * max len is 1<<15
 */
// #include <chrono>

#include "binfhecontext.cuh"
#include "openfhe.h"
#include "phantom.h"
#include "utils.h"

using namespace lbcrypto;
using namespace phantom::bitwise;
// using namespace std::chrono;

void CPUBool(BINFHE_METHOD method, BINFHE_PARAMSET set) {
    auto cc = BinFHEContext();

    cc.GenerateBinFHEContext(STD128);
    auto sk = cc.KeyGen();

    std::cout << "Generating the bootstrapping keys..." << std::endl;

    // Generate the bootstrapping keys (refresh and switching keys)
    cc.BTKeyGen(sk);

    std::cout << "Completed the key generation." << std::endl;

    std::vector<LWECiphertext> ct_cube;
    std::vector<LWECiphertext> v_ct0;
    std::vector<LWECiphertext> v_ct1;

    size_t loop = 2;
    for (int exp = 0; exp < loop; exp++) {
        auto batch_size = 1 << exp;
        for (size_t j = 0; j < batch_size; j++) {
            v_ct0.push_back(cc.Encrypt(sk, 1));
            v_ct1.push_back(cc.Encrypt(sk, 0));
        }

        auto timer = Timer();
        {
            for (size_t j = 0; j < batch_size; j++) {
                ct_cube.push_back(cc.EvalBinGate(AND, v_ct0[j], v_ct1[j]));
            }
        }
        timer.stop();
        std::cout << timer.timer_duration<std::chrono::milliseconds>() << "ms\n";
    }

    LWEPlaintext result;
    cc.Decrypt(sk, ct_cube[0], &result);

    std::cout << "Result of encrypted computation of (1 AND 1) OR (1 AND (NOT 1)) = " << result << std::endl;
}

void CPUPBS(BINFHE_METHOD method, BINFHE_PARAMSET set) {
    // Sample Program: Step 1: Set CryptoContext
    auto cc = BinFHEContext();
    cc.GenerateBinFHEContext(set, true, 11, 0, method);

    // Sample Program: Step 2: Key Generation

    // Generate the secret key
    auto sk = cc.KeyGen();

    std::cout << "Generating the bootstrapping keys on CPU..." << std::endl;

    // Generate the bootstrapping keys (refresh and switching keys)
    cc.BTKeyGen(sk);

    std::cout << "Completed the key generation." << std::endl;

    // Sample Program: Step 3: Create the to-be-evaluated function and obtain its corresponding LUT
    int p = cc.GetMaxPlaintextSpace().ConvertToInt(); // Obtain the maximum plaintext space

    // Initialize Function f(x) = x^3 % p
    auto fp = [](NativeInteger m, NativeInteger p1) -> NativeInteger {
        if (m < p1)
            return (m * m * m) % p1;
        else
            return ((m - p1 / 2) * (m - p1 / 2) * (m - p1 / 2)) % p1;
    };

    // Generate LUT from function f(x)
    auto lut = cc.GenerateLUTviaFunction(fp, p);
    std::cout << "Evaluate x^3%" << p << " on CPU." << std::endl;

    // Sample Program: Step 4: evaluate f(x) homomorphically and decrypt
    // Note that we check for all the possible plaintexts.

    size_t loop = 2;
    for (int exp = 0; exp < loop; exp++) {
        size_t batch_size = 1 << 2;

        std::cout << "batch is: " << batch_size << std::endl;

        std::vector<LWECiphertext> ct_cube;
        std::vector<LWECiphertext> v_ct;
        for (size_t j = 0; j < batch_size; j++) {
            v_ct.push_back(cc.Encrypt(sk, 1 % p, FRESH, p)); // i < p
        }

        auto timer = Timer();
        {
            for (size_t i = 0; i < batch_size; i++) {
                ct_cube.push_back(cc.EvalFunc(v_ct[i], lut));
            }
        }
        timer.stop();
        std::cout << timer.timer_duration<std::chrono::milliseconds>() << "ms\n";

        std::cout << "lut end. " << std::endl;

        // for (size_t j = 0; j < batch_size; j++) {
        //     LWEPlaintext result;
        //     cc.Decrypt(sk, ct_cube[j], &result, p);
        //     std::cout << "Input: " << i << ". Expected: " << fp(i, p) << ". Evaluated = " << result << std::endl;
        //     if (fp(i, p) != result)
        //         throw std::logic_error("Error: Evaluated result does not match the expected result on CPU.");
        // }
    }
}

void GPUCMUX(BINFHE_METHOD method, BINFHE_PARAMSET set) {

    // Sample Program: Step 1: Set CryptoContext
    auto cc = BinFHEContext();
    cc.GenerateBinFHEContext(set, true, 11, 0, method);
    GPUBinFHEContext gpu_cc(cc);

    // Sample Program: Step 2: Key Generation

    // Generate the secret key
    auto sk = cc.KeyGen();

    std::cout << "Generating the bootstrapping keys on GPU..." << std::endl;

    // Generate the bootstrapping keys (refresh and switching keys)
    gpu_cc.GPUBTKeyGen(sk);

    std::cout << "Completed the key generation." << std::endl;

    int p = cc.GetMaxPlaintextSpace().ConvertToInt(); // Obtain the maximum plaintext space
    std::cout << "p: " << p << std::endl;

    size_t log_max_batch = 17;
    for (int exp = 0; exp < log_max_batch; exp++) {
        size_t batch_size = 1 << exp;

        std::cout << "batch is: " << batch_size << std::endl;

        std::vector<LWECiphertext> v_ct;
        for (size_t j = 0; j < batch_size; j++) {
            v_ct.push_back(cc.Encrypt(sk, 1 % p, FRESH, p)); // i < p
        }

        auto v_ct_cmux = gpu_cc.BatchGPUEvalCMUX(v_ct);

        std::cout << "cmux end. " << std::endl;
        // for (size_t j = 0; j < batch_size; j++) {
        //     LWEPlaintext result;
        //     cc.Decrypt(sk, v_ct_cmux[j], &result, p);
        //     std::cout << ". Evaluated = " << result << std::endl;
        // }
    }
}

void GPUPBS(BINFHE_METHOD method, BINFHE_PARAMSET set) {
    // Sample Program: Step 1: Set CryptoContext
    auto cc = BinFHEContext();
    cc.GenerateBinFHEContext(set, true, 11, 0, method);
    GPUBinFHEContext gpu_cc(cc);

    std::cout << "LWE RingDim: " << gpu_cc.GetParams()->GetLWEParams()->GetN() << std::endl;

    // Sample Program: Step 2: Key Generation

    // Generate the secret key
    auto sk = cc.KeyGen();

    std::cout << "Generating the bootstrapping keys on GPU..." << std::endl;

    // Generate the bootstrapping keys (refresh and switching keys)
    gpu_cc.GPUBTKeyGen(sk);

    std::cout << "Completed the key generation." << std::endl;

    // Sample Program: Step 3: Create the to-be-evaluated function and obtain its corresponding LUT
    int p = cc.GetMaxPlaintextSpace().ConvertToInt();                      // Obtain the maximum plaintext space
    std::cout << "Obtain the maximum plaintext space: " << p << std::endl; // p = 4

    // Initialize Function f(x) = x^3 % p
    auto fp = [](NativeInteger m, NativeInteger p1) -> NativeInteger {
        if (m < p1)
            return (m * m * m) % p1;
        else
            return ((m - p1 / 2) * (m - p1 / 2) * (m - p1 / 2)) % p1;
    };

    // Generate LUT from function f(x)
    auto lut = cc.GenerateLUTviaFunction(fp, p);
    std::cout << "Evaluate x^3%" << p << " on GPU." << std::endl;

    // Sample Program: Step 4: evaluate f(x) homomorphically and decrypt
    // Note that we check for all the possible plaintexts.
    for (size_t logbatch = 0; logbatch < 16; logbatch++) {
        /* code */
        std::cout << std::endl
                  << "logbatch is: " << logbatch << std::endl;

        size_t batch_size = 1 << logbatch;
        int input = 1 % p;
        std::vector<LWECiphertext> v_ct;
        for (size_t j = 0; j < batch_size; j++) {
            v_ct.push_back(cc.Encrypt(sk, input, FRESH, p));
        }

        // std::cout << "type is " << sizeof(typeid(v_ct[0]->GetB()).name()) << " size: " << sizeof(uint64_t) << std::endl; // 8 8
        // exit(0);
        std::vector<LWECiphertext> v_ct_cube;

        {
            phantom::util::CUDATimer timer("BatchGPUEvalFunc_" + std::to_string(batch_size), gpu_cc.GetStreamWrapper().get_stream());
            timer.start();
            v_ct_cube = gpu_cc.BatchGPUEvalFunc(v_ct, lut);
            timer.stop();
            CHECK_CUDA_LAST_ERROR();
        }

        for (size_t j = 0; j < batch_size; j++) {
            LWEPlaintext result;
            cc.Decrypt(sk, v_ct_cube[j], &result, p);
            // std::cout << "Input: " << input << ". Expected: " << fp(input, p) << ". Evaluated = " << result << std::endl;
            if (fp(input, p) != result)
                throw std::logic_error("Error: Evaluated result does not match the expected result on GPU.");
        }
    }
}

void GPUboolean(BINFHE_METHOD method, BINFHE_PARAMSET set) {
    // Sample Program: Step 1: Set CryptoContext

    auto cc = BinFHEContext();

    // STD128 is the security level of 128 bits of security based on LWE Estimator
    // and HE standard. Other common options are TOY, MEDIUM, STD192, and STD256.
    // MEDIUM corresponds to the level of more than 100 bits for both quantum and
    // classical computer attacks.
    cc.GenerateBinFHEContext(set, method);
    GPUBinFHEContext gpu_cc(cc);

    // Sample Program: Step 2: Key Generation

    // Generate the secret key
    auto sk = cc.KeyGen();

    std::cout << "Generating the bootstrapping keys on GPU..." << std::endl;

    // Generate the bootstrapping keys (refresh and switching keys)
    gpu_cc.GPUBTKeyGen(sk);

    std::cout << "Completed the key generation." << std::endl;

    std::vector<std::pair<LWEPlaintext, LWEPlaintext>> bin_gates;
    bin_gates.emplace_back(0, 0);
    // bin_gates.emplace_back(0, 1);
    // bin_gates.emplace_back(1, 0);
    // bin_gates.emplace_back(1, 1);

    size_t log_max_batch = 6;
    for (size_t logbatch = 0; logbatch < log_max_batch; logbatch++) {

        size_t batch_size = 1 << logbatch;

        for (const auto &pair : bin_gates) {
            std::vector<LWECiphertext> v_ct1;
            std::vector<LWECiphertext> v_ct2;
            for (size_t j = 0; j < batch_size; j++) {
                v_ct1.push_back(cc.Encrypt(sk, pair.first));
                v_ct2.push_back(cc.Encrypt(sk, pair.second));
            }

            phantom::util::CUDATimer timer("AND_" + std::to_string(batch_size), gpu_cc.GetStreamWrapper().get_stream());
            timer.start();
            auto v_ct = gpu_cc.BatchGPUEvalBinGate(AND, v_ct1, v_ct2);
            timer.stop();
            CHECK_CUDA_LAST_ERROR();

            for (size_t j = 0; j < batch_size; j++) {
                LWEPlaintext result;
                cc.Decrypt(sk, v_ct[j], &result);
                if (result != (pair.first & pair.second)) {
                    std::cout << pair.first << " AND " << pair.second << " = " << result << std::endl;
                    throw std::logic_error("Error: AND result does not match the expected result on GPU.");
                }
            }
        }
        std::cout << "AND test passed." << std::endl;
    }
}

void GPUADD(BINFHE_METHOD method, BINFHE_PARAMSET set) {
    // Sample Program: Step 1: Set CryptoContext
    auto cc = BinFHEContext();
    cc.GenerateBinFHEContext(set, true, 11, 0, method);
    GPUBinFHEContext gpu_cc(cc);

    std::cout << "LWE RingDim: " << gpu_cc.GetParams()->GetLWEParams()->GetN() << std::endl;

    // Sample Program: Step 2: Key Generation

    // Generate the secret key
    auto sk = cc.KeyGen();

    std::cout << "Generating the bootstrapping keys on GPU..." << std::endl;

    // Generate the bootstrapping keys (refresh and switching keys)
    gpu_cc.GPUBTKeyGen(sk);

    std::cout << "Completed the key generation." << std::endl;

    int p = cc.GetMaxPlaintextSpace().ConvertToInt();                      // Obtain the maximum plaintext space
    std::cout << "Obtain the maximum plaintext space: " << p << std::endl; // p = 4

    // Sample Program: Step 4: evaluate f(x) homomorphically and decrypt
    // Note that we check for all the possible plaintexts.
    for (size_t logbatch = 0; logbatch < 16; logbatch++) {
        /* code */
        std::cout << std::endl
                  << "logbatch is: " << logbatch << std::endl;

        size_t batch_size = 1 << logbatch;
        int input = 1 % p;
        std::vector<LWECiphertext> v_ct0;
        for (size_t j = 0; j < batch_size; j++) {
            v_ct0.push_back(cc.Encrypt(sk, input, FRESH, p));
        }
        std::vector<LWECiphertext> v_ct1;
        for (size_t j = 0; j < batch_size; j++) {
            v_ct1.push_back(cc.Encrypt(sk, input, FRESH, p));
        }

        // std::cout << "type is " << sizeof(typeid(v_ct[0]->GetB()).name()) << " size: " << sizeof(uint64_t) << std::endl; // 8 8
        std::cout << "v_ct0 size: " << v_ct0.size() << std::endl;
        // exit(0);
        std::vector<LWECiphertext> v_ct_cube;

        {
            phantom::util::CUDATimer timer("BatchGPUADD_out_" + std::to_string(batch_size), gpu_cc.GetStreamWrapper().get_stream());
            timer.start();
            v_ct_cube = gpu_cc.BatchGPUEvalADD(v_ct0, v_ct1);
            timer.stop();
            // std::cout << "BatchGPUADD end. " << std::endl;
            cudaDeviceSynchronize();
            CHECK_CUDA_LAST_ERROR();
        }

        for (size_t j = 0; j < 1; j++) {
            LWEPlaintext result;
            cc.Decrypt(sk, v_ct_cube[j], &result, p);
            std::cout << "Input: " << input + input << ". Evaluated = " << result << std::endl;
        }
    }
}

int main() {
    // cpu eval
    // CPUPBS(GINX, STD128_Binary);
    // CPUBool(GINX, STD128_Binary);

    // gpu eval
    // GPUPBS(GINX, STD128_Binary);
    // GPUPBS(GINX, TOY);
    // GPUCMUX(GINX, STD128);
    // GPUCMUX(GINX, STD128_Binary);
    // GPUboolean(GINX, STD128_Binary);
    GPUADD(GINX, STD128_Binary);
    return 0;
}
