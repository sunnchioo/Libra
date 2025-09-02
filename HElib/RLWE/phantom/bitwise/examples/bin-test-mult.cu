// batch tfhe mult
/**
 * max len is 1<<15
 */
#include <chrono>

#include "binfhecontext.cuh"
#include "openfhe.h"
#include "phantom.h"

using namespace lbcrypto;
using namespace phantom::bitwise;
using namespace std::chrono;

void example_eval_func_gpu(BINFHE_METHOD method, BINFHE_PARAMSET set, size_t batch_size) {
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

    // Sample Program: Step 3: Create the to-be-evaluated function and obtain its corresponding LUT
    int p = cc.GetMaxPlaintextSpace().ConvertToInt(); // Obtain the maximum plaintext space
    std::cout << "p in tfhe: " << p << std::endl;

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
    for (int i = 0; i < p; i++) {

        i = 3;

        for (batch_size = 1; batch_size < (1 << 16) + 1; batch_size = (batch_size << 1)) {
            std::cout << "batch: " << batch_size << std::endl;

            std::vector<LWECiphertext> v_ct;
            for (size_t j = 0; j < batch_size; j++) {
                v_ct.push_back(cc.Encrypt(sk, i % p, FRESH, p));
            }

            auto start = system_clock::now();
            auto v_ct_cube = gpu_cc.BatchGPUEvalFunc(v_ct, lut);
            duration<double> sec = system_clock::now() - start;
            std::cout << "Batch Func took: " << sec.count() * 1000 << " ms" << std::endl;

            for (size_t j = 0; j < batch_size; j++) {
                LWEPlaintext result;
                cc.Decrypt(sk, v_ct_cube[j], &result, p);
                // std::cout << "Input: " << i << ". Expected: " << fp(i, p) << ". Evaluated = " << result << std::endl;
                if (fp(i, p) != result)
                    throw std::logic_error("Error: Evaluated result does not match the expected result on GPU.");
            }
        }

        break;
    }
}

int main() {
    size_t batch_size = 1 << 16;
    example_eval_func_gpu(GINX, STD128, batch_size);
    return 0;
}
