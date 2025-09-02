//==================================================================================
// BSD 2-Clause License
//
// Copyright (c) 2014-2022, NJIT, Duality Technologies Inc. and other contributors
//
// All rights reserved.
//
// Author TPOC: contact@openfhe.org
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//==================================================================================

/*
  Example for the FHEW scheme using the default bootstrapping method (GINX)
 */

#include "openfhe.h"
#include "binfhecontext.cuh"
#include "phantom.h"

using namespace lbcrypto;
using namespace phantom::bitwise;

void example_boolean_gpu(BINFHE_METHOD method, BINFHE_PARAMSET set, size_t batch_size) {
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
    bin_gates.emplace_back(0, 1);
    bin_gates.emplace_back(1, 0);
    bin_gates.emplace_back(1, 1);

    for (const auto &pair: bin_gates) {
        std::vector<LWECiphertext> v_ct1;
        std::vector<LWECiphertext> v_ct2;
        for (size_t j = 0; j < batch_size; j++) {
            v_ct1.push_back(cc.Encrypt(sk, pair.first));
            v_ct2.push_back(cc.Encrypt(sk, pair.second));
        }

        auto v_ct = gpu_cc.BatchGPUEvalBinGate(AND, v_ct1, v_ct2);

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

    for (const auto &pair: bin_gates) {
        std::vector<LWECiphertext> v_ct1;
        std::vector<LWECiphertext> v_ct2;
        for (size_t j = 0; j < batch_size; j++) {
            v_ct1.push_back(cc.Encrypt(sk, pair.first));
            v_ct2.push_back(cc.Encrypt(sk, pair.second));
        }

        auto v_ct = gpu_cc.BatchGPUEvalBinGate(NAND, v_ct1, v_ct2);

        for (size_t j = 0; j < batch_size; j++) {
            LWEPlaintext result;
            cc.Decrypt(sk, v_ct[j], &result);
            if (result != (!(pair.first & pair.second))) {
                std::cout << pair.first << " NAND " << pair.second << " = " << result << std::endl;
                throw std::logic_error("Error: NAND result does not match the expected result on GPU.");
            }
        }
    }
    std::cout << "NAND test passed." << std::endl;

    for (const auto &pair: bin_gates) {
        std::vector<LWECiphertext> v_ct1;
        std::vector<LWECiphertext> v_ct2;
        for (size_t j = 0; j < batch_size; j++) {
            v_ct1.push_back(cc.Encrypt(sk, pair.first));
            v_ct2.push_back(cc.Encrypt(sk, pair.second));
        }

        auto v_ct = gpu_cc.BatchGPUEvalBinGate(OR, v_ct1, v_ct2);

        for (size_t j = 0; j < batch_size; j++) {
            LWEPlaintext result;
            cc.Decrypt(sk, v_ct[j], &result);
            if (result != (pair.first | pair.second)) {
                std::cout << pair.first << " OR " << pair.second << " = " << result << std::endl;
                throw std::logic_error("Error: OR result does not match the expected result on GPU.");
            }
        }
    }
    std::cout << "OR test passed." << std::endl;

    for (const auto &pair: bin_gates) {
        std::vector<LWECiphertext> v_ct1;
        std::vector<LWECiphertext> v_ct2;
        for (size_t j = 0; j < batch_size; j++) {
            v_ct1.push_back(cc.Encrypt(sk, pair.first));
            v_ct2.push_back(cc.Encrypt(sk, pair.second));
        }

        auto v_ct = gpu_cc.BatchGPUEvalBinGate(XOR, v_ct1, v_ct2);

        for (size_t j = 0; j < batch_size; j++) {
            LWEPlaintext result;
            cc.Decrypt(sk, v_ct[j], &result);
            if (result != (!pair.first != !pair.second)) {
                std::cout << pair.first << " XOR " << pair.second << " = " << result << std::endl;
                throw std::logic_error("Error: XOR result does not match the expected result on GPU.");
            }
        }
    }
    std::cout << "XOR test passed." << std::endl;
}

int main() {
    example_boolean_gpu(AP, STD128, 10);
    example_boolean_gpu(AP, STD128_Binary, 10);
    example_boolean_gpu(GINX, STD128, 10);
    example_boolean_gpu(GINX, STD128_Binary, 10);
//    example_boolean_gpu(AP, T_1024_30, 10);
//    example_boolean_gpu(GINX, T_1024_30, 10);
    if (std::is_same<BasicInteger, uint64_t>::value) {
        example_boolean_gpu(AP, STD128Q_3, 10);
        example_boolean_gpu(GINX, STD128Q_3, 10);
        example_boolean_gpu(GINX, STD128Q_3_Binary, 10);
        example_boolean_gpu(AP, T_1024_36, 10);
        example_boolean_gpu(GINX, T_1024_36, 10);
        example_boolean_gpu(GINX, T_1024_36_Binary, 10);
        example_boolean_gpu(AP, T_2048_50, 10);
        example_boolean_gpu(GINX, T_2048_50, 10);
        example_boolean_gpu(GINX, T_2048_50_Binary, 10);
    }
    return 0;
}
