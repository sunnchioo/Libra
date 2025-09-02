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

void example_boolean_cpu(BINFHE_METHOD method, BINFHE_PARAMSET set) {
    // Sample Program: Step 1: Set CryptoContext

    auto cc = BinFHEContext();

    // STD128 is the security level of 128 bits of security based on LWE Estimator
    // and HE standard. Other common options are TOY, MEDIUM, STD192, and STD256.
    // MEDIUM corresponds to the level of more than 100 bits for both quantum and
    // classical computer attacks.
    cc.GenerateBinFHEContext(set, method);

    // Sample Program: Step 2: Key Generation

    // Generate the secret key
    auto sk = cc.KeyGen();

    std::cout << "Generating the bootstrapping keys on CPU..." << std::endl;

    // Generate the bootstrapping keys (refresh and switching keys)
    cc.BTKeyGen(sk);

    std::cout << "Completed the key generation." << std::endl;

    std::vector<std::pair<LWEPlaintext, LWEPlaintext>> bin_gates;
    bin_gates.emplace_back(0, 0);
    bin_gates.emplace_back(0, 1);
    bin_gates.emplace_back(1, 0);
    bin_gates.emplace_back(1, 1);

    for (const auto &pair: bin_gates) {
        auto ct1 = cc.Encrypt(sk, pair.first);
        auto ct2 = cc.Encrypt(sk, pair.second);
        auto ct = cc.EvalBinGate(AND, ct1, ct2);
        LWEPlaintext result;
        cc.Decrypt(sk, ct, &result);
        std::cout << pair.first << " AND " << pair.second << " = " << result << std::endl;
        if (result != (pair.first & pair.second))
            throw std::logic_error("Error: Evaluated result does not match the expected result on CPU.");
    }

    for (const auto &pair: bin_gates) {
        auto ct1 = cc.Encrypt(sk, pair.first);
        auto ct2 = cc.Encrypt(sk, pair.second);
        auto ct = cc.EvalBinGate(OR, ct1, ct2);
        LWEPlaintext result;
        cc.Decrypt(sk, ct, &result);
        std::cout << pair.first << " OR " << pair.second << " = " << result << std::endl;
        if (result != (pair.first | pair.second))
            throw std::logic_error("Error: Evaluated result does not match the expected result on CPU.");
    }

    for (const auto &pair: bin_gates) {
        auto ct1 = cc.Encrypt(sk, pair.first);
        auto ct2 = cc.Encrypt(sk, pair.second);
        auto ct = cc.EvalBinGate(XOR, ct1, ct2);
        LWEPlaintext result;
        cc.Decrypt(sk, ct, &result);
        std::cout << pair.first << " XOR " << pair.second << " = " << result << std::endl;
        if (result != (!pair.first != !pair.second))
            throw std::logic_error("Error: Evaluated result does not match the expected result on CPU.");
    }
}

void example_boolean_gpu(BINFHE_METHOD method, BINFHE_PARAMSET set) {
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
        auto ct1 = cc.Encrypt(sk, pair.first);
        auto ct2 = cc.Encrypt(sk, pair.second);
        auto ct = gpu_cc.GPUEvalBinGate(AND, ct1, ct2);
        LWEPlaintext result;
        cc.Decrypt(sk, ct, &result);
        std::cout << pair.first << " AND " << pair.second << " = " << result << std::endl;
        if (result != (pair.first & pair.second))
            throw std::logic_error("Error: Evaluated result does not match the expected result on GPU.");
    }

    for (const auto &pair: bin_gates) {
        auto ct1 = cc.Encrypt(sk, pair.first);
        auto ct2 = cc.Encrypt(sk, pair.second);
        auto ct = gpu_cc.GPUEvalBinGate(NAND, ct1, ct2);
        LWEPlaintext result;
        cc.Decrypt(sk, ct, &result);
        std::cout << pair.first << " NAND " << pair.second << " = " << result << std::endl;
        if (result != (!(pair.first & pair.second)))
            throw std::logic_error("Error: Evaluated result does not match the expected result on GPU.");
    }

    for (const auto &pair: bin_gates) {
        auto ct1 = cc.Encrypt(sk, pair.first);
        auto ct2 = cc.Encrypt(sk, pair.second);
        auto ct = gpu_cc.GPUEvalBinGate(OR, ct1, ct2);
        LWEPlaintext result;
        cc.Decrypt(sk, ct, &result);
        std::cout << pair.first << " OR " << pair.second << " = " << result << std::endl;
        if (result != (pair.first | pair.second))
            throw std::logic_error("Error: Evaluated result does not match the expected result on GPU.");
    }

    for (const auto &pair: bin_gates) {
        auto ct1 = cc.Encrypt(sk, pair.first);
        auto ct2 = cc.Encrypt(sk, pair.second);
        auto ct = gpu_cc.GPUEvalBinGate(XOR, ct1, ct2);
        LWEPlaintext result;
        cc.Decrypt(sk, ct, &result);
        std::cout << pair.first << " XOR " << pair.second << " = " << result << std::endl;
        if (result != (!pair.first != !pair.second))
            throw std::logic_error("Error: Evaluated result does not match the expected result on GPU.");
    }
}

int main() {
    example_boolean_cpu(AP, STD128);
    example_boolean_cpu(GINX, STD128);
    if (std::is_same<BasicInteger, uint64_t>::value) {
        example_boolean_cpu(AP, STD128Q_3);
        example_boolean_cpu(GINX, STD128Q_3);
    }

    example_boolean_gpu(AP, STD128);
    example_boolean_gpu(GINX, STD128);
    example_boolean_gpu(GINX, STD128_Binary);
//    example_boolean_gpu(AP, T_1024_30);
//    example_boolean_gpu(GINX, T_1024_30);
    if (std::is_same<BasicInteger, uint64_t>::value) {
        example_boolean_gpu(AP, STD128Q_3);
        example_boolean_gpu(GINX, STD128Q_3);
        example_boolean_gpu(GINX, STD128Q_3_Binary);
        example_boolean_gpu(AP, T_1024_36);
        example_boolean_gpu(GINX, T_1024_36);
        example_boolean_gpu(GINX, T_1024_36_Binary);
        example_boolean_gpu(AP, T_2048_50);
        example_boolean_gpu(GINX, T_2048_50);
        example_boolean_gpu(GINX, T_2048_50_Binary);
    }
    return 0;
}
