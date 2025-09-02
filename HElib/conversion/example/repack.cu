#include <algorithm>
#include <functional>
#include <random>
#include <vector>

#include "ckks_evaluator.cuh"
#include "conversion.cuh"
#include "cutfhe++.h"
#include "fileio.h"
#include "phantom.h"
#include "tlwevaluator.cuh"

// #include "extract.cuh"
#include "repack.h"

using namespace rlwe;
using namespace cuTFHEpp;
using namespace cuTFHEpp::util;
using namespace conver;

using CUDATimer = phantom::util::CUDATimer;

void print_matrix(std::vector<std::vector<double>> &matrix, std::string str) {
    std::cout << str << " : " << std::endl;
    for (size_t i = 0; i < matrix.size(); i++) {
        std::cout << "line " << i << " : ";
        for (size_t j = 0; j < 10; j++) {
            std::cout << matrix[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main() {
    // tfhe init
    std::cout << "Setting LWE Parameters..." << endl;
    using lwe_enc_lvl = Lvl1;
    int scale_bits = 26;
    double lwe_scale = pow(2.0, scale_bits);
    // double lwe_scale = lwe_enc_lvl::Δ;

    TFHESecretKey sk;  // 定义时已经初始化
    TFHEEvalKey ek;
    load_keys<BootstrappingKeyFFTLvl01, BootstrappingKeyFFTLvl02,
              KeySwitchingKeyLvl10, KeySwitchingKeyLvl20, KeySwitchingKeyLvl21>(sk, ek);

    // repack (all to one)
    std::vector<uint32_t> msg = {0, 1, 2, 3};

    PhantomRLWE rlwer(msg.size());
    rlwer.genLWE2RLWEGaloisKeys();
    PhantomCiphertext results;

    std::vector<TLWELvl1> h_lwes(msg.size());
    for (size_t i = 0; i < msg.size(); i++) {
        h_lwes[i] = TFHEpp::tlweSymInt32Encrypt<lwe_enc_lvl>(msg[i], lwe_enc_lvl::α, lwe_scale, sk.key.get<lwe_enc_lvl>());
        std::cout << "check encrypt: " << TFHEpp::tlweSymInt32Decrypt<lwe_enc_lvl>(h_lwes[i], lwe_scale, sk.key.get<lwe_enc_lvl>()) << std::endl;
    }

    {
        CUDATimer timer("Repack", 0);
        timer.start();
        conver::repack(results, h_lwes, rlwer, sk);  // 11 levels
        timer.stop();
    }

    rlwer.print_decrypted_ct(results, 10, "Repack results");
    std::cout << "conversion level: " << results.coeff_modulus_size() << " chain: " << results.chain_index() << std::endl;  // 13 12
}