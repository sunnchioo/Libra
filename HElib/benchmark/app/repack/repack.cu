// #include "../utils/utils.h"
#include <algorithm>
#include <functional>
#include <random>
#include <vector>

#include "ckks_evaluator.cuh"
#include "conversion.cuh"
#include "cutfhe++.h"
#include "extract.cuh"
#include "fileio.h"
#include "phantom.h"
#include "repack.h"
#include "tlwevaluator.cuh"

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
    // int scale_bits = std::numeric_limits<typename lwe_enc_lvl::T>::digits - lwe_enc_lvl::plain_modulus_bit - 1;
    // double lwe_scale = pow(2.0, scale_bits);
    double lwe_scale = lwe_enc_lvl::Δ;

    TFHESecretKey sk;
    TFHEEvalKey ek;
    load_keys<BootstrappingKeyFFTLvl01, BootstrappingKeyFFTLvl02,
              KeySwitchingKeyLvl10, KeySwitchingKeyLvl20, KeySwitchingKeyLvl21>(sk, ek);

    // std::vector<int> lwe_num = {32, 256, 1024, 2048};
    // std::vector<int> lwe_num = {64, 512, 2048, 4096};
    std::vector<int> lwe_num = {32, 64, 128, 512, 1024, 2048};

    for (size_t i = 0; i < lwe_num.size(); i++) {
        int len = lwe_num[i];

        // repack (all to one)
        PhantomRLWE rlwer(1);
        rlwer.genLWE2RLWEGaloisKeys(len);
        PhantomCiphertext results;

        std::vector<TLWELvl1> h_lwes(len);
        std::vector<uint32_t> msg(h_lwes.size(), 1);
        for (size_t i = 0; i < msg.size(); i++) {
            h_lwes[i] = TFHEpp::tlweSymInt32Encrypt<lwe_enc_lvl>(msg[i], lwe_enc_lvl::α, lwe_scale, sk.key.get<lwe_enc_lvl>());
        }
        {
            CUDATimer timer("Repack", 0);
            timer.start();
            conver::repack(results, h_lwes, rlwer, sk);  // 11 levels
            timer.stop();
        }
    }

    // std::cout << "conversion level: " << results.coeff_modulus_size() << " chain: " << results.chain_index() << std::endl;
}