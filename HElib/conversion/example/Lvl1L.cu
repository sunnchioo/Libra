#include <algorithm>
#include <functional>
#include <random>
#include <vector>

#include "conversion.cuh"
#include "tlwevaluator.cuh"
#include "trlwevaluator.cuh"

using namespace rlwe;
using namespace cuTFHEpp;
using namespace cuTFHEpp::util;
using namespace phantom;

void random_real(std::vector<double> &vec, size_t size) {
    // std::random_device rn;
    // std::mt19937_64 rnd(rn());
    std::mt19937_64 rnd(42);
    thread_local std::uniform_real_distribution<double> distribution(-1, 1);

    vec.reserve(size);

    for (size_t i = 0; i < size; i++) {
        vec[i] = distribution(rnd);
    }
}

int main() {
    std::cout << "Setting LWE..." << endl;
    using lwe_enc_lvl = Lvl1L;
    // double lwe_scale = lwe_enc_lvl::Δ;
    double lwe_scale = std::pow(2.0, 51);

    TFHESecretKey lwe_sk;
    TFHEEvalKey lwe_ek;
    load_keys<BootstrappingKeyFFTLvl01, BootstrappingKeyFFTLvl02,
              KeySwitchingKeyLvl10, KeySwitchingKeyLvl20, KeySwitchingKeyLvl21>(lwe_sk, lwe_ek);
    tlwevaluator<lwe_enc_lvl> tlwer(&lwe_sk, &lwe_ek, lwe_scale);

    std::vector<uint32_t> msg = {0, 1, 2, 3};

    std::vector<TLWELvl1L> h_lwes(msg.size());
    for (size_t i = 0; i < msg.size(); i++) {
        h_lwes[i] = tlwer.tlweSymIntEncrypt(msg[i], lwe_enc_lvl::α, lwe_scale, lwe_sk.key.get<lwe_enc_lvl>());
        std::cout << "check encrypt: " << tlwer.tlweSymIntDecrypt(h_lwes[i], lwe_scale, lwe_sk.key.get<lwe_enc_lvl>()) << std::endl;
        // tlwer.print_lwe_ct_vec(h_lwes, lwe_scale, "check: ");
    }
}