#include <cmath>
#include <cstdio>
#include <cutfhe++.h>
#include <iostream>
#include <random>

#include "cutfhe++.h"

using namespace cuTFHEpp;
using namespace cuTFHEpp::util;

template <typename LvlXY, typename LvlYZ,
          typename LvlX = LvlXY::domainP, typename LvlY = LvlXY::targetP, typename LvlZ = LvlYZ::targetP>
void TestGateBoostrapping(const Pointer<Context> &context, const TFHESecretKey &sk, const size_t num_test) {
    static_assert(std::is_same<LvlY, typename LvlYZ::domainP>::value, "Invalid LvlY");

    std::random_device seed_gen;
    std::default_random_engine engine(seed_gen());
    std::uniform_int_distribution<uint32_t> binary(0, 1);

    using P = std::conditional_t<isLvlCover<LvlX, LvlZ>(), LvlX, LvlZ>;

    Pointer<BootstrappingData<LvlYZ>> bs_data(num_test);
    std::vector<Pointer<cuTLWE<P>>> tlwe_data;
    tlwe_data.reserve(2);
    for (size_t i = 0; i < 2; ++i) {
        tlwe_data.emplace_back(num_test);
    }

    TFHEpp::TLWE<LvlX> *d_tlwe = tlwe_data[0]->template get<LvlX>();
    TFHEpp::TLWE<LvlZ> *d_res = tlwe_data[1]->template get<LvlZ>();

    std::vector<TFHEpp::TLWE<LvlX>> tlwe(num_test);
    std::vector<TFHEpp::TLWE<LvlZ>> res(num_test);
    std::vector<bool> p(num_test);

    for (int test = 0; test < num_test; test++) {
        p[test] = binary(engine) > 0;
        std::cout << "p[test]: " << p[test] << std::endl;

        tlwe[test] = TFHEpp::tlweSymEncrypt<LvlX>(p[test] ? LvlX::μ : -LvlX::μ, LvlX::α, sk.key.get<LvlX>());
    }

    CUDA_CHECK_RETURN(cudaMemcpy(d_tlwe, tlwe.data(), sizeof(TFHEpp::TLWE<LvlX>) * num_test, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    RECORD_TIME_START(start, stop);
    GateBootstrapping<LvlXY, LvlYZ>(context.get(), bs_data, d_res, d_tlwe, num_test);
    float time = RECORD_TIME_END(start, stop);
    CUDA_CHECK_ERROR();

    std::cout << std::fixed << "GateBootstrapping: " << time << "ms, per gate = " << time / num_test << "ms" << std::endl;

    CUDA_CHECK_RETURN(cudaMemcpy(res.data(), d_res, sizeof(TFHEpp::TLWE<LvlZ>) * num_test, cudaMemcpyDeviceToHost));

    for (int test = 0; test < num_test; test++) {
        bool p2 = TFHEpp::tlweSymDecrypt<LvlZ>(res[test], sk.key.get<LvlZ>());
        std::cout << "decrypt boot: " << p2 << std::endl;

        assert(p2 == p[test]);
    }
}

template <typename LvlXY, typename LvlYZ,
          typename LvlX = LvlXY::domainP, typename LvlY = LvlXY::targetP, typename LvlZ = LvlYZ::targetP>
void TestIntBoostrapping(const Pointer<Context> &context, const TFHESecretKey &sk) {
    static_assert(std::is_same<LvlY, typename LvlYZ::domainP>::value, "Invalid LvlY");

    using lwe_enc_lvl = Lvl1;
    using P = std::conditional_t<isLvlCover<LvlX, LvlZ>(), LvlX, LvlZ>;

    std::vector<lwe_enc_lvl::T> msg = {0, 1, 2, 3};
    size_t len_lwe = msg.size();
    std::vector<TLWELvl1> h_lwes(len_lwe);
    Pointer<cuTLWE<lwe_enc_lvl>> d_lwes(len_lwe);
    Pointer<cuTLWE<Lvl1>> d_lwes_res(len_lwe);

    for (size_t i = 0; i < len_lwe; i++) {
        h_lwes[i] = TFHEpp::tlweSymIntEncrypt<lwe_enc_lvl>(msg[i], lwe_enc_lvl::α, sk.key.get<lwe_enc_lvl>());
        // auto lwe_dec_num = TFHEpp::tlweSymIntDecrypt<lwe_enc_lvl>(h_lwes[i], sk.key.get<lwe_enc_lvl>());
        // std::cout << "decrypt: " << lwe_dec_num << " ground: " << msg[i] << std::endl;
    }

    TFHEpp::TLWE<lwe_enc_lvl> *d_dest = d_lwes->template get<lwe_enc_lvl>();
    TFHEpp::TLWE<lwe_enc_lvl> *h_src = h_lwes.data();
    CUDA_CHECK_RETURN(cudaMemcpy(d_dest, h_src, len_lwe * sizeof(TFHEpp::TLWE<lwe_enc_lvl>), cudaMemcpyHostToDevice));

    Pointer<BootstrappingData<LvlYZ>> bs_data(len_lwe);

    TFHEpp::TLWE<LvlX> *d_tlwe = d_lwes->template get<LvlX>();
    TFHEpp::TLWE<LvlZ> *d_res = d_lwes_res->template get<LvlZ>();

    cudaEvent_t start, stop;
    RECORD_TIME_START(start, stop);
    GateBootstrapping<LvlXY, LvlYZ>(context.get(), bs_data, d_res, d_tlwe, len_lwe);
    float time = RECORD_TIME_END(start, stop);
    CUDA_CHECK_ERROR();

    std::cout << std::fixed << "GateBootstrapping: " << time << "ms, per gate = " << time / len_lwe << "ms" << std::endl;

    CUDA_CHECK_RETURN(cudaMemcpy(h_lwes.data(), d_res, sizeof(TFHEpp::TLWE<LvlZ>) * len_lwe, cudaMemcpyDeviceToHost));

    for (int test = 0; test < len_lwe; test++) {
        auto tmep = TFHEpp::tlweSymIntDecrypt<LvlZ>(h_lwes[test], sk.key.get<LvlZ>());
        std::cout << "decrypt boot: " << tmep << std::endl;

        bool p2 = (tmep != 0);
        assert(p2 == p[test]);
    }
}

int main(int argc, char **argv) {
    cudaSetDevice(DEVICE_ID);

    TFHESecretKey sk;
    TFHEEvalKey ek;

    load_keys<BootstrappingKeyFFTLvl01, BootstrappingKeyFFTLvl02,
              KeySwitchingKeyLvl10, KeySwitchingKeyLvl20>(sk, ek);
    // load_keys<BootstrappingKeyFFTLvl01, KeySwitchingKeyLvl10>(sk, ek);

    std::cout << "copy eval key to GPU" << std::endl;
    Pointer<Context> context(ek);
    std::cout << "eval key is copied to GPU" << std::endl;

    // const size_t num_test = 4;

    // for (size_t i = 1; i < (1 << 16); i = i * 2) {
    //     TestGateBoostrapping<Lvl10, Lvl01>(context, sk, i);
    // }

    // TestGateBoostrapping<Lvl10, Lvl01>(context, sk, num_test);
    // TestGateBoostrapping<Lvl20, Lvl02>(context, sk, num_test);
    TestIntBoostrapping<Lvl10, Lvl01>(context, sk);
    return 0;
}
