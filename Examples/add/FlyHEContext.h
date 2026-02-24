#ifndef FLYHE_CONTEXT_H
#define FLYHE_CONTEXT_H

#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "bootstrapping/Bootstrapper.cuh"
#include "ckks_evaluator.cuh"
#include "phantom.h"
#include "utils.cuh"

#include "cutfhe++.h"
#include "tlwevaluator.cuh"

#include "conversion.cuh"
#include "trlwevaluator.cuh"

// Namespaces & Aliases
using namespace std;
using namespace phantom;
using namespace phantom::arith;
using namespace phantom::util;
using namespace rlwe;

using namespace cuTFHEpp;
using namespace cuTFHEpp::util;
using namespace conver;

using CUDATimer = phantom::util::CUDATimer;

// namespace for lvl1
using lwe_enc_lvl = Lvl1;
using lwe_res_lvl = Lvl1;
using lwe_pbs_flvl = Lvl10;
using lwe_pbs_llvl = Lvl01;

// 1. 配置对象：FlyHEConfig (管理所有的同态参数)
enum class FlyMode {
    SIMD,
    SISD,
    CROSS
};

struct FlyHEConfig {
    FlyMode mode = FlyMode::SIMD;
    long logN = 0;
    long logn = 0;
    int remaining_levels = 16;
    bool bootstrapping_enabled = false;

    // === CKKS 基础参数 ===
    int logp = 46;
    int logq = 51;
    int log_special_prime = 51;
    int secret_key_hamming_weight = 192;
    int boot_levels = 14;
    int special_prime_len = 4;

    // === CKKS 自举参数 ===
    long boundary_K = 25;
    long deg = 59;
    long scale_factor = 2;
    long inverse_deg = 1;
    long loge = 10;

    // 静态工厂方法
    static FlyHEConfig CreateSIMD(long logN, long logn, int remaining_lvl = 16, bool enable_boot = false) {
        FlyHEConfig cfg;
        cfg.mode = FlyMode::SIMD;
        cfg.logN = logN;
        cfg.logn = logn;
        cfg.remaining_levels = remaining_lvl;
        cfg.bootstrapping_enabled = enable_boot;
        return cfg;
    }

    static FlyHEConfig CreateSISD() {
        FlyHEConfig cfg;
        cfg.mode = FlyMode::SISD;
        return cfg;
    }

    static FlyHEConfig CreateCROSS(long logN, long logn, int remaining_lvl = 16, bool enable_boot = true) {
        FlyHEConfig cfg;
        cfg.mode = FlyMode::CROSS;
        cfg.logN = logN;
        cfg.logn = logn;
        cfg.remaining_levels = remaining_lvl;
        cfg.bootstrapping_enabled = enable_boot;
        return cfg;
    }
};

// 2. 上下文对象：FlyHEContext
template <typename TFHELevel = lwe_enc_lvl>
class FlyHEContext {
public:
    using TFHELevelType = TFHELevel;

    FlyHEConfig config;
    FlyMode mode;

    shared_ptr<PhantomContext> context = nullptr;
    shared_ptr<PhantomSecretKey> secret_key = nullptr;
    shared_ptr<PhantomPublicKey> public_key = nullptr;
    shared_ptr<PhantomRelinKey> relin_keys = nullptr;
    shared_ptr<PhantomGaloisKey> galois_keys = nullptr;
    shared_ptr<PhantomCKKSEncoder> encoder = nullptr;
    shared_ptr<CKKSEvaluator> ckks_evaluator = nullptr;
    shared_ptr<Bootstrapper> bootstrapper = nullptr;

    shared_ptr<TFHESecretKey> tfhe_secret_key = nullptr;
    shared_ptr<TFHEEvalKey> tfhe_eval_key = nullptr;
    shared_ptr<tlwevaluator<TFHELevelType>> tfhe_evaluator = nullptr;

    double ckks_scale = 0.0;
    double lwe_scale = 0.0;

    long logN = 0;
    long logn = 0;
    int remaining_levels = 0;
    bool bootstrapping_enabled = false;
    int total_levels = 0;

    cudaStream_t stream = nullptr;

    FlyHEContext(const FlyHEConfig& cfg)
        : config(cfg),
          mode(cfg.mode),
          logN(cfg.logN),
          logn(cfg.logn),
          remaining_levels(cfg.remaining_levels),
          bootstrapping_enabled(cfg.bootstrapping_enabled) {

        cout << "\n========================================" << endl;

        switch (mode) {
        case FlyMode::SIMD:
            cout << ">>> [Init] Mode: SIMD" << endl;
            check_ckks_params();
            init_ckks_context();
            break;

        case FlyMode::SISD:
            cout << ">>> [Init] Mode: SISD" << endl;
            init_tfhectx_cross();
            break;

        case FlyMode::CROSS:
            cout << ">>> [Init] Mode: CROSS" << endl;
            check_ckks_params();
            init_ckks_context();
            init_tfhectx_cross();
            break;
        }

        cout << ">>> Context Ready." << endl;
        cout << "========================================\n"
             << endl;
    }

private:
    void check_ckks_params() {
        if (logN == 0 || logn == 0) {
            throw std::invalid_argument("[Error] SIMD/CROSS mode requires valid logN and logn parameters!");
        }
    }

    bool has_ckks() const { return ckks_evaluator != nullptr; }
    bool has_tfhe() const { return tfhe_evaluator != nullptr; }

    inline void find_galois_steps_for_bsgs(std::vector<int>& steps, size_t row, bool aggr_step = false) {
        if (aggr_step) {
            size_t log_slots = ceil(log2(row));
            for (size_t j = 1; j < (1 << log_slots); j <<= 1) {
                int val = (int)j;
                if (std::find(steps.begin(), steps.end(), val) == steps.end()) {
                    steps.push_back(val);
                }
            }
        } else {
            int val_1 = 1;
            if (std::find(steps.begin(), steps.end(), val_1) == steps.end()) {
                steps.push_back(val_1);
            }

            size_t min_len = std::min(row, (size_t)lwe_enc_lvl::n);
            size_t g_tilde = CeilSqrt(min_len);
            size_t b_tilde = CeilDiv(min_len, g_tilde);

            for (size_t b = 1; b < b_tilde && g_tilde * b < min_len; ++b) {
                int val = (int)(b * g_tilde);
                if (std::find(steps.begin(), steps.end(), val) == steps.end()) {
                    steps.push_back(val);
                }
            }
            if (row < (size_t)lwe_enc_lvl::n) {
                size_t gama = std::log2((size_t)lwe_enc_lvl::n / row);
                for (size_t j = 0; j < gama; j++) {
                    int val = (int)((1U << j) * row);
                    if (std::find(steps.begin(), steps.end(), val) == steps.end()) {
                        steps.push_back(val);
                    }
                }
            }
        }
    }

    void init_ckks_context() {
        cout << "  [CKKS] Setting up Poly Degree N=2^" << logN << "..." << endl;

        long sparse_slots = (1 << (logN - 1));

        int boot_level_cnt = bootstrapping_enabled ? config.boot_levels : 0;
        total_levels = remaining_levels + boot_level_cnt;

        vector<int> coeff_bit_vec;
        coeff_bit_vec.push_back(config.logq);
        for (int i = 0; i < remaining_levels; i++)
            coeff_bit_vec.push_back(config.logp);
        for (int i = 0; i < boot_level_cnt; i++)
            coeff_bit_vec.push_back(config.logq);
        for (int i = 0; i < config.special_prime_len; i++)
            coeff_bit_vec.push_back(config.log_special_prime);

        EncryptionParameters params(scheme_type::ckks);
        size_t poly_modulus_degree = (size_t)(1 << logN);
        ckks_scale = pow(2.0, config.logp);

        params.set_poly_modulus_degree(poly_modulus_degree);
        params.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, coeff_bit_vec));
        params.set_secret_key_hamming_weight(config.secret_key_hamming_weight);
        params.set_sparse_slots(sparse_slots);
        params.set_special_modulus_size(config.special_prime_len);

        context = make_shared<PhantomContext>(params);
        secret_key = make_shared<PhantomSecretKey>(*context);
        public_key = make_shared<PhantomPublicKey>(secret_key->gen_publickey(*context));
        relin_keys = make_shared<PhantomRelinKey>(secret_key->gen_relinkey(*context));
        galois_keys = make_shared<PhantomGaloisKey>(secret_key->create_galois_keys(*context));

        encoder = make_shared<PhantomCKKSEncoder>(*context);
        ckks_evaluator = make_shared<CKKSEvaluator>(context.get(), public_key.get(), secret_key.get(),
                                                    encoder.get(), relin_keys.get(), galois_keys.get(), ckks_scale);

        vector<int> gal_steps_vector;
        gal_steps_vector.push_back(0);
        for (int i = 0; i < logn; i++) {
            gal_steps_vector.push_back((1 << i));
        }

        if (bootstrapping_enabled) {
            bootstrapper = make_shared<Bootstrapper>(
                config.loge, logn, logN - 1, total_levels, ckks_scale,
                config.boundary_K, config.deg, config.scale_factor, config.inverse_deg, ckks_evaluator.get());

            bootstrapper->prepare_mod_polynomial();
            bootstrapper->addLeftRotKeys_Linear_to_vector_3(gal_steps_vector);
            bootstrapper->slot_vec.push_back(logn);
            bootstrapper->generate_LT_coefficient_ori_3();
        }

        if (mode == FlyMode::CROSS) {
            find_galois_steps_for_bsgs(gal_steps_vector, 128);
        }

        ckks_evaluator->decryptor.create_galois_keys_from_steps(gal_steps_vector, *(ckks_evaluator->galois_keys));
        cout << "  [CKKS] Initialized." << endl;

        if (phantom::util::global_variables::default_stream) {
            this->stream = phantom::util::global_variables::default_stream->get_stream();
        } else {
            std::cerr << "[Warning] Default stream is nullptr! Using legacy default stream (0)." << std::endl;
            this->stream = 0;
        }
    }

    void init_tfhectx_cross() {
        cout << "  [TFHE] Initializing LWE Context..." << endl;

        lwe_scale = TFHELevelType::Δ;
        cout << "  -> LWE Scale: " << lwe_scale << endl;

        tfhe_secret_key = make_shared<TFHESecretKey>();
        tfhe_eval_key = make_shared<TFHEEvalKey>();

        if constexpr (std::is_same_v<TFHELevelType, Lvl1>) {
            cout << "  -> Loading Keys for Lvl1..." << endl;
            load_keys<BootstrappingKeyFFTLvl01, KeySwitchingKeyLvl10>(*tfhe_secret_key, *tfhe_eval_key);
        } else {
            cout << "  -> Loading Keys for Lvl2 (Full Set)..." << endl;
            load_keys<BootstrappingKeyFFTLvl01, BootstrappingKeyFFTLvl02,
                      KeySwitchingKeyLvl10, KeySwitchingKeyLvl20, KeySwitchingKeyLvl21>(*tfhe_secret_key, *tfhe_eval_key);
        }

        tfhe_evaluator = make_shared<tlwevaluator<TFHELevelType>>(tfhe_secret_key.get(), tfhe_eval_key.get(), lwe_scale);

        cout << "  [TFHE] Initialized." << endl;
    }
};

#endif // FLYHE_CONTEXT_H