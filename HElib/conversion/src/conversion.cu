#include "conversion.cuh"

// This code was modified based on HE3DB: https://github.com/zhouzhangwalker/HE3DB

namespace conver {
    using namespace cuTFHEpp::util;

    void repack(
        std::vector<PhantomCiphertext> &results,
        std::vector<std::vector<TLWELvl1>> &lwes,
        std::vector<std::vector<uint32_t>> &pred_res,
        PhantomRLWE &rlwe,
        TFHESecretKey &sk,
        double &conversion_time,
        bool nocheck) {
        size_t groupby_num = lwes.size();
        size_t slots_count = lwes[0].size();

        results.resize(groupby_num);
        LTPreKey pre_key = rlwe.genPreKey(sk, Lvl1::n);
        // conversion
        std::cout << "Starting Conversion..." << std::endl;
        std::chrono::system_clock::time_point start, end;
        start = std::chrono::system_clock::now();
        for (size_t i = 0; i < groupby_num; i++) {
            LWEsToRLWE(*rlwe.context, results[i], lwes[i], pre_key, rlwe.scale,
                       std::pow(2., rlwe.modq_bits), std::pow(2., rlwe.modulus_bits - rlwe.modq_bits),
                       *rlwe.ckks_encoder, *rlwe.galois_keys, *rlwe.relin_keys);
            HomRound(*rlwe.context, results[i], results[i].scale(), *rlwe.ckks_encoder,
                     *rlwe.relin_keys);
        }
        end = std::chrono::system_clock::now();
        conversion_time =
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
                .count();

        std::cout << "[LWEsToRLWE] Conversion Time: " << conversion_time << " ms" << std::endl;

        if (!nocheck) {
            std::vector<PhantomPlaintext> plain(groupby_num);
            std::vector<std::vector<double>> computed(groupby_num,
                                                      std::vector<double>(slots_count));
            for (size_t i = 0; i < groupby_num; i++) {
                rlwe.secret_key->decrypt(*rlwe.context, results[i], plain[i]);
                pack_decode(*rlwe.context, computed[i], plain[i], *rlwe.ckks_encoder);
            }

            std::vector<double> err(groupby_num, 0.);
            for (size_t i = 0; i < groupby_num; i++) {
                for (size_t j = 0; j < slots_count; ++j) {
                    err[i] += std::abs(computed[i][j] - pred_res[i][j]);
                }

                printf("Repack average error in %02ld = %f ~ 2^%.1f\n", i,
                       err[i] / slots_count, std::log2(err[i] / slots_count));
            }
        }
    }

    void repack(
        std::vector<PhantomCiphertext> &results,
        std::vector<std::vector<TLWELvl1>> &lwes,
        std::vector<std::vector<uint32_t>> &pred_res,
        PhantomRLWE &rlwe,
        TFHESecretKey &sk,
        double &conversion_time) {
        repack(results, lwes, pred_res, rlwe, sk, conversion_time, true);
    }

    void repack(
        PhantomCiphertext &results,
        std::vector<TLWELvl1> &lwes,
        PhantomRLWE &rlwe,
        TFHESecretKey &sk) {
        LTPreKey pre_key = rlwe.genPreKey(sk, Lvl1::n);

        LWEsToRLWE(*rlwe.context, results, lwes, pre_key, rlwe.scale,
                   std::pow(2., rlwe.modq_bits), std::pow(2., rlwe.modulus_bits - rlwe.modq_bits),
                   *rlwe.ckks_encoder, *rlwe.galois_keys, *rlwe.relin_keys);
        // LWEsToRLWE(*rlwe.context, results, lwes, pre_key, rlwe.scale,
        //            std::pow(2., rlwe.modq_bits), std::pow(2., rlwe.modulus_bits - rlwe.modq_bits),
        //            *rlwe.ckks_encoder, *rlwe.galois_keys, *rlwe.relin_keys, *rlwe.secret_key);
    }
} // namespace conver
