#pragma once

#include <chrono>

#include "extract.cuh"
#include "repack.h"
#include "rlwe.cuh"
#include "tlwe.h"

namespace conver {
    using namespace cuTFHEpp::util;

    void repack(std::vector<PhantomCiphertext> &results,
                std::vector<std::vector<TLWELvl1>> &pred_cres,
                std::vector<std::vector<uint32_t>> &pred_res, PhantomRLWE &rlwe,
                TFHESecretKey &sk,
                double &conversion_time,
                bool nocheck);

    void repack(std::vector<PhantomCiphertext> &results,
                std::vector<std::vector<TLWELvl1>> &pred_cres,
                std::vector<std::vector<uint32_t>> &pred_res, PhantomRLWE &rlwe,
                TFHESecretKey &sk,
                double &conversion_time);

    void repack(PhantomCiphertext &results,
                std::vector<TLWELvl1> &pred_cres,
                PhantomRLWE &rlwe,
                TFHESecretKey &sk);

    template <typename Lvl>
    void extract(trlwevaluator &trlwer, LWEContext *lwe_context, PhantomCiphertext &rlwe_cipher, Pointer<cuTLWE<Lvl>> &lwe_ciphers, std::vector<size_t> &extract_indices, GPUDecomposedLWEKSwitchKey &extractKey) {
        RLWEToLWEs<Lvl>(trlwer, lwe_context, rlwe_cipher, lwe_ciphers, extract_indices, extractKey);
    }
    // template void extract<TFHEpp::lvl1param>(trlwevaluator &, PhantomCiphertext &, cuTFHEpp::util::Pointer<cuTFHEpp::cuTLWE<TFHEpp::lvl1param>> &, std::vector<size_t> &, GPUDecomposedLWEKSwitchKey &);
    template void extract<TFHEpp::lvl1Lparam>(trlwevaluator &, LWEContext *, PhantomCiphertext &, cuTFHEpp::util::Pointer<cuTFHEpp::cuTLWE<TFHEpp::lvl1Lparam>> &, std::vector<size_t> &, GPUDecomposedLWEKSwitchKey &);
}  // namespace conver
