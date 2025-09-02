#include "binfhecontext.cuh"
#include "kernel.cuh"
#include "ntt.cuh"
#include "openfhe.h"
#include <cassert>

using namespace lbcrypto;

namespace phantom::bitwise {

    void GPUBinFHEContext::GPUBTKeyGen(lbcrypto::ConstLWEPrivateKey &sk, lbcrypto::KEYGEN_MODE keygenMode) {
        if (d_BTKey_map_.empty()) {
            auto temp = params_->GetRingGSWParams()->GetBaseG(); // m_baseG
            d_BTKey_map_[temp] = gpu_binfhescheme_->GPUKeyGen(params_, sk, stream_wrapper_.get_stream(), keygenMode);
        }
    }

    LWECiphertext GPUBinFHEContext::GPUEvalBinGate(const BINGATE gate,
                                                   ConstLWECiphertext &ct1,
                                                   ConstLWECiphertext &ct2) const {
        const auto &EK = d_BTKey_map_.at(params_->GetRingGSWParams()->GetBaseG());
        return gpu_binfhescheme_->GPUEvalBinGate(params_, gate, EK, ct1, ct2, stream_wrapper_.get_stream());
    }

    std::vector<LWECiphertext> GPUBinFHEContext::BatchGPUEvalBinGate(const BINGATE gate,
                                                                     const std::vector<LWECiphertext> &v_ct1,
                                                                     const std::vector<LWECiphertext> &v_ct2) const {
        const auto &EK = d_BTKey_map_.at(params_->GetRingGSWParams()->GetBaseG());
        return gpu_binfhescheme_->BatchGPUEvalBinGate(params_, gate, EK, v_ct1, v_ct2, stream_wrapper_.get_stream());
    }

    LWECiphertext GPUBinFHEContext::GPUEvalFunc(ConstLWECiphertext &ct, const std::vector<NativeInteger> &LUT) const {
        const auto &EK = d_BTKey_map_.at(params_->GetRingGSWParams()->GetBaseG());
        return gpu_binfhescheme_->GPUEvalFunc(params_, EK, ct, LUT, GetBeta(), stream_wrapper_.get_stream());
    }

    std::vector<LWECiphertext> GPUBinFHEContext::BatchGPUEvalFunc(const std::vector<LWECiphertext> &v_ct,
                                                                  const std::vector<NativeInteger> &LUT) const {
        const auto &EK = d_BTKey_map_.at(params_->GetRingGSWParams()->GetBaseG());
        return gpu_binfhescheme_->BatchGPUEvalFunc(params_, EK, v_ct, LUT, GetBeta(), stream_wrapper_.get_stream());
    }

    LWECiphertext GPUBinFHEContext::GPUEvalFloor(ConstLWECiphertext &ct, uint32_t roundbits) const {
        const auto &EK = d_BTKey_map_.at(params_->GetRingGSWParams()->GetBaseG());
        return gpu_binfhescheme_->GPUEvalFloor(params_, EK, ct, GetBeta(), stream_wrapper_.get_stream(),
                                               roundbits);
    }

    LWECiphertext GPUBinFHEContext::GPUEvalSign(ConstLWECiphertext &ct, bool schemeSwitch) {
        return gpu_binfhescheme_->GPUEvalSign(params_, d_BTKey_map_, ct, GetBeta(), stream_wrapper_.get_stream(),
                                              schemeSwitch);
    }

    /*********************** new add *******************************/
    std::vector<LWECiphertext> GPUBinFHEContext::BatchGPUEvalCMUX(const std::vector<lbcrypto::LWECiphertext> &v_ct) const {
        const auto &EK = d_BTKey_map_.at(params_->GetRingGSWParams()->GetBaseG());
        return gpu_binfhescheme_->BatchGPUEvalCMUX(params_, EK, v_ct, stream_wrapper_.get_stream());
    }

    std::vector<LWECiphertext> GPUBinFHEContext::BatchGPUEvalADD(std::vector<lbcrypto::LWECiphertext> &input0, const std::vector<lbcrypto::LWECiphertext> &input1) const {
        return gpu_binfhescheme_->BatchGPUEvalADD(params_, input0, input1, stream_wrapper_.get_stream());
    }
    /*********************** new add *******************************/
}
