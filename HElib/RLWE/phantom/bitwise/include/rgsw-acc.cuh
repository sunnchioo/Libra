#pragma once

#include "openfhe.h"
#include "phantom.h"

#include "compact_ntt.cuh"

namespace phantom::bitwise {

    typedef std::vector<std::vector<std::vector<phantom::util::cuda_auto_ptr<BasicInteger>>>> GPURingGSWACCKey;

    class GPURingGSWBTKey {
    public:
        // refreshing key
        GPURingGSWACCKey RGSWACCKey;

        // switching key
        phantom::util::cuda_auto_ptr<BasicInteger> LWESwitchKey_A;
        std::vector<std::vector<std::vector<NativeInteger>>> cpu_keyB;
    };

    class GPURingGSWAccumulator : public lbcrypto::RingGSWAccumulator {
    private:
        [[nodiscard]] GPURingGSWACCKey
        GPUKeyGenAccDM(const std::shared_ptr<lbcrypto::RingGSWCryptoParams> &params,
                       const phantom::util::cuda_auto_ptr<BasicInteger> &d_skNTT,
                       lbcrypto::ConstLWEPrivateKey &LWEsk,
                       const cudaStream_t &stream) const;

        [[nodiscard]] GPURingGSWACCKey
        GPUKeyGenAccCGGI(const std::shared_ptr<lbcrypto::RingGSWCryptoParams> &params,
                         const phantom::util::cuda_auto_ptr<BasicInteger> &d_skNTT,
                         lbcrypto::ConstLWEPrivateKey &LWEsk,
                         const cudaStream_t &stream) const;

        [[nodiscard]] phantom::util::cuda_auto_ptr<BasicInteger>
        GPUKeyGenDM(const std::shared_ptr<lbcrypto::RingGSWCryptoParams> &params,
                    const phantom::util::cuda_auto_ptr<BasicInteger> &d_skNTT,
                    lbcrypto::LWEPlaintext m,
                    const cudaStream_t &stream) const;

        [[nodiscard]] phantom::util::cuda_auto_ptr<BasicInteger>
        GPUKeyGenCGGI(const std::shared_ptr<lbcrypto::RingGSWCryptoParams> &params,
                      const phantom::util::cuda_auto_ptr<BasicInteger> &d_skNTT,
                      lbcrypto::LWEPlaintext m,
                      const cudaStream_t &stream) const;

        void GPUEvalAccDM(const std::shared_ptr<lbcrypto::BinFHECryptoParams> &params, const GPURingGSWBTKey &EK,
                          const NativeVector &a, const phantom::util::cuda_auto_ptr<BasicInteger> &d_acc,
                          const cudaStream_t &s) const;

        void BatchGPUEvalAccDM(const std::shared_ptr<lbcrypto::BinFHECryptoParams> &params, const GPURingGSWBTKey &EK,
                               const std::vector<NativeVector> &v_a,
                               const phantom::util::cuda_auto_ptr<BasicInteger> &d_acc,
                               const cudaStream_t &s) const;

        void GPUEvalAccCGGI(const std::shared_ptr<lbcrypto::BinFHECryptoParams> &params, const GPURingGSWBTKey &EK,
                            const NativeVector &a, const phantom::util::cuda_auto_ptr<BasicInteger> &d_acc,
                            const cudaStream_t &s) const;

        void BatchGPUEvalAccCGGI(const std::shared_ptr<lbcrypto::BinFHECryptoParams> &params, const GPURingGSWBTKey &EK,
                                 const std::vector<NativeVector> &v_a,
                                 const phantom::util::cuda_auto_ptr<BasicInteger> &d_acc,
                                 const cudaStream_t &s) const;

        void BatchGPUEvalAccCGGI_nttopt(const std::shared_ptr<lbcrypto::BinFHECryptoParams> &params, const GPURingGSWBTKey &EK,
                                        const std::vector<NativeVector> &v_a,
                                        const phantom::util::cuda_auto_ptr<BasicInteger> &d_acc,
                                        const cudaStream_t &s) const;

    protected:
        phantom::util::cuda_auto_ptr<BasicInteger> d_monic_polys_;
        std::shared_ptr<FourStepNTT> ntt_;

    public:
        GPURingGSWAccumulator() = default;

        /**
         * Key generation for internal Ring GSW
         *
         * @param params a shared pointer to RingGSW scheme parameters
         * @param sk secret key polynomial in the COEFFICIENT representation
         * @param LWEsk the secret key
         * @param stream CUDA stream
         * @return a shared pointer to the resulting keys
         */
        [[nodiscard]] GPURingGSWACCKey
        GPUKeyGenAcc(const std::shared_ptr<lbcrypto::RingGSWCryptoParams> &params,
                     const lbcrypto::NativeVector &sk, lbcrypto::ConstLWEPrivateKey &LWEsk,
                     const cudaStream_t &stream);

        /**
         * Main accumulator function used in bootstrapping
         *
         * @param params a shared pointer to RingGSW scheme parameters
         * @param ek the RGSW bootstrapping key
         * @param a ciphertext
         * @param acc previous value of the accumulator
         * @param s CUDA stream
         */
        void GPUEvalAcc(const std::shared_ptr<lbcrypto::BinFHECryptoParams> &params, const GPURingGSWBTKey &EK,
                        const NativeVector &a, const phantom::util::cuda_auto_ptr<BasicInteger> &d_acc,
                        const cudaStream_t &s) const;

        void BatchGPUEvalAcc(const std::shared_ptr<lbcrypto::BinFHECryptoParams> &params, const GPURingGSWBTKey &EK,
                             const std::vector<NativeVector> &v_a,
                             const phantom::util::cuda_auto_ptr<BasicInteger> &d_acc,
                             const cudaStream_t &s) const;

        /***************** new add *****************/
        void BatchGPUEvalCMUX(const std::shared_ptr<lbcrypto::BinFHECryptoParams> &params, const GPURingGSWBTKey &EK,
                              const std::vector<NativeVector> &v_a, const phantom::util::cuda_auto_ptr<BasicInteger> &d_acc,
                              const cudaStream_t &s) const;
        void BatchGPUEvalADD(const std::shared_ptr<lbcrypto::BinFHECryptoParams> &params, std::vector<lbcrypto::LWECiphertext> &input0,
                             const std::vector<lbcrypto::LWECiphertext> &input1, const cudaStream_t &s) const;
        /***************** new add *****************/
    };
} // namespace lbcrypto
