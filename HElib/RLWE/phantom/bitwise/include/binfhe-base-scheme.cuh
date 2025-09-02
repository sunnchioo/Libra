#pragma once

#include "openfhe.h"
#include "phantom.h"

#include "compact_ntt.cuh"
#include "rgsw-acc.cuh"

namespace phantom::bitwise {

    class GPUBinFHEScheme : public lbcrypto::BinFHEScheme {
    private:
        /**
         * Bootstraps a fresh ciphertext
         *
         * @param params a shared pointer to RingGSW scheme parameters
         * @param EK a shared pointer to the bootstrapping keys
         * @param ct input ciphertext
         * @param f function to evaluate in the functional bootstrapping
         * @param fmod modulus over which the function is defined
         * @return the output RingLWE accumulator
         */
        template <typename Func>
        lbcrypto::LWECiphertext
        GPUBootstrapFunc(const std::shared_ptr<lbcrypto::BinFHECryptoParams> &params, const GPURingGSWBTKey &EK,
                         lbcrypto::ConstLWECiphertext &ct, Func f, const NativeInteger &fmod,
                         const cudaStream_t &s) const;

        template <typename Func>
        std::vector<lbcrypto::LWECiphertext>
        BatchGPUBootstrapFunc(const std::shared_ptr<lbcrypto::BinFHECryptoParams> &params, const GPURingGSWBTKey &EK,
                              const std::vector<lbcrypto::LWECiphertext> &v_ct, Func f, const NativeInteger &fmod,
                              const cudaStream_t &s) const;

        /**
         * Core bootstrapping operation
         *
         * @param params a shared pointer to RingGSW scheme parameters
         * @param gate the gate; can be AND, OR, NAND, NOR, XOR, or XOR
         * @param EK a shared pointer to the bootstrapping keys
         * @param ct input ciphertext
         * @param s CUDA stream
         * @return the output RingLWE accumulator
         */
        [[nodiscard]] phantom::util::cuda_auto_ptr<BasicInteger>
        GPUBootstrapGateCore(const std::shared_ptr<lbcrypto::BinFHECryptoParams> &params,
                             lbcrypto::BINGATE gate,
                             const GPURingGSWBTKey &EK,
                             lbcrypto::ConstLWECiphertext &ctprep,
                             const cudaStream_t &s) const;

        [[nodiscard]] phantom::util::cuda_auto_ptr<BasicInteger>
        BatchGPUBootstrapGateCore(const std::shared_ptr<lbcrypto::BinFHECryptoParams> &params,
                                  lbcrypto::BINGATE gate,
                                  const GPURingGSWBTKey &EK,
                                  const std::vector<lbcrypto::LWECiphertext> &v_ctprep,
                                  const cudaStream_t &s) const;

        /**
         * Core bootstrapping operation
         *
         * @param params a shared pointer to RingGSW scheme parameters
         * @param EK a shared pointer to the bootstrapping keys
         * @param ct input ciphertext
         * @param f function to evaluate in the functional bootstrapping
         * @param fmod modulus over which the function is defined
         * @param s CUDA stream
         * @return a shared pointer to the resulting ciphertext
         */
        template <typename Func>
        [[nodiscard]] phantom::util::cuda_auto_ptr<BasicInteger>
        GPUBootstrapFuncCore(const std::shared_ptr<lbcrypto::BinFHECryptoParams> &params,
                             const GPURingGSWBTKey &EK,
                             lbcrypto::ConstLWECiphertext &ct,
                             Func f,
                             const NativeInteger &fmod,
                             const cudaStream_t &s) const;

        template <typename Func>
        [[nodiscard]] phantom::util::cuda_auto_ptr<BasicInteger>
        BatchGPUBootstrapFuncCore(const std::shared_ptr<lbcrypto::BinFHECryptoParams> &params,
                                  const GPURingGSWBTKey &EK,
                                  const std::vector<lbcrypto::LWECiphertext> &v_ct,
                                  Func f,
                                  const NativeInteger &fmod,
                                  const cudaStream_t &s) const;

    protected:
        std::shared_ptr<lbcrypto::LWEEncryptionScheme> LWEscheme_{nullptr};
        std::shared_ptr<lbcrypto::RingGSWAccumulator> ACCscheme_{nullptr};
        std::shared_ptr<GPURingGSWAccumulator> GPUACCscheme_{nullptr};

        lbcrypto::BINFHE_METHOD method_;

        /**
         * Checks type of input function
         *
         * @param lut look up table for the input function
         * @param mod modulus over which the function is defined
         * @return the function type: 0 for negacyclic, 1 for periodic, 2 for arbitrary
         */
        static uint32_t checkInputFunction(const std::vector<NativeInteger> &lut, NativeInteger mod) {
            size_t mid{lut.size() / 2};
            if (lut[0] == (mod - lut[mid])) {
                for (size_t i = 1; i < mid; ++i)
                    if (lut[i] != (mod - lut[mid + i]))
                        return 2;
                return 0;
            }
            if (lut[0] == lut[mid]) {
                for (size_t i = 1; i < mid; ++i)
                    if (lut[i] != lut[mid + i])
                        return 2;
                return 1;
            }
            return 2;
        }

    public:
        explicit GPUBinFHEScheme(const std::shared_ptr<lbcrypto::BinFHECryptoParams> &params, const cudaStream_t &s) {
            method_ = params->GetRingGSWParams()->GetMethod();

            LWEscheme_ = std::make_shared<lbcrypto::LWEEncryptionScheme>();

            if (method_ == lbcrypto::BINFHE_METHOD::AP)
                ACCscheme_ = std::make_shared<lbcrypto::RingGSWAccumulatorDM>();
            else if (method_ == lbcrypto::BINFHE_METHOD::GINX)
                ACCscheme_ = std::make_shared<lbcrypto::RingGSWAccumulatorCGGI>();
            else if (method_ == lbcrypto::BINFHE_METHOD::LMKCDEY)
                ACCscheme_ = std::make_shared<lbcrypto::RingGSWAccumulatorLMKCDEY>();
            else
                OPENFHE_THROW("method is invalid");

            GPUACCscheme_ = std::make_shared<GPURingGSWAccumulator>();
        }

        /**
         * Generates a refresh key
         *
         * @param params a shared pointer to RingGSW scheme parameters
         * @param LWEsk a shared pointer to the secret key of the underlying additive
         * @param keygenMode enum to indicate generation of secret key only (SYM_ENCRYPT) or
         * secret key, public key pair (PUB_ENCRYPT)
         * @return a shared pointer to the refresh key
         */
        [[nodiscard]] GPURingGSWBTKey
        GPUKeyGen(const std::shared_ptr<lbcrypto::BinFHECryptoParams> &params,
                  lbcrypto::ConstLWEPrivateKey &LWEsk,
                  const cudaStream_t &s,
                  lbcrypto::KEYGEN_MODE keygenMode = lbcrypto::SYM_ENCRYPT) const;

        /**
         * Evaluates a binary gate (calls bootstrapping as a subroutine)
         *
         * @param params a shared pointer to RingGSW scheme parameters
         * @param gate the gate; can be AND, OR, NAND, NOR, XOR, or XOR
         * @param EK a shared pointer to the bootstrapping keys
         * @param ct1 first ciphertext
         * @param ct2 second ciphertext
         * @return a shared pointer to the resulting ciphertext
         */
        [[nodiscard]] lbcrypto::LWECiphertext
        GPUEvalBinGate(const std::shared_ptr<lbcrypto::BinFHECryptoParams> &params,
                       lbcrypto::BINGATE gate, const GPURingGSWBTKey &EK,
                       lbcrypto::ConstLWECiphertext &ct1, lbcrypto::ConstLWECiphertext &ct2,
                       const cudaStream_t &s) const;

        [[nodiscard]] std::vector<lbcrypto::LWECiphertext>
        BatchGPUEvalBinGate(const std::shared_ptr<lbcrypto::BinFHECryptoParams> &params,
                            lbcrypto::BINGATE gate, const GPURingGSWBTKey &EK,
                            const std::vector<lbcrypto::LWECiphertext> &v_ct1,
                            const std::vector<lbcrypto::LWECiphertext> &v_ct2,
                            const cudaStream_t &s) const;

        /**
         * Evaluate an arbitrary function
         *
         * @param params a shared pointer to RingGSW scheme parameters
         * @param EK a shared pointer to the bootstrapping keys
         * @param ct input ciphertext
         * @param LUT the look-up table of the to-be-evaluated function
         * @param beta the error bound
         * @return a shared pointer to the resulting ciphertext
         */
        [[nodiscard]] lbcrypto::LWECiphertext
        GPUEvalFunc(const std::shared_ptr<lbcrypto::BinFHECryptoParams> &params, const GPURingGSWBTKey &EK,
                    lbcrypto::ConstLWECiphertext &ct, const std::vector<NativeInteger> &LUT,
                    const NativeInteger &beta, const cudaStream_t &s) const;

        [[nodiscard]] std::vector<lbcrypto::LWECiphertext>
        BatchGPUEvalFunc(const std::shared_ptr<lbcrypto::BinFHECryptoParams> &params, const GPURingGSWBTKey &EK,
                         const std::vector<lbcrypto::LWECiphertext> &v_ct, const std::vector<NativeInteger> &LUT,
                         const NativeInteger &beta, const cudaStream_t &s) const;

        /**
         * Evaluate a round down function
         *
         * @param params a shared pointer to RingGSW scheme parameters
         * @param EK a shared pointer to the bootstrapping keys
         * @param ct input ciphertext
         * @param beta the error bound
         * @param roundbits by how many bits to round down
         * @return a shared pointer to the resulting ciphertext
         */
        [[nodiscard]] lbcrypto::LWECiphertext
        GPUEvalFloor(const std::shared_ptr<lbcrypto::BinFHECryptoParams> &params, const GPURingGSWBTKey &EK,
                     lbcrypto::ConstLWECiphertext &ct, const NativeInteger &beta, const cudaStream_t &s,
                     uint32_t roundbits = 0) const;

        /**
         * Evaluate a sign function over large precision
         *
         * @param params a shared pointer to RingGSW scheme parameters
         * @param EK a shared pointer to the bootstrapping keys map
         * @param ct input ciphertext
         * @param beta the error bound
         * @param schemeSwitch flag that indicates if it should be compatible to scheme switching
         * @return a shared pointer to the resulting ciphertext
         */
        [[nodiscard]] lbcrypto::LWECiphertext
        GPUEvalSign(const std::shared_ptr<lbcrypto::BinFHECryptoParams> &params,
                    const std::map<uint32_t, GPURingGSWBTKey> &EKs, lbcrypto::ConstLWECiphertext &ct,
                    const NativeInteger &beta, const cudaStream_t &s, bool schemeSwitch = false) const;

        /******************** new add ********************/
        [[nodiscard]] std::vector<lbcrypto::LWECiphertext>
        BatchGPUEvalCMUX(const std::shared_ptr<lbcrypto::BinFHECryptoParams> &params, const GPURingGSWBTKey &EK,
                         const std::vector<lbcrypto::LWECiphertext> &v_ct, const cudaStream_t &s) const;

        [[nodiscard]] std::vector<lbcrypto::LWECiphertext>
        BatchGPUEvalADD(const std::shared_ptr<lbcrypto::BinFHECryptoParams> &params, std::vector<lbcrypto::LWECiphertext> &res,
                        const std::vector<lbcrypto::LWECiphertext> &input, const cudaStream_t &s) const;
        /******************** new add ********************/
    };
}
