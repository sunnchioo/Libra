#pragma once

#include "binfhe-base-scheme.cuh"
#include "openfhe.h"
#include "phantom.h"

namespace phantom::bitwise {

    class GPUBinFHEContext : public lbcrypto::BinFHEContext {

    private:
        phantom::util::cuda_stream_wrapper stream_wrapper_;

        std::shared_ptr<lbcrypto::BinFHECryptoParams> params_{nullptr};

        // Shared pointer to the underlying RingGSW/RLWE scheme
        std::shared_ptr<GPUBinFHEScheme> gpu_binfhescheme_{nullptr}; // 由 context 调用 scheme

        // Struct containing the bootstrapping keys
        std::map<uint32_t, GPURingGSWBTKey> d_BTKey_map_;

    public:
        explicit GPUBinFHEContext(BinFHEContext &cc) {
            // GPU setup
            int device;
            cudaGetDevice(&device);
            cudaMemPool_t mempool;
            cudaDeviceGetDefaultMemPool(&mempool, device);
            uint64_t threshold = UINT64_MAX;
            cudaMemPoolSetAttribute(mempool, cudaMemPoolAttrReleaseThreshold, &threshold);

            params_ = cc.GetParams();
            gpu_binfhescheme_ = std::make_shared<GPUBinFHEScheme>(params_, stream_wrapper_.get_stream());
        }

        /**
         * Generates bootstrapping keys
         *
         * @param sk secret key
         * @param keygenMode key generation mode for symmetric or public encryption
         */
        void GPUBTKeyGen(lbcrypto::ConstLWEPrivateKey &sk, lbcrypto::KEYGEN_MODE keygenMode = lbcrypto::SYM_ENCRYPT);

        /**
         * Evaluates a binary gate (calls bootstrapping as a subroutine)
         *
         * @param gate the gate; can be AND, OR, NAND, NOR, XOR, or XNOR
         * @param ct1 first ciphertext
         * @param ct2 second ciphertext
         * @return a shared pointer to the resulting ciphertext
         */
        [[nodiscard]] lbcrypto::LWECiphertext
        GPUEvalBinGate(lbcrypto::BINGATE gate,
                       lbcrypto::ConstLWECiphertext &ct1, lbcrypto::ConstLWECiphertext &ct2) const;

        [[nodiscard]] std::vector<lbcrypto::LWECiphertext>
        BatchGPUEvalBinGate(lbcrypto::BINGATE gate,
                            const std::vector<lbcrypto::LWECiphertext> &v_ct1,
                            const std::vector<lbcrypto::LWECiphertext> &v_ct2) const;

        /**
         * Evaluate an arbitrary function
         *
         * @param ct ciphertext to be bootstrapped
         * @param LUT the look-up table of the to-be-evaluated function
         * @return a shared pointer to the resulting ciphertext
         */
        [[nodiscard]] lbcrypto::LWECiphertext
        GPUEvalFunc(lbcrypto::ConstLWECiphertext &ct, const std::vector<NativeInteger> &LUT) const;

        [[nodiscard]] std::vector<lbcrypto::LWECiphertext>
        BatchGPUEvalFunc(const std::vector<lbcrypto::LWECiphertext> &v_ct, const std::vector<NativeInteger> &LUT) const;

        /**
         * Evaluate a round down function
         *
         * @param ct ciphertext to be bootstrapped
         * @param roundbits number of bits to be rounded
         * @return a shared pointer to the resulting ciphertext
         */
        [[nodiscard]] lbcrypto::LWECiphertext
        GPUEvalFloor(lbcrypto::ConstLWECiphertext &ct, uint32_t roundbits = 0) const;

        /**
         * Evaluate a sign function over large precisions
         *
         * @param ct ciphertext to be bootstrapped
         * @param schemeSwitch flag that indicates if it should be compatible to scheme switching
         * @return a shared pointer to the resulting ciphertext
         */
        [[nodiscard]] lbcrypto::LWECiphertext
        GPUEvalSign(lbcrypto::ConstLWECiphertext &ct, bool schemeSwitch = false);

        /**************************** new add ********************************/
        /**
         * Getter for params
         * @return
         */
        const std::shared_ptr<lbcrypto::BinFHECryptoParams> &GetParams() {
            return params_;
        }

        /**
         * @return return stream wrapper
         */
        const phantom::util::cuda_stream_wrapper &GetStreamWrapper() const noexcept {
            return stream_wrapper_;
        }

        /**
         * Evaluate a cmux
         */
        [[nodiscard]] std::vector<lbcrypto::LWECiphertext>
        BatchGPUEvalCMUX(const std::vector<lbcrypto::LWECiphertext> &v_ct) const;

        /**
         * Evaluate a add
         */
        [[nodiscard]] std::vector<lbcrypto::LWECiphertext>
        BatchGPUEvalADD(std::vector<lbcrypto::LWECiphertext> &v_ct0, const std::vector<lbcrypto::LWECiphertext> &v_ct1) const;

        /**************************** new add ********************************/

    private:
    };
}
