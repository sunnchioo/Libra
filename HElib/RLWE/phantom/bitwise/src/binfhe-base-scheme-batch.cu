#include "binfhe-base-scheme.cuh"
#include "kernel.cuh"

using namespace lbcrypto;

namespace phantom::bitwise {

    // the main rounding operation used in ModSwitch (as described in Section 3 of
    // https://eprint.iacr.org/2014/816) The idea is that Round(x) = 0.5 + Floor(x)
    static NativeInteger RoundqQ(const NativeInteger &v, const NativeInteger &q, const NativeInteger &Q) {
        return NativeInteger(
                   static_cast<BasicInteger>(
                       std::floor(0.5 + v.ConvertToDouble() * q.ConvertToDouble() / Q.ConvertToDouble())))
            .Mod(q);
    }

    phantom::util::cuda_auto_ptr<BasicInteger>
    GPUBinFHEScheme::BatchGPUBootstrapGateCore(const std::shared_ptr<BinFHECryptoParams> &params,
                                               BINGATE gate,
                                               const GPURingGSWBTKey &EK,
                                               const std::vector<lbcrypto::LWECiphertext> &v_ctprep,
                                               const cudaStream_t &s) const {

        auto &LWEParams = params->GetLWEParams();
        auto &RGSWParams = params->GetRingGSWParams();
        auto polyParams = RGSWParams->GetPolyParams();

        // Specifies the range [q1,q2) that will be used for mapping
        NativeInteger p = v_ctprep[0]->GetptModulus();
        NativeInteger q = v_ctprep[0]->GetModulus();
        uint32_t qHalf = q.ConvertToInt() >> 1;
        NativeInteger q1 = RGSWParams->GetGateConst()[static_cast<size_t>(gate)];
        NativeInteger q2 = q1.ModAddFast(NativeInteger(qHalf), q);

        // depending on whether the value is the range, it will be set
        // to either Q/8 or -Q/8 to match binary arithmetic
        NativeInteger Q = RGSWParams->GetQ();
        NativeInteger Q2p = Q / NativeInteger(2 * p) + 1;
        NativeInteger Q2pNeg = Q - Q2p;

        uint32_t N = LWEParams->GetN();

        // Since q | (2*N), we deal with a sparse embedding of Z_Q[x]/(X^{q/2}+1) to
        // Z_Q[x]/(X^N+1)
        uint32_t factor = (2 * N / q.ConvertToInt());

        size_t batch_size = v_ctprep.size();

        std::vector<NativeVector> v_m(batch_size, NativeVector(N, Q));

        for (size_t i = 0; i < batch_size; ++i) {
            const NativeInteger &b = v_ctprep[i]->GetB();
            for (size_t j = 0; j < qHalf; ++j) {
                NativeInteger temp = b.ModSub(j, q);
                if (q1 < q2)
                    v_m[i][j * factor] = ((temp >= q1) && (temp < q2)) ? Q2pNeg : Q2p;
                else
                    v_m[i][j * factor] = ((temp >= q2) && (temp < q1)) ? Q2p : Q2pNeg;
            }
        }

        cudaStreamSynchronize(s);

        auto d_acc_batch = phantom::util::make_cuda_auto_ptr<BasicInteger>(2 * N * batch_size, s);

        for (size_t i = 0; i < batch_size; ++i) {
            cudaMemsetAsync(d_acc_batch.get() + i * 2 * N, 0, sizeof(BasicInteger) * N, s);
            cudaMemcpyAsync(d_acc_batch.get() + i * 2 * N + N, &v_m[i].at(0), sizeof(BasicInteger) * N,
                            cudaMemcpyHostToDevice, s);
        }

        // main accumulation computation
        // the following loop is the bottleneck of bootstrapping/binary gate
        // evaluation
        std::vector<NativeVector> v_ctprep_A;
        for (size_t i = 0; i < batch_size; ++i) {
            v_ctprep_A.push_back(v_ctprep[i]->GetA());
        }

        GPUACCscheme_->BatchGPUEvalAcc(params, EK, v_ctprep_A, d_acc_batch, s);

        return std::move(d_acc_batch);
    }

    std::vector<LWECiphertext> GPUBinFHEScheme::BatchGPUEvalBinGate(const std::shared_ptr<BinFHECryptoParams> &params,
                                                                    BINGATE gate,
                                                                    const GPURingGSWBTKey &EK,
                                                                    const std::vector<LWECiphertext> &v_ct1,
                                                                    const std::vector<LWECiphertext> &v_ct2,
                                                                    const cudaStream_t &s) const {
        if (v_ct1.size() != v_ct2.size())
            OPENFHE_THROW("Input ciphertexts should have the same size");

        for (size_t i = 0; i < v_ct1.size(); ++i) {
            if (v_ct1[i] == v_ct2[i])
                OPENFHE_THROW("Input ciphertexts should be independent");
        }

        size_t batch_size = v_ct1.size();

        std::vector<LWECiphertext> v_ctprep;

        for (size_t i = 0; i < batch_size; ++i) {
            LWECiphertext ctprep = std::make_shared<LWECiphertextImpl>(*v_ct1[i]);
            // the additive homomorphic operation for XOR/NXOR is different from the other gates we compute
            // 2*(ct1 + ct2) mod 4 for XOR, 0 -> 0, 2 -> 1
            // XOR_FAST and XNOR_FAST are included for backwards compatibility; they map to XOR and XNOR
            if ((gate == XOR) || (gate == XNOR) || (gate == XOR_FAST) || (gate == XNOR_FAST)) {
                LWEscheme->EvalAddEq(ctprep, v_ct2[i]);
                LWEscheme->EvalAddEq(ctprep, ctprep);
            } else {
                // for all other gates, we simply compute (ct1 + ct2) mod 4
                // for AND: 0,1 -> 0 and 2,3 -> 1
                // for OR: 1,2 -> 1 and 3,0 -> 0
                LWEscheme->EvalAddEq(ctprep, v_ct2[i]);
            }
            v_ctprep.push_back(ctprep);
        }

        auto d_acc_batch = BatchGPUBootstrapGateCore(params, gate, EK, v_ctprep, s);

        auto &LWEParams = params->GetLWEParams();
        auto &RGSWParams = params->GetRingGSWParams();
        auto polyParams = RGSWParams->GetPolyParams();

        uint32_t N = LWEParams->GetN();
        NativeInteger Q = LWEParams->GetQ();

        // Sample extract fuses modulus switching from Q to qKS
        NativeInteger qKS = LWEParams->GetqKS();
        const int logN = phantom::arith::get_power_of_two(N);
        auto d_tmp_batch = phantom::util::make_cuda_auto_ptr<BasicInteger>(2 * N * batch_size, s);
        cudaMemcpyAsync(d_tmp_batch.get(), d_acc_batch.get(), 2 * N * batch_size * sizeof(BasicInteger),
                        cudaMemcpyDeviceToDevice, s);
        kernel_automorphism_modSwitch<<<dim3(N / ThreadsPerBlock, batch_size), ThreadsPerBlock, 0, s>>>(
            d_acc_batch.get(), d_tmp_batch.get(), qKS.ConvertToInt(), Q.ConvertToInt(), logN);

        // copy acc_a to host
        std::vector<NativeVector> v_acc_a(batch_size, NativeVector(N, qKS));
        for (size_t i = 0; i < batch_size; ++i) {
            cudaMemcpyAsync(&v_acc_a[i].at(0), d_acc_batch.get() + i * 2 * N, N * sizeof(BasicInteger),
                            cudaMemcpyDeviceToHost, s);
        }

        // copy acc_b to host
        std::vector<NativeInteger> v_b(batch_size, 0);
        for (size_t i = 0; i < batch_size; ++i) {
            cudaMemcpyAsync((void *)&v_b[i], d_acc_batch.get() + i * 2 * N + N,
                            1 * sizeof(BasicInteger), cudaMemcpyDeviceToHost, s);
        }

        // explicitly synchronize to ensure that the data is copied to the host
        cudaStreamSynchronize(s);

        // Key switching reduces dimension from N to n
        // Fuses modulus switching from qKS to fmod
        uint32_t n = LWEParams->Getn();
        NativeInteger::Integer baseKS(LWEParams->GetBaseKS());
        const auto log_baseKS = GetMSB(baseKS) - 1;
        const auto digitCount = static_cast<size_t>(std::ceil(
            log(qKS.ConvertToDouble()) / log(static_cast<double>(baseKS))));

        auto d_LWE_ct_A_batch = phantom::util::make_cuda_auto_ptr<BasicInteger>(n * batch_size, s);

        kernel_LWEKeySwitch_modSwitch<<<dim3((n + ThreadsPerBlock - 1) / ThreadsPerBlock, batch_size),
                                        ThreadsPerBlock, 0, s>>>(
            d_LWE_ct_A_batch.get(), d_acc_batch.get(), EK.LWESwitchKey_A.get(),
            n, N, v_ct1[0]->GetModulus().ConvertToInt(), qKS.ConvertToInt(), log_baseKS, digitCount);

        // copy LWE ciphertext part A to host
        std::vector<NativeVector> v_A(batch_size, NativeVector(n, v_ct1[0]->GetModulus()));
        for (size_t i = 0; i < batch_size; ++i) {
            cudaMemcpyAsync(&v_A[i].at(0), d_LWE_ct_A_batch.get() + i * n, n * sizeof(BasicInteger),
                            cudaMemcpyDeviceToHost, s);
        }

        // asynchronously compute key switching and mod switching for b on CPU

        for (size_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
            auto &b = v_b[batch_idx];

            // we add Q/8 to "b" to map back to Q/4 (i.e., mod 2) arithmetic.
            b.ModAddFastEq((Q >> 3) + 1, Q);

            // mod switch
            b = RoundqQ(b, qKS, Q);

            // key switch
            for (size_t i = 0; i < N; ++i) {
                NativeInteger::Integer atmp(v_acc_a[batch_idx][i].ConvertToInt());
                for (size_t j = 0; j < digitCount; ++j) {
                    const auto a0 = (atmp & (baseKS - 1));
                    atmp >>= log_baseKS;
                    b.ModSubFastEq(EK.cpu_keyB[i][a0][j], qKS);
                }
            }

            // mod switch
            b = RoundqQ(b, v_ct1[0]->GetModulus().ConvertToInt(), qKS);
        }

        // explicitly synchronize to ensure that CPU and GPU all finish the computation
        cudaStreamSynchronize(s);

        std::vector<LWECiphertext> v_result;

        for (size_t i = 0; i < batch_size; ++i) {
            v_result.push_back(std::make_shared<LWECiphertextImpl>(std::move(v_A[i]), v_b[i]));
        }

        return v_result;
    }

    // Functions below are for large-precision sign evaluation,
    // flooring, homomorphic digit decomposition, and arbitrary
    // funciton evaluation, from https://eprint.iacr.org/2021/1337
    template <typename Func>
    phantom::util::cuda_auto_ptr<BasicInteger>
    GPUBinFHEScheme::BatchGPUBootstrapFuncCore(const std::shared_ptr<BinFHECryptoParams> &params,
                                               const GPURingGSWBTKey &EK,
                                               const std::vector<lbcrypto::LWECiphertext> &v_ct,
                                               Func f,
                                               const NativeInteger &fmod,
                                               const cudaStream_t &s) const {
        auto &LWEParams = params->GetLWEParams();
        auto &RGSWParams = params->GetRingGSWParams();
        auto polyParams = RGSWParams->GetPolyParams();

        NativeInteger Q = LWEParams->GetQ();
        uint32_t N = LWEParams->GetN();

        // For specific function evaluation instead of general bootstrapping
        NativeInteger ctMod = v_ct[0]->GetModulus();
        uint32_t factor = (2 * N / ctMod.ConvertToInt());

        size_t batch_size = v_ct.size();
        std::vector<NativeVector> v_m(batch_size, NativeVector(N, Q));
        for (size_t i = 0; i < batch_size; ++i) {
            const NativeInteger &b = v_ct[i]->GetB();
            for (size_t j = 0; j < (ctMod >> 1); ++j) {
                NativeInteger temp = b.ModSub(j, ctMod);
                v_m[i][j * factor] = Q.ConvertToInt() / fmod.ConvertToInt() * f(temp, ctMod, fmod);
            }
        }

        cudaStreamSynchronize(s);

        auto d_acc_batch = phantom::util::make_cuda_auto_ptr<BasicInteger>(2 * N * batch_size, s);

        for (size_t i = 0; i < batch_size; ++i) {
            cudaMemsetAsync(d_acc_batch.get() + i * 2 * N, 0, sizeof(BasicInteger) * N, s);
            cudaMemcpyAsync(d_acc_batch.get() + i * 2 * N + N, &v_m[i].at(0), sizeof(BasicInteger) * N,
                            cudaMemcpyHostToDevice, s); // 存储了 b
        } // 这是 rlwe，(0,0, ..., v_m)

        // main accumulation computation
        // the following loop is the bottleneck of bootstrapping/binary gate
        // evaluation
        std::vector<NativeVector> v_ct_A;
        for (size_t i = 0; i < batch_size; ++i) {
            v_ct_A.push_back(v_ct[i]->GetA());
        }
        GPUACCscheme_->BatchGPUEvalAcc(params, EK, v_ct_A, d_acc_batch, s);

        return std::move(d_acc_batch);
    }

    // Full evaluation as described in https://eprint.iacr.org/2020/086
    // Functions below are for large-precision sign evaluation,
    // flooring, homomorphic digit decomposition, and arbitrary
    // function evaluation, from https://eprint.iacr.org/2021/1337
    template <typename Func>
    std::vector<LWECiphertext> GPUBinFHEScheme::BatchGPUBootstrapFunc(const std::shared_ptr<BinFHECryptoParams> &params,
                                                                      const GPURingGSWBTKey &EK,
                                                                      const std::vector<LWECiphertext> &v_ct,
                                                                      Func f,
                                                                      const NativeInteger &fmod,
                                                                      const cudaStream_t &s) const {
        size_t batch_size = v_ct.size();

        // phantom::util::CUDATimer timer_br("blind rotate: " + std::to_string(batch_size), s);
        // timer_br.start();
        auto d_acc_batch = BatchGPUBootstrapFuncCore(params, EK, v_ct, f, fmod, s); // blind rotate
        // timer_br.stop();
        // CHECK_CUDA_LAST_ERROR();

        auto &LWEParams = params->GetLWEParams();
        auto &RGSWParams = params->GetRingGSWParams();
        auto polyParams = RGSWParams->GetPolyParams();

        NativeInteger Q = LWEParams->GetQ();
        uint32_t N = LWEParams->GetN();

        // Sample extract fuses modulus switching from Q to qKS
        // phantom::util::CUDATimer timer_se("sample extract: " + std::to_string(batch_size), s);
        // timer_se.start();
        NativeInteger qKS = LWEParams->GetqKS();
        const int logN = phantom::arith::get_power_of_two(N);
        auto d_tmp_batch = phantom::util::make_cuda_auto_ptr<BasicInteger>(2 * N * batch_size, s);
        cudaMemcpyAsync(d_tmp_batch.get(), d_acc_batch.get(), 2 * N * batch_size * sizeof(BasicInteger),
                        cudaMemcpyDeviceToDevice, s);
        kernel_automorphism_modSwitch<<<dim3(N / ThreadsPerBlock, batch_size), ThreadsPerBlock, 0, s>>>(
            d_acc_batch.get(), d_tmp_batch.get(), qKS.ConvertToInt(), Q.ConvertToInt(), logN);

        // copy acc_a to host
        std::vector<NativeVector> v_acc_a(batch_size, NativeVector(N, qKS));
        for (size_t i = 0; i < batch_size; ++i) {
            cudaMemcpyAsync(&v_acc_a[i].at(0), d_acc_batch.get() + i * 2 * N, N * sizeof(BasicInteger),
                            cudaMemcpyDeviceToHost, s);
        }

        // copy acc_b to host
        std::vector<NativeInteger> v_b(batch_size, 0);
        for (size_t i = 0; i < batch_size; ++i) {
            cudaMemcpyAsync((void *)&v_b[i], d_acc_batch.get() + i * 2 * N + N,
                            1 * sizeof(BasicInteger), cudaMemcpyDeviceToHost, s);
        }

        // explicitly synchronize to ensure that the data is copied to the host
        cudaStreamSynchronize(s);
        // timer_se.stop();
        // CHECK_CUDA_LAST_ERROR();

        // phantom::util::CUDATimer timer("key switching " + std::to_string(batch_size), s);
        // timer.start();
        // Key switching reduces dimension from N to n
        // Fuses modulus switching from qKS to fmod
        uint32_t n = LWEParams->Getn();
        NativeInteger::Integer baseKS(LWEParams->GetBaseKS());
        const auto log_baseKS = GetMSB(baseKS) - 1;
        const auto digitCount = static_cast<size_t>(std::ceil(
            log(qKS.ConvertToDouble()) / log(static_cast<double>(baseKS))));

        auto d_LWE_ct_A_batch = phantom::util::make_cuda_auto_ptr<BasicInteger>(n * batch_size, s);

        kernel_LWEKeySwitch_modSwitch<<<dim3((n + ThreadsPerBlock - 1) / ThreadsPerBlock, batch_size),
                                        ThreadsPerBlock, 0, s>>>(
            d_LWE_ct_A_batch.get(), d_acc_batch.get(), EK.LWESwitchKey_A.get(),
            n, N, fmod.ConvertToInt(), qKS.ConvertToInt(), log_baseKS, digitCount);

        // copy LWE ciphertext part A to host
        std::vector<NativeVector> v_A(batch_size, NativeVector(n, fmod));
        for (size_t i = 0; i < batch_size; ++i) {
            cudaMemcpyAsync(&v_A[i].at(0), d_LWE_ct_A_batch.get() + i * n, n * sizeof(BasicInteger),
                            cudaMemcpyDeviceToHost, s);
        }

        // asynchronously compute key switching and mod switching for b on CPU

        for (size_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) { // 可能因为 b 很小，所以不用 GPU 计算
            auto &b = v_b[batch_idx];

            // mod switch
            b = RoundqQ(b, qKS, Q);

            // key switch
            for (size_t i = 0; i < N; ++i) {
                NativeInteger::Integer atmp(v_acc_a[batch_idx][i].ConvertToInt());
                for (size_t j = 0; j < digitCount; ++j) {
                    const auto a0 = (atmp & (baseKS - 1));
                    atmp >>= log_baseKS;
                    b.ModSubFastEq(EK.cpu_keyB[i][a0][j], qKS);
                }
            }

            // mod switch
            b = RoundqQ(b, fmod, qKS);
        }

        // explicitly synchronize to ensure that CPU and GPU all finish the computation
        cudaStreamSynchronize(s);
        // timer.stop();
        // CHECK_CUDA_LAST_ERROR();

        std::vector<LWECiphertext> v_result;

        for (size_t i = 0; i < batch_size; ++i) {
            v_result.push_back(std::make_shared<LWECiphertextImpl>(std::move(v_A[i]), v_b[i]));
        }

        return v_result;
    }

    std::vector<LWECiphertext> GPUBinFHEScheme::BatchGPUEvalFunc(const std::shared_ptr<BinFHECryptoParams> &params,
                                                                 const GPURingGSWBTKey &EK,
                                                                 const std::vector<LWECiphertext> &v_ct,
                                                                 const std::vector<NativeInteger> &LUT,
                                                                 const NativeInteger &beta,
                                                                 const cudaStream_t &s) const {
        size_t batch_size = v_ct.size();
        std::vector<LWECiphertext> v_ct1;
        for (size_t i = 0; i < batch_size; ++i) {
            LWECiphertext ct1 = std::make_shared<LWECiphertextImpl>(*v_ct[i]);
            v_ct1.push_back(ct1);
        }

        NativeInteger q{v_ct[0]->GetModulus()};
        uint32_t functionProperty{checkInputFunction(LUT, q)};

        if (functionProperty == 0) { // negacyclic function only needs one bootstrap
            auto fLUT = [LUT](NativeInteger x, NativeInteger q, NativeInteger Q) -> NativeInteger {
                return LUT[x.ConvertToInt()];
            };
            for (size_t i = 0; i < batch_size; ++i)
                LWEscheme->EvalAddConstEq(v_ct1[i], beta);
            return BatchGPUBootstrapFunc(params, EK, v_ct1, fLUT, q, s);
        }

        if (functionProperty == 2) { // arbitrary function
            const auto &LWEParams = params->GetLWEParams();
            uint32_t N{LWEParams->GetN()};
            if (q.ConvertToInt() > N) { // need q to be at most = N for arbitrary function
                std::string errMsg =
                    "ERROR: ciphertext modulus q needs to be <= ring dimension for arbitrary function evaluation";
                OPENFHE_THROW(errMsg);
            }

            // TODO: figure out a way to not do this :(

            // repeat the LUT to make it periodic
            std::vector<NativeInteger> LUT2;
            LUT2.reserve(LUT.size() + LUT.size());
            LUT2.insert(LUT2.end(), LUT.begin(), LUT.end());
            LUT2.insert(LUT2.end(), LUT.begin(), LUT.end());

            NativeInteger dq{q << 1};
            // raise the modulus of ct1 : q -> 2q
            for (size_t i = 0; i < batch_size; ++i)
                v_ct1[i]->GetA().SetModulus(dq);

            std::vector<LWECiphertext> v_ct2;
            for (size_t i = 0; i < batch_size; ++i) {
                LWECiphertext ct2 = std::make_shared<LWECiphertextImpl>(*v_ct1[i]);
                LWEscheme->EvalAddConstEq(ct2, beta);
                v_ct2.push_back(ct2);
            }

            // this is 1/4q_small or -1/4q_small mod q
            auto f0 = [](NativeInteger x, NativeInteger q, NativeInteger Q) -> NativeInteger {
                if (x < (q >> 1))
                    return Q - (q >> 2);
                else
                    return (q >> 2);
            };
            auto v_ct3 = BatchGPUBootstrapFunc(params, EK, v_ct2, f0, dq, s);
            for (size_t i = 0; i < batch_size; ++i) {
                LWEscheme->EvalSubEq2(v_ct1[i], v_ct3[i]);
                LWEscheme->EvalAddConstEq(v_ct3[i], beta);
                LWEscheme->EvalSubConstEq(v_ct3[i], q >> 1);
            }

            // Now the input is within the range [0, q/2).
            // Note that for non-periodic function, the input q is boosted up to 2q
            auto fLUT2 = [LUT2](NativeInteger x, NativeInteger q, NativeInteger Q) -> NativeInteger {
                if (x < (q >> 1))
                    return LUT2[x.ConvertToInt()];
                else
                    return Q - LUT2[x.ConvertToInt() - q.ConvertToInt() / 2];
            };
            auto v_ct4 = BatchGPUBootstrapFunc(params, EK, v_ct3, fLUT2, dq, s);
            for (size_t i = 0; i < batch_size; ++i)
                v_ct4[i]->SetModulus(q);
            return v_ct4;
        }

        // Else it's periodic function so we evaluate directly
        for (size_t i = 0; i < batch_size; ++i)
            LWEscheme->EvalAddConstEq(v_ct1[i], beta);
        // this is 1/4q_small or -1/4q_small mod q
        auto f0 = [](NativeInteger x, NativeInteger q, NativeInteger Q) -> NativeInteger {
            if (x < (q >> 1))
                return Q - (q >> 2);
            else
                return (q >> 2);
        };
        auto v_ct2 = BatchGPUBootstrapFunc(params, EK, v_ct1, f0, q, s);
        for (size_t i = 0; i < batch_size; ++i) {
            LWEscheme->EvalSubEq2(v_ct[i], v_ct2[i]);
            LWEscheme->EvalAddConstEq(v_ct2[i], beta);
            LWEscheme->EvalSubConstEq(v_ct2[i], q >> 2);
        }

        // Now the input is within the range [0, q/2).
        // Note that for non-periodic function, the input q is boosted up to 2q
        auto fLUT1 = [LUT](NativeInteger x, NativeInteger q, NativeInteger Q) -> NativeInteger {
            if (x < (q >> 1))
                return LUT[x.ConvertToInt()];
            else
                return Q - LUT[x.ConvertToInt() - q.ConvertToInt() / 2];
        };
        return BatchGPUBootstrapFunc(params, EK, v_ct2, fLUT1, q, s);
    }

    /************************* new add *************************/
    std::vector<LWECiphertext> GPUBinFHEScheme::BatchGPUEvalCMUX(const std::shared_ptr<lbcrypto::BinFHECryptoParams> &params, const GPURingGSWBTKey &EK,
                                                                 const std::vector<lbcrypto::LWECiphertext> &v_ct, const cudaStream_t &s) const {
        auto &LWEParams = params->GetLWEParams();
        auto &RGSWParams = params->GetRingGSWParams();
        auto polyParams = RGSWParams->GetPolyParams();

        NativeInteger Q = LWEParams->GetQ();
        uint32_t N = LWEParams->GetN();

        // For specific function evaluation instead of general bootstrapping
        NativeInteger ctMod = v_ct[0]->GetModulus();
        uint32_t factor = (2 * N / ctMod.ConvertToInt());

        size_t batch_size = v_ct.size();
        std::vector<NativeVector> v_m(batch_size, NativeVector(N, Q));
        for (size_t i = 0; i < batch_size; ++i) {
            const NativeInteger &b = v_ct[i]->GetB();
            for (size_t j = 0; j < (ctMod >> 1); ++j) {
                NativeInteger temp = b.ModSub(j, ctMod);
                v_m[i][j * factor] = temp;
            }
        }

        cudaStreamSynchronize(s);

        auto d_acc_batch = phantom::util::make_cuda_auto_ptr<BasicInteger>(2 * N * batch_size, s);

        for (size_t i = 0; i < batch_size; ++i) {
            cudaMemsetAsync(d_acc_batch.get() + i * 2 * N, 0, sizeof(BasicInteger) * N, s);
            cudaMemcpyAsync(d_acc_batch.get() + i * 2 * N + N, &v_m[i].at(0), sizeof(BasicInteger) * N,
                            cudaMemcpyHostToDevice, s);
        }

        std::vector<NativeVector> v_ct_A;
        for (size_t i = 0; i < batch_size; ++i) {
            v_ct_A.push_back(v_ct[i]->GetA());
        }

        GPUACCscheme_->BatchGPUEvalCMUX(params, EK, v_ct_A, d_acc_batch, s);

        // copy acc_a to host
        std::vector<NativeVector> v_acc_a(batch_size, NativeVector(N, 0));
        for (size_t i = 0; i < batch_size; ++i) {
            cudaMemcpyAsync(&v_acc_a[i].at(0), d_acc_batch.get() + i * 2 * N, N * sizeof(BasicInteger),
                            cudaMemcpyDeviceToHost, s);
        }

        // copy acc_b to host
        std::vector<NativeInteger> v_b(batch_size, 0);
        for (size_t i = 0; i < batch_size; ++i) {
            cudaMemcpyAsync((void *)&v_b[i], d_acc_batch.get() + i * 2 * N + N,
                            1 * sizeof(BasicInteger), cudaMemcpyDeviceToHost, s);
        }

        std::vector<LWECiphertext> v_result;
        for (size_t i = 0; i < batch_size; ++i) {
            v_result.push_back(std::make_shared<LWECiphertextImpl>(std::move(v_acc_a[i]), v_b[i]));
        }

        return v_result;
    }

    std::vector<LWECiphertext> GPUBinFHEScheme::BatchGPUEvalADD(const std::shared_ptr<lbcrypto::BinFHECryptoParams> &params, std::vector<lbcrypto::LWECiphertext> &input0,
                                                                const std::vector<lbcrypto::LWECiphertext> &input1, const cudaStream_t &s) const {

        // size_t batch_size = input0.size();

        // auto copy0{input0};
        // auto copy1{input1};

        // phantom::util::CUDATimer timer_add("BatchGPUEvalADD: " + std::to_string(batch_size), s);
        // timer_add.start();
        GPUACCscheme_->BatchGPUEvalADD(params, input0, input1, s);
        // timer_add.stop();

        // std::cout << "copy0: " << copy0[0]->GetA().at(0) << std::endl;
        // std::cout << "copy1: " << copy1[0]->GetA().at(0) << std::endl;

        // std::cout << "GetModulus: " << input0[0]->GetModulus() << std::endl;

        // for (size_t i = 0; i < batch_size; i++) {
        //     LWEscheme->EvalAddEq(copy0[i], copy1[i]);
        // }

        // std::cout << "result: " << copy0[0]->GetA().at(0) << std::endl;

        // cudaDeviceSynchronize();
        // CHECK_CUDA_LAST_ERROR();
        // std::cout << " BatchGPUEvalADD done" << std::endl;

        // std::vector<LWECiphertext> v_result;
        // for (size_t i = 0; i < batch_size; ++i) {
        //     v_result.push_back(std::make_shared<LWECiphertextImpl>(input0[i]->GetA(), input0[i]->GetB()));
        // }

        // return v_result;
        return input0;
    }
    /************************* new add *************************/
}
