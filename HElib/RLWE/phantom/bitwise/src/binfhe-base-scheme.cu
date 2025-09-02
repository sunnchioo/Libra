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

    GPURingGSWBTKey GPUBinFHEScheme::GPUKeyGen(const std::shared_ptr<BinFHECryptoParams> &params,
                                               ConstLWEPrivateKey &LWEsk,
                                               const cudaStream_t &s,
                                               KEYGEN_MODE keygenMode) const {
        GPURingGSWBTKey gpu_ek;

        /***************************************************************************************************************
         * generate LWE switching key on CPU
         **************************************************************************************************************/
        const auto &LWEParams = params->GetLWEParams();
        const auto &RGSWParams = params->GetRingGSWParams();

        RingGSWBTKey ek;
        LWEPrivateKey skN;
        if (keygenMode == SYM_ENCRYPT) {
            skN = LWEscheme_->KeyGen(RGSWParams->GetN(), RGSWParams->GetQ());
        } else if (keygenMode == PUB_ENCRYPT) {
            throw std::runtime_error("PUB_ENCRYPT keygen not supported yet");
        } else {
            OPENFHE_THROW("Invalid KeyGen mode");
        }

        ek.KSkey = LWEscheme_->KeySwitchGen(LWEParams, LWEsk, skN);

        /***************************************************************************************************************
         * copy LWE switching key from CPU to GPU
         **************************************************************************************************************/
        gpu_ek.cpu_keyB = ek.KSkey->GetElementsB();

        size_t dim1_A = ek.KSkey->GetElementsA().size();
        size_t dim2_A = ek.KSkey->GetElementsA()[0].size();
        size_t dim3_A = ek.KSkey->GetElementsA()[0][0].size();
        size_t dim4_A = ek.KSkey->GetElementsA()[0][0][0].GetLength();

        size_t dim_A = dim1_A * dim2_A * dim3_A * dim4_A;
        gpu_ek.LWESwitchKey_A = phantom::util::make_cuda_auto_ptr<BasicInteger>(dim_A, s);

        for (size_t i = 0; i < dim1_A; ++i) {
            for (size_t j = 0; j < dim2_A; ++j) {
                for (size_t k = 0; k < dim3_A; ++k) {
                    cudaMemcpyAsync(
                        gpu_ek.LWESwitchKey_A.get() + i * dim2_A * dim3_A * dim4_A + j * dim3_A * dim4_A +
                            k * dim4_A,
                        &ek.KSkey->GetElementsA()[i][j][k].at(0),
                        sizeof(BasicInteger) * dim4_A, cudaMemcpyHostToDevice, s);
                }
            }
        }

        /***************************************************************************************************************
         * generate RGSW ACC key on GPU
         **************************************************************************************************************/
        if (RGSWParams->IsCompositeNTT()) {
            NativeVector new_skN(RGSWParams->GetN(), RGSWParams->GetPQ());
            for (size_t i = 0; i < RGSWParams->GetN(); i++) {
                if (skN->GetElement()[i] == RGSWParams->GetQ() - 1)
                    new_skN[i] = RGSWParams->GetPQ() - 1;
                else
                    new_skN[i] = skN->GetElement()[i];
            }
            gpu_ek.RGSWACCKey = GPUACCscheme_->GPUKeyGenAcc(RGSWParams, new_skN, LWEsk, s);
        } else {
            gpu_ek.RGSWACCKey = GPUACCscheme_->GPUKeyGenAcc(RGSWParams, skN->GetElement(), LWEsk, s);
        }

        return gpu_ek;
    }

    phantom::util::cuda_auto_ptr<BasicInteger>
    GPUBinFHEScheme::GPUBootstrapGateCore(const std::shared_ptr<BinFHECryptoParams> &params,
                                          BINGATE gate,
                                          const GPURingGSWBTKey &EK,
                                          ConstLWECiphertext &ctprep,
                                          const cudaStream_t &s) const {

        auto &LWEParams = params->GetLWEParams();
        auto &RGSWParams = params->GetRingGSWParams();
        auto polyParams = RGSWParams->GetPolyParams();

        // Specifies the range [q1,q2) that will be used for mapping
        NativeInteger p = ctprep->GetptModulus();
        NativeInteger q = ctprep->GetModulus();
        uint32_t qHalf = q.ConvertToInt() >> 1;
        NativeInteger q1 = RGSWParams->GetGateConst()[static_cast<size_t>(gate)];
        NativeInteger q2 = q1.ModAddFast(NativeInteger(qHalf), q);

        // depending on whether the value is the range, it will be set
        // to either Q/8 or -Q/8 to match binary arithmetic
        NativeInteger Q = RGSWParams->GetQ();
        NativeInteger Q2p = Q / NativeInteger(2 * p) + 1;
        NativeInteger Q2pNeg = Q - Q2p;

        uint32_t N = LWEParams->GetN();
        NativeVector m(N, Q);
        // Since q | (2*N), we deal with a sparse embedding of Z_Q[x]/(X^{q/2}+1) to
        // Z_Q[x]/(X^N+1)
        uint32_t factor = (2 * N / q.ConvertToInt());

        const NativeInteger &b = ctprep->GetB();
        for (size_t j = 0; j < qHalf; ++j) {
            NativeInteger temp = b.ModSub(j, q);
            if (q1 < q2)
                m[j * factor] = ((temp >= q1) && (temp < q2)) ? Q2pNeg : Q2p;
            else
                m[j * factor] = ((temp >= q2) && (temp < q1)) ? Q2p : Q2pNeg;
        }

        cudaStreamSynchronize(s);

        auto d_acc = phantom::util::make_cuda_auto_ptr<BasicInteger>(2 * N, s);
        cudaMemsetAsync(d_acc.get(), 0, sizeof(BasicInteger) * N, s);
        cudaMemcpyAsync(d_acc.get() + N, &m.at(0), sizeof(BasicInteger) * N, cudaMemcpyHostToDevice, s);

        // main accumulation computation
        // the following loop is the bottleneck of bootstrapping/binary gate
        // evaluation
        GPUACCscheme_->GPUEvalAcc(params, EK, ctprep->GetA(), d_acc, s);

        return std::move(d_acc);
    }

    LWECiphertext GPUBinFHEScheme::GPUEvalBinGate(const std::shared_ptr<BinFHECryptoParams> &params,
                                                  BINGATE gate,
                                                  const GPURingGSWBTKey &EK,
                                                  ConstLWECiphertext &ct1,
                                                  ConstLWECiphertext &ct2,
                                                  const cudaStream_t &s) const {
        if (ct1 == ct2)
            OPENFHE_THROW("Input ciphertexts should be independent");

        LWECiphertext ctprep = std::make_shared<LWECiphertextImpl>(*ct1);
        // the additive homomorphic operation for XOR/NXOR is different from the other gates we compute
        // 2*(ct1 + ct2) mod 4 for XOR, 0 -> 0, 2 -> 1
        // XOR_FAST and XNOR_FAST are included for backwards compatibility; they map to XOR and XNOR
        if ((gate == XOR) || (gate == XNOR) || (gate == XOR_FAST) || (gate == XNOR_FAST)) {
            LWEscheme->EvalAddEq(ctprep, ct2);
            LWEscheme->EvalAddEq(ctprep, ctprep);
        } else {
            // for all other gates, we simply compute (ct1 + ct2) mod 4
            // for AND: 0,1 -> 0 and 2,3 -> 1
            // for OR: 1,2 -> 1 and 3,0 -> 0
            LWEscheme->EvalAddEq(ctprep, ct2);
        }

        auto d_acc = GPUBootstrapGateCore(params, gate, EK, ctprep, s);

        auto &LWEParams = params->GetLWEParams();
        auto &RGSWParams = params->GetRingGSWParams();
        auto polyParams = RGSWParams->GetPolyParams();

        uint32_t N = LWEParams->GetN();
        NativeInteger Q = LWEParams->GetQ();

        // Sample extract fuses modulus switching from Q to qKS
        NativeInteger qKS = LWEParams->GetqKS();
        const int logN = phantom::arith::get_power_of_two(N);
        auto d_tmp = phantom::util::make_cuda_auto_ptr<BasicInteger>(N, s);
        cudaMemcpyAsync(d_tmp.get(), d_acc.get(), N * sizeof(BasicInteger), cudaMemcpyDeviceToDevice, s);
        kernel_automorphism_modSwitch<<<N / ThreadsPerBlock, ThreadsPerBlock, 0, s>>>(
            d_acc.get(), d_tmp.get(), qKS.ConvertToInt(), Q.ConvertToInt(), logN);

        // copy acc_a to host
        NativeVector acc_a(N, qKS);
        cudaMemcpyAsync(&acc_a.at(0), d_acc.get(), N * sizeof(BasicInteger), cudaMemcpyDeviceToHost, s);

        // copy acc_b to host
        NativeInteger b{0};
        cudaMemcpyAsync((void *)&b, d_acc.get() + N,
                        1 * sizeof(BasicInteger), cudaMemcpyDeviceToHost, s);

        // explicitly synchronize to ensure that the data is copied to the host
        cudaStreamSynchronize(s);

        // Key switching reduces dimension from N to n
        // Fuses modulus switching from qKS to fmod
        uint32_t n = LWEParams->Getn();
        NativeInteger::Integer baseKS(LWEParams->GetBaseKS());
        const auto log_baseKS = GetMSB(baseKS) - 1;
        const auto digitCount = static_cast<size_t>(std::ceil(
            log(qKS.ConvertToDouble()) / log(static_cast<double>(baseKS))));

        auto d_LWE_ct_A = phantom::util::make_cuda_auto_ptr<BasicInteger>(n, s);

        kernel_LWEKeySwitch_modSwitch<<<(n + ThreadsPerBlock - 1) / ThreadsPerBlock, ThreadsPerBlock, 0, s>>>(
            d_LWE_ct_A.get(), d_acc.get(), EK.LWESwitchKey_A.get(),
            n, N, ct1->GetModulus().ConvertToInt(), qKS.ConvertToInt(), log_baseKS, digitCount);

        // copy LWE ciphertext part A to host
        NativeVector A(n, ct1->GetModulus());
        cudaMemcpyAsync(&A.at(0), d_LWE_ct_A.get(), n * sizeof(BasicInteger), cudaMemcpyDeviceToHost, s);

        // asynchronously compute key switching and mod switching for b on CPU

        // we add Q/8 to "b" to map back to Q/4 (i.e., mod 2) arithmetic.
        b.ModAddFastEq((Q >> 3) + 1, Q);

        // mod switch
        b = RoundqQ(b, qKS, Q);

        // key switch
        for (size_t i = 0; i < N; ++i) {
            NativeInteger::Integer atmp(acc_a[i].ConvertToInt());
            for (size_t j = 0; j < digitCount; ++j) {
                const auto a0 = (atmp & (baseKS - 1));
                atmp >>= log_baseKS;
                b.ModSubFastEq(EK.cpu_keyB[i][a0][j], qKS);
            }
        }

        // mod switch
        b = RoundqQ(b, ct1->GetModulus().ConvertToInt(), qKS);

        // explicitly synchronize to ensure that CPU and GPU all finish the computation
        cudaStreamSynchronize(s);

        return std::make_shared<LWECiphertextImpl>(std::move(A), b);
    }

    // Functions below are for large-precision sign evaluation,
    // flooring, homomorphic digit decomposition, and arbitrary
    // funciton evaluation, from https://eprint.iacr.org/2021/1337
    template <typename Func>
    phantom::util::cuda_auto_ptr<BasicInteger>
    GPUBinFHEScheme::GPUBootstrapFuncCore(const std::shared_ptr<BinFHECryptoParams> &params,
                                          const GPURingGSWBTKey &EK,
                                          ConstLWECiphertext &ct,
                                          Func f,
                                          const NativeInteger &fmod,
                                          const cudaStream_t &s) const {
        auto &LWEParams = params->GetLWEParams();
        auto &RGSWParams = params->GetRingGSWParams();
        auto polyParams = RGSWParams->GetPolyParams();

        NativeInteger Q = LWEParams->GetQ();
        uint32_t N = LWEParams->GetN();
        NativeVector m(N, Q);
        // For specific function evaluation instead of general bootstrapping
        NativeInteger ctMod = ct->GetModulus();
        uint32_t factor = (2 * N / ctMod.ConvertToInt());
        const NativeInteger &b = ct->GetB(); // 一个 NativeInteger
        for (size_t j = 0; j < (ctMod >> 1); ++j) {
            NativeInteger temp = b.ModSub(j, ctMod);
            m[j * factor] = Q.ConvertToInt() / fmod.ConvertToInt() * f(temp, ctMod, fmod);
        }

        cudaStreamSynchronize(s);
        auto d_acc = phantom::util::make_cuda_auto_ptr<BasicInteger>(2 * N, s); // 包含 b 的信息
        cudaMemsetAsync(d_acc.get(), 0, sizeof(BasicInteger) * N, s);
        cudaMemcpyAsync(d_acc.get() + N, &m.at(0), sizeof(BasicInteger) * N, cudaMemcpyHostToDevice, s);

        // main accumulation computation
        // the following loop is the bottleneck of bootstrapping/binary gate
        // evaluation
        GPUACCscheme_->GPUEvalAcc(params, EK, ct->GetA(), d_acc, s);

        return std::move(d_acc);
    }

    // boot in tfhe
    // Full evaluation as described in https://eprint.iacr.org/2020/086
    // Functions below are for large-precision sign evaluation,
    // flooring, homomorphic digit decomposition, and arbitrary
    // function evaluation, from https://eprint.iacr.org/2021/1337
    template <typename Func>
    LWECiphertext GPUBinFHEScheme::GPUBootstrapFunc(const std::shared_ptr<BinFHECryptoParams> &params,
                                                    const GPURingGSWBTKey &EK,
                                                    ConstLWECiphertext &ct,
                                                    Func f,
                                                    const NativeInteger &fmod,
                                                    const cudaStream_t &s) const {
        auto d_acc = GPUBootstrapFuncCore(params, EK, ct, f, fmod, s); // ACC <- b; ACC <- c_i · ek_i. GINX (mod switch + blind rotaton)

        auto &LWEParams = params->GetLWEParams();
        auto &RGSWParams = params->GetRingGSWParams();
        auto polyParams = RGSWParams->GetPolyParams();

        NativeInteger Q = LWEParams->GetQ();
        uint32_t N = LWEParams->GetN();

        // Sample extract fuses modulus switching from Q to qKS
        NativeInteger qKS = LWEParams->GetqKS();
        const int logN = phantom::arith::get_power_of_two(N);
        auto d_tmp = phantom::util::make_cuda_auto_ptr<BasicInteger>(N, s);
        cudaMemcpyAsync(d_tmp.get(), d_acc.get(), N * sizeof(BasicInteger), cudaMemcpyDeviceToDevice, s);
        kernel_automorphism_modSwitch<<<N / ThreadsPerBlock, ThreadsPerBlock, 0, s>>>( // Sample extract
            d_acc.get(), d_tmp.get(), qKS.ConvertToInt(), Q.ConvertToInt(), logN);

        // copy acc_a to host
        NativeVector acc_a(N, qKS);
        cudaMemcpyAsync(&acc_a.at(0), d_acc.get(), N * sizeof(BasicInteger), cudaMemcpyDeviceToHost, s);

        // copy acc_b to host
        NativeInteger b{0};
        cudaMemcpyAsync((void *)&b, d_acc.get() + N,
                        1 * sizeof(BasicInteger), cudaMemcpyDeviceToHost, s);

        // explicitly synchronize to ensure that the data is copied to the host
        cudaStreamSynchronize(s);

        // Key switching reduces dimension from N to n
        // Fuses modulus switching from qKS to fmod
        uint32_t n = LWEParams->Getn();
        NativeInteger::Integer baseKS(LWEParams->GetBaseKS());
        const auto log_baseKS = GetMSB(baseKS) - 1;
        const auto digitCount = static_cast<size_t>(std::ceil(
            log(qKS.ConvertToDouble()) / log(static_cast<double>(baseKS))));

        auto d_LWE_ct_A = phantom::util::make_cuda_auto_ptr<BasicInteger>(n, s);

        kernel_LWEKeySwitch_modSwitch<<<(n + ThreadsPerBlock - 1) / ThreadsPerBlock, ThreadsPerBlock, 0, s>>>( // key switch (recover the key)
            d_LWE_ct_A.get(), d_acc.get(), EK.LWESwitchKey_A.get(),
            n, N, fmod.ConvertToInt(), qKS.ConvertToInt(), log_baseKS, digitCount);

        // copy LWE ciphertext part A to host
        NativeVector A(n, fmod);
        cudaMemcpyAsync(&A.at(0), d_LWE_ct_A.get(), n * sizeof(BasicInteger), cudaMemcpyDeviceToHost, s);

        // asynchronously compute key switching and mod switching for b on CPU

        // mod switch
        b = RoundqQ(b, qKS, Q);

        // key switch
        for (size_t i = 0; i < N; ++i) {
            NativeInteger::Integer atmp(acc_a[i].ConvertToInt());
            for (size_t j = 0; j < digitCount; ++j) {
                const auto a0 = (atmp & (baseKS - 1));
                atmp >>= log_baseKS;
                b.ModSubFastEq(EK.cpu_keyB[i][a0][j], qKS);
            }
        }

        // mod switch
        b = RoundqQ(b, fmod, qKS);

        // explicitly synchronize to ensure that CPU and GPU all finish the computation
        cudaStreamSynchronize(s);

        return std::make_shared<LWECiphertextImpl>(std::move(A), b); // b is one
    }

    // boot in tfhe
    LWECiphertext GPUBinFHEScheme::GPUEvalFunc(const std::shared_ptr<BinFHECryptoParams> &params,
                                               const GPURingGSWBTKey &EK,
                                               ConstLWECiphertext &ct,
                                               const std::vector<NativeInteger> &LUT,
                                               const NativeInteger &beta,
                                               const cudaStream_t &s) const {
        auto ct1 = std::make_shared<LWECiphertextImpl>(*ct);
        NativeInteger q{ct->GetModulus()};
        uint32_t functionProperty{checkInputFunction(LUT, q)};

        if (functionProperty == 0) { // negacyclic function only needs one bootstrap
            auto fLUT = [LUT](NativeInteger x, NativeInteger q, NativeInteger Q) -> NativeInteger {
                return LUT[x.ConvertToInt()];
            };
            LWEscheme->EvalAddConstEq(ct1, beta);
            return GPUBootstrapFunc(params, EK, ct1, fLUT, q, s);
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
            ct1->GetA().SetModulus(dq);

            auto ct2 = std::make_shared<LWECiphertextImpl>(*ct1);
            LWEscheme->EvalAddConstEq(ct2, beta);
            // this is 1/4q_small or -1/4q_small mod q
            auto f0 = [](NativeInteger x, NativeInteger q, NativeInteger Q) -> NativeInteger {
                if (x < (q >> 1))
                    return Q - (q >> 2);
                else
                    return (q >> 2);
            };
            auto ct3 = GPUBootstrapFunc(params, EK, ct2, f0, dq, s);
            LWEscheme->EvalSubEq2(ct1, ct3);
            LWEscheme->EvalAddConstEq(ct3, beta);
            LWEscheme->EvalSubConstEq(ct3, q >> 1);

            // Now the input is within the range [0, q/2).
            // Note that for non-periodic function, the input q is boosted up to 2q
            auto fLUT2 = [LUT2](NativeInteger x, NativeInteger q, NativeInteger Q) -> NativeInteger {
                if (x < (q >> 1))
                    return LUT2[x.ConvertToInt()];
                else
                    return Q - LUT2[x.ConvertToInt() - q.ConvertToInt() / 2];
            };
            auto ct4 = GPUBootstrapFunc(params, EK, ct3, fLUT2, dq, s);
            ct4->SetModulus(q);
            return ct4;
        }

        // Else it's periodic function so we evaluate directly
        LWEscheme->EvalAddConstEq(ct1, beta);
        // this is 1/4q_small or -1/4q_small mod q
        auto f0 = [](NativeInteger x, NativeInteger q, NativeInteger Q) -> NativeInteger {
            if (x < (q >> 1))
                return Q - (q >> 2);
            else
                return (q >> 2);
        };
        auto ct2 = GPUBootstrapFunc(params, EK, ct1, f0, q, s);
        LWEscheme->EvalSubEq2(ct, ct2);
        LWEscheme->EvalAddConstEq(ct2, beta);
        LWEscheme->EvalSubConstEq(ct2, q >> 2);

        // Now the input is within the range [0, q/2).
        // Note that for non-periodic function, the input q is boosted up to 2q
        auto fLUT1 = [LUT](NativeInteger x, NativeInteger q, NativeInteger Q) -> NativeInteger {
            if (x < (q >> 1))
                return LUT[x.ConvertToInt()];
            else
                return Q - LUT[x.ConvertToInt() - q.ConvertToInt() / 2];
        };
        return GPUBootstrapFunc(params, EK, ct2, fLUT1, q, s);
    }

    // Evaluate Homomorphic Flooring
    LWECiphertext GPUBinFHEScheme::GPUEvalFloor(const std::shared_ptr<BinFHECryptoParams> &params,
                                                const GPURingGSWBTKey &EK,
                                                ConstLWECiphertext &ct,
                                                const NativeInteger &beta,
                                                const cudaStream_t &s,
                                                uint32_t roundbits) const {
        const auto &LWEParams = params->GetLWEParams();
        NativeInteger q{roundbits == 0 ? LWEParams->Getq() : beta * (1 << roundbits + 1)};
        NativeInteger mod{ct->GetModulus()};

        auto ct1 = std::make_shared<LWECiphertextImpl>(*ct);
        LWEscheme->EvalAddConstEq(ct1, beta);

        auto ct1Modq = std::make_shared<LWECiphertextImpl>(*ct1);
        ct1Modq->SetModulus(q);
        // this is 1/4q_small or -1/4q_small mod q
        auto f1 = [](NativeInteger x, NativeInteger q, NativeInteger Q) -> NativeInteger {
            if (x < (q >> 1))
                return Q - (q >> 2);
            else
                return (q >> 2);
        };
        auto ct2 = GPUBootstrapFunc(params, EK, ct1Modq, f1, mod, s);
        LWEscheme->EvalSubEq(ct1, ct2);

        auto ct2Modq = std::make_shared<LWECiphertextImpl>(*ct1);
        ct2Modq->SetModulus(q);

        // now the input is only within the range [0, q/2)
        auto f2 = [](NativeInteger x, NativeInteger q, NativeInteger Q) -> NativeInteger {
            if (x < (q >> 2))
                return Q - (q >> 1) - x;
            else if (((q >> 2) <= x) && (x < 3 * (q >> 2)))
                return x;
            else
                return Q + (q >> 1) - x;
        };
        auto ct3 = GPUBootstrapFunc(params, EK, ct2Modq, f2, mod, s);
        LWEscheme->EvalSubEq(ct1, ct3);

        return ct1;
    }

    LWECiphertext GPUBinFHEScheme::GPUEvalSign(const std::shared_ptr<lbcrypto::BinFHECryptoParams> &params,
                                               const std::map<uint32_t, GPURingGSWBTKey> &EKs,
                                               ConstLWECiphertext &ct, const NativeInteger &beta,
                                               const cudaStream_t &s, bool schemeSwitch) const {
        auto mod{ct->GetModulus()};
        const auto &LWEParams = params->GetLWEParams();
        auto q{LWEParams->Getq()};
        if (mod <= q) {
            std::string errMsg =
                "ERROR: EvalSign is only for large precision. For small precision, please use bootstrapping directly";
            OPENFHE_THROW(errMsg);
        }

        const auto &RGSWParams = params->GetRingGSWParams();
        const auto curBase = RGSWParams->GetBaseG();
        auto search = EKs.find(curBase);
        if (search == EKs.end()) {
            std::string errMsg("ERROR: No key [" + std::to_string(curBase) + "] found in the map");
            OPENFHE_THROW(errMsg);
        }
        GPURingGSWBTKey curEK(search->second);

        auto cttmp = std::make_shared<LWECiphertextImpl>(*ct);
        while (mod > q) {
            cttmp = GPUEvalFloor(params, curEK, cttmp, beta, s);
            // round Q to 2betaQ/q
            //  mod   = mod / q * 2 * beta;
            mod = (mod << 1) * beta / q;
            cttmp = LWEscheme->ModSwitch(mod, cttmp);

            // if dynamic
            if (EKs.size() == 3) {
                // TODO: use GetMSB()?
                uint32_t binLog = static_cast<uint32_t>(ceil(GetMSB(mod.ConvertToInt()) - 1));
                uint32_t base{0};
                if (binLog <= static_cast<uint32_t>(17))
                    base = static_cast<uint32_t>(1) << 27;
                else if (binLog <= static_cast<uint32_t>(26))
                    base = static_cast<uint32_t>(1) << 18;

                if (0 != base) { // if base is to change ...
                    RGSWParams->Change_BaseG(base);

                    auto search = EKs.find(base);
                    if (search == EKs.end()) {
                        std::string errMsg("ERROR: No key [" + std::to_string(curBase) + "] found in the map");
                        OPENFHE_THROW(errMsg);
                    }
                    curEK = search->second;
                }
            }
        }
        LWEscheme->EvalAddConstEq(cttmp, beta);

        if (!schemeSwitch) {
            // if the ended q is smaller than q, we need to change the param for the final boostrapping
            auto f3 = [](NativeInteger x, NativeInteger q, NativeInteger Q) -> NativeInteger {
                return (x < q / 2) ? (Q / 4) : (Q - Q / 4);
            };
            cttmp = GPUBootstrapFunc(params, curEK, cttmp, f3, q, s); // this is 1/4q_small or -1/4q_small mod q
            LWEscheme->EvalSubConstEq(cttmp, q >> 2);
        } else { // return the negated f3 and do not subtract q/4 for a more natural encoding in scheme switching
            // if the ended q is smaller than q, we need to change the param for the final boostrapping
            auto f3 = [](NativeInteger x, NativeInteger q, NativeInteger Q) -> NativeInteger {
                return (x < q / 2) ? (Q - Q / 4) : (Q / 4);
            };
            cttmp = GPUBootstrapFunc(params, curEK, cttmp, f3, q, s); // this is 1/4q_small or -1/4q_small mod q
        }
        RGSWParams->Change_BaseG(curBase);
        return cttmp;
    }
}
