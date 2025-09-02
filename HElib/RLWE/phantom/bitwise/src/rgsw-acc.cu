#include "kernel.cuh"
#include "rgsw-acc.cuh"

#include "math/distributiongenerator.h"

#include "arith/big_integer_modop.h"

using namespace lbcrypto;

namespace phantom::bitwise {
    GPURingGSWACCKey GPURingGSWAccumulator::GPUKeyGenAcc(const std::shared_ptr<RingGSWCryptoParams> &params,
                                                         const NativeVector &sk,
                                                         ConstLWEPrivateKey &LWEsk,
                                                         const cudaStream_t &stream) {
        // initialize NTT
        const size_t N = params->GetN();
        const BasicInteger Q = params->GetQ().ConvertToInt();
        const BasicInteger P = params->GetP().ConvertToInt();
        const BasicInteger PQ = params->GetPQ().ConvertToInt();

        const auto &polyParams = (params->IsCompositeNTT())
                                     ? params->GetCompositePolyParams()
                                     : params->GetPolyParams();

        if (params->IsCompositeNTT()) {
            ntt_ = std::make_shared<FourStepNTT>(N, P, Q, stream);
        } else {
            ntt_ = std::make_shared<FourStepNTT>(N, Q, stream);
        }

        // initialize monic polynomials for CGGI
        if (params->GetMethod() == lbcrypto::BINFHE_METHOD::GINX) {
            // Precomputed polynomials in Format::EVALUATION representation for X^m - 1
            // (used only for CGGI bootstrapping)
            std::vector<NativePoly> monomials;

            constexpr NativeInteger one{1};
            monomials.reserve(2 * N);
            for (uint32_t i = 0; i < N; ++i) {
                NativePoly aPoly(polyParams, Format::COEFFICIENT, true);
                if (params->IsCompositeNTT()) {
                    // composite NTT
                    aPoly[0].ModSubFastEq(one, PQ); // -1
                    aPoly[i].ModAddFastEq(one, PQ); // X^m
                } else {
                    // gadget decompose
                    aPoly[0].ModSubFastEq(one, Q); // -1
                    aPoly[i].ModAddFastEq(one, Q); // X^m 第几个单项式(-1, ... -x^i)
                }
                monomials.push_back(std::move(aPoly));
            }
            for (uint32_t i = 0; i < N; ++i) {
                NativePoly aPoly(polyParams, Format::COEFFICIENT, true);
                if (params->IsCompositeNTT()) {
                    // composite NTT
                    aPoly[0].ModSubFastEq(one, PQ); // -1
                    aPoly[i].ModSubFastEq(one, PQ); // -X^m
                } else {
                    // gadget decompose
                    aPoly[0].ModSubFastEq(one, Q); // -1
                    aPoly[i].ModSubFastEq(one, Q); // -X^m 第几个单项式(-1, ... -x^i)
                }
                monomials.push_back(std::move(aPoly));
            }

            d_monic_polys_ = phantom::util::make_cuda_auto_ptr<BasicInteger>(2 * N * N, stream);
            for (size_t i = 0; i < 2 * N; i++) {
                cudaMemcpyAsync(d_monic_polys_.get() + i * N, &monomials[i].GetValues().at(0),
                                N * sizeof(BasicInteger), cudaMemcpyHostToDevice, stream);
            }
            ntt_->forward(d_monic_polys_.get(), d_monic_polys_.get(), 1024, 2 * N, stream); // ntt 域
        }

        auto d_skNTT = phantom::util::make_cuda_auto_ptr<BasicInteger>(N, stream);
        cudaMemcpyAsync(d_skNTT.get(), &sk.at(0), N * sizeof(BasicInteger), cudaMemcpyHostToDevice, stream);
        ntt_->forward(d_skNTT.get(), d_skNTT.get(), 1024, 1, stream);

        switch (params->GetMethod()) {
        case BINFHE_METHOD::AP:
            return GPUKeyGenAccDM(params, d_skNTT, LWEsk, stream);
        case BINFHE_METHOD::GINX:
            return GPUKeyGenAccCGGI(params, d_skNTT, LWEsk, stream);
        //            case BINFHE_METHOD::LMKCDEY:
        //                return GPUKeyGenLMKCDEY(params, d_skNTT, LWEsk, stream);
        default:
            throw std::invalid_argument("ERROR: Invalid ACC method");
        }
    }

    // Key generation as described in Section 4 of https://eprint.iacr.org/2014/816
    GPURingGSWACCKey GPURingGSWAccumulator::GPUKeyGenAccDM(const std::shared_ptr<RingGSWCryptoParams> &params,
                                                           const phantom::util::cuda_auto_ptr<BasicInteger> &d_skNTT,
                                                           ConstLWEPrivateKey &LWEsk,
                                                           const cudaStream_t &stream) const {
        auto sv{LWEsk->GetElement()};
        auto mod{sv.GetModulus().ConvertToInt<int32_t>()};
        auto modHalf{mod >> 1};
        uint32_t n(sv.GetLength());
        int32_t baseR(params->GetBaseR());
        const auto &digitsR = params->GetDigitsR();

        GPURingGSWACCKey d_ek;
        d_ek.resize(n, std::vector<std::vector<phantom::util::cuda_auto_ptr<BasicInteger>>>(
                           baseR, std::vector<phantom::util::cuda_auto_ptr<BasicInteger>>(digitsR.size())));

        for (uint32_t i = 0; i < n; ++i) {
            for (int32_t j = 0; j < baseR; ++j) {
                for (size_t k = 0; k < digitsR.size(); ++k) {
                    auto s{sv[i].ConvertToInt<int32_t>()};
                    LWEPlaintext m = (s > modHalf ? s - mod : s) * j * digitsR[k].ConvertToInt<int32_t>();
                    d_ek[i][j][k] = GPUKeyGenDM(params, d_skNTT, m, stream);
                }
            }
        }
        return d_ek;
    }

    // Key generation as described in Section 4 of https://eprint.iacr.org/2014/816
    GPURingGSWACCKey GPURingGSWAccumulator::GPUKeyGenAccCGGI(const std::shared_ptr<RingGSWCryptoParams> &params,
                                                             const phantom::util::cuda_auto_ptr<BasicInteger> &d_skNTT,
                                                             ConstLWEPrivateKey &LWEsk,
                                                             const cudaStream_t &stream) const {
        auto sv = LWEsk->GetElement();
        auto neg = sv.GetModulus().ConvertToInt() - 1;
        uint32_t n = sv.GetLength();

        GPURingGSWACCKey d_ek;
        d_ek.resize(1, std::vector<std::vector<phantom::util::cuda_auto_ptr<BasicInteger>>>(
                           2, std::vector<phantom::util::cuda_auto_ptr<BasicInteger>>(n)));

        if (params->GetKeyDist() == UNIFORM_TERNARY) {
            // handles ternary secrets using signed mod 3 arithmetic
            // 0 -> {0,0}, 1 -> {1,0}, -1 -> {0,1}
            for (uint32_t i = 0; i < n; ++i) {
                auto s = sv[i].ConvertToInt();
                d_ek[0][0][i] = GPUKeyGenCGGI(params, d_skNTT, s == 1 ? 1 : 0, stream);
                d_ek[0][1][i] = GPUKeyGenCGGI(params, d_skNTT, s == neg ? 1 : 0, stream);
            }
        } else if (params->GetKeyDist() == BINARY) {
            for (uint32_t i = 0; i < n; ++i) {
                auto s = sv[i].ConvertToInt();
                d_ek[0][0][i] = GPUKeyGenCGGI(params, d_skNTT, s, stream);
            }
        } else {
            throw std::invalid_argument("ERROR: Invalid key distribution for CGGI");
        }

        return d_ek;
    }

    // Encryption as described in Section 5 of https://eprint.iacr.org/2014/816
    // skNTT corresponds to the secret key z
    phantom::util::cuda_auto_ptr<BasicInteger>
    GPURingGSWAccumulator::GPUKeyGenDM(const std::shared_ptr<RingGSWCryptoParams> &params,
                                       const phantom::util::cuda_auto_ptr<BasicInteger> &d_skNTT,
                                       LWEPlaintext m,
                                       const cudaStream_t &stream) const {
        // composite NTT
        const auto &polyParams = params->IsCompositeNTT()
                                     ? params->GetCompositePolyParams()
                                     : params->GetPolyParams();

        DiscreteUniformGeneratorImpl<NativeVector> dug;
        NativeInteger Q{params->GetQ()};
        NativeInteger P{params->GetP()};
        NativeInteger PQ{params->GetPQ()};

        // Reduce mod q (dealing with negative number as well)
        uint64_t q = params->Getq().ConvertToInt();
        uint32_t N = params->GetN();
        int64_t mm = (((m % q) + q) % q) * (2 * N / q);
        bool isReducedMM;
        if ((isReducedMM = (mm >= N)))
            mm -= N;

        uint32_t digitsG2{(params->GetDigitsG() - 1) << 1};
        RingGSWEvalKeyImpl result(digitsG2, 2);
        auto d_result = phantom::util::make_cuda_auto_ptr<BasicInteger>(digitsG2 * 2 * N, stream);

        // generate GPowers
        std::vector<NativeInteger> Gpow;
        Gpow.reserve(params->GetDigitsG());
        NativeInteger vTemp = 1;
        for (uint32_t i = 0; i < params->GetDigitsG(); ++i) {
            Gpow.push_back(vTemp);
            vTemp = vTemp.ModMulFast(NativeInteger(params->GetBaseG()), polyParams->GetModulus());
        }

        for (uint32_t i = 0; i < digitsG2; ++i) {
            NativePoly tempA = NativePoly(dug, polyParams, Format::COEFFICIENT);
            auto d_tempA = phantom::util::make_cuda_auto_ptr<BasicInteger>(N, stream);
            cudaMemcpyAsync(d_tempA.get(), &tempA.GetValues().at(0),
                            N * sizeof(BasicInteger), cudaMemcpyHostToDevice, stream);

            result[i][0] = tempA;
            result[i][1] = NativePoly(params->GetDgg(), polyParams, Format::COEFFICIENT);

            // 2: hybrid; 1: composite; 0: gadget decompose
            if (!isReducedMM) {
                if (params->IsCompositeNTT() == 2)
                    result[i][i & 0x1][mm].ModAddFastEq(Gpow[(i >> 1) + 1], PQ);
                else if (params->IsCompositeNTT() == 1)
                    result[i][i & 0x1][mm].ModAddFastEq(P, PQ);
                else
                    result[i][i & 0x1][mm].ModAddFastEq(Gpow[(i >> 1) + 1], Q);
            } else {
                if (params->IsCompositeNTT() == 2)
                    result[i][i & 0x1][mm].ModSubFastEq(Gpow[(i >> 1) + 1], PQ);
                else if (params->IsCompositeNTT() == 1)
                    result[i][i & 0x1][mm].ModSubFastEq(P, PQ);
                else
                    result[i][i & 0x1][mm].ModSubFastEq(Gpow[(i >> 1) + 1], Q);
            }

            BasicInteger *d_result_i_0 = d_result.get() + i * 2 * N;
            cudaMemcpyAsync(d_result_i_0, &result[i][0].GetValues().at(0),
                            N * sizeof(BasicInteger), cudaMemcpyHostToDevice, stream);

            BasicInteger *d_result_i_1 = d_result_i_0 + N;
            cudaMemcpyAsync(d_result_i_1, &result[i][1].GetValues().at(0),
                            N * sizeof(BasicInteger), cudaMemcpyHostToDevice, stream);

            ntt_->forward(d_result_i_0, d_result_i_0, 1024, 2, stream);
            ntt_->forward(d_tempA.get(), d_tempA.get(), 1024, 1, stream);
            ntt_->multiply_and_accumulate(d_result_i_1, d_tempA.get(), d_skNTT.get(), stream);
        }

        cudaStreamSynchronize(stream);
        return d_result;
    }

    // Encryption for the CGGI variant, as described in https://eprint.iacr.org/2020/086
    phantom::util::cuda_auto_ptr<BasicInteger>
    GPURingGSWAccumulator::GPUKeyGenCGGI(const std::shared_ptr<RingGSWCryptoParams> &params,
                                         const phantom::util::cuda_auto_ptr<BasicInteger> &d_skNTT,
                                         LWEPlaintext m,
                                         const cudaStream_t &stream) const {
        // composite NTT
        const auto &polyParams = params->IsCompositeNTT()
                                     ? params->GetCompositePolyParams()
                                     : params->GetPolyParams();

        DiscreteUniformGeneratorImpl<NativeVector> dug;
        NativeInteger Q{params->GetQ()};
        NativeInteger P{params->GetP()};
        NativeInteger PQ{params->GetPQ()};
        uint32_t N = params->GetN();

        uint32_t digitsG2{(params->GetDigitsG() - 1) << 1};
        RingGSWEvalKeyImpl result(digitsG2, 2);
        auto d_result = phantom::util::make_cuda_auto_ptr<BasicInteger>(digitsG2 * 2 * N, stream);

        // generate GPowers
        std::vector<NativeInteger> Gpow;
        Gpow.reserve(params->GetDigitsG());
        NativeInteger vTemp = 1;
        for (uint32_t i = 0; i < params->GetDigitsG(); ++i) {
            Gpow.push_back(vTemp);
            vTemp = vTemp.ModMulFast(NativeInteger(params->GetBaseG()), polyParams->GetModulus()); // 在这里 乘以 baseG，所以在GPU上可以直接相乘
        }

        for (uint32_t i = 0; i < digitsG2; ++i) {
            result[i][0] = NativePoly(dug, polyParams, Format::COEFFICIENT);
            result[i][1] = NativePoly(params->GetDgg(), polyParams, Format::COEFFICIENT);

            auto d_tempA = phantom::util::make_cuda_auto_ptr<BasicInteger>(N, stream);
            cudaMemcpyAsync(d_tempA.get(), &result[i][0].GetValues().at(0),
                            N * sizeof(BasicInteger), cudaMemcpyHostToDevice, stream);

            // 2: hybrid; 1: composite; 0: gadget decompose
            if (m) {
                if (params->IsCompositeNTT() == 2) {
                    result[i][i & 0x1][0].ModAddFastEq(Gpow[(i >> 1) + 1], PQ);
                } else if (params->IsCompositeNTT() == 1)
                    result[i][i & 0x1][0].ModAddFastEq(P, PQ);
                else
                    result[i][i & 0x1][0].ModAddFastEq(Gpow[(i >> 1) + 1], Q);
            }

            BasicInteger *d_result_i_0 = d_result.get() + i * 2 * N;
            cudaMemcpyAsync(d_result_i_0, &result[i][0].GetValues().at(0),
                            N * sizeof(BasicInteger), cudaMemcpyHostToDevice, stream);

            BasicInteger *d_result_i_1 = d_result_i_0 + N;
            cudaMemcpyAsync(d_result_i_1, &result[i][1].GetValues().at(0),
                            N * sizeof(BasicInteger), cudaMemcpyHostToDevice, stream);

            ntt_->forward(d_result_i_0, d_result_i_0, 1024, 2, stream);
            ntt_->forward(d_tempA.get(), d_tempA.get(), 1024, 1, stream);
            ntt_->multiply_and_accumulate(d_result_i_1, d_tempA.get(), d_skNTT.get(), stream);
        }

        cudaStreamSynchronize(stream);
        return d_result;
    }

    void GPURingGSWAccumulator::GPUEvalAcc(const std::shared_ptr<BinFHECryptoParams> &params,
                                           const GPURingGSWBTKey &EK,
                                           const NativeVector &a,
                                           const phantom::util::cuda_auto_ptr<BasicInteger> &d_acc,
                                           const cudaStream_t &s) const {
        switch (params->GetRingGSWParams()->GetMethod()) {
        case BINFHE_METHOD::AP:
            GPUEvalAccDM(params, EK, a, d_acc, s);
            break;
        case BINFHE_METHOD::GINX:
            GPUEvalAccCGGI(params, EK, a, d_acc, s);
            break;
        //            case BINFHE_METHOD::LMKCDEY:
        //                GPUEvalAccLMKCDEY(params, EK, a, d_acc, s);
        //                break;
        default:
            throw std::invalid_argument("ERROR: Invalid ACC method");
        }
    }

    void GPURingGSWAccumulator::GPUEvalAccDM(const std::shared_ptr<BinFHECryptoParams> &params,
                                             const GPURingGSWBTKey &EK,
                                             const NativeVector &a,
                                             const phantom::util::cuda_auto_ptr<BasicInteger> &d_acc,
                                             const cudaStream_t &s) const {
        auto &LWEParams = params->GetLWEParams();
        auto &RGSWParams = params->GetRingGSWParams();
        auto polyParams = RGSWParams->GetPolyParams();
        const size_t n = LWEParams->Getn();
        auto q = LWEParams->Getq().ConvertToInt();
        auto Q = RGSWParams->GetQ().ConvertToInt();
        auto P = RGSWParams->GetP().ConvertToInt();
        auto PQ = RGSWParams->GetPQ().ConvertToInt();
        const size_t N = RGSWParams->GetN();
        NativeInteger baseR = RGSWParams->GetBaseR();
        const auto &digitsR = RGSWParams->GetDigitsR();
        uint32_t digitsG2 = (RGSWParams->GetDigitsG() - 1) << 1;

        auto d_acc_ntt = phantom::util::make_cuda_auto_ptr<BasicInteger>(2 * N, s);

        // approximate gadget decomposition is used; the first digit is ignored
        auto d_dct = phantom::util::make_cuda_auto_ptr<BasicInteger>(digitsG2 * N, s);

        for (size_t i = 0; i < n; ++i) {
            auto aI = NativeInteger(0).ModSubFast(a[i], q);
            for (size_t k = 0; k < digitsR.size(); ++k, aI /= baseR) {
                auto a0 = (aI.Mod(baseR)).ConvertToInt<uint32_t>();
                if (a0) {
                    // AP Accumulation as described in https://eprint.iacr.org/2020/086
                    if (RGSWParams->IsCompositeNTT() == 2) {
                        // hybrid
                        // temporary storage for hybrid method
                        auto d_acc_temp = phantom::util::make_cuda_auto_ptr<BasicInteger>(2 * N, s);
                        ntt_->multiply_scalar(d_acc_temp.get(), d_acc.get(), P, s);
                        kernel_SignedDigitDecompose<<<digitsG2, 1024, 0, s>>>(
                            d_dct.get(), d_acc_temp.get(),
                            RGSWParams->GetPQ().ConvertToInt(), RGSWParams->GetBaseG(), N);
                        ntt_->forward(d_dct.get(), d_dct.get(), 1024, digitsG2, s);
                    } else if (RGSWParams->IsCompositeNTT() == 1) {
                        // composite NTT
                        ntt_->forward(d_dct.get(), d_acc.get(), 1024, 2, s);
                    } else {
                        // gadget decompose
                        //                        if (N == 1024) {
                        //                            constexpr size_t n1 = 32;
                        //                            constexpr size_t n2 = 32;
                        //                            size_t sMemSize = n1 * (n2 + 1) + 2 * n1;
                        //                            kernel_SignedDigitDecompose_fuse_1024<<<digitsG2, 1024, sMemSize, s>>>(
                        //                                    d_dct.get(), d_acc.get(),
                        //                                    RGSWParams->GetQ().ConvertToInt(), RGSWParams->GetBaseG(),
                        //                                    ntt_->getTwRoot2n1(), ntt_->getTwRoot2n1Shoup(),
                        //                                    ntt_->getTwRoot2n(), ntt_->getTwRoot2nShoup());
                        //                        } else {
                        kernel_SignedDigitDecompose<<<digitsG2, 1024, 0, s>>>(
                            d_dct.get(), d_acc.get(),
                            RGSWParams->GetQ().ConvertToInt(), RGSWParams->GetBaseG(), N);
                        ntt_->forward(d_dct.get(), d_dct.get(), 1024, digitsG2, s);
                        //                        }
                    }

                    // acc = dct * ek (matrix product);
                    BasicInteger *d_ACCKey = EK.RGSWACCKey[i][a0][k].get();

                    //                    if (N == 1024) {
                    //                        constexpr size_t n1 = 32;
                    //                        constexpr size_t n2 = 32;
                    //                        size_t sMemSize = n1 * (n2 + 1) + 2 * n1;
                    //                        kernel_EvalAccCoreDM_fuse_1024<<<2, 1024, sMemSize, s>>>(
                    //                                d_acc.get(), d_dct.get(), d_ACCKey, N,
                    //                                ntt_->getMod(), ntt_->getMu()[0], ntt_->getMu()[1], digitsG2,
                    //                                ntt_->getTwInvRoot2n(), ntt_->getTwInvRoot2nShoup(),
                    //                                ntt_->getTwInvRoot2n1(), ntt_->getTwInvRoot2n1Shoup(),
                    //                                ntt_->getInvn(), ntt_->getInvnShoup(),
                    //                                RGSWParams->IsCompositeNTT(), P);
                    //                    } else {
                    kernel_EvalAccCoreDM<<<2, 1024, 0, s>>>(
                        d_acc_ntt.get(), d_dct.get(), d_ACCKey,
                        N, ntt_->getMod(), ntt_->getMu()[0], ntt_->getMu()[1], digitsG2);

                    ntt_->inverse(d_acc.get(), d_acc_ntt.get(), 1024, 2, s);

                    // composite NTT
                    if (RGSWParams->IsCompositeNTT()) {
                        // scale P
                        kernel_scale_by_p<<<2, 1024, 0, s>>>(
                            d_acc.get(), d_acc.get(), N, Q, PQ);
                    }
                    //                    }
                }
            }
        }
    }

    void GPURingGSWAccumulator::GPUEvalAccCGGI(const std::shared_ptr<BinFHECryptoParams> &params,
                                               const GPURingGSWBTKey &EK,
                                               const NativeVector &a,
                                               const phantom::util::cuda_auto_ptr<BasicInteger> &d_acc,
                                               const cudaStream_t &s) const {
        auto &LWEParams = params->GetLWEParams();
        auto &RGSWParams = params->GetRingGSWParams();
        auto polyParams = RGSWParams->GetPolyParams();
        const size_t n = LWEParams->Getn();
        const size_t N = RGSWParams->GetN();
        auto Q = RGSWParams->GetQ().ConvertToInt();
        auto P = RGSWParams->GetP().ConvertToInt();
        auto PQ = RGSWParams->GetPQ().ConvertToInt();
        uint32_t digitsG2 = (RGSWParams->GetDigitsG() - 1) << 1;
        const auto &mod = a.GetModulus();
        const NativeInteger M{2 * RGSWParams->GetN()};
        const auto MbyMod{2 * RGSWParams->GetN() / a.GetModulus()};
        // std::cout << "params: " << n << " " << N << std::endl; // 1305 1024
        auto d_acc_ntt = phantom::util::make_cuda_auto_ptr<BasicInteger>(2 * N, s);

        // approximate gadget decomposition is used; the first digit is ignored
        auto d_dct = phantom::util::make_cuda_auto_ptr<BasicInteger>(digitsG2 * N, s);

        for (size_t n_idx = 0; n_idx < n; ++n_idx) { // 密钥的长度是 n
            // handles -a*E(1) and handles -a*E(-1) = a*E(1)
            // CGGI Accumulation as described in https://eprint.iacr.org/2020/086
            // Added ternary MUX introduced in paper https://eprint.iacr.org/2022/074.pdf section 5
            // We optimize the algorithm by multiplying the monomial after the external product
            // This reduces the number of polynomial multiplications which further reduces the runtime
            if (RGSWParams->IsCompositeNTT() == 2) {
                // hybrid
                // temporary storage for hybrid method
                auto d_acc_temp = phantom::util::make_cuda_auto_ptr<BasicInteger>(2 * N, s);
                ntt_->multiply_scalar(d_acc_temp.get(), d_acc.get(), P, s);
                kernel_SignedDigitDecompose<<<digitsG2, 1024, 0, s>>>(
                    d_dct.get(), d_acc_temp.get(),
                    RGSWParams->GetPQ().ConvertToInt(), RGSWParams->GetBaseG(), N);
                ntt_->forward(d_dct.get(), d_dct.get(), 1024, digitsG2, s);
            } else if (RGSWParams->IsCompositeNTT() == 1) {
                // composite NTT
                ntt_->forward(d_dct.get(), d_acc.get(), 1024, 2, s);
            } else {
                // gadget decompose
                //                if (N == 1024) {
                //                    constexpr size_t n1 = 32;
                //                    constexpr size_t n2 = 32;
                //                    size_t sMemSize = n1 * (n2 + 1) + 2 * n1;
                //                    kernel_SignedDigitDecompose_fuse_1024<<<digitsG2, 1024, sMemSize, s>>>(
                //                            d_dct.get(), d_acc.get(),
                //                            RGSWParams->GetQ().ConvertToInt(), RGSWParams->GetBaseG(),
                //                            ntt_->getTwRoot2n1(), ntt_->getTwRoot2n1Shoup(),
                //                            ntt_->getTwRoot2n(), ntt_->getTwRoot2nShoup());
                //                } else {
                kernel_SignedDigitDecompose<<<digitsG2, 1024, 0, s>>>(
                    d_dct.get(), d_acc.get(),
                    RGSWParams->GetQ().ConvertToInt(), RGSWParams->GetBaseG(), N);
                ntt_->forward(d_dct.get(), d_dct.get(), 1024, digitsG2, s);
                //                }
            }

            //            if (N == 1024) {
            //                constexpr size_t n1 = 32;
            //                constexpr size_t n2 = 32;
            //                size_t sMemSize = n1 * (n2 + 1) + 2 * n1;
            //                kernel_EvalAccCoreCGGI_fuse_1024<<<2, 1024, sMemSize, s>>>(
            //                        d_acc.get(), d_dct.get(), d_ACCKey0, d_ACCKey1, d_monic_polys_.get(), N,
            //                        ntt_->getMod(), ntt_->getMu()[0], ntt_->getMu()[1], digitsG2, indexPos, indexNeg,
            //                        ntt_->getTwInvRoot2n(), ntt_->getTwInvRoot2nShoup(),
            //                        ntt_->getTwInvRoot2n1(), ntt_->getTwInvRoot2n1Shoup(),
            //                        ntt_->getInvn(), ntt_->getInvnShoup(),
            //                        RGSWParams->IsCompositeNTT(), P, Q);
            //            } else {
            if (params->GetLWEParams()->GetKeyDist() == UNIFORM_TERNARY) { // {-1, 0, 1}
                // std::cout << "UNIFORM_TERNARY" << std::endl;
                // obtain both monomial(index) for sk = 1 and monomial(-index) for sk = -1
                // index is in range [0,m] - so we need to adjust the edge case when index == m to index = 0
                NativeInteger ai = NativeInteger(0).ModSubFast(a[n_idx], mod) * MbyMod;
                auto indexPos = ai.ConvertToInt<uint32_t>();
                auto indexNeg = NativeInteger(0).ModSubFast(ai, M).ConvertToInt<uint32_t>();
                if (indexPos >= 2 * N || indexNeg >= 2 * N)
                    throw std::invalid_argument("ERROR: indexPos or indexNeg out of bound");

                // acc = acc + dct * ek1 * monomial + dct * ek2 * negative_monomial;
                // Needs to be done using two loops for ternary secrets.
                BasicInteger *d_ACCKey0 = EK.RGSWACCKey[0][0][n_idx].get();
                BasicInteger *d_ACCKey1 = EK.RGSWACCKey[0][1][n_idx].get();

                kernel_EvalAccCoreCGGI<<<2, 1024, 0, s>>>( // cmux
                    d_acc_ntt.get(), d_dct.get(), d_ACCKey0, d_ACCKey1, d_monic_polys_.get(), N,
                    ntt_->getMod(), ntt_->getMu()[0], ntt_->getMu()[1], digitsG2, indexPos, indexNeg);
            } else if (params->GetLWEParams()->GetKeyDist() == BINARY) { // {0, 1}
                // std::cout << "BINARY" << std::endl;
                // obtain both monomial(index) for sk = 1 and monomial(-index) for sk = -1
                // index is in range [0,m] - so we need to adjust the edge case when index == m to index = 0
                NativeInteger ai = NativeInteger(0).ModSubFast(a[n_idx], mod) * MbyMod;
                auto indexPos = ai.ConvertToInt<uint32_t>();
                if (indexPos >= 2 * N)
                    throw std::invalid_argument("ERROR: indexPos out of bound");

                // acc = acc + dct * ek1 * monomial + dct * ek2 * negative_monomial;
                // Needs to be done using two loops for ternary secrets.
                BasicInteger *d_ACCKey = EK.RGSWACCKey[0][0][n_idx].get();

                kernel_EvalAccCoreCGGI_binary<<<2, 1024, 0, s>>>( // one cmux
                    d_acc_ntt.get(), d_dct.get(), d_ACCKey, d_monic_polys_.get(), N,
                    ntt_->getMod(), ntt_->getMu()[0], ntt_->getMu()[1], digitsG2, indexPos);
            } else {
                throw std::invalid_argument("ERROR: Invalid key distribution for CGGI");
            }

            ntt_->inverse(d_acc_ntt.get(), d_acc_ntt.get(), 1024, 2, s);

            if (RGSWParams->IsCompositeNTT()) {
                // composite NTT
                kernel_scale_by_p<<<2, 1024, 0, s>>>(
                    d_acc_ntt.get(), d_acc_ntt.get(), N, Q, PQ);
            }

            // accumulate to acc
            kernel_element_add<<<2, 1024, 0, s>>>(
                d_acc.get(), d_acc.get(), d_acc_ntt.get(), N, Q); // N: dim
            //            }
        }
    }

    // /******************** new add ************************/
    // void GPURingGSWAccumulator::GPUEvalCMUX(const std::shared_ptr<lbcrypto::BinFHECryptoParams> &params, const GPURingGSWBTKey &EK,
    //                                         const NativeVector &a, const phantom::util::cuda_auto_ptr<BasicInteger> &d_acc,
    //                                         const cudaStream_t &s) {
    // }
    // /******************** new add ************************/
}
