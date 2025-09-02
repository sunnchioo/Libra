#include "fileio.h"
#include "kernel.cuh"
#include "math/distributiongenerator.h"
#include "rgsw-acc.cuh"

#include "arith/big_integer_modop.h"

using namespace lbcrypto;

namespace phantom::bitwise {
    void GPURingGSWAccumulator::BatchGPUEvalAcc(const std::shared_ptr<BinFHECryptoParams> &params,
                                                const GPURingGSWBTKey &EK,
                                                const std::vector<NativeVector> &v_a,
                                                const phantom::util::cuda_auto_ptr<BasicInteger> &d_acc,
                                                const cudaStream_t &s) const {
        switch (params->GetRingGSWParams()->GetMethod()) {
        case BINFHE_METHOD::AP:
            BatchGPUEvalAccDM(params, EK, v_a, d_acc, s);
            break;
        case BINFHE_METHOD::GINX:
            // BatchGPUEvalAccCGGI(params, EK, v_a, d_acc, s);
            BatchGPUEvalAccCGGI_nttopt(params, EK, v_a, d_acc, s);
            break;
        //            case BINFHE_METHOD::LMKCDEY:
        //                BatchGPUEvalAccLMKCDEY(params, EK, v_a, d_acc, s);
        //                break;
        default:
            throw std::invalid_argument("ERROR: Invalid ACC method");
        }
    }

    void GPURingGSWAccumulator::BatchGPUEvalAccDM(const std::shared_ptr<BinFHECryptoParams> &params,
                                                  const GPURingGSWBTKey &EK,
                                                  const std::vector<NativeVector> &v_a,
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

        size_t batch_size = v_a.size();

        std::vector<BasicInteger *> v_d_ACCKey(n * digitsR.size() * batch_size);
        for (size_t batch_idx = 0; batch_idx < batch_size; batch_idx++) {
            for (size_t n_idx = 0; n_idx < n; ++n_idx) {
                auto aI = NativeInteger(0).ModSubFast(v_a[batch_idx][n_idx], q);
                for (size_t k = 0; k < digitsR.size(); ++k, aI /= baseR) {
                    auto a0 = (aI.Mod(baseR)).ConvertToInt<uint32_t>();
                    BasicInteger *d_ACCKey = EK.RGSWACCKey[n_idx][a0][k].get();
                    v_d_ACCKey[n_idx * digitsR.size() * batch_size + k * batch_size + batch_idx] = d_ACCKey;
                }
            }
        }

        auto d_v_d_ACCKey = phantom::util::make_cuda_auto_ptr<BasicInteger *>(n * digitsR.size() * batch_size, s);
        cudaMemcpyAsync(d_v_d_ACCKey.get(), v_d_ACCKey.data(), n * digitsR.size() * batch_size * sizeof(BasicInteger *),
                        cudaMemcpyHostToDevice, s);
        cudaStreamSynchronize(s);

        auto d_acc_ntt = phantom::util::make_cuda_auto_ptr<BasicInteger>(2 * N * batch_size, s);

        // approximate gadget decomposition is used; the first digit is ignored
        auto d_dct = phantom::util::make_cuda_auto_ptr<BasicInteger>(digitsG2 * N * batch_size, s);

        for (size_t n_idx = 0; n_idx < n; ++n_idx) {
            for (size_t k = 0; k < digitsR.size(); ++k) {
                // AP Accumulation as described in https://eprint.iacr.org/2020/086
                if (RGSWParams->IsCompositeNTT() == 1) {
                    // composite NTT
                    ntt_->forward(d_dct.get(), d_acc.get(), 256, 2 * batch_size, s);
                } else if (RGSWParams->IsCompositeNTT() == 0) {
                    // gadget decompose
                    //                            if (N == 1024) {
                    //                                constexpr size_t n1 = 32;
                    //                                constexpr size_t n2 = 32;
                    //                                size_t sMemSize = n1 * (n2 + 1) + 2 * n1;
                    //                                kernel_SignedDigitDecompose_fuse_1024<<<digitsG2, 1024, sMemSize, s>>>(
                    //                                        d_dct.get() + digitsG2 * N * batch_idx, d_acc.get() + 2 * N * batch_idx,
                    //                                        RGSWParams->GetQ().ConvertToInt(), RGSWParams->GetBaseG(),
                    //                                        ntt_->getTwRoot2n1(), ntt_->getTwRoot2n1Shoup(),
                    //                                        ntt_->getTwRoot2n(), ntt_->getTwRoot2nShoup());
                    //                            } else {
                    kernel_SignedDigitDecompose<<<dim3(digitsG2, batch_size), 256, 0, s>>>(
                        d_dct.get(), d_acc.get(), RGSWParams->GetQ().ConvertToInt(), RGSWParams->GetBaseG(), N);
                    ntt_->forward(d_dct.get(), d_dct.get(), 256, digitsG2 * batch_size, s);
                    //                            }
                } else {
                    throw std::invalid_argument("ERROR: Unsupported ACC technique");
                }

                // acc = dct * ek (matrix product);
                BasicInteger **d_ACCKeys = d_v_d_ACCKey.get() + n_idx * digitsR.size() * batch_size + k * batch_size;

                if (N == 1024) {
                    constexpr size_t n1 = 32;
                    constexpr size_t n2 = 32;
                    size_t sMemSize = n1 * (n2 + 1) + 2 * n1;
                    kernel_EvalAccCoreDM_1024_batch_fuse<<<dim3(2, batch_size), 256, sMemSize, s>>>(
                        d_acc.get(), d_dct.get(), d_ACCKeys,
                        N, ntt_->getMod(), ntt_->getMu()[0], ntt_->getMu()[1], digitsG2,
                        ntt_->getTwInvRoot2n(), ntt_->getTwInvRoot2nShoup(),
                        ntt_->getTwInvRoot2n1(), ntt_->getTwInvRoot2n1Shoup(),
                        ntt_->getInvn(), ntt_->getInvnShoup(),
                        RGSWParams->IsCompositeNTT(), P);
                } else {
                    kernel_EvalAccCoreDM_batch<<<dim3(2, batch_size), 256, 0, s>>>(
                        d_acc_ntt.get(), d_dct.get(), d_ACCKeys,
                        N, ntt_->getMod(), ntt_->getMu()[0], ntt_->getMu()[1], digitsG2);

                    ntt_->inverse(d_acc.get(), d_acc_ntt.get(), 256, 2 * batch_size, s);

                    // composite NTT
                    if (RGSWParams->IsCompositeNTT()) {
                        // scale P
                        kernel_scale_by_p<<<2 * batch_size, 256, 0, s>>>(
                            d_acc.get(), d_acc.get(), N, Q, PQ);
                    }
                }
            }
        }
    }

    void GPURingGSWAccumulator::BatchGPUEvalAccCGGI(const std::shared_ptr<BinFHECryptoParams> &params,
                                                    const GPURingGSWBTKey &EK,
                                                    const std::vector<NativeVector> &v_a,
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
        const auto &mod = v_a[0].GetModulus();
        const NativeInteger M{2 * RGSWParams->GetN()};
        const auto MbyMod{2 * RGSWParams->GetN() / v_a[0].GetModulus()};

        // std::cout << "n: " << n << " N: " << N << " Q: " << Q << " P: " << P << " PQ: " << PQ
        //           << " digitsG2: " << digitsG2 << std::endl;

        size_t batch_size = v_a.size();

        std::vector<uint32_t> v_indexPos(n * batch_size);
        std::vector<uint32_t> v_indexNeg(n * batch_size);
        for (size_t batch_idx = 0; batch_idx < batch_size; batch_idx++) {
            for (size_t n_idx = 0; n_idx < n; ++n_idx) {
                // obtain both monomial(index) for sk = 1 and monomial(-index) for sk = -1
                // index is in range [0,m] - so we need to adjust the edge case when index == m to index = 0
                NativeInteger ai = NativeInteger(0).ModSubFast(v_a[batch_idx][n_idx], mod) * MbyMod;
                auto indexPos = ai.ConvertToInt<uint32_t>();
                auto indexNeg = NativeInteger(0).ModSubFast(ai, M).ConvertToInt<uint32_t>();
                if (indexPos >= 2 * N || indexNeg >= 2 * N)
                    throw std::invalid_argument("ERROR: indexPos or indexNeg out of bound");
                v_indexPos[n_idx * batch_size + batch_idx] = indexPos;
                if (params->GetLWEParams()->GetKeyDist() == UNIFORM_TERNARY)
                    v_indexNeg[n_idx * batch_size + batch_idx] = indexNeg;
            }
        }

        auto d_v_indexPos = phantom::util::make_cuda_auto_ptr<uint32_t>(n * batch_size, s);
        auto d_v_indexNeg = phantom::util::make_cuda_auto_ptr<uint32_t>(n * batch_size, s);
        cudaMemcpyAsync(d_v_indexPos.get(), v_indexPos.data(), n * batch_size * sizeof(uint32_t),
                        cudaMemcpyHostToDevice, s);
        cudaMemcpyAsync(d_v_indexNeg.get(), v_indexNeg.data(), n * batch_size * sizeof(uint32_t),
                        cudaMemcpyHostToDevice, s);
        cudaStreamSynchronize(s);

        auto d_acc_ntt = phantom::util::make_cuda_auto_ptr<BasicInteger>(2 * N * batch_size, s);

        // approximate gadget decomposition is used; the first digit is ignored
        auto d_dct = phantom::util::make_cuda_auto_ptr<BasicInteger>(digitsG2 * N * batch_size, s);

        std::vector<BasicInteger> h_dct(d_dct.get_n(), 0);
        std::vector<BasicInteger> h_acc(d_acc.get_n(), 0);

        // cudaMemcpy(h_acc.data(), d_acc.get(), d_acc.get_n() * sizeof(BasicInteger), cudaMemcpyDeviceToHost);
        // FileIO<BasicInteger>::SaveText("/mnt/data2/home/syt/data/fhec/log/pbs/acc0.txt", h_acc);

        for (size_t n_idx = 0; n_idx < n; ++n_idx) {
            // handles -a*E(1) and handles -a*E(-1) = a*E(1)
            // CGGI Accumulation as described in https://eprint.iacr.org/2020/086
            // Added ternary MUX introduced in paper https://eprint.iacr.org/2022/074.pdf section 5
            // We optimize the algorithm by multiplying the monomial after the external product
            // This reduces the number of polynomial multiplications which further reduces the runtime
            if (RGSWParams->IsCompositeNTT() == 1) {
                // std::cout << "IsCompositeNTT: " << RGSWParams->IsCompositeNTT() << std::endl;
                // exit(0);
                // composite NTT
                ntt_->forward(d_dct.get(), d_acc.get(), 256, 2 * batch_size, s); // ntt
            } else if (RGSWParams->IsCompositeNTT() == 0) {                      // this
                // std::cout << "IsCompositeNTT: " << RGSWParams->IsCompositeNTT() << std::endl;
                // exit(0);
                // gadget decompose
                //                    if (N == 1024) {
                //                        constexpr size_t n1 = 32;
                //                        constexpr size_t n2 = 32;
                //                        size_t sMemSize = n1 * (n2 + 1) + 2 * n1;
                //                        kernel_SignedDigitDecompose_fuse_1024<<<digitsG2, 1024, sMemSize, s>>>(
                //                                d_dct.get() + digitsG2 * N * batch_idx, d_acc.get() + 2 * N * batch_idx,
                //                                RGSWParams->GetQ().ConvertToInt(), RGSWParams->GetBaseG(),
                //                                ntt_->getTwRoot2n1(), ntt_->getTwRoot2n1Shoup(),
                //                                ntt_->getTwRoot2n(), ntt_->getTwRoot2nShoup());
                //                    } else {
                kernel_SignedDigitDecompose<<<dim3(digitsG2, batch_size), 256, 0, s>>>(
                    d_dct.get(), d_acc.get(), RGSWParams->GetQ().ConvertToInt(), RGSWParams->GetBaseG(), N); // out / in
                ntt_->forward(d_dct.get(), d_dct.get(), 256, digitsG2 * batch_size, s);

                // cudaMemcpy(h_dct.data(), d_dct.get(), d_dct.get_n() * sizeof(BasicInteger), cudaMemcpyDeviceToHost);
                // FileIO<BasicInteger>::SaveText("/mnt/data2/home/syt/data/fhec/log/pbs/dct0.txt", h_dct);
                // return;
                //                    }
            } else {
                throw std::invalid_argument("ERROR: Unsupported ACC technique");
            }

            if (params->GetLWEParams()->GetKeyDist() == UNIFORM_TERNARY) {
                // obtain both monomial(index) for sk = 1 and monomial(-index) for sk = -1
                // index is in range [0,m] - so we need to adjust the edge case when index == m to index = 0
                uint32_t *d_indexPos = d_v_indexPos.get() + n_idx * batch_size;
                uint32_t *d_indexNeg = d_v_indexNeg.get() + n_idx * batch_size;

                // acc = acc + dct * ek1 * monomial + dct * ek2 * negative_monomial;
                // Needs to be done using two loops for ternary secrets.
                const BasicInteger *d_ACCKey0 = EK.RGSWACCKey[0][0][n_idx].get();
                const BasicInteger *d_ACCKey1 = EK.RGSWACCKey[0][1][n_idx].get();

                if (N == 1024) {
                    constexpr size_t n1 = 32;
                    constexpr size_t n2 = 32;
                    size_t sMemSize = n1 * (n2 + 1) + 2 * n1;
                    kernel_EvalAccCoreCGGI_1024_batch_fuse<<<dim3(2, batch_size), 256, sMemSize, s>>>(
                        d_acc.get(), d_dct.get(), d_ACCKey0, d_ACCKey1, d_monic_polys_.get(),
                        N, ntt_->getMod(), ntt_->getMu()[0], ntt_->getMu()[1], digitsG2, d_indexPos, d_indexNeg,
                        ntt_->getTwInvRoot2n(), ntt_->getTwInvRoot2nShoup(),
                        ntt_->getTwInvRoot2n1(), ntt_->getTwInvRoot2n1Shoup(),
                        ntt_->getInvn(), ntt_->getInvnShoup(),
                        RGSWParams->IsCompositeNTT(), P, Q);
                } else {
                    kernel_EvalAccCoreCGGI_batch<<<dim3(2, batch_size), 256, 0, s>>>(
                        d_acc_ntt.get(), d_dct.get(), d_ACCKey0, d_ACCKey1, d_monic_polys_.get(), N,
                        ntt_->getMod(), ntt_->getMu()[0], ntt_->getMu()[1], digitsG2, d_indexPos, d_indexNeg);

                    ntt_->inverse(d_acc_ntt.get(), d_acc_ntt.get(), 256, 2 * batch_size, s);

                    if (RGSWParams->IsCompositeNTT()) {
                        // composite NTT
                        kernel_scale_by_p<<<2 * batch_size, 256, 0, s>>>(
                            d_acc_ntt.get(), d_acc_ntt.get(), N, Q, PQ);
                    }

                    // accumulate to acc
                    kernel_element_add<<<2 * batch_size, 256, 0, s>>>(
                        d_acc.get(), d_acc.get(), d_acc_ntt.get(), N, Q);
                }
            } else if (params->GetLWEParams()->GetKeyDist() == BINARY) {
                // obtain both monomial(index) for sk = 1 and monomial(-index) for sk = -1
                // index is in range [0,m] - so we need to adjust the edge case when index == m to index = 0
                uint32_t *d_indexPos = d_v_indexPos.get() + n_idx * batch_size;

                // acc = acc + dct * ek1 * monomial + dct * ek2 * negative_monomial;
                // Needs to be done using two loops for ternary secrets.
                const BasicInteger *d_ACCKey = EK.RGSWACCKey[0][0][n_idx].get();

                if (N == 1024) {
                    constexpr size_t n1 = 32;
                    constexpr size_t n2 = 32;
                    size_t sMemSize = n1 * (n2 + 1) + 2 * n1;
                    kernel_EvalAccCoreCGGI_1024_binary_batch_fuse<<<dim3(2, batch_size), 256, sMemSize, s>>>(
                        d_acc.get(), d_dct.get(), d_ACCKey, d_monic_polys_.get(),
                        N, ntt_->getMod(), ntt_->getMu()[0], ntt_->getMu()[1], digitsG2, d_indexPos,
                        ntt_->getTwInvRoot2n(), ntt_->getTwInvRoot2nShoup(),
                        ntt_->getTwInvRoot2n1(), ntt_->getTwInvRoot2n1Shoup(),
                        ntt_->getInvn(), ntt_->getInvnShoup(),
                        RGSWParams->IsCompositeNTT(), P, Q);
                } else {
                    kernel_EvalAccCoreCGGI_binary_batch<<<dim3(2, batch_size), 256, 0, s>>>( // cmux batch
                        d_acc_ntt.get(), d_dct.get(), d_ACCKey, d_monic_polys_.get(), N,
                        ntt_->getMod(), ntt_->getMu()[0], ntt_->getMu()[1], digitsG2, d_indexPos);

                    ntt_->inverse(d_acc_ntt.get(), d_acc_ntt.get(), 256, 2 * batch_size, s);

                    if (RGSWParams->IsCompositeNTT()) {
                        // composite NTT
                        kernel_scale_by_p<<<2 * batch_size, 256, 0, s>>>(
                            d_acc_ntt.get(), d_acc_ntt.get(), N, Q, PQ);
                    }

                    // accumulate to acc
                    kernel_element_add<<<2 * batch_size, 256, 0, s>>>(
                        d_acc.get(), d_acc.get(), d_acc_ntt.get(), N, Q);
                }
            } else {
                throw std::invalid_argument("ERROR: Invalid key distribution for CGGI");
            }
        }
    }

    /****************** new add **********************/
    void GPURingGSWAccumulator::BatchGPUEvalAccCGGI_nttopt(const std::shared_ptr<BinFHECryptoParams> &params,
                                                           const GPURingGSWBTKey &EK,
                                                           const std::vector<NativeVector> &v_a,
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
        const auto &mod = v_a[0].GetModulus();
        const NativeInteger M{2 * RGSWParams->GetN()};
        const auto MbyMod{2 * RGSWParams->GetN() / v_a[0].GetModulus()};

        // std::cout << "n: " << n << " N: " << N << " Q: " << Q << " P: " << P << " PQ: " << PQ
        //           << " digitsG2: " << digitsG2 << " BaseG: " << RGSWParams->GetBaseG() << std::endl;

        size_t batch_size = v_a.size();

        std::vector<uint32_t> v_indexPos(n * batch_size);
        std::vector<uint32_t> v_indexNeg(n * batch_size);
        for (size_t batch_idx = 0; batch_idx < batch_size; batch_idx++) {
            for (size_t n_idx = 0; n_idx < n; ++n_idx) {
                // obtain both monomial(index) for sk = 1 and monomial(-index) for sk = -1
                // index is in range [0,m] - so we need to adjust the edge case when index == m to index = 0
                NativeInteger ai = NativeInteger(0).ModSubFast(v_a[batch_idx][n_idx], mod) * MbyMod;
                auto indexPos = ai.ConvertToInt<uint32_t>();
                auto indexNeg = NativeInteger(0).ModSubFast(ai, M).ConvertToInt<uint32_t>();
                if (indexPos >= 2 * N || indexNeg >= 2 * N)
                    throw std::invalid_argument("ERROR: indexPos or indexNeg out of bound");
                v_indexPos[n_idx * batch_size + batch_idx] = indexPos;
                if (params->GetLWEParams()->GetKeyDist() == UNIFORM_TERNARY)
                    v_indexNeg[n_idx * batch_size + batch_idx] = indexNeg;
            }
        }

        auto d_v_indexPos = phantom::util::make_cuda_auto_ptr<uint32_t>(n * batch_size, s);
        auto d_v_indexNeg = phantom::util::make_cuda_auto_ptr<uint32_t>(n * batch_size, s);
        cudaMemcpyAsync(d_v_indexPos.get(), v_indexPos.data(), n * batch_size * sizeof(uint32_t),
                        cudaMemcpyHostToDevice, s);
        cudaMemcpyAsync(d_v_indexNeg.get(), v_indexNeg.data(), n * batch_size * sizeof(uint32_t),
                        cudaMemcpyHostToDevice, s);
        cudaStreamSynchronize(s);

        auto d_acc_ntt = phantom::util::make_cuda_auto_ptr<BasicInteger>(2 * N * batch_size, s);

        // approximate gadget decomposition is used; the first digit is ignored
        auto d_dct = phantom::util::make_cuda_auto_ptr<BasicInteger>(digitsG2 * N * batch_size, s);

        std::vector<BasicInteger> h_dct(d_dct.get_n(), 0);
        std::vector<BasicInteger> h_acc(d_acc.get_n(), 0);

        // cudaMemcpy(h_acc.data(), d_acc.get(), d_acc.get_n() * sizeof(BasicInteger), cudaMemcpyDeviceToHost);
        // FileIO<BasicInteger>::SaveText("/mnt/data2/home/syt/data/fhec/log/pbs/acc1.txt", h_acc);

        ntt_->forward(d_acc_ntt.get(), d_acc.get(), 256, 2 * batch_size, s); // 先 ntt 然后 decomp

        for (size_t n_idx = 0; n_idx < n; ++n_idx) {
            // handles -a*E(1) and handles -a*E(-1) = a*E(1)
            // CGGI Accumulation as described in https://eprint.iacr.org/2020/086
            // Added ternary MUX introduced in paper https://eprint.iacr.org/2022/074.pdf section 5
            // We optimize the algorithm by multiplying the monomial after the external product
            // This reduces the number of polynomial multiplications which further reduces the runtime
            if (RGSWParams->IsCompositeNTT() == 1) {
                // composite NTT
                ntt_->forward(d_dct.get(), d_acc.get(), 256, 2 * batch_size, s); // ntt
            } else if (RGSWParams->IsCompositeNTT() == 0) {                      // this
                // std::cout << "IsCompositeNTT: " << RGSWParams->IsCompositeNTT() << std::endl;
                kernel_SignedDigitDecompose_opt<<<dim3(digitsG2, batch_size), 256, 0, s>>>(
                    d_dct.get(), d_acc_ntt.get(), RGSWParams->GetQ().ConvertToInt(), RGSWParams->GetBaseG(), N); // out / in
                // ntt_->forward(d_dct.get(), d_dct.get(), 256, digitsG2 * batch_size, s);
                // cudaMemcpy(h_dct.data(), d_dct.get(), d_dct.get_n() * sizeof(BasicInteger), cudaMemcpyDeviceToHost);
                // FileIO<BasicInteger>::SaveText("/mnt/data2/home/syt/data/fhec/log/pbs/dct1.txt", h_dct);
                // exit(0);
            } else {
                throw std::invalid_argument("ERROR: Unsupported ACC technique");
            }

            if (params->GetLWEParams()->GetKeyDist() == UNIFORM_TERNARY) {
                // obtain both monomial(index) for sk = 1 and monomial(-index) for sk = -1
                // index is in range [0,m] - so we need to adjust the edge case when index == m to index = 0
                uint32_t *d_indexPos = d_v_indexPos.get() + n_idx * batch_size;
                uint32_t *d_indexNeg = d_v_indexNeg.get() + n_idx * batch_size;

                // acc = acc + dct * ek1 * monomial + dct * ek2 * negative_monomial;
                // Needs to be done using two loops for ternary secrets.
                const BasicInteger *d_ACCKey0 = EK.RGSWACCKey[0][0][n_idx].get();
                const BasicInteger *d_ACCKey1 = EK.RGSWACCKey[0][1][n_idx].get();

                kernel_EvalAccCoreCGGI_batch<<<dim3(2, batch_size), 256, 0, s>>>(
                    d_acc.get(), d_dct.get(), d_ACCKey0, d_ACCKey1, d_monic_polys_.get(), N,
                    ntt_->getMod(), ntt_->getMu()[0], ntt_->getMu()[1], digitsG2, d_indexPos, d_indexNeg);
                // ntt_->inverse(d_acc_ntt.get(), d_acc_ntt.get(), 256, 2 * batch_size, s);

                if (RGSWParams->IsCompositeNTT()) {
                    // composite NTT
                    kernel_scale_by_p<<<2 * batch_size, 256, 0, s>>>(
                        d_acc_ntt.get(), d_acc_ntt.get(), N, Q, PQ);
                }

                // accumulate to acc
                kernel_element_add<<<2 * batch_size, 256, 0, s>>>(
                    d_acc.get(), d_acc.get(), d_acc_ntt.get(), N, Q);

            } else if (params->GetLWEParams()->GetKeyDist() == BINARY) {
                // obtain both monomial(index) for sk = 1 and monomial(-index) for sk = -1
                // index is in range [0,m] - so we need to adjust the edge case when index == m to index = 0
                uint32_t *d_indexPos = d_v_indexPos.get() + n_idx * batch_size;

                // acc = acc + dct * ek1 * monomial + dct * ek2 * negative_monomial;
                // Needs to be done using two loops for ternary secrets.
                const BasicInteger *d_ACCKey = EK.RGSWACCKey[0][0][n_idx].get();

                kernel_EvalAccCoreCGGI_binary_batch<<<dim3(2, batch_size), 256, 0, s>>>( // cmux batch
                    d_acc.get(), d_dct.get(), d_ACCKey, d_monic_polys_.get(), N,
                    ntt_->getMod(), ntt_->getMu()[0], ntt_->getMu()[1], digitsG2, d_indexPos);

                // ntt_->inverse(d_acc_ntt.get(), d_acc_ntt.get(), 256, 2 * batch_size, s);

                if (RGSWParams->IsCompositeNTT()) {
                    // composite NTT
                    kernel_scale_by_p<<<2 * batch_size, 256, 0, s>>>(
                        d_acc.get(), d_acc.get(), N, Q, PQ);
                }

                // accumulate to acc
                kernel_element_add<<<2 * batch_size, 256, 0, s>>>(
                    d_acc_ntt.get(), d_acc.get(), d_acc_ntt.get(), N, Q);

            } else {
                throw std::invalid_argument("ERROR: Invalid key distribution for CGGI");
            }
        }
        ntt_->inverse(d_acc.get(), d_acc_ntt.get(), 256, 2 * batch_size, s);
    }

    void GPURingGSWAccumulator::BatchGPUEvalCMUX(const std::shared_ptr<lbcrypto::BinFHECryptoParams> &params, const GPURingGSWBTKey &EK,
                                                 const std::vector<NativeVector> &v_a, const phantom::util::cuda_auto_ptr<BasicInteger> &d_acc,
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
        const auto &mod = v_a[0].GetModulus();
        const NativeInteger M{2 * RGSWParams->GetN()};
        const auto MbyMod{2 * RGSWParams->GetN() / v_a[0].GetModulus()};

        size_t batch_size = v_a.size();

        std::vector<uint32_t> v_indexPos(n * batch_size);
        std::vector<uint32_t> v_indexNeg(n * batch_size);
        for (size_t batch_idx = 0; batch_idx < batch_size; batch_idx++) {
            for (size_t n_idx = 0; n_idx < n; ++n_idx) {
                // obtain both monomial(index) for sk = 1 and monomial(-index) for sk = -1
                // index is in range [0,m] - so we need to adjust the edge case when index == m to index = 0
                NativeInteger ai = NativeInteger(0).ModSubFast(v_a[batch_idx][n_idx], mod) * MbyMod;
                auto indexPos = ai.ConvertToInt<uint32_t>();
                auto indexNeg = NativeInteger(0).ModSubFast(ai, M).ConvertToInt<uint32_t>();
                if (indexPos >= 2 * N || indexNeg >= 2 * N)
                    throw std::invalid_argument("ERROR: indexPos or indexNeg out of bound");
                v_indexPos[n_idx * batch_size + batch_idx] = indexPos;
                if (params->GetLWEParams()->GetKeyDist() == UNIFORM_TERNARY)
                    v_indexNeg[n_idx * batch_size + batch_idx] = indexNeg;
            }
        }

        auto d_v_indexPos = phantom::util::make_cuda_auto_ptr<uint32_t>(n * batch_size, s);
        auto d_v_indexNeg = phantom::util::make_cuda_auto_ptr<uint32_t>(n * batch_size, s);
        cudaMemcpyAsync(d_v_indexPos.get(), v_indexPos.data(), n * batch_size * sizeof(uint32_t),
                        cudaMemcpyHostToDevice, s);
        cudaMemcpyAsync(d_v_indexNeg.get(), v_indexNeg.data(), n * batch_size * sizeof(uint32_t),
                        cudaMemcpyHostToDevice, s);
        cudaStreamSynchronize(s);

        auto d_acc_ntt = phantom::util::make_cuda_auto_ptr<BasicInteger>(2 * N * batch_size, s);

        // approximate gadget decomposition is used; the first digit is ignored
        auto d_dct = phantom::util::make_cuda_auto_ptr<BasicInteger>(digitsG2 * N * batch_size, s);

        // handles -a*E(1) and handles -a*E(-1) = a*E(1)
        // CGGI Accumulation as described in https://eprint.iacr.org/2020/086
        // Added ternary MUX introduced in paper https://eprint.iacr.org/2022/074.pdf section 5
        // We optimize the algorithm by multiplying the monomial after the external product
        // This reduces the number of polynomial multiplications which further reduces the runtime
        if (RGSWParams->IsCompositeNTT() == 1) {
            // composite NTT
            ntt_->forward(d_dct.get(), d_acc.get(), 256, 2 * batch_size, s);
        } else if (RGSWParams->IsCompositeNTT() == 0) {
            // gadget decompose
            kernel_SignedDigitDecompose<<<dim3(digitsG2, batch_size), 256, 0, s>>>(
                d_dct.get(), d_acc.get(), RGSWParams->GetQ().ConvertToInt(), RGSWParams->GetBaseG(), N);
            ntt_->forward(d_dct.get(), d_dct.get(), 256, digitsG2 * batch_size, s);
            //                    }
        } else {
            throw std::invalid_argument("ERROR: Unsupported ACC technique");
        }

        if (params->GetLWEParams()->GetKeyDist() == UNIFORM_TERNARY) {
            // obtain both monomial(index) for sk = 1 and monomial(-index) for sk = -1
            // index is in range [0,m] - so we need to adjust the edge case when index == m to index = 0
            uint32_t *d_indexPos = d_v_indexPos.get() + 0 * batch_size;
            uint32_t *d_indexNeg = d_v_indexNeg.get() + 0 * batch_size;

            // acc = acc + dct * ek1 * monomial + dct * ek2 * negative_monomial;
            // Needs to be done using two loops for ternary secrets.
            const BasicInteger *d_ACCKey0 = EK.RGSWACCKey[0][0][0].get();
            const BasicInteger *d_ACCKey1 = EK.RGSWACCKey[0][1][0].get();

            if (N == 1024) {
                constexpr size_t n1 = 32;
                constexpr size_t n2 = 32;
                size_t sMemSize = n1 * (n2 + 1) + 2 * n1;
                phantom::util::CUDATimer timer("CMUX UNIFORM_TERNARY N=1024", s);
                timer.start();
                kernel_EvalAccCoreCGGI_1024_batch_fuse<<<dim3(2, batch_size), 256, sMemSize, s>>>( // 融合了 ntt
                    d_acc.get(), d_dct.get(), d_ACCKey0, d_ACCKey1, d_monic_polys_.get(),
                    N, ntt_->getMod(), ntt_->getMu()[0], ntt_->getMu()[1], digitsG2, d_indexPos, d_indexNeg,
                    ntt_->getTwInvRoot2n(), ntt_->getTwInvRoot2nShoup(),
                    ntt_->getTwInvRoot2n1(), ntt_->getTwInvRoot2n1Shoup(),
                    ntt_->getInvn(), ntt_->getInvnShoup(),
                    RGSWParams->IsCompositeNTT(), P, Q);
                timer.stop();
                CHECK_CUDA_LAST_ERROR();
            } else {
                {
                    phantom::util::CUDATimer timer("CMUX UNIFORM_TERNARY", s);
                    timer.start();
                    kernel_EvalAccCoreCGGI_batch<<<dim3(2, batch_size), 256, 0, s>>>(
                        d_acc_ntt.get(), d_dct.get(), d_ACCKey0, d_ACCKey1, d_monic_polys_.get(), N,
                        ntt_->getMod(), ntt_->getMu()[0], ntt_->getMu()[1], digitsG2, d_indexPos, d_indexNeg);
                    timer.stop();
                }
                ntt_->inverse(d_acc_ntt.get(), d_acc_ntt.get(), 256, 2 * batch_size, s);

                if (RGSWParams->IsCompositeNTT()) { // intt 之后乘以 p
                    // composite NTT
                    kernel_scale_by_p<<<2 * batch_size, 256, 0, s>>>(
                        d_acc_ntt.get(), d_acc_ntt.get(), N, Q, PQ);
                }

                // accumulate to acc
                kernel_element_add<<<2 * batch_size, 256, 0, s>>>(
                    d_acc.get(), d_acc.get(), d_acc_ntt.get(), N, Q);
            }
        } else if (params->GetLWEParams()->GetKeyDist() == BINARY) {
            // obtain both monomial(index) for sk = 1 and monomial(-index) for sk = -1
            // index is in range [0,m] - so we need to adjust the edge case when index == m to index = 0
            uint32_t *d_indexPos = d_v_indexPos.get() + 0 * batch_size;

            // acc = acc + dct * ek1 * monomial + dct * ek2 * negative_monomial;
            // Needs to be done using two loops for ternary secrets.
            const BasicInteger *d_ACCKey = EK.RGSWACCKey[0][0][0].get();

            if (N == 1024) {
                constexpr size_t n1 = 32;
                constexpr size_t n2 = 32;
                size_t sMemSize = n1 * (n2 + 1) + 2 * n1;

                // cuda time
                phantom::util::CUDATimer timer("CMUX BINARY N=1024", s);
                timer.start();
                kernel_EvalAccCoreCGGI_1024_binary_batch_fuse<<<dim3(2, batch_size), 256, sMemSize, s>>>(
                    d_acc.get(), d_dct.get(), d_ACCKey, d_monic_polys_.get(),
                    N, ntt_->getMod(), ntt_->getMu()[0], ntt_->getMu()[1], digitsG2, d_indexPos,
                    ntt_->getTwInvRoot2n(), ntt_->getTwInvRoot2nShoup(),
                    ntt_->getTwInvRoot2n1(), ntt_->getTwInvRoot2n1Shoup(),
                    ntt_->getInvn(), ntt_->getInvnShoup(),
                    RGSWParams->IsCompositeNTT(), P, Q);
                timer.stop();

            } else {
                phantom::util::CUDATimer timer("CMUX BINARY", s);
                timer.start();

                kernel_EvalAccCoreCGGI_binary_batch<<<dim3(2, batch_size), 256, 0, s>>>( // cmux batch
                    d_acc_ntt.get(), d_dct.get(), d_ACCKey, d_monic_polys_.get(), N,
                    ntt_->getMod(), ntt_->getMu()[0], ntt_->getMu()[1], digitsG2, d_indexPos);

                ntt_->inverse(d_acc_ntt.get(), d_acc_ntt.get(), 256, 2 * batch_size, s);
                timer.stop();

                if (RGSWParams->IsCompositeNTT()) {
                    // composite NTT
                    kernel_scale_by_p<<<2 * batch_size, 256, 0, s>>>(
                        d_acc_ntt.get(), d_acc_ntt.get(), N, Q, PQ);
                }

                // accumulate to acc
                kernel_element_add<<<2 * batch_size, 256, 0, s>>>(
                    d_acc.get(), d_acc.get(), d_acc_ntt.get(), N, Q);
            }
        } else {
            throw std::invalid_argument("ERROR: Invalid key distribution for CGGI");
        }
    }

    void GPURingGSWAccumulator::BatchGPUEvalADD(const std::shared_ptr<lbcrypto::BinFHECryptoParams> &params, std::vector<lbcrypto::LWECiphertext> &input0,
                                                const std::vector<lbcrypto::LWECiphertext> &input1, const cudaStream_t &s) const {
        size_t batch_size = input0.size();
        auto &LWEParams = params->GetLWEParams();
        // auto &RGSWParams = params->GetRingGSWParams();
        // auto polyParams = RGSWParams->GetPolyParams();

        auto modulus = input0[0]->GetModulus().ConvertToInt();
        uint32_t N = LWEParams->GetN();

        // std::cout << "params. " << std::endl;
        // std::cout << "N: " << N << std::endl;
        // std::cout << "GetModulus: " << input0[0]->GetModulus() << std::endl;

        // std::cout << "input0: " << input0[0]->GetA().at(0) << " " << input0[0]->GetB() << std::endl;
        // std::cout << "input1: " << input1[0]->GetA().at(0) << " " << input1[0]->GetB() << std::endl;

        // phantom::util::CUDATimer timer("BatchGPUADD_" + std::to_string(batch_size), s);
        // timer.start();

        std::vector<NativeVector> v_ct_A(batch_size, NativeVector(N, input0[0]->GetModulus()));

        auto d_input0 = phantom::util::make_cuda_auto_ptr<BasicInteger>(N * batch_size, s);
        auto d_input1 = phantom::util::make_cuda_auto_ptr<BasicInteger>(N * batch_size, s);

        for (size_t i = 0; i < batch_size; i++) {
            cudaMemcpyAsync(d_input0.get() + i * N, &input0[i]->GetA().at(0), N * sizeof(BasicInteger), cudaMemcpyHostToDevice, s);
            cudaMemcpyAsync(d_input1.get() + i * N, &input1[i]->GetA().at(0), N * sizeof(BasicInteger), cudaMemcpyHostToDevice, s);
        }

        {
            phantom::util::CUDATimer timer("BatchGPUADD_" + std::to_string(batch_size), s);
            timer.start();

            kernel_element_add<<<batch_size, 256, 0, s>>>(d_input0.get(), d_input0.get(), d_input1.get(), N, modulus);

            for (size_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
                input0[batch_idx]->GetB().ModAddFastEq(input1[batch_idx]->GetB(), modulus);
            }

            timer.stop();
            CHECK_CUDA_LAST_ERROR();
        }

        for (size_t i = 0; i < batch_size; i++) {
            CHECK_CUDA_ERROR(cudaMemcpyAsync(&v_ct_A[i].at(0), d_input0.get() + i * N, N * sizeof(BasicInteger), cudaMemcpyDeviceToHost, s));
        }
        cudaStreamSynchronize(s);

        // std::cout << "v_ct_A: " << v_ct_A[0].at(0) << " " << input0[0]->GetB() << std::endl;
        // cudaStreamSynchronize(s);

        for (size_t i = 0; i < batch_size; i++) {
            input0[i]->SetA(v_ct_A[i]);
        }

        // timer.stop();
        // CHECK_CUDA_LAST_ERROR();
    }
    /****************** new add **********************/
}
