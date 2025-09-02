#pragma once

#include <cutfhe++.h>
#include <phantom.h>

#include <complex>

#include "polyeval_bsgs.h"
#include "trlwevaluator.cuh"
#include "utils.h"

using namespace cuTFHEpp;

namespace conver {
    using namespace cuTFHEpp::util;
    using namespace phantom;
    using namespace phantom::arith;
    using namespace phantom::util;

    using LWEContext = PhantomContext;
    using LWEParams = EncryptionParameters;

    struct RLWE2LWECt {
        PhantomPlaintext a;
        phantom::util::cuda_auto_ptr<uint64_t> b;  // modulus_size
        double scale = 1.;
    };

    // Used to swtich LWE(s_N, N, p) tto LWE(s_n, n, p)
    struct GPUDecomposedLWEKSwitchKey {
        uint32_t decompose_base_ = 7;
        uint32_t ndigits_ = 0;                  // w^D ~ p0
        std::vector<PhantomCiphertext> parts_;  // ceil(N/n) * D
    };

    // extern unique_ptr<LWEContext> lwe_context;

    /************************ cuda kernel **************************/
    __global__ void multiply_scalar_poly(const uint64_t *operand, const uint64_t scalar, const DModulus *modulus, uint64_t *result, const uint64_t poly_degree, const uint64_t coeff_mod_size);
    __global__ void extract_lwe_part(uint64_t *lwe_ct_a, uint64_t *lwe_ct_b,
                                     const uint64_t *rlwe_ct_1, const uint64_t *rlwe_ct_0, const DModulus *modulus, size_t degree, size_t nmoduli,
                                     const size_t extract_indices);
    // __global__ void add_uint_mod_and_trans_type(uint32_t *lwe_ct_data0, uint64_t *lwe_ct_a, size_t n, uint64_t *lwe_ct_b, Modulus modulus);
    __global__ void add_uint_mod(uint64_t *lwe_ct_a, uint64_t *lwe_ct_b, uint64_t q0);
    __global__ void reverse_and_negate_kernel(uint64_t *arr, int extract_index, int n, uint64_t q0);
    __global__ void trans_type(uint32_t *dst, uint64_t *src, int len, double rescale);
    __global__ void rescale_and_trans(uint64_t *dst, uint64_t *src, int len, double rescale);
    __global__ void shift_mod_fusion(const uint64_t *input, uint64_t *output, size_t size, uint64_t w_mod, uint64_t shift);
    __global__ void reverse_transform_fusion(uint64_t *lwe_ct0, size_t n, uint64_t p0);
    __global__ void dummy_kernel(int sleep_ms);

    /************************ alg **************************/
    void SampleExtract(trlwevaluator &trlwer, std::vector<RLWE2LWECt> &lwe_ct, PhantomCiphertext &rlwe_cipher, std::vector<size_t> &extract_indices);
    void ExternalProduct(LWEContext *lwe_context, PhantomCiphertext &ct, const PhantomCiphertext *key_array, const int ndigits, const uint32_t decompose_base);
    void ExternalProduct(LWEContext *lwe_context, PhantomCiphertext &ct, const PhantomCiphertext *key_array, const int ndigits, const uint32_t decompose_base, const cuda_stream_wrapper &stream_wrapper);

    template <typename Lvl>
    void SetAndExtractFirst(LWEContext *lwe_context, TFHEpp::TLWE<Lvl> *d_lwe_n, const PhantomCiphertext &rlwe_n, const RLWE2LWECt &lwe_N) {
        int extract_index = 0;

        const auto &s = phantom::util::global_variables::default_stream->get_stream();

        if (rlwe_n.size() != 2) {
            throw std::invalid_argument(
                "SampleExtract: require rlwe_n cipher of size 2");
        }

        if (rlwe_n.is_ntt_form()) {
            throw std::invalid_argument("SampleExtract: require non_ntt cipher");
        }

        const auto &working_context = lwe_context->last_context_data();
        const size_t n = working_context.parms().poly_modulus_degree();
        auto mod_q0 = working_context.parms().coeff_modulus().front();
        const uint64_t q0 = mod_q0.value();

        if (n != Lvl::n) {
            throw std::invalid_argument("SampleExtract: LWE.n mismatch");
        }

        auto lwe_n_data = make_cuda_auto_ptr<uint64_t>(n + 1, s);
        // size_t gridDimGlb = (n + blockDimGlb.x - 1) / blockDimGlb.x;
        // add_uint_mod_and_trans_type<<<gridDimGlb, blockDimGlb, 0, s>>>(lwe_n_data, rlwe_n.data(), n, lwe_N.b.get(), mod_q0);
        add_uint_mod<<<1, 1, 0, s>>>(rlwe_n.data(), lwe_N.b.get(), q0);

        uint64_t *lwe_ct_ptr = lwe_n_data.get();
        CHECK_CUDA_ERROR(cudaMemcpyAsync(lwe_ct_ptr, rlwe_n.data(1), n * sizeof(uint64_t), cudaMemcpyDeviceToDevice, s));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(lwe_ct_ptr + n, rlwe_n.data(0), sizeof(uint64_t), cudaMemcpyDeviceToDevice, s));

        int max_threads_needed = (n - 1) / 2;
        int numBlocks = (max_threads_needed + blockDimGlb.x - 1) / blockDimGlb.x;
        reverse_and_negate_kernel<<<numBlocks, blockDimGlb, 0, s>>>(lwe_ct_ptr, extract_index, n, q0);

        double rescale = static_cast<double>(1ULL << 58) / q0;
        size_t gridDimGlb = (n + 1 + blockDimGlb.x - 1) / blockDimGlb.x;
        // trans_type<<<gridDimGlb, blockDimGlb, 0, s>>>(d_lwe_n->data(), lwe_ct_ptr, n + 1, rescale);
        uint64_t *dst = d_lwe_n->data();
        rescale_and_trans<<<gridDimGlb, blockDimGlb, 0, s>>>(dst, lwe_ct_ptr, n + 1, rescale);
    }

    template <typename Lvl>
    void SetAndExtractFirst(LWEContext *lwe_context, TFHEpp::TLWE<Lvl> *d_lwe_n, const PhantomCiphertext &rlwe_n, const RLWE2LWECt &lwe_N, const cuda_stream_wrapper &stream_wrapper) {
        const auto &s = stream_wrapper.get_stream();
        if (rlwe_n.size() != 2) {
            throw std::invalid_argument(
                "SampleExtract: require rlwe_n cipher of size 2");
        }

        if (rlwe_n.is_ntt_form()) {
            throw std::invalid_argument("SampleExtract: require non_ntt cipher");
        }

        const auto &working_context = lwe_context->last_context_data();
        const size_t n = working_context.parms().poly_modulus_degree();
        auto mod_q0 = working_context.parms().coeff_modulus().front();
        const uint64_t q0 = mod_q0.value();

        if (n != Lvl::n) {
            throw std::invalid_argument("SampleExtract: LWE.n mismatch");
        }

        // auto lwe_n_data = make_cuda_auto_ptr<uint64_t>(n + 1, s);
        // size_t gridDimGlb = (n + blockDimGlb.x - 1) / blockDimGlb.x;
        // add_uint_mod_and_trans_type<<<gridDimGlb, blockDimGlb, 0, s>>>(lwe_n_data, rlwe_n.data(), n, lwe_N.b.get(), mod_q0);
        add_uint_mod<<<1, 1, 0, s>>>(rlwe_n.data(0), lwe_N.b.get(), q0);

        // uint64_t *lwe_ct_ptr = lwe_n_data.get();
        uint64_t *lwe_ct_ptr = d_lwe_n->data();
        CHECK_CUDA_ERROR(cudaMemcpyAsync(lwe_ct_ptr, rlwe_n.data(1), n * sizeof(uint64_t), cudaMemcpyDeviceToDevice, s));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(lwe_ct_ptr + n, rlwe_n.data(0), sizeof(uint64_t), cudaMemcpyDeviceToDevice, s));

        int extract_index = 0;
        int max_threads_needed = (n - 1) / 2;
        int numBlocks = (max_threads_needed + blockDimGlb.x - 1) / blockDimGlb.x;
        reverse_and_negate_kernel<<<numBlocks, blockDimGlb, 0, s>>>(lwe_ct_ptr, extract_index, n, q0);  // only extract 0

        // double rescale = static_cast<double>(1ULL << 58) / q0;
        // size_t gridDimGlb = (n + 1 + blockDimGlb.x - 1) / blockDimGlb.x;
        // uint64_t *dst = d_lwe_n->data();
        // rescale_and_trans<<<gridDimGlb, blockDimGlb, 0, s>>>(dst, lwe_ct_ptr, n + 1, rescale);
    }

    template <typename Lvl>
    void LWEKeySwitch(LWEContext *lwe_context, TFHEpp::TLWE<Lvl> *d_lwe_n, const RLWE2LWECt &lwe_N, const GPUDecomposedLWEKSwitchKey &key) {
        const auto &s = phantom::util::global_variables::default_stream->get_stream();

        const size_t N = lwe_N.a.coeff_count();
        const size_t n = Lvl::n;
        const size_t key_sze = N / n;
        const size_t ndigits = key.ndigits_;

        const auto &working_context = lwe_context->last_context_data();
        // std::cout << "working_context.chain_index(): " << working_context.chain_index() << std::endl;

        if (key_sze * ndigits != key.parts_.size()) {
            std::cerr << key_sze << "*" << ndigits << " != " << key.parts_.size() << "\n";
            throw std::invalid_argument("LWEKeySwitch: invalid key size.");
        }

        if (lwe_N.b.get_n() != 1) {
            throw std::invalid_argument("LWEKeySwitch: invalid lwe_N cipher");
        }

        if (working_context.parms().poly_modulus_degree() != n) {
            throw std::invalid_argument("LWEKeySwitch: mismatch lwe_rt_mod_q0");
        }

        PhantomCiphertext trivial;
        trivial.resize(*lwe_context, lwe_context->last_context_data().chain_index(), 2, s);
        trivial.set_ntt_form(false);

        PhantomCiphertext accum;
        accum.resize(*lwe_context, lwe_context->last_context_data().chain_index(), 2, s);
        accum.set_ntt_form(false);

        // std::cout << "chain index: " << lwe_context->last_context_data().chain_index() << " modulus size: " << lwe_context->last_context_data().parms().coeff_modulus().size() << std::endl;

        auto mod_p0 = working_context.parms().coeff_modulus().front();
        const int64_t p0 = mod_p0.value();

        Evaluator lwe_evaluator(lwe_context);

        const size_t coeff_size = trivial.coeff_modulus_size() * trivial.poly_modulus_degree();
        size_t gridDimGlb = (coeff_size + blockDimGlb.x - 1) / blockDimGlb.x;
        for (size_t i = 0; i < N / n; i++) {
            // std::cout << "loop: " << i << std::endl;

            CHECK_CUDA_ERROR(cudaMemcpyAsync(trivial.data(0), lwe_N.a.data() + i * n, n * sizeof(uint64_t), cudaMemcpyDeviceToDevice, s));
            reverse_transform_fusion<<<gridDimGlb, blockDimGlb, 0, s>>>(trivial.data(0), n, p0);
            CHECK_CUDA_ERROR(cudaMemsetAsync(trivial.data(1), 0, n * sizeof(uint64_t), s));
            trivial.set_ntt_form(false);
            // std::cout << "reverse_transform_fusion done" << std::endl;

            ExternalProduct(lwe_context, trivial, &(key.parts_.at(i * ndigits)), ndigits, key.decompose_base_);  // 合并？
            // std::cout << "ExternalProduct done" << std::endl;

            lwe_evaluator.transform_from_ntt_inplace(trivial.data(), 1, 0, s);
            trivial.set_ntt_form(false);
            // std::cout << "transform_from_ntt_inplace done" << std::endl;

            lwe_evaluator.add_inplace(accum, trivial);
            // std::cout << "add_inplace done" << std::endl;
        }
        // std::cout << "loop done" << std::endl;

        // set and extract b
        SetAndExtractFirst<Lvl>(lwe_context, d_lwe_n, accum, lwe_N);
        // std::cout << "SetAndExtractFirst done" << std::endl;
    }

    template <typename Lvl>
    void LWEKeySwitch(LWEContext *lwe_context, TFHEpp::TLWE<Lvl> *d_lwe_n, const RLWE2LWECt &lwe_N, const GPUDecomposedLWEKSwitchKey &key, const cuda_stream_wrapper &stream_wrapper) {
        const auto &s = stream_wrapper.get_stream();

        // std::cout << "Stream Address: " << s << std::endl;

        const size_t N = lwe_N.a.coeff_count();
        const size_t n = Lvl::n;
        const size_t key_sze = N / n;
        const size_t ndigits = key.ndigits_;

        const auto &working_context = lwe_context->last_context_data();
        // std::cout << "working_context.chain_index(): " << working_context.chain_index() << std::endl;

        if (key_sze * ndigits != key.parts_.size()) {
            std::cerr << key_sze << "*" << ndigits << " != " << key.parts_.size() << "\n";
            throw std::invalid_argument("LWEKeySwitch: invalid key size.");
        }

        if (lwe_N.b.get_n() != 1) {
            throw std::invalid_argument("LWEKeySwitch: invalid lwe_N cipher");
        }

        if (working_context.parms().poly_modulus_degree() != n) {
            throw std::invalid_argument("LWEKeySwitch: mismatch lwe_rt_mod_q0");
        }

        PhantomCiphertext trivial;
        trivial.resize(*lwe_context, lwe_context->last_context_data().chain_index(), 2, s);
        trivial.set_ntt_form(false);

        PhantomCiphertext accum;
        // accum.resize(*lwe_context, lwe_context->last_context_data().chain_index(), 2, s);
        // accum.set_ntt_form(false);

        // std::cout << "chain index: " << lwe_context->last_context_data().chain_index() << " modulus size: " << lwe_context->last_context_data().parms().coeff_modulus().size() << std::endl;

        auto mod_p0 = working_context.parms().coeff_modulus().front();
        const int64_t p0 = mod_p0.value();

        Evaluator lwe_evaluator(lwe_context);

        const size_t coeff_size = trivial.coeff_modulus_size() * trivial.poly_modulus_degree();
        // std::cout << "coeff size: " << coeff_size << "  blockDimGlb.x: " << blockDimGlb.x << std::endl;
        size_t gridDimGlb = (coeff_size + blockDimGlb.x - 1) / blockDimGlb.x;
        // std::cout << "gridDimGlb: " << gridDimGlb << "  blockDimGlb.x: " << blockDimGlb.x << std::endl;

        for (size_t i = 0; i < N / n; i++) {
            // std::cout << "loop: " << i << std::endl;

            CHECK_CUDA_ERROR(cudaMemcpyAsync(trivial.data(0), lwe_N.a.data() + i * n, n * sizeof(uint64_t), cudaMemcpyDeviceToDevice, s));
            reverse_transform_fusion<<<gridDimGlb, blockDimGlb, 0, s>>>(trivial.data(0), n, p0);
            CHECK_CUDA_ERROR(cudaMemsetAsync(trivial.data(1), 0, n, s));
            trivial.set_ntt_form(false);
            // std::cout << "reverse_transform_fusion done" << std::endl;

            ExternalProduct(lwe_context, trivial, &(key.parts_.at(i * ndigits)), ndigits, key.decompose_base_, stream_wrapper);  // 合并？
            // std::cout << "ExternalProduct done" << std::endl;

            lwe_evaluator.transform_from_1dntt_inplace(trivial.data(0), 1, 0, s);  // ntt 提到外部去
            lwe_evaluator.transform_from_1dntt_inplace(trivial.data(1), 1, 0, s);  // ntt 提到外部去
            trivial.set_ntt_form(false);
            // std::cout << "transform_from_ntt_inplace done" << std::endl;

            if (i > 0) {
                lwe_evaluator.add_inplace(accum, trivial, stream_wrapper);
            } else {
                accum = trivial;
            }
            // std::cout << "add_inplace done" << std::endl;
        }
        // std::cout << "loop done" << std::endl;

        // set and extract b
        SetAndExtractFirst<Lvl>(lwe_context, d_lwe_n, accum, lwe_N, stream_wrapper);
        // std::cout << "SetAndExtractFirst done" << std::endl;
    }

    template <typename Lvl>
    void ExtractCoeffs(trlwevaluator &trlwer, LWEContext *lwe_context, PhantomCiphertext &rlwe_cipher, Pointer<cuTLWE<Lvl>> &lwe_ciphers, std::vector<size_t> &extract_indices, GPUDecomposedLWEKSwitchKey &extractKey) {
        // Pointer<cuTLWE<LvlR>> lwe_N_ciphers(extract_indices.size());
        std::vector<RLWE2LWECt> lwe_N_ct(extract_indices.size());
        // std::cout << "Extract Coeffs " << std::endl;
        // std::cout << "rlwe_cipher.is_ntt_form(): " << rlwe_cipher.is_ntt_form() << std::endl;
        const auto &stream_wrapper = phantom::util::global_variables::default_stream;
        const auto &s = stream_wrapper->get_stream();
        {
            CUDATimer timer("SampleExtract", s);
            timer.start();
            if (!rlwe_cipher.is_ntt_form()) {
                SampleExtract(trlwer, lwe_N_ct, rlwe_cipher, extract_indices);
            } else {
                trlwer.ckks->evaluator.transform_from_ntt_inplace(rlwe_cipher);
                SampleExtract(trlwer, lwe_N_ct, rlwe_cipher, extract_indices);
            }
            timer.stop();
        }
        std::cout << "SampleExtract done" << std::endl;
        // {
        //     auto index = 0;
        //     // auto ct_copy{lwe_N_ct[index]};  // 已经在非ntt
        //     RLWE2LWECt ct_copy = lwe_N_ct[index];
        //     // rlwe::SwitchNTTForm(ct_copy.a.data(), NTTDir::FromNTT, 1, runtime_->SEALRunTime());

        //     const auto &modulus = trlwer.ckks->context->last_context_data().parms().coeff_modulus().front();
        //     uint64_t half = modulus.value() >> 1;

        //     const uint64_t *sk_data = trlwer.ckks->decryptor.get_secretKey()->get_secret_key_non_ntt_array();
        //     const uint64_t *ct_data = ct_copy.a.data();
        //     const uint64_t *ct_data_b = ct_copy.b.get();

        //     uint64_t *h_sk = new uint64_t[ct_copy.a.coeff_count()];
        //     uint64_t *h_ct = new uint64_t[ct_copy.a.coeff_count()];
        //     uint64_t *h_ct_b = new uint64_t[1];
        //     // std::cout << "get ptr " << std::endl;

        //     cudaMemcpyAsync(h_sk, sk_data, ct_copy.a.coeff_count() * sizeof(uint64_t), cudaMemcpyDeviceToHost, s);
        //     cudaMemcpyAsync(h_ct, ct_data, ct_copy.a.coeff_count() * sizeof(uint64_t), cudaMemcpyDeviceToHost, s);
        //     cudaMemcpyAsync(h_ct_b, ct_data_b, sizeof(uint64_t), cudaMemcpyDeviceToHost, s);
        //     cudaStreamSynchronize(s);
        //     // std::cout << "cudaMemcpyAsync " << std::endl;

        //     uint64_t acc = dot_product_mod(h_sk, h_ct, ct_copy.a.coeff_count(), modulus);
        //     acc = add_uint_mod(acc, h_ct_b[0], modulus);
        //     // std::cout << "acc " << "rlwe_cipher.scale(): " << rlwe_cipher.scale() << std::endl;

        //     int64_t v = acc;
        //     if (acc >= half)
        //         v -= modulus.value();
        //     double res = static_cast<double>(v / rlwe_cipher.scale());
        //     std::cout << "res of exctract: " << res << std::endl;
        // }

        {
            CUDATimer timer("LWEKeySwitch", s);
            timer.start();
            TFHEpp::TLWE<Lvl> *lwe_ciphers_ptr = lwe_ciphers->template get<Lvl>();
            for (size_t i = 0; i < extract_indices.size(); i++) {
                LWEKeySwitch<Lvl>(lwe_context, lwe_ciphers_ptr + i, lwe_N_ct[i], extractKey, *stream_wrapper);
            }
            timer.stop();
        }
        // std::cout << "LWEKeySwitch done" << std::endl;

        // multi stream
        // {
        //     cudaDeviceSynchronize();
        //     size_t num_streams = extract_indices.size();
        //     std::vector<std::unique_ptr<cuda_stream_wrapper>> streams;
        //     for (size_t i = 0; i < num_streams; i++) {
        //         streams.push_back(std::make_unique<cuda_stream_wrapper>());
        //     }

        //     // for (int i = 0; i < num_streams; i++) {
        //     //     cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
        //     // }
        //     CUDATimer timer("LWEKeySwitch", s);
        //     timer.start();
        //     TFHEpp::TLWE<Lvl> *lwe_ciphers_ptr = lwe_ciphers->template get<Lvl>();
        //     for (size_t i = 0; i < extract_indices.size(); i++) {
        //         LWEKeySwitch<Lvl>(lwe_context, lwe_ciphers_ptr + i, lwe_N_ct[i], extractKey, *streams[i]);
        //         // dummy_kernel<<<8, 128, 0, streams[i]->get_stream()>>>(100);
        //     }
        //     cudaDeviceSynchronize();
        //     timer.stop();
        // }
    }

    /************************ function ****************************/
    void GenExtractKey(const trlwevaluator &trlwer, LWEContext *lwe_context, GPUDecomposedLWEKSwitchKey &extractKey, const TFHESecretKey &lwe_sk, const PhantomSecretKey &rlwe_sk);

    void RLWEToLWEs(const PhantomContext &context, PhantomCiphertext &rlwe_cipher, std::vector<std::vector<TLWELvl1>> &lwe_ciphers);

    template <typename Lvl>
    void RLWEToLWEs(trlwevaluator &trlwer, LWEContext *lwe_context, PhantomCiphertext &rlwe_cipher, Pointer<cuTLWE<Lvl>> &lwe_ciphers, std::vector<size_t> &extract_indices, GPUDecomposedLWEKSwitchKey &extractKey) {
        std::cout << "RLWEToLWEs " << std::endl;

        auto &context_data = trlwer.ckks->context->get_context_data(rlwe_cipher.chain_index());

        PhantomCiphertext tmpct0, tmpct1;
        // slot to coeff
        trlwer.ckks->evaluator.mod_switch_to_inplace(rlwe_cipher, context_data.parms().coeff_modulus().size() - 3);
        std::cout << "modulus size: " << rlwe_cipher.coeff_modulus_size() << std::endl;
        // trlwer.ckks->print_decrypted_ct(rlwe_cipher, 10, "after mod switch");

        // trlwer.ckksboot->slottocoeff_full_3(tmpct0, rlwe_cipher);

        const auto &s = phantom::util::global_variables::default_stream->get_stream();
        {
            CUDATimer timer("slottocoeff_3", s);
            timer.start();
            trlwer.ckksboot->slottocoeff_full_3(tmpct0, rlwe_cipher);
            timer.stop();
        }

        // trlwer.ckks->print_decrypted_ct(tmpct0, true, 10, 10, "after slottocoeff_full_3");
        trlwer.ckks->print_decrypted_nodecode_ct(tmpct0, 10, 10, "after slottocoeff_sparse_3");

        // sample extract and key switch
        // {
        //     CUDATimer timer("ExtractCoeffs", s);
        //     timer.start();
        //     ExtractCoeffs<Lvl>(trlwer, lwe_context, tmpct0, lwe_ciphers, extract_indices, extractKey);
        //     timer.stop();
        // }

        {
            CUDATimer timer("ExtractCoeffs", s);
            timer.start();
            ExtractCoeffs<Lvl>(trlwer, lwe_context, tmpct0, lwe_ciphers, extract_indices, extractKey);
            timer.stop();
        }
    }

    // template define
    // template void LWEKeySwitch<TFHEpp::lvl1param>(TFHEpp::TLWE<TFHEpp::lvl1param> *, const RLWE2LWECt &, const GPUDecomposedLWEKSwitchKey &);
    // template void ExtractCoeffs<TFHEpp::lvl1param>(trlwevaluator &, PhantomCiphertext &, Pointer<cuTLWE<TFHEpp::lvl1param>> &, std::vector<size_t> &, GPUDecomposedLWEKSwitchKey &);
    // template void RLWEToLWEs<TFHEpp::lvl1param>(trlwevaluator &, PhantomCiphertext &, Pointer<cuTLWE<TFHEpp::lvl1param>> &, std::vector<uint64_t> &, GPUDecomposedLWEKSwitchKey &);
    // template void SetAndExtractFirst<TFHEpp::lvl1param>(TFHEpp::TLWE<TFHEpp::lvl1param> *, const PhantomCiphertext &, const RLWE2LWECt &);

    template void LWEKeySwitch<TFHEpp::lvl1Lparam>(LWEContext *, TFHEpp::TLWE<TFHEpp::lvl1Lparam> *, const RLWE2LWECt &, const GPUDecomposedLWEKSwitchKey &);
    template void LWEKeySwitch<TFHEpp::lvl1Lparam>(LWEContext *, TFHEpp::TLWE<TFHEpp::lvl1Lparam> *, const RLWE2LWECt &, const GPUDecomposedLWEKSwitchKey &, const cuda_stream_wrapper &);
    template void ExtractCoeffs<TFHEpp::lvl1Lparam>(trlwevaluator &, LWEContext *, PhantomCiphertext &, Pointer<cuTLWE<TFHEpp::lvl1Lparam>> &, std::vector<size_t> &, GPUDecomposedLWEKSwitchKey &);
    template void RLWEToLWEs<TFHEpp::lvl1Lparam>(trlwevaluator &, LWEContext *, PhantomCiphertext &, Pointer<cuTLWE<TFHEpp::lvl1Lparam>> &, std::vector<uint64_t> &, GPUDecomposedLWEKSwitchKey &);
    template void SetAndExtractFirst<TFHEpp::lvl1Lparam>(LWEContext *, TFHEpp::TLWE<TFHEpp::lvl1Lparam> *, const PhantomCiphertext &, const RLWE2LWECt &);
    template void SetAndExtractFirst<TFHEpp::lvl1Lparam>(LWEContext *, TFHEpp::TLWE<TFHEpp::lvl1Lparam> *, const PhantomCiphertext &, const RLWE2LWECt &, const cuda_stream_wrapper &);
}