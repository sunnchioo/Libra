#include "extract.cuh"
#include "polymath.cuh"
#include "trlwe.cuh"

namespace conver {
    using namespace cuTFHEpp::util;

    __global__ void multiply_scalar_poly(const uint64_t *operand,
                                         const uint64_t scalar,
                                         const DModulus *modulus,
                                         uint64_t *result,
                                         const uint64_t poly_degree,
                                         const uint64_t coeff_mod_size) {
        for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
             tid < poly_degree * coeff_mod_size;
             tid += blockDim.x * gridDim.x) {
            result[tid] = multiply_and_barrett_reduce_uint64(operand[tid], scalar, modulus[0].value(), modulus[0].const_ratio());
        }
    }

    __global__ void dummy_kernel(int sleep_ms) {
        // 模拟长时间运行
        clock_t start_clock = clock();
        clock_t clocks_per_ms = CLOCKS_PER_SEC / 1000;
        while ((clock() - start_clock) < (clock_t)sleep_ms * clocks_per_ms) {
        }
    }

    // unique_ptr<LWEContext> lwe_context;
    __global__ void shift_mod_fusion(const uint64_t *input, uint64_t *output, size_t size, uint64_t w_mod, uint64_t shift) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            output[idx] = (input[idx] >> shift) & w_mod;
        }
    }

    __global__ void reverse_transform_fusion(uint64_t *lwe_ct0, size_t n, uint64_t p0) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        // if (idx == 0)
        // printf("reverse_transform_fusion.\n");
        // __syncthreads();
        if (idx > 0 && idx < n / 2) {
            size_t reverse_idx = n - idx;
            uint64_t temp0 = lwe_ct0[idx];
            uint64_t temp1 = lwe_ct0[reverse_idx];

            temp0 = (temp0 > 0) ? (p0 - temp0) : 0;
            temp1 = (temp1 > 0) ? (p0 - temp1) : 0;

            lwe_ct0[idx] = temp1;
            lwe_ct0[reverse_idx] = temp0;
        }
    }

    __global__ void extract_lwe_part(uint64_t *lwe_ct_a, uint64_t *lwe_ct_b,
                                     const uint64_t *rlwe_ct_1, const uint64_t *rlwe_ct_0, const DModulus *modulus, size_t degree, size_t nmoduli,
                                     const size_t extract_indices) {
        for (int tid = blockIdx.x * blockDim.x + threadIdx.x;
             tid < degree * nmoduli;
             tid += gridDim.x * blockDim.x) {
            size_t local_tid = tid % (degree * nmoduli);

            size_t i = local_tid / degree;  // modulus index
            size_t j = local_tid % degree;  // coefficient index
            size_t extract_index = extract_indices;

            uint64_t qi = modulus[i].value();

            // a
            const uint64_t *rlwe_ct_ptr = rlwe_ct_1 + i * degree;
            uint64_t *lwe_ct_ptr = lwe_ct_a + i * degree;
            // b
            const uint64_t *rlwe_ct_ptr_0 = rlwe_ct_0 + i * degree;

            // Extract and modify the a part
            if (j > extract_index) {
                uint64_t temp = rlwe_ct_ptr[degree + extract_index - j];
                lwe_ct_ptr[j] = temp > 0 ? qi - temp : 0;
            }
            if (j <= extract_index) {
                lwe_ct_ptr[j] = rlwe_ct_ptr[extract_index - j];
            }

            if (j == extract_index && i < nmoduli) {
                lwe_ct_b[i] = rlwe_ct_ptr_0[extract_index];
            }
        }
    }

    // __global__ void add_uint_mod_and_trans_type(uint32_t *lwe_ct_data, uint64_t *lwe_ct_a, size_t n, uint64_t *lwe_ct_b, Modulus modulus) {  // 正确的是 rescale
    //     for (int tid = blockIdx.x * blockDim.x + threadIdx.x;
    //          tid < n;
    //          tid += gridDim.x * blockDim.x) {
    //         if (tid == 0) {
    //             lwe_ct_a[0] += lwe_ct_b[0];
    //             return lwe_ct_a[0] - (modulus.value() & static_cast<std::uint64_t>(-static_cast<std::int64_t>(lwe_ct_a[0] >= modulus.value())));
    //         }

    //         lwe_ct_data[tid] = static_cast<uint32_t>(static_cast<double>(lwe_ct_a[tid]));
    //         lwe_ct_data[tid + n] = static_cast<uint32_t>(static_cast<double>(lwe_ct_a[tid + n]));
    //     }
    // }

    __global__ void add_uint_mod(uint64_t *lwe_ct_a, uint64_t *lwe_ct_b, uint64_t q0) {  // 正确的是 rescale
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid == 0) {
            lwe_ct_a[0] += lwe_ct_b[0];
            lwe_ct_a[0] -= (q0 & static_cast<std::uint64_t>(-static_cast<std::int64_t>(lwe_ct_a[0] >= q0)));
        }
    }

    __global__ void reverse_and_negate_kernel(uint64_t *arr, int extract_index, int n, uint64_t q0) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;

        // Reverse the first segment (0 to extract_index)
        if (idx <= (extract_index + 1) / 2) {
            int i = idx;
            int j = extract_index - idx;
            if (i < j) {
                uint64_t temp = arr[i];
                arr[i] = arr[j];
                arr[j] = temp;
            }
        }

        // Reverse and negate the second segment (extract_index + 1 to n - 1)
        if (idx <= (n - extract_index - 1) / 2) {
            int i = extract_index + 1 + idx;
            int j = n - 1 - idx;
            if (i < j) {
                uint64_t temp0 = arr[i];
                uint64_t temp1 = arr[j];
                // Negate the element
                if (temp0 != 0) {
                    temp0 = q0 - temp0;
                }
                if (temp1 != 0) {
                    temp1 = q0 - temp1;
                }
                arr[i] = temp1;
                arr[j] = temp0;
            }
        }
    }

    __global__ void trans_type(uint32_t *dst, uint64_t *src, int len, double rescale) {
        for (int tid = blockIdx.x * blockDim.x + threadIdx.x;
             tid < len;
             tid += gridDim.x * blockDim.x) {
            dst[tid] = static_cast<uint32_t>(src[tid] * rescale);
        }
    }

    __global__ void rescale_and_trans(uint64_t *dst, uint64_t *src, int len, double rescale) {
        for (int tid = blockIdx.x * blockDim.x + threadIdx.x;
             tid < len;
             tid += gridDim.x * blockDim.x) {
            dst[tid] = static_cast<uint64_t>(src[tid] * rescale);
        }
    }

    void RLWEToLWEs(const PhantomContext &context, PhantomCiphertext &rlwe_cipher, std::vector<std::vector<TLWELvl1>> &lwe_ciphers) {}

    // template <typename Lvl>
    // void RLWEToLWEs(trlwevaluator &trlwer, PhantomCiphertext &rlwe_cipher, Pointer<cuTLWE<Lvl>> &lwe_ciphers) {
    // PhantomCiphertext tmpct0;
    // // slot to coeff
    // trlwer.ckksboot->coefftoslot_3(tmpct0, rlwe_cipher);

    // // sample extract

    // // key switch
    // }

    void GenExtractKey(const trlwevaluator &trlwer, LWEContext *lwe_context, GPUDecomposedLWEKSwitchKey &extractKey, const TFHESecretKey &lwe_sk_no_ntt, const PhantomSecretKey &rlwe_sk) {
        // std::cout << "Gen Extract Key" << std::endl;

        const auto &s = phantom::util::global_variables::default_stream->get_stream();

        const auto &rlwe_rt = trlwer.ckks->context;
        const size_t N = rlwe_rt->last_context_data().parms().poly_modulus_degree();
        const size_t n = lwe_context->last_context_data().parms().poly_modulus_degree();
        const uint64_t rlwe_q0 = rlwe_rt->last_context_data().parms().coeff_modulus()[0].value();
        const auto mod_q0 = lwe_context->last_context_data().parms().coeff_modulus()[0];
        const auto &dmod_q0 = lwe_context->gpu_rns_tables().modulus();
        size_t chain_index = rlwe_rt->last_context_data().chain_index();

        auto &rlwe_ntt_tables = rlwe_rt->gpu_rns_tables();
        auto &lwe_ntt_tables = lwe_context->gpu_rns_tables();

        DModulus tmp_mod;
        cudaMemcpyAsync(&tmp_mod, dmod_q0, 1 * sizeof(DModulus), cudaMemcpyDeviceToHost, s);

        // std::cout << "LWEKSKeyInit: " << " N: " << N << " n: " << n << " rlwe_q0: " << rlwe_q0 << " mod_q0: " << mod_q0.value() << std::endl;
        // std::cout << "LWEKSKeyInit: " << " gpu_rns_tables: " << lwe_context->gpu_rns_tables().n() << "  " << lwe_context->gpu_rns_tables().size() << " " << tmp_mod.value() << std::endl;

        PhantomSecretKey lwe_sk;
        lwe_sk.resize(chain_index, 1, n, 1);
        // std::vector<uint64_t> lwe_key_data(n, 0);
        // for (size_t i = 0; i < n; ++i) {
        //     lwe_key_data[i] = static_cast<uint64_t>(lwe_sk_no_ntt.key.get<Lvl1L>()[i]);
        // }
        uint64_t *lwe_sk_no_ntt_ptr = lwe_sk_no_ntt.key.get<Lvl1L>().data();
        // {
        //     std::cout << "lwe_sk_non_ntt: " << std::endl;
        //     std::cout << "key: ";
        //     for (int index = 0; index < n; index++) {
        //         std::cout << lwe_sk_no_ntt_ptr[index] << " ";
        //     }
        //     std::cout << std::endl;
        // }

        // cudaMemcpyAsync(lwe_sk.get_secret_key_array(), lwe_key_data.data(), n * sizeof(uint64_t), cudaMemcpyHostToDevice, s);  // 用这个加密rlwe_key
        cudaMemcpyAsync(lwe_sk.get_secret_key_array(), lwe_sk_no_ntt_ptr, n * sizeof(uint64_t), cudaMemcpyHostToDevice, s);  // 用这个加密rlwe_key
        // nwt_2d_radix8_forward_inplace(lwe_sk.get_secret_key_array(), lwe_context->gpu_rns_tables(), 1, 0, s);
        fnwt_1d_opt(lwe_sk.get_secret_key_array(), lwe_ntt_tables.twiddle(), lwe_ntt_tables.twiddle_shoup(), lwe_ntt_tables.modulus(), n, 1, 0, s);
        // {
        //     std::cout << "lwe_sk_ntt: " << std::endl;
        //     uint64_t *lwe_sk_re = new uint64_t[n];
        //     cudaMemcpyAsync(lwe_sk_re, lwe_sk.get_secret_key_array(), n * sizeof(uint64_t), cudaMemcpyDeviceToHost, s);

        //     std::cout << "key: ";
        //     for (int index = 0; index < n; index++) {
        //         std::cout << lwe_sk_re[index] << " ";
        //     }
        //     std::cout << std::endl;
        // }

        // inwt_1d_opt(lwe_sk.get_secret_key_array(), lwe_ntt_tables.itwiddle(), lwe_ntt_tables.itwiddle_shoup(), lwe_ntt_tables.modulus(),
        //             lwe_ntt_tables.n_inv_mod_q(), lwe_ntt_tables.n_inv_mod_q_shoup(), n, 1, 0, s);
        // {
        //     std::cout << "lwe_sk_non_ntt: " << std::endl;
        //     uint64_t *lwe_sk_re = new uint64_t[n];
        //     cudaMemcpyAsync(lwe_sk_re, lwe_sk.get_secret_key_array(), n * sizeof(uint64_t), cudaMemcpyDeviceToHost, s);

        //     std::cout << "key: ";
        //     for (int index = 0; index < n; index++) {
        //         std::cout << lwe_sk_re[index] << " ";
        //     }
        //     std::cout << std::endl;
        // }

        PhantomSecretKey rlwe_sk_non_ntt;  // 加密这个，直接喂进去，直接set一个n的secret_key,key in ntt
        rlwe_sk_non_ntt.resize(chain_index, 1, N, 1);
        cudaMemcpyAsync(rlwe_sk_non_ntt.get_secret_key_non_ntt_array(), rlwe_sk.get_secret_key_non_ntt_array(), N * sizeof(uint64_t), cudaMemcpyDeviceToDevice, s);
        // nwt_2d_radix8_forward_inplace(rlwe_sk_non_ntt.get_secret_key_non_ntt_array(), rlwe_rt->gpu_rns_tables(), 1, 0, s);
        // trlwer.ckks->evaluator.transform_from_ntt_inplace(rlwe_sk_non_ntt.get_secret_key_array(), 1, 0, s);
        // {
        //     std::cout << "rlwe_sk_non_ntt: " << std::endl;
        //     uint64_t *rlwe_sk_re = new uint64_t[n];
        //     cudaMemcpyAsync(rlwe_sk_re, rlwe_sk_non_ntt.get_secret_key_non_ntt_array(), n * sizeof(uint64_t), cudaMemcpyDeviceToHost, s);

        //     std::cout << "key: ";
        //     for (int index = 0; index < n; index++) {
        //         std::cout << rlwe_sk_re[index] << " ";
        //     }
        //     std::cout << std::endl;
        // }
        // nwt_2d_radix8_backward_inplace(rlwe_sk_non_ntt.get_secret_key_non_ntt_array(), rlwe_rt->gpu_rns_tables(), 1, 0, s);
        // {
        //     std::cout << "rlwe_sk_non_ntt: " << std::endl;
        //     uint64_t *rlwe_sk_re = new uint64_t[n];
        //     cudaMemcpyAsync(rlwe_sk_re, rlwe_sk_non_ntt.get_secret_key_non_ntt_array(), n * sizeof(uint64_t), cudaMemcpyDeviceToHost, s);

        //     std::cout << "key: ";
        //     for (int index = 0; index < n; index++) {
        //         std::cout << rlwe_sk_re[index] << " ";
        //     }
        //     std::cout << std::endl;
        // }

        if (extractKey.decompose_base_ > 16) {
            throw std::invalid_argument("LWEKSKeyInit: decompose_base out-of-bound");
        }

        if (n > N || rlwe_q0 != mod_q0.value() || (N % n != 0)) {
            throw std::invalid_argument("LWEKSKeyInit: meta mismatch");
        }

        if (rlwe_sk_non_ntt.coeff_count() != N) {
            throw std::invalid_argument("LWEKSKeyInit: key meta mismatch");
        }

        int ndigits = std::ceil(std::log2(static_cast<double>(rlwe_q0)) / extractKey.decompose_base_);
        extractKey.ndigits_ = ndigits;
        if (ndigits == 0) {
            throw std::invalid_argument("LWEKSKeyInit: ndigits > 0");
        }

        // std::cout << "decompose_base: " << extractKey.decompose_base_ << " ndigits: " << ndigits
        //           << " std::log2(static_cast<double>(rlwe_q0): " << std::log2(static_cast<double>(rlwe_q0)) << std::endl;

        const size_t key_sze = N / n;
        extractKey.parts_.resize(key_sze * ndigits);

        PhantomPlaintext rlwe_sk_part;
        rlwe_sk_part.resize(1, n, s);
        rlwe_sk_part.set_chain_index(lwe_context->last_context_data().chain_index());
        const uint64_t *rlwe_sk_non_ntt_ptr = rlwe_sk_non_ntt.get_secret_key_non_ntt_array();

        std::vector<uint64_t> factors(ndigits);
        uint64_t w = 1;
        // std::cout << "factor: ";
        for (uint64_t k = 0; k < ndigits; ++k) {
            // std::cout << w << " ";
            factors[k] = w;
            w <<= extractKey.decompose_base_;
        }
        // std::cout << std::endl;

        uint64_t gridDimGlb = N / blockDimGlb.x;
        auto gsw_iter = extractKey.parts_.begin();
        for (size_t i = 0; i < key_sze; ++i, rlwe_sk_non_ntt_ptr += n) {
            for (size_t k = 0; k < ndigits; ++k, ++gsw_iter) {
                PhantomCiphertext &ct = *gsw_iter;
                uint64_t *rlwe_sk_ntt_ptr = rlwe_sk_part.data();
                multiply_scalar_poly<<<gridDimGlb, blockDimGlb, 0, s>>>(rlwe_sk_non_ntt_ptr, factors[k], dmod_q0, rlwe_sk_ntt_ptr, n, 1);
                // nwt_2d_radix8_forward_inplace(rlwe_sk_ntt_ptr, lwe_context->gpu_rns_tables(), 1, 0, s);
                fnwt_1d_opt(rlwe_sk_ntt_ptr, lwe_ntt_tables.twiddle(), lwe_ntt_tables.twiddle_shoup(), lwe_ntt_tables.modulus(), n, 1, 0, s);
                // inwt_1d_opt(rlwe_sk_ntt_ptr, lwe_ntt_tables.itwiddle(), lwe_ntt_tables.itwiddle_shoup(), lwe_ntt_tables.modulus(),
                //             lwe_ntt_tables.n_inv_mod_q(), lwe_ntt_tables.n_inv_mod_q_shoup(), n, 1, 0, s);
                // nwt_2d_radix8_backward_inplace(rlwe_sk_ntt_ptr, lwe_context->gpu_rns_tables(), 1, 0, s);
                // {
                //     uint64_t *rlwe_sk_non_ntt;
                //     cudaMallocAsync(&rlwe_sk_non_ntt, n * sizeof(uint64_t), s);
                //     cudaMemcpyAsync(rlwe_sk_non_ntt, rlwe_sk_ntt_ptr, n * sizeof(uint64_t), cudaMemcpyDeviceToDevice, s);

                //     inwt_1d_opt(rlwe_sk_non_ntt, lwe_ntt_tables.itwiddle(), lwe_ntt_tables.itwiddle_shoup(), lwe_ntt_tables.modulus(),
                //                 lwe_ntt_tables.n_inv_mod_q(), lwe_ntt_tables.n_inv_mod_q_shoup(), n, 1, 0, s);
                //     std::cout << "rlwe_sk_ntt_ptr: " << std::endl;
                //     uint64_t *rlwe_sk_re = new uint64_t[n];
                //     cudaMemcpyAsync(rlwe_sk_re, rlwe_sk_non_ntt, n * sizeof(uint64_t), cudaMemcpyDeviceToHost, s);

                //     std::cout << "key: ";
                //     for (int index = 0; index < n; index++) {
                //         std::cout << rlwe_sk_re[index] << " ";
                //     }
                //     std::cout << std::endl;
                // }

                lwe_sk.encrypt_symmetric(*lwe_context, rlwe_sk_part, ct, s);

                // {
                //     std::cout << "ct len: " << ct.data_ptr().get_n() << std::endl;
                //     std::cout << "decrypt: " << std::endl;
                //     PhantomPlaintext rlwe_sk_pt = lwe_sk.decrypt(*lwe_context, ct);

                //     inwt_1d_opt(rlwe_sk_pt.data(), lwe_ntt_tables.itwiddle(), lwe_ntt_tables.itwiddle_shoup(), lwe_ntt_tables.modulus(),
                //                 lwe_ntt_tables.n_inv_mod_q(), lwe_ntt_tables.n_inv_mod_q_shoup(), n, 1, 0, s);

                //     uint64_t *rlwe_sk_re = new uint64_t[n];
                //     cudaMemcpyAsync(rlwe_sk_re, rlwe_sk_pt.data(), n * sizeof(uint64_t), cudaMemcpyDeviceToHost, s);

                //     std::cout << "key: ";
                //     for (int index = 0; index < n; index++) {
                //         std::cout << rlwe_sk_re[index] << " ";
                //     }
                //     std::cout << std::endl;
                // }

                // exit(0);
            }
        }

        if (rlwe_sk_non_ntt_ptr != (rlwe_sk_non_ntt.get_secret_key_non_ntt_array() + N)) {
            throw std::runtime_error("rlwe sk Not End");
        }

        if (gsw_iter != extractKey.parts_.end()) {
            throw std::runtime_error("key part Not End");
        }
    }

    void SampleExtract(trlwevaluator &trlwer, std::vector<RLWE2LWECt> &lwe_ct, PhantomCiphertext &rlwe_cipher, std::vector<size_t> &extract_indices) {
        // std::cout << "sample extract " << std::endl;

        const auto &s = phantom::util::global_variables::default_stream->get_stream();

        const auto &context_data = trlwer.ckks->context->get_context_data(rlwe_cipher.chain_index());
        const size_t poly_modulus_degree = rlwe_cipher.poly_modulus_degree();
        const size_t modulus_size = rlwe_cipher.coeff_modulus_size();
        const auto modulus = context_data.parms().coeff_modulus();

        for (int extract_iter = 0; extract_iter < extract_indices.size(); extract_iter++) {
            size_t extract_index = extract_indices[extract_iter];
            std::cout << "extract_index: " << extract_index << std::endl;
            lwe_ct[extract_iter].a.parms_id() = phantom::parms_id_zero;
            lwe_ct[extract_iter].a.resize(modulus_size, poly_modulus_degree, s);
            lwe_ct[extract_iter].b = phantom::util::make_cuda_auto_ptr<uint64_t>(modulus_size, s);

            uint64_t *lwe_ct_ptr = lwe_ct[extract_iter].a.data();
            uint64_t *rlwe_ct_ptr_1 = rlwe_cipher.data(1);
            uint64_t *rlwe_ct_ptr_0 = rlwe_cipher.data(0);

            // extract rlwe on GPU
            uint64_t *d_lwe_ct_a = lwe_ct_ptr;
            uint64_t *d_lwe_ct_b = lwe_ct[extract_iter].b.get();
            uint64_t *d_rlwe_ct_1 = rlwe_ct_ptr_1;
            uint64_t *d_rlwe_ct_0 = rlwe_ct_ptr_0;
            auto d_modulus = trlwer.ckks->context->gpu_rns_tables().modulus();

            dim3 blockdim(128);
            size_t total_threads = poly_modulus_degree * modulus_size;  // 一次处理的数据
            size_t gridDimGlb = (total_threads + blockdim.x - 1) / blockdim.x;
            extract_lwe_part<<<gridDimGlb, blockdim, 0, s>>>(d_lwe_ct_a, d_lwe_ct_b, d_rlwe_ct_1, d_rlwe_ct_0, d_modulus, poly_modulus_degree, modulus_size, extract_index);
        }

        // std::cout << "SampleExtract end." << std::endl;
    }

    void ExternalProduct(LWEContext *lwe_context, PhantomCiphertext &ct, const PhantomCiphertext *key_array, const int ndigits, const uint32_t decompose_base) {
        const auto &s = phantom::util::global_variables::default_stream->get_stream();

        if (ct.size() != 2) {
            throw std::invalid_argument("ExternalProduct: require |ct| = 2");
        }

        if (ct.is_ntt_form()) {
            throw std::invalid_argument("ExternalProduct: require non_ntt ct");
        }

        auto &ct_cntxt = lwe_context->last_context_data();
        const size_t degree = ct_cntxt.parms().poly_modulus_degree();
        size_t chain_index = ct_cntxt.chain_index();
        // need nSpecials=0, nCtModuli=1

        const uint64_t w = (1ULL << decompose_base);
        const uint64_t w_mod = w - 1;

        PhantomCiphertext out;
        Evaluator evaluator(lwe_context);

        PhantomPlaintext decomposed;
        decomposed.resize(1, degree, s);
        decomposed.set_chain_index(chain_index);

        size_t shift = 0;
        size_t gridDimGlb = (degree + blockDimGlb.x - 1) / blockDimGlb.x;
        for (size_t k = 0; k < ndigits; ++k, shift += decompose_base) {
            shift_mod_fusion<<<gridDimGlb, blockDimGlb, 0, s>>>(ct.data(0), decomposed.data(), degree, w_mod, shift);
            evaluator.transform_to_ntt_inplace(ct.data(0), 1, 0, s);

            PhantomCiphertext ct_copy = key_array[k];
            evaluator.multiply_plain_inplace(ct_copy, decomposed);

            if (out.size() > 0) {
                evaluator.add_inplace(out, ct_copy);
            } else {
                out = ct_copy;
            }
        }

        ct = out;
        ct.set_ntt_form(true);
    }

    void ExternalProduct(LWEContext *lwe_context, PhantomCiphertext &ct, const PhantomCiphertext *key_array, const int ndigits, const uint32_t decompose_base, const cuda_stream_wrapper &stream_wrapper) {
        const auto &s = stream_wrapper.get_stream();

        if (ct.size() != 2) {
            throw std::invalid_argument("ExternalProduct: require |ct| = 2");
        }

        if (ct.is_ntt_form()) {
            throw std::invalid_argument("ExternalProduct: require non_ntt ct");
        }

        auto &ct_cntxt = lwe_context->last_context_data();
        const size_t degree = ct_cntxt.parms().poly_modulus_degree();
        size_t chain_index = ct_cntxt.chain_index();
        // need nSpecials=0, nCtModuli=1

        const uint64_t w = (1ULL << decompose_base);
        const uint64_t w_mod = w - 1;

        PhantomCiphertext out;
        Evaluator evaluator(lwe_context);

        PhantomPlaintext decomposed;
        decomposed.resize(1, degree, s);
        decomposed.set_chain_index(chain_index);

        size_t shift = 0;
        size_t gridDimGlb = (degree + blockDimGlb.x - 1) / blockDimGlb.x;
        for (size_t k = 0; k < ndigits; ++k, shift += decompose_base) {
            shift_mod_fusion<<<gridDimGlb, blockDimGlb, 0, s>>>(ct.data(0), decomposed.data(), degree, w_mod, shift);
            evaluator.transform_to_1dntt_inplace(decomposed.data(), 1, 0, s);

            PhantomCiphertext ct_copy = key_array[k];
            evaluator.multiply_plain_inplace(ct_copy, decomposed, stream_wrapper);

            if (k > 0) {
                evaluator.add_inplace(out, ct_copy, stream_wrapper);
            } else {
                out = ct_copy;
                // std::cout << "out size: " << out.size() << std::endl;

                // exit(0);
            }
        }

        ct = out;
        ct.set_ntt_form(true);
    }
}