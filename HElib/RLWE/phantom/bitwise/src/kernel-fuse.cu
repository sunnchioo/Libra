#include "kernel.cuh"
#include "compact_ntt.cuh"

using namespace lbcrypto;

__device__ BasicInteger device_ScaleP_fast(const BasicInteger &v, const BasicInteger &P) {
    BasicInteger tmp = std::floor(0.5 + static_cast<double>(v) / static_cast<double>(P));
    return tmp;
}

__global__ void kernel_SignedDigitDecompose_fuse_1024(BasicInteger *g_dct, const BasicInteger *g_acc,
                                                      BasicInteger Q, uint32_t baseG,
                                                      const BasicInteger *tw_root_2n1,
                                                      const BasicInteger *tw_root_2n1_shoup,
                                                      const BasicInteger *tw_root_2n,
                                                      const BasicInteger *tw_root_2n_shoup) {
    constexpr
    size_t n1 = 32;
    constexpr
    size_t n2 = 32;
    constexpr
    size_t N = n1 * n2;

    __shared__ BasicInteger s_n1n2[n1][n2 + 1];

    __shared__ BasicInteger s_tw_root_2n1[n1];
    __shared__ BasicInteger s_tw_root_2n1_shoup[n1];

    // load twiddle factors into shared memory
    for (size_t tid = threadIdx.x; tid < n1; tid += blockDim.x) {
        s_tw_root_2n1[tid] = tw_root_2n1[tid];
        s_tw_root_2n1_shoup[tid] = tw_root_2n1_shoup[tid];
    }
    __syncthreads();

    const size_t warpsPerBlock = blockDim.x / warpSize;
    const size_t warp_id = threadIdx.x >> 5;
    const size_t lane_id = threadIdx.x & 31;
    const size_t poly_idx = blockIdx.x;

    const uint32_t QHalf = Q >> 1;
    const auto Q_int = static_cast<NativeInteger::SignedNativeInt>(Q);
    const int gBits = __clz(static_cast<int32_t>(__brev(baseG)));
    const int maxBits = sizeof(NativeInteger) * 8;
    const int gBitsMaxBits = maxBits - gBits;

    for (size_t n1_idx = warp_id; n1_idx < n1; n1_idx += warpsPerBlock) {
        for (size_t n2_idx = lane_id; n2_idx < n2; n2_idx += warpSize) {
            const size_t N_idx = n1_idx * n2 + lane_id;

            auto t = g_acc[(poly_idx & 1) * N + N_idx];
            auto d = static_cast<NativeInteger::SignedNativeInt>(t < QHalf ? t : t - Q_int);
            auto r = (d << gBitsMaxBits) >> gBitsMaxBits;
            d = (d - r) >> gBits;

            for (size_t i = 0; i < (poly_idx >> 1) + 1; i++) {
                r = (d << gBitsMaxBits) >> gBitsMaxBits;
                d = (d - r) >> gBits;
                r += static_cast<NativeInteger::SignedNativeInt>(
                        (static_cast<BasicInteger>(r) >> (sizeof(BasicInteger) * 8 - 1)) * Q_int);
            }

            s_n1n2[n1_idx][n2_idx] = r;
        }
    }
    __syncthreads();

    // TODO: merge into above loop
    device_4step_ntt_forward_phase1_ws<32, 32>(s_n1n2, s_tw_root_2n1, s_tw_root_2n1_shoup,
                                               tw_root_2n, tw_root_2n_shoup, Q);
    __syncthreads();

    device_4step_ntt_forward_phase2_ws<32, 32>(g_dct + poly_idx * N, s_n1n2, s_tw_root_2n1, s_tw_root_2n1_shoup, Q);
}

__global__ void kernel_EvalAccCoreDM_1024_batch_fuse(
        BasicInteger *acc, const BasicInteger *dct, BasicInteger **acc_keys,
        size_t N, BasicInteger mod, BasicInteger mu0, BasicInteger mu1,
        uint32_t digitsG2,
        const BasicInteger *tw_inv_root_2n,
        const BasicInteger *tw_inv_root_2n_shoup,
        const BasicInteger *tw_inv_root_2n1,
        const BasicInteger *tw_inv_root_2n1_shoup,
        BasicInteger inv_n, BasicInteger inv_n_shoup,
        int IsCompositeNTT, BasicInteger P) {
    constexpr size_t n1 = 32;
    constexpr size_t n2 = 32;

    __shared__ BasicInteger s_n1n2[n1][n2 + 1];
    __shared__ BasicInteger s_tw_inv_root_2n1[n1];
    __shared__ BasicInteger s_tw_inv_root_2n1_shoup[n1];

    // load twiddle factors into shared memory
    for (size_t tid = threadIdx.x; tid < n1; tid += blockDim.x) {
        s_tw_inv_root_2n1[tid] = tw_inv_root_2n1[tid];
        s_tw_inv_root_2n1_shoup[tid] = tw_inv_root_2n1_shoup[tid];
    }
    __syncthreads();

    const size_t poly_idx = blockIdx.x;
    const size_t batch_idx = blockIdx.y;
    const BasicInteger *acc_key = acc_keys[batch_idx];

    const size_t warpsPerBlock = blockDim.x / warpSize;
    const size_t warp_id = threadIdx.x >> 5;
    const size_t lane_id = threadIdx.x & 31;

    wide_type<BasicInteger> mod_mu(mu0, mu1);

    for (size_t n1_idx = warp_id; n1_idx < n1; n1_idx += warpsPerBlock) {
        const size_t N_idx = n1_idx * n2 + lane_id;
        BasicInteger dct_first = dct[batch_idx * digitsG2 * N + N_idx];
        BasicInteger reg = modMulBarrett(dct_first, acc_key[poly_idx * N + N_idx], mod, mod_mu);
        for (size_t digitsG2_index = 1; digitsG2_index < digitsG2; ++digitsG2_index) {
            BasicInteger dct_coeff = dct[batch_idx * digitsG2 * N + digitsG2_index * N + N_idx];
            reg = modAdd(modMulBarrett(dct_coeff, acc_key[digitsG2_index * 2 * N + poly_idx * N + N_idx], mod, mod_mu),
                         reg, mod);
        }

        // iNTT
        for (int log_m = 4; log_m >= 0; log_m--) {
            size_t log_step = 4 - log_m;
            size_t w_idx = lane_id >> (log_step + 1);

            BasicInteger reg_new = __shfl_xor_sync(0xffffffff, reg, 1 << log_step);
            size_t C = (lane_id >> log_step) & 1;
            BasicInteger left = (1 - C) * (reg - reg_new) + reg_new;  // C = 0 choose reg, C = 1 choose reg_new
            BasicInteger right = C * (reg - reg_new) + reg_new;  // C = 0 choose reg_new, C = 1 choose reg
            gs_butterfly_shoup(left, right,
                               s_tw_inv_root_2n1[w_idx], s_tw_inv_root_2n1_shoup[w_idx], mod);
            reg = (1 - C) * (left - right) + right;
        }
        // elementwise multiply and transpose store into shared memory
        s_n1n2[n1_idx][lane_id] = modMulShoupLazy(reg,
                                                  tw_inv_root_2n[n1_idx * n2 + lane_id],
                                                  tw_inv_root_2n_shoup[n1_idx * n2 + lane_id],
                                                  mod);
    }
    __syncthreads();

    for (size_t n2_idx = warp_id; n2_idx < n2; n2_idx += warpsPerBlock) {
        BasicInteger reg = s_n1n2[lane_id][n2_idx];
        for (int log_m = 4; log_m >= 0; log_m--) {
            size_t log_step = 4 - log_m;
            size_t w_idx = lane_id >> (log_step + 1);

            BasicInteger reg_new = __shfl_xor_sync(0xffffffff, reg, 1 << log_step);
            size_t C = (lane_id >> log_step) & 1;
            BasicInteger left = (1 - C) * (reg - reg_new) + reg_new;  // C = 0 choose reg, C = 1 choose reg_new
            BasicInteger right = C * (reg - reg_new) + reg_new;  // C = 0 choose reg_new, C = 1 choose reg
            gs_butterfly_shoup(left, right, s_tw_inv_root_2n1[(1 << log_m) + w_idx],
                               s_tw_inv_root_2n1_shoup[(1 << log_m) + w_idx], mod);
            reg = (1 - C) * (left - right) + right;
        }
        // transpose write back to s_out
        BasicInteger tmp = modMulShoup(reg, inv_n, inv_n_shoup, mod);
        if (IsCompositeNTT)
            s_n1n2[lane_id][n2_idx] = device_ScaleP_fast(tmp, P);
        else
            s_n1n2[lane_id][n2_idx] = tmp;
    }
    __syncthreads();

    // store into global memory
    for (size_t n1_idx = warp_id; n1_idx < n1; n1_idx += warpsPerBlock) {
        for (size_t n2_idx = lane_id; n2_idx < n2; n2_idx += warpSize) {
            acc[batch_idx * 2 * N + poly_idx * N + n1_idx * n2 + n2_idx] = s_n1n2[n1_idx][n2_idx];
        }
    }
}

__global__ void kernel_EvalAccCoreCGGI_1024_batch_fuse(
        BasicInteger *acc, const BasicInteger *dct,
        const BasicInteger *acc_key0, const BasicInteger *acc_key1,
        const BasicInteger *monic_polys,
        size_t N,
        BasicInteger mod, BasicInteger mu0, BasicInteger mu1,
        uint32_t digitsG2,
        const uint32_t *indexPos_batch,
        const uint32_t *indexNeg_batch,
        const BasicInteger *tw_inv_root_2n,
        const BasicInteger *tw_inv_root_2n_shoup,
        const BasicInteger *tw_inv_root_2n1,
        const BasicInteger *tw_inv_root_2n1_shoup,
        BasicInteger inv_n, BasicInteger inv_n_shoup,
        int IsCompositeNTT, BasicInteger P, BasicInteger Q) {
    constexpr
    size_t n1 = 32;
    constexpr
    size_t n2 = 32;

    __shared__ BasicInteger s_n1n2[n1][n2 + 1];
    __shared__ BasicInteger s_tw_inv_root_2n1[n1];
    __shared__ BasicInteger s_tw_inv_root_2n1_shoup[n1];

    // load twiddle factors into shared memory
    for (size_t tid = threadIdx.x; tid < n1; tid += blockDim.x) {
        s_tw_inv_root_2n1[tid] = tw_inv_root_2n1[tid];
        s_tw_inv_root_2n1_shoup[tid] = tw_inv_root_2n1_shoup[tid];
    }
    __syncthreads();

    const size_t poly_idx = blockIdx.x;
    const size_t batch_idx = blockIdx.y;

    uint32_t indexPos = indexPos_batch[batch_idx];
    uint32_t indexNeg = indexNeg_batch[batch_idx];

    const size_t warpsPerBlock = blockDim.x / warpSize;
    const size_t warp_id = threadIdx.x >> 5;
    const size_t lane_id = threadIdx.x & 31;

    wide_type<BasicInteger> mod_mu(mu0, mu1);

    for (size_t n1_idx = warp_id; n1_idx < n1; n1_idx += warpsPerBlock) {
        // acc = acc + dct * ek1 * monomial + dct * ek2 * negative_monomial
        const size_t N_idx = n1_idx * n2 + lane_id;
        const auto pos_monic_coeff = monic_polys[indexPos * N + N_idx];
        const auto neg_monic_coeff = monic_polys[indexNeg * N + N_idx];

        BasicInteger tmp0 = 0;
        BasicInteger tmp1 = 0;

        for (uint32_t digitsG2_idx = 0; digitsG2_idx < digitsG2; ++digitsG2_idx) {
            BasicInteger dct_coeff = dct[batch_idx * digitsG2 * N + digitsG2_idx * N + N_idx];

            tmp0 = modAdd(modMulBarrett(dct_coeff, acc_key0[digitsG2_idx * 2 * N + poly_idx * N + N_idx], mod, mod_mu),
                          tmp0, mod);

            tmp1 = modAdd(modMulBarrett(dct_coeff, acc_key1[digitsG2_idx * 2 * N + poly_idx * N + N_idx], mod, mod_mu),
                          tmp1, mod);
        }

        wide_type<BasicInteger> acc_wide = addWide(mulWide(tmp0, pos_monic_coeff),
                                                   mulWide(tmp1, neg_monic_coeff));

        BasicInteger reg = modWideBarrett(acc_wide, mod, mod_mu);

        // iNTT
        for (int log_m = 4; log_m >= 0; log_m--) {
            size_t log_step = 4 - log_m;
            size_t w_idx = lane_id >> (log_step + 1);

            BasicInteger reg_new = __shfl_xor_sync(0xffffffff, reg, 1 << log_step);
            size_t C = (lane_id >> log_step) & 1;
            BasicInteger left = (1 - C) * (reg - reg_new) + reg_new;  // C = 0 choose reg, C = 1 choose reg_new
            BasicInteger right = C * (reg - reg_new) + reg_new;  // C = 0 choose reg_new, C = 1 choose reg
            gs_butterfly_shoup(left, right,
                               s_tw_inv_root_2n1[w_idx], s_tw_inv_root_2n1_shoup[w_idx], mod);
            reg = (1 - C) * (left - right) + right;
        }
        // elementwise multiply and transpose store into shared memory
        s_n1n2[n1_idx][lane_id] = modMulShoupLazy(reg,
                                                  tw_inv_root_2n[n1_idx * n2 + lane_id],
                                                  tw_inv_root_2n_shoup[n1_idx * n2 + lane_id],
                                                  mod);
    }
    __syncthreads();

    for (size_t n2_idx = warp_id; n2_idx < n2; n2_idx += warpsPerBlock) {
        BasicInteger reg = s_n1n2[lane_id][n2_idx];
        for (int log_m = 4; log_m >= 0; log_m--) {
            size_t log_step = 4 - log_m;
            size_t w_idx = lane_id >> (log_step + 1);

            BasicInteger reg_new = __shfl_xor_sync(0xffffffff, reg, 1 << log_step);
            size_t C = (lane_id >> log_step) & 1;
            BasicInteger left = (1 - C) * (reg - reg_new) + reg_new;  // C = 0 choose reg, C = 1 choose reg_new
            BasicInteger right = C * (reg - reg_new) + reg_new;  // C = 0 choose reg_new, C = 1 choose reg
            gs_butterfly_shoup(left, right, s_tw_inv_root_2n1[(1 << log_m) + w_idx],
                               s_tw_inv_root_2n1_shoup[(1 << log_m) + w_idx], mod);
            reg = (1 - C) * (left - right) + right;
        }
        // transpose write back to s_out
        BasicInteger tmp = modMulShoup(reg, inv_n, inv_n_shoup, mod);
        if (IsCompositeNTT)
            s_n1n2[lane_id][n2_idx] = device_ScaleP_fast(tmp, P);
        else
            s_n1n2[lane_id][n2_idx] = tmp;
    }
    __syncthreads();

    // store into global memory
    BasicInteger new_mod = (IsCompositeNTT) ? Q : mod;
    for (size_t n1_idx = warp_id; n1_idx < n1; n1_idx += warpsPerBlock) {
        for (size_t n2_idx = lane_id; n2_idx < n2; n2_idx += warpSize) {
            acc[batch_idx * 2 * N + blockIdx.x * N + n1_idx * n2 + n2_idx] =
                    modAdd(acc[batch_idx * 2 * N + blockIdx.x * N + n1_idx * n2 + n2_idx],
                           s_n1n2[n1_idx][n2_idx], new_mod);
        }
    }
}

__global__ void kernel_EvalAccCoreCGGI_1024_binary_batch_fuse(
        BasicInteger *acc, const BasicInteger *dct,
        const BasicInteger *acc_key,
        const BasicInteger *monic_polys,
        size_t N,
        BasicInteger mod, BasicInteger mu0, BasicInteger mu1,
        uint32_t digitsG2,
        const uint32_t *indexPos_batch,
        const BasicInteger *tw_inv_root_2n,
        const BasicInteger *tw_inv_root_2n_shoup,
        const BasicInteger *tw_inv_root_2n1,
        const BasicInteger *tw_inv_root_2n1_shoup,
        BasicInteger inv_n, BasicInteger inv_n_shoup,
        int IsCompositeNTT, BasicInteger P, BasicInteger Q) {
    constexpr
    size_t n1 = 32;
    constexpr
    size_t n2 = 32;

    __shared__ BasicInteger s_n1n2[n1][n2 + 1];
    __shared__ BasicInteger s_tw_inv_root_2n1[n1];
    __shared__ BasicInteger s_tw_inv_root_2n1_shoup[n1];

    // load twiddle factors into shared memory
    for (size_t tid = threadIdx.x; tid < n1; tid += blockDim.x) {
        s_tw_inv_root_2n1[tid] = tw_inv_root_2n1[tid];
        s_tw_inv_root_2n1_shoup[tid] = tw_inv_root_2n1_shoup[tid];
    }
    __syncthreads();

    const size_t poly_idx = blockIdx.x;
    const size_t batch_idx = blockIdx.y;

    uint32_t indexPos = indexPos_batch[batch_idx];

    const size_t warpsPerBlock = blockDim.x / warpSize;
    const size_t warp_id = threadIdx.x >> 5;
    const size_t lane_id = threadIdx.x & 31;

    wide_type<BasicInteger> mod_mu(mu0, mu1);

    for (size_t n1_idx = warp_id; n1_idx < n1; n1_idx += warpsPerBlock) {
        // acc = acc + dct * ek1 * monomial + dct * ek2 * negative_monomial
        const size_t N_idx = n1_idx * n2 + lane_id;
        const auto pos_monic_coeff = monic_polys[indexPos * N + N_idx];

        BasicInteger tmp = 0;

        for (uint32_t digitsG2_idx = 0; digitsG2_idx < digitsG2; ++digitsG2_idx) {
            BasicInteger dct_coeff = dct[batch_idx * digitsG2 * N + digitsG2_idx * N + N_idx];

            tmp = modAdd(modMulBarrett(dct_coeff, acc_key[digitsG2_idx * 2 * N + poly_idx * N + N_idx], mod, mod_mu),
                          tmp, mod);
        }

        BasicInteger reg = modMulBarrett(tmp, pos_monic_coeff, mod, mod_mu);

        // iNTT
        for (int log_m = 4; log_m >= 0; log_m--) {
            size_t log_step = 4 - log_m;
            size_t w_idx = lane_id >> (log_step + 1);

            BasicInteger reg_new = __shfl_xor_sync(0xffffffff, reg, 1 << log_step);
            size_t C = (lane_id >> log_step) & 1;
            BasicInteger left = (1 - C) * (reg - reg_new) + reg_new;  // C = 0 choose reg, C = 1 choose reg_new
            BasicInteger right = C * (reg - reg_new) + reg_new;  // C = 0 choose reg_new, C = 1 choose reg
            gs_butterfly_shoup(left, right,
                               s_tw_inv_root_2n1[w_idx], s_tw_inv_root_2n1_shoup[w_idx], mod);
            reg = (1 - C) * (left - right) + right;
        }
        // elementwise multiply and transpose store into shared memory
        s_n1n2[n1_idx][lane_id] = modMulShoupLazy(reg,
                                                  tw_inv_root_2n[n1_idx * n2 + lane_id],
                                                  tw_inv_root_2n_shoup[n1_idx * n2 + lane_id],
                                                  mod);
    }
    __syncthreads();

    for (size_t n2_idx = warp_id; n2_idx < n2; n2_idx += warpsPerBlock) {
        BasicInteger reg = s_n1n2[lane_id][n2_idx];
        for (int log_m = 4; log_m >= 0; log_m--) {
            size_t log_step = 4 - log_m;
            size_t w_idx = lane_id >> (log_step + 1);

            BasicInteger reg_new = __shfl_xor_sync(0xffffffff, reg, 1 << log_step);
            size_t C = (lane_id >> log_step) & 1;
            BasicInteger left = (1 - C) * (reg - reg_new) + reg_new;  // C = 0 choose reg, C = 1 choose reg_new
            BasicInteger right = C * (reg - reg_new) + reg_new;  // C = 0 choose reg_new, C = 1 choose reg
            gs_butterfly_shoup(left, right, s_tw_inv_root_2n1[(1 << log_m) + w_idx],
                               s_tw_inv_root_2n1_shoup[(1 << log_m) + w_idx], mod);
            reg = (1 - C) * (left - right) + right;
        }
        // transpose write back to s_out
        BasicInteger tmp = modMulShoup(reg, inv_n, inv_n_shoup, mod);
        if (IsCompositeNTT)
            s_n1n2[lane_id][n2_idx] = device_ScaleP_fast(tmp, P);
        else
            s_n1n2[lane_id][n2_idx] = tmp;
    }
    __syncthreads();

    // store into global memory
    BasicInteger new_mod = (IsCompositeNTT) ? Q : mod;
    for (size_t n1_idx = warp_id; n1_idx < n1; n1_idx += warpsPerBlock) {
        for (size_t n2_idx = lane_id; n2_idx < n2; n2_idx += warpSize) {
            acc[batch_idx * 2 * N + blockIdx.x * N + n1_idx * n2 + n2_idx] =
                    modAdd(acc[batch_idx * 2 * N + blockIdx.x * N + n1_idx * n2 + n2_idx],
                           s_n1n2[n1_idx][n2_idx], new_mod);
        }
    }
}
