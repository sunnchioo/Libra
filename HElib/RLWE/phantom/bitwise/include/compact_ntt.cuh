#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include "openfhe.h"

#include "cuda_wrapper.cuh"
#include "modarith.cuh"

class FourStepNTT {
private:
    BasicInteger q_{};
    std::vector<BasicInteger> mu_{};
    BasicInteger root_2n_{};
    BasicInteger inv_root_2n_{};
    BasicInteger inv_n_{};
    BasicInteger inv_n_shoup_{};
    size_t n_{};
    size_t n1_{};
    size_t n2_{};
    size_t log_n_{};
    size_t log_n1_{};
    size_t log_n2_{};

    phantom::util::cuda_auto_ptr<BasicInteger> tw_root_2n1_;
    phantom::util::cuda_auto_ptr<BasicInteger> tw_root_2n1_shoup_;
    phantom::util::cuda_auto_ptr<BasicInteger> tw_inv_root_2n1_;
    phantom::util::cuda_auto_ptr<BasicInteger> tw_inv_root_2n1_shoup_;

    phantom::util::cuda_auto_ptr<BasicInteger> tw_root_2n_;
    phantom::util::cuda_auto_ptr<BasicInteger> tw_root_2n_shoup_;
    phantom::util::cuda_auto_ptr<BasicInteger> tw_inv_root_2n_;
    phantom::util::cuda_auto_ptr<BasicInteger> tw_inv_root_2n_shoup_;

    phantom::util::cuda_auto_ptr<BasicInteger> tw_root_n2_;
    phantom::util::cuda_auto_ptr<BasicInteger> tw_root_n2_shoup_;
    phantom::util::cuda_auto_ptr<BasicInteger> tw_inv_root_n2_;
    phantom::util::cuda_auto_ptr<BasicInteger> tw_inv_root_n2_shoup_;

    static void gen_tw_table(BasicInteger *tw_root, BasicInteger *tw_root_shoup, BasicInteger root,
                             size_t len, size_t log_len, BasicInteger mod, const cudaStream_t &stream);

public:

    /**
     * Constructor for 4step NTT
     * @param n ring dimension
     * @param mod modulus
     * @param stream CUDA stream
     */
    FourStepNTT(size_t n, BasicInteger mod, const cudaStream_t &stream);

    /**
     * Constructor for composite 4step NTT
     * @param n ring dimension
     * @param q1 modulus 1
     * @param q2 modulus 2
     * @param stream CUDA stream
     */
    FourStepNTT(size_t n, BasicInteger q1, BasicInteger q2, const cudaStream_t &stream);

    void forward(BasicInteger *output, const BasicInteger *input, size_t threadsPerBlock, size_t n_poly,
                 const cudaStream_t &stream);

    void inverse(BasicInteger *output, const BasicInteger *input, size_t threadsPerBlock, size_t n_poly,
                 const cudaStream_t &stream);

    void multiply(BasicInteger *output, const BasicInteger *input1, const BasicInteger *input2,
                  const cudaStream_t &stream);

    void multiply_and_accumulate(BasicInteger *output, const BasicInteger *input1, const BasicInteger *input2,
                                 const cudaStream_t &stream);

    void multiply_scalar(BasicInteger *output, const BasicInteger *input, BasicInteger scalar,
                         const cudaStream_t &stream);

    void add(BasicInteger *output, const BasicInteger *input1, const BasicInteger *input2, const cudaStream_t &stream);

    [[nodiscard]] auto getMod() const -> BasicInteger {
        return q_;
    }

    [[nodiscard]] auto getMu() const -> const std::vector<BasicInteger> & {
        return mu_;
    }

    [[nodiscard]] auto getRoot2n() const -> BasicInteger {
        return root_2n_;
    }

    [[nodiscard]] auto getInvRoot2n() const -> BasicInteger {
        return inv_root_2n_;
    }

    [[nodiscard]] auto getInvn() const -> BasicInteger {
        return inv_n_;
    }

    [[nodiscard]] auto getInvnShoup() const -> BasicInteger {
        return inv_n_shoup_;
    }

    [[nodiscard]] auto getTwRoot2n1() const -> const BasicInteger * {
        return tw_root_2n1_.get();
    }

    [[nodiscard]] auto getTwRoot2n1Shoup() const -> const BasicInteger * {
        return tw_root_2n1_shoup_.get();
    }

    [[nodiscard]] auto getTwInvRoot2n1() const -> const BasicInteger * {
        return tw_inv_root_2n1_.get();
    }

    [[nodiscard]] auto getTwInvRoot2n1Shoup() const -> const BasicInteger * {
        return tw_inv_root_2n1_shoup_.get();
    }

    [[nodiscard]] auto getTwRoot2n() const -> const BasicInteger * {
        return tw_root_2n_.get();
    }

    [[nodiscard]] auto getTwRoot2nShoup() const -> const BasicInteger * {
        return tw_root_2n_shoup_.get();
    }

    [[nodiscard]] auto getTwInvRoot2n() const -> const BasicInteger * {
        return tw_inv_root_2n_.get();
    }

    [[nodiscard]] auto getTwInvRoot2nShoup() const -> const BasicInteger * {
        return tw_inv_root_2n_shoup_.get();
    }

    [[nodiscard]] auto getTwRootn2() const -> const BasicInteger * {
        return tw_root_n2_.get();
    }

    [[nodiscard]] auto getTwRootn2Shoup() const -> const BasicInteger * {
        return tw_root_n2_shoup_.get();
    }

    [[nodiscard]] auto getTwInvRootn2() const -> const BasicInteger * {
        return tw_inv_root_n2_.get();
    }

    [[nodiscard]] auto getTwInvRootn2Shoup() const -> const BasicInteger * {
        return tw_inv_root_n2_shoup_.get();
    }
};

/** Cooley-Tukey butterfly using Shoup's modmul
* input: x [0, 4q), y [0, 4q)
* output: x [0, 4q), y [0, 4q)
*/
__device__ inline void ct_butterfly_shoup(BasicInteger &x, BasicInteger &y,
                                          BasicInteger tw, BasicInteger tw_shoup,
                                          BasicInteger mod) {
    const BasicInteger mod2 = 2 * mod;
    const BasicInteger t = modMulShoupLazy(y, tw, tw_shoup, mod);
    x = modFast(x, mod2);
    y = x + mod2 - t;
    x += t;
}

/** Gentleman-Sande butterfly using Shoup's modmul
 * input: x [0, 2q), y [0, 2q)
 * output: x [0, 2q), y [0, 2q)
 */
__device__ inline void gs_butterfly_shoup(BasicInteger &x, BasicInteger &y,
                                          BasicInteger tw, BasicInteger tw_shoup,
                                          BasicInteger mod) {
    const BasicInteger mod2 = 2 * mod;
    const BasicInteger t = x + mod2 - y; // [0, 4q)
    x += y; // [0, 4q)
    x = modFast(x, mod2); // [0, 2q)
    y = modMulShoupLazy(t, tw, tw_shoup, mod);
}

template<size_t n1, size_t n2>
__device__ inline void device_4step_ntt_forward_phase1_ws(BasicInteger s_n1n2[n1][n2 + 1],
                                                          const BasicInteger *s_tw_root_2n1,
                                                          const BasicInteger *s_tw_root_2n1_shoup,
                                                          const BasicInteger *tw_root_2n,
                                                          const BasicInteger *tw_root_2n_shoup,
                                                          BasicInteger mod) {
    const size_t warpsPerBlock = blockDim.x / warpSize;
    const size_t warp_id = threadIdx.x >> 5;
    const size_t lane_id = threadIdx.x & 31;

    for (size_t n2_idx = warp_id; n2_idx < n2; n2_idx += warpsPerBlock) {
        BasicInteger reg = s_n1n2[lane_id][n2_idx];
        for (size_t log_m = 0; log_m < 5; log_m++) {
            size_t log_step = 4 - log_m;
            size_t w_idx = lane_id >> (log_step + 1);

            BasicInteger reg_new = __shfl_xor_sync(0xffffffff, reg, 1 << log_step);
            size_t C = (lane_id >> log_step) & 1;
            BasicInteger left = (1 - C) * (reg - reg_new) + reg_new;  // C = 0 choose reg, C = 1 choose reg_new
            BasicInteger right = C * (reg - reg_new) + reg_new;  // C = 0 choose reg_new, C = 1 choose reg
            ct_butterfly_shoup(left, right, s_tw_root_2n1[(1 << log_m) + w_idx],
                               s_tw_root_2n1_shoup[(1 << log_m) + w_idx], mod);
            reg = (1 - C) * (left - right) + right;
        }

        // elementwise multiply and transpose store into shared memory
        s_n1n2[lane_id][n2_idx] = modMulShoupLazy(reg, tw_root_2n[n2_idx * n1 + lane_id],
                                                  tw_root_2n_shoup[n2_idx * n1 + lane_id], mod);
    }
}

template<size_t n1, size_t n2>
__device__ inline void device_4step_ntt_forward_phase2_ws(BasicInteger *g_out,
                                                          const BasicInteger s_in[n1][n2 + 1],
                                                          const BasicInteger *s_tw_root_n2,
                                                          const BasicInteger *s_tw_root_n2_shoup, BasicInteger mod) {
    const size_t warpsPerBlock = blockDim.x / warpSize;
    const size_t warp_id = threadIdx.x >> 5;
    const size_t lane_id = threadIdx.x & 31;

    for (size_t n1_idx = warp_id; n1_idx < n1; n1_idx += warpsPerBlock) {
        BasicInteger reg = s_in[n1_idx][lane_id];
        for (size_t log_m = 0; log_m < 5; log_m++) {
            size_t log_step = 4 - log_m;
            size_t w_idx = lane_id >> (log_step + 1);

            BasicInteger reg_new = __shfl_xor_sync(0xffffffff, reg, 1 << log_step);
            size_t C = (lane_id >> log_step) & 1;
            BasicInteger left = (1 - C) * (reg - reg_new) + reg_new;  // C = 0 choose reg, C = 1 choose reg_new
            BasicInteger right = C * (reg - reg_new) + reg_new;  // C = 0 choose reg_new, C = 1 choose reg
            ct_butterfly_shoup(left, right, s_tw_root_n2[w_idx],
                               s_tw_root_n2_shoup[w_idx], mod);
            reg = (1 - C) * (left - right) + right;
        }
        // reduce and write back to global memory
        BasicInteger tmp = modFast(reg, mod * 2);
        g_out[n1_idx * n2 + lane_id] = modFast(tmp, mod);
    }
}

template<size_t n1, size_t n2>
__device__ inline void device_4step_ntt_forward_phase1_naive(BasicInteger s_n1n2[n1][n2 + 1],
                                                             const BasicInteger *s_tw_root_2n1,
                                                             const BasicInteger *s_tw_root_2n1_shoup,
                                                             const BasicInteger *tw_root_2n,
                                                             const BasicInteger *tw_root_2n_shoup,
                                                             BasicInteger mod) {
    const size_t warpsPerBlock = blockDim.x / warpSize;
    const size_t warp_id = threadIdx.x >> 5;
    const size_t lane_id = threadIdx.x & 31;

    for (size_t n2_idx = warp_id; n2_idx < n2; n2_idx += warpsPerBlock) {
        for (size_t log_m = 0; log_m < 6; log_m++) {
            size_t log_step = 5 - log_m;
            size_t w_idx = lane_id >> log_step;
            size_t butt_idx = ((w_idx << log_step) + lane_id) & 0x3f;
            ct_butterfly_shoup(s_n1n2[butt_idx][n2_idx], s_n1n2[butt_idx + (1 << log_step)][n2_idx],
                               s_tw_root_2n1[(1 << log_m) + w_idx], s_tw_root_2n1_shoup[(1 << log_m) + w_idx], mod);
        }
        // elementwise multiply and transpose store into shared memory
        s_n1n2[lane_id][n2_idx] = modMulShoupLazy(s_n1n2[lane_id][n2_idx],
                                                  tw_root_2n[n2_idx * n1 + lane_id],
                                                  tw_root_2n_shoup[n2_idx * n1 + lane_id], mod);
        s_n1n2[lane_id + warpSize][n2_idx] = modMulShoupLazy(s_n1n2[lane_id + warpSize][n2_idx],
                                                             tw_root_2n[n2_idx * n1 + lane_id + warpSize],
                                                             tw_root_2n_shoup[n2_idx * n1 + lane_id + warpSize], mod);
    }
}

template<size_t n1, size_t n2>
__device__ inline void device_4step_ntt_forward_phase2_naive(BasicInteger *g_out,
                                                             BasicInteger s_in[n1][n2 + 1],
                                                             const BasicInteger *s_tw_root_n2,
                                                             const BasicInteger *s_tw_root_n2_shoup, BasicInteger mod) {
    const size_t warpsPerBlock = blockDim.x / warpSize;
    const size_t warp_id = threadIdx.x >> 5;
    const size_t lane_id = threadIdx.x & 31;

    for (size_t n1_idx = warp_id; n1_idx < n1; n1_idx += warpsPerBlock) {
        for (size_t log_m = 0; log_m < 6; log_m++) {
            size_t log_step = 5 - log_m;
            size_t w_idx = lane_id >> log_step;
            size_t butt_idx = ((w_idx << log_step) + lane_id) & 0x3f;
            ct_butterfly_shoup(s_in[n1_idx][butt_idx], s_in[n1_idx][butt_idx + (1 << log_step)],
                               s_tw_root_n2[w_idx], s_tw_root_n2_shoup[w_idx], mod);
        }
        // reduce and write back to global memory
        BasicInteger tmp = modFast(s_in[n1_idx][lane_id], mod * 2);
        g_out[n1_idx * n2 + lane_id] = modFast(tmp, mod);
        tmp = modFast(s_in[n1_idx][lane_id + warpSize], mod * 2);
        g_out[n1_idx * n2 + lane_id + warpSize] = modFast(tmp, mod);
    }
}

template<size_t n1, size_t n2>
__device__ inline void device_4step_ntt_inverse_phase1_ws(BasicInteger s_n1n2[n1][n2 + 1],
                                                          const BasicInteger *g_in,
                                                          const BasicInteger *s_tw_inv_root_n2,
                                                          const BasicInteger *s_tw_inv_root_n2_shoup,
                                                          const BasicInteger *tw_inv_root_2n,
                                                          const BasicInteger *tw_inv_root_2n_shoup,
                                                          BasicInteger mod) {
    const size_t warpsPerBlock = blockDim.x / warpSize;
    const size_t warp_id = threadIdx.x >> 5;
    const size_t lane_id = threadIdx.x & 31;

    for (size_t n1_idx = warp_id; n1_idx < n1; n1_idx += warpsPerBlock) {
        BasicInteger reg = g_in[n1_idx * n2 + lane_id];
        for (int log_m = 4; log_m >= 0; log_m--) {
            size_t log_step = 4 - log_m;
            size_t w_idx = lane_id >> (log_step + 1);

            BasicInteger reg_new = __shfl_xor_sync(0xffffffff, reg, 1 << log_step);
            size_t C = (lane_id >> log_step) & 1;
            BasicInteger left = (1 - C) * (reg - reg_new) + reg_new;  // C = 0 choose reg, C = 1 choose reg_new
            BasicInteger right = C * (reg - reg_new) + reg_new;  // C = 0 choose reg_new, C = 1 choose reg
            gs_butterfly_shoup(left, right,
                               s_tw_inv_root_n2[w_idx], s_tw_inv_root_n2_shoup[w_idx], mod);
            reg = (1 - C) * (left - right) + right;
        }
        // elementwise multiply and transpose store into shared memory
        s_n1n2[n1_idx][lane_id] = modMulShoupLazy(reg,
                                                  tw_inv_root_2n[n1_idx * n2 + lane_id],
                                                  tw_inv_root_2n_shoup[n1_idx * n2 + lane_id],
                                                  mod);
    }
}

template<size_t n1, size_t n2>
__device__ inline void device_4step_ntt_inverse_phase2_ws(BasicInteger s_n1n2[n1][n2 + 1],
                                                          const BasicInteger *s_tw_inv_root_2n1,
                                                          const BasicInteger *s_tw_inv_root_2n1_shoup,
                                                          BasicInteger inv_n, BasicInteger inv_n_shoup,
                                                          BasicInteger mod) {
    const size_t warpsPerBlock = blockDim.x / warpSize;
    const size_t warp_id = threadIdx.x >> 5;
    const size_t lane_id = threadIdx.x & 31;

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
        BasicInteger tmp = modMulShoupLazy(reg, inv_n, inv_n_shoup, mod);
        s_n1n2[lane_id][n2_idx] = modFast(tmp, mod);
    }
}

template<size_t n1, size_t n2>
__device__ inline void device_4step_ntt_inverse_phase1_naive(BasicInteger s_n1n2[n1][n2 + 1],
                                                             const BasicInteger *s_tw_inv_root_n2,
                                                             const BasicInteger *s_tw_inv_root_n2_shoup,
                                                             const BasicInteger *tw_inv_root_2n,
                                                             const BasicInteger *tw_inv_root_2n_shoup,
                                                             BasicInteger mod) {
    const size_t warpsPerBlock = blockDim.x / warpSize;
    const size_t warp_id = threadIdx.x >> 5;
    const size_t lane_id = threadIdx.x & 31;

    for (size_t n1_idx = warp_id; n1_idx < n1; n1_idx += warpsPerBlock) {
        for (int log_m = 5; log_m >= 0; log_m--) {
            size_t log_step = 5 - log_m;
            size_t w_idx = lane_id >> log_step;
            size_t butt_idx = ((w_idx << log_step) + lane_id) & 0x3f;
            gs_butterfly_shoup(s_n1n2[n1_idx][butt_idx], s_n1n2[n1_idx][butt_idx + (1 << log_step)],
                               s_tw_inv_root_n2[w_idx], s_tw_inv_root_n2_shoup[w_idx], mod);
        }
        // elementwise multiply and transpose store into shared memory
        s_n1n2[n1_idx][lane_id] = modMulShoupLazy(s_n1n2[n1_idx][lane_id],
                                                  tw_inv_root_2n[n1_idx * n2 + lane_id],
                                                  tw_inv_root_2n_shoup[n1_idx * n2 + lane_id],
                                                  mod);
        s_n1n2[n1_idx][lane_id + warpSize] = modMulShoupLazy(s_n1n2[n1_idx][lane_id + warpSize],
                                                             tw_inv_root_2n[n1_idx * n2 + lane_id + warpSize],
                                                             tw_inv_root_2n_shoup[n1_idx * n2 + lane_id + warpSize],
                                                             mod);
    }
}

template<size_t n1, size_t n2>
__device__ inline void device_4step_ntt_inverse_phase2_naive(BasicInteger s_n1n2[n1][n2 + 1],
                                                             const BasicInteger *s_tw_inv_root_2n1,
                                                             const BasicInteger *s_tw_inv_root_2n1_shoup,
                                                             BasicInteger inv_n, BasicInteger inv_n_shoup,
                                                             BasicInteger mod) {
    const size_t warpsPerBlock = blockDim.x / warpSize;
    const size_t warp_id = threadIdx.x >> 5;
    const size_t lane_id = threadIdx.x & 31;

    for (size_t n2_idx = warp_id; n2_idx < n2; n2_idx += warpsPerBlock) {
        for (int log_m = 5; log_m >= 0; log_m--) {
            size_t log_step = 5 - log_m;
            size_t w_idx = lane_id >> log_step;
            size_t butt_idx = ((w_idx << log_step) + lane_id) & 0x3f;
            gs_butterfly_shoup(s_n1n2[butt_idx][n2_idx], s_n1n2[butt_idx + (1 << log_step)][n2_idx],
                               s_tw_inv_root_2n1[(1 << log_m) + w_idx],
                               s_tw_inv_root_2n1_shoup[(1 << log_m) + w_idx],
                               mod);
        }
        // transpose write back to s_out
        BasicInteger tmp = modMulShoupLazy(s_n1n2[lane_id][n2_idx], inv_n, inv_n_shoup, mod);
        s_n1n2[lane_id][n2_idx] = modFast(tmp, mod);
        tmp = modMulShoupLazy(s_n1n2[lane_id + warpSize][n2_idx], inv_n, inv_n_shoup, mod);
        s_n1n2[lane_id + warpSize][n2_idx] = modFast(tmp, mod);
    }
}
