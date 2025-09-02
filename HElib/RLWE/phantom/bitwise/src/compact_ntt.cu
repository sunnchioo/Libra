#include "compact_ntt.cuh"

#include "arith/nbtheory.h"
#include "arith/big_integer_modop.h"

__global__ void
kernel_4step_ntt_forward_1024(BasicInteger *out, const BasicInteger *in,
                              const BasicInteger *tw_root_2n1, const BasicInteger *tw_root_2n1_shoup,
                              const BasicInteger *tw_root_2n, const BasicInteger *tw_root_2n_shoup,
                              BasicInteger mod) {
    constexpr size_t n1 = 32;
    constexpr size_t n2 = 32;
    constexpr size_t n = 1024;

    const size_t warpsPerBlock = blockDim.x / warpSize;

    __shared__ BasicInteger s_n1n2[n1][n2 + 1];
    __shared__ BasicInteger s_tw_root_2n1[n1];
    __shared__ BasicInteger s_tw_root_2n1_shoup[n1];

    // load twiddle factors into shared memory
    for (size_t tid = threadIdx.x; tid < n1; tid += blockDim.x) {
        s_tw_root_2n1[tid] = tw_root_2n1[tid];
        s_tw_root_2n1_shoup[tid] = tw_root_2n1_shoup[tid];
    }

    // transpose store into shared memory
    const size_t warp_id = threadIdx.x >> 5;
    const size_t lane_id = threadIdx.x & 31;
    for (size_t n1_idx = warp_id; n1_idx < n1; n1_idx += warpsPerBlock)
        for (size_t n2_idx = lane_id; n2_idx < n2; n2_idx += warpSize)
            s_n1n2[n1_idx][n2_idx] = in[blockIdx.x * n + n1_idx * n2 + n2_idx];
    __syncthreads();

    device_4step_ntt_forward_phase1_ws<32, 32>(s_n1n2, s_tw_root_2n1, s_tw_root_2n1_shoup,
                                               tw_root_2n, tw_root_2n_shoup, mod);
    __syncthreads();

    device_4step_ntt_forward_phase2_ws<32, 32>(out + blockIdx.x * n, s_n1n2, s_tw_root_2n1, s_tw_root_2n1_shoup, mod);
}

__global__ void
kernel_4step_ntt_forward_2048(BasicInteger *out, const BasicInteger *in,
                              const BasicInteger *tw_root_2n1, const BasicInteger *tw_root_2n1_shoup,
                              const BasicInteger *tw_root_2n, const BasicInteger *tw_root_2n_shoup,
                              const BasicInteger *tw_root_n2, const BasicInteger *tw_root_n2_shoup,
                              BasicInteger mod) {
    constexpr size_t n1 = 64;
    constexpr size_t n2 = 32;
    constexpr size_t n = 2048;

    const size_t warpsPerBlock = blockDim.x / warpSize;

    __shared__ BasicInteger s_n1n2[n1][n2 + 1];
    __shared__ BasicInteger s_tw_root_2n1[n1];
    __shared__ BasicInteger s_tw_root_2n1_shoup[n1];
    __shared__ BasicInteger s_tw_root_n2[n2 / 2];
    __shared__ BasicInteger s_tw_root_n2_shoup[n2 / 2];

    // load twiddle factors into shared memory
    for (size_t tid = threadIdx.x; tid < n1; tid += blockDim.x) {
        s_tw_root_2n1[tid] = tw_root_2n1[tid];
        s_tw_root_2n1_shoup[tid] = tw_root_2n1_shoup[tid];
    }

    for (size_t tid = threadIdx.x; tid < n2 / 2; tid += blockDim.x) {
        s_tw_root_n2[tid] = tw_root_n2[tid];
        s_tw_root_n2_shoup[tid] = tw_root_n2_shoup[tid];
    }

    // transpose store into shared memory
    const size_t warp_id = threadIdx.x >> 5;
    const size_t lane_id = threadIdx.x & 31;
    for (size_t n1_idx = warp_id; n1_idx < n1; n1_idx += warpsPerBlock)
        for (size_t n2_idx = lane_id; n2_idx < n2; n2_idx += warpSize)
            s_n1n2[n1_idx][n2_idx] = in[blockIdx.x * n + n1_idx * n2 + n2_idx];
    __syncthreads();

    device_4step_ntt_forward_phase1_naive<64, 32>(s_n1n2, s_tw_root_2n1, s_tw_root_2n1_shoup,
                                                  tw_root_2n, tw_root_2n_shoup, mod);
    __syncthreads();

    device_4step_ntt_forward_phase2_ws<64, 32>(out + blockIdx.x * n, s_n1n2, s_tw_root_n2, s_tw_root_n2_shoup, mod);
}

__global__ void
kernel_4step_ntt_forward_4096(BasicInteger *out, const BasicInteger *in,
                              const BasicInteger *tw_root_2n1, const BasicInteger *tw_root_2n1_shoup,
                              const BasicInteger *tw_root_2n, const BasicInteger *tw_root_2n_shoup,
                              BasicInteger mod) {
    constexpr size_t n1 = 64;
    constexpr size_t n2 = 64;
    constexpr size_t n = 4096;

    const size_t warpsPerBlock = blockDim.x / warpSize;

    __shared__ BasicInteger s_n1n2[n1][n2 + 1];
    __shared__ BasicInteger s_tw_root_2n1[n1];
    __shared__ BasicInteger s_tw_root_2n1_shoup[n1];

    // load twiddle factors into shared memory
    for (size_t tid = threadIdx.x; tid < n1; tid += blockDim.x) {
        s_tw_root_2n1[tid] = tw_root_2n1[tid];
        s_tw_root_2n1_shoup[tid] = tw_root_2n1_shoup[tid];
    }

    // transpose store into shared memory
    const size_t warp_id = threadIdx.x >> 5;
    const size_t lane_id = threadIdx.x & 31;
    for (size_t n1_idx = warp_id; n1_idx < n1; n1_idx += warpsPerBlock)
        for (size_t n2_idx = lane_id; n2_idx < n2; n2_idx += warpSize)
            s_n1n2[n1_idx][n2_idx] = in[blockIdx.x * n + n1_idx * n2 + n2_idx];
    __syncthreads();

    device_4step_ntt_forward_phase1_naive<64, 64>(s_n1n2,
                                                  s_tw_root_2n1, s_tw_root_2n1_shoup,
                                                  tw_root_2n, tw_root_2n_shoup, mod);
    __syncthreads();

    device_4step_ntt_forward_phase2_naive<64, 64>(out + blockIdx.x * n, s_n1n2,
                                                  s_tw_root_2n1, s_tw_root_2n1_shoup, mod);
}

__global__ void
kernel_4step_ntt_inverse_1024(BasicInteger *out, const BasicInteger *in,
                              const BasicInteger *tw_inv_root_2n, const BasicInteger *tw_inv_root_2n_shoup,
                              const BasicInteger *tw_inv_root_2n1, const BasicInteger *tw_inv_root_2n1_shoup,
                              BasicInteger inv_n, BasicInteger inv_n_shoup, BasicInteger mod) {
    constexpr size_t n1 = 32;
    constexpr size_t n2 = 32;
    constexpr size_t n = 1024;

    __shared__ BasicInteger s_n1n2[n1][n2 + 1];
    __shared__ BasicInteger s_tw_inv_root_2n1[n1];
    __shared__ BasicInteger s_tw_inv_root_2n1_shoup[n1];

    // load twiddle factors into shared memory
    for (size_t tid = threadIdx.x; tid < n1; tid += blockDim.x) {
        s_tw_inv_root_2n1[tid] = tw_inv_root_2n1[tid];
        s_tw_inv_root_2n1_shoup[tid] = tw_inv_root_2n1_shoup[tid];
    }
    __syncthreads();

    device_4step_ntt_inverse_phase1_ws<32, 32>(s_n1n2, in + blockIdx.x * n,
                                               s_tw_inv_root_2n1, s_tw_inv_root_2n1_shoup,
                                               tw_inv_root_2n, tw_inv_root_2n_shoup, mod);
    __syncthreads();

    device_4step_ntt_inverse_phase2_ws<32, 32>(s_n1n2, s_tw_inv_root_2n1, s_tw_inv_root_2n1_shoup,
                                               inv_n, inv_n_shoup, mod);
    __syncthreads();

    // store into global memory
    const size_t warpsPerBlock = blockDim.x / warpSize;
    const size_t warp_id = threadIdx.x >> 5;
    const size_t lane_id = threadIdx.x & 31;
    for (size_t n1_idx = warp_id; n1_idx < n1; n1_idx += warpsPerBlock)
        for (size_t n2_idx = lane_id; n2_idx < n2; n2_idx += warpSize)
            out[blockIdx.x * n + n1_idx * n2 + n2_idx] = s_n1n2[n1_idx][n2_idx];
}

__global__ void
kernel_4step_ntt_inverse_2048(BasicInteger *out, const BasicInteger *in,
                              const BasicInteger *tw_inv_root_n2, const BasicInteger *tw_inv_root_n2_shoup,
                              const BasicInteger *tw_inv_root_2n, const BasicInteger *tw_inv_root_2n_shoup,
                              const BasicInteger *tw_inv_root_2n1, const BasicInteger *tw_inv_root_2n1_shoup,
                              BasicInteger inv_n, BasicInteger inv_n_shoup, BasicInteger mod) {
    constexpr size_t n1 = 64;
    constexpr size_t n2 = 32;
    constexpr size_t n = 2048;

    __shared__ BasicInteger s_n1n2[n1][n2 + 1];
    __shared__ BasicInteger s_tw_inv_root_2n1[n1];
    __shared__ BasicInteger s_tw_inv_root_2n1_shoup[n1];
    __shared__ BasicInteger s_tw_inv_root_n2[n2 / 2];
    __shared__ BasicInteger s_tw_inv_root_n2_shoup[n2 / 2];

    // load twiddle factors into shared memory
    for (size_t tid = threadIdx.x; tid < n1; tid += blockDim.x) {
        s_tw_inv_root_2n1[tid] = tw_inv_root_2n1[tid];
        s_tw_inv_root_2n1_shoup[tid] = tw_inv_root_2n1_shoup[tid];
    }

    for (size_t tid = threadIdx.x; tid < n2 / 2; tid += blockDim.x) {
        s_tw_inv_root_n2[tid] = tw_inv_root_n2[tid];
        s_tw_inv_root_n2_shoup[tid] = tw_inv_root_n2_shoup[tid];
    }
    __syncthreads();

    device_4step_ntt_inverse_phase1_ws<64, 32>(s_n1n2, in + blockIdx.x * n, s_tw_inv_root_n2, s_tw_inv_root_n2_shoup,
                                               tw_inv_root_2n, tw_inv_root_2n_shoup, mod);
    __syncthreads();

    device_4step_ntt_inverse_phase2_naive<64, 32>(s_n1n2, s_tw_inv_root_2n1, s_tw_inv_root_2n1_shoup,
                                                  inv_n, inv_n_shoup, mod);
    __syncthreads();

    // store into global memory
    const size_t warpsPerBlock = blockDim.x / warpSize;
    const size_t warp_id = threadIdx.x >> 5;
    const size_t lane_id = threadIdx.x & 31;
    for (size_t n1_idx = warp_id; n1_idx < n1; n1_idx += warpsPerBlock)
        for (size_t n2_idx = lane_id; n2_idx < n2; n2_idx += warpSize)
            out[blockIdx.x * n + n1_idx * n2 + n2_idx] = s_n1n2[n1_idx][n2_idx];
}

__global__ void
kernel_4step_ntt_inverse_4096(BasicInteger *out, const BasicInteger *in,
                              const BasicInteger *tw_inv_root_2n, const BasicInteger *tw_inv_root_2n_shoup,
                              const BasicInteger *tw_inv_root_2n1, const BasicInteger *tw_inv_root_2n1_shoup,
                              BasicInteger inv_n, BasicInteger inv_n_shoup, BasicInteger mod) {
    constexpr size_t n1 = 64;
    constexpr size_t n2 = 64;
    constexpr size_t n = 4096;

    __shared__ BasicInteger s_n1n2[n1][n2 + 1];
    __shared__ BasicInteger s_tw_inv_root_2n1[n1];
    __shared__ BasicInteger s_tw_inv_root_2n1_shoup[n1];

    // load twiddle factors into shared memory
    for (size_t tid = threadIdx.x; tid < n1; tid += blockDim.x) {
        s_tw_inv_root_2n1[tid] = tw_inv_root_2n1[tid];
        s_tw_inv_root_2n1_shoup[tid] = tw_inv_root_2n1_shoup[tid];
    }
    __syncthreads();

    // read from global memory to shared memory
    const size_t warpsPerBlock = blockDim.x / warpSize;
    const size_t warp_id = threadIdx.x >> 5;
    const size_t lane_id = threadIdx.x & 31;
    for (size_t n1_idx = warp_id; n1_idx < n1; n1_idx += warpsPerBlock)
        for (size_t n2_idx = lane_id; n2_idx < n2; n2_idx += warpSize)
            s_n1n2[n1_idx][n2_idx] = in[blockIdx.x * n + n1_idx * n2 + n2_idx];
    __syncthreads();

    device_4step_ntt_inverse_phase1_naive<64, 64>(s_n1n2,
                                                  s_tw_inv_root_2n1, s_tw_inv_root_2n1_shoup,
                                                  tw_inv_root_2n, tw_inv_root_2n_shoup, mod);
    __syncthreads();

    device_4step_ntt_inverse_phase2_naive<64, 64>(s_n1n2,
                                                  s_tw_inv_root_2n1, s_tw_inv_root_2n1_shoup,
                                                  inv_n, inv_n_shoup, mod);
    __syncthreads();

    // store into global memory
    for (size_t n1_idx = warp_id; n1_idx < n1; n1_idx += warpsPerBlock)
        for (size_t n2_idx = lane_id; n2_idx < n2; n2_idx += warpSize)
            out[blockIdx.x * n + n1_idx * n2 + n2_idx] = s_n1n2[n1_idx][n2_idx];
}

void FourStepNTT::forward(BasicInteger *output, const BasicInteger *input, size_t threadsPerBlock, size_t n_poly,
                          const cudaStream_t &stream) {
    size_t sMemSize;
    switch (n_) {
        case 1024:
            sMemSize = n1_ * (n2_ + 1) + 2 * n1_;
            kernel_4step_ntt_forward_1024<<<n_poly, threadsPerBlock, sMemSize, stream>>>(
                    output, input,
                    tw_root_2n1_.get(), tw_root_2n1_shoup_.get(), // twiddle factors for negacyclic convolution
                    tw_root_2n_.get(), tw_root_2n_shoup_.get(), // twiddle factors for correction step
                    q_);
            break;
        case 2048:
            sMemSize = n1_ * (n2_ + 1) + 2 * n1_ + n2_;
            kernel_4step_ntt_forward_2048<<<n_poly, threadsPerBlock, sMemSize, stream>>>(
                    output, input,
                    tw_root_2n1_.get(), tw_root_2n1_shoup_.get(), // twiddle factors for negacyclic convolution
                    tw_root_2n_.get(), tw_root_2n_shoup_.get(), // twiddle factors for correction step
                    tw_root_n2_.get(), tw_root_n2_shoup_.get(), // twiddle factors for cyclic convolution
                    q_);
            break;
        case 4096:
            sMemSize = n1_ * (n2_ + 1) + 2 * n1_;
            cudaFuncSetAttribute(kernel_4step_ntt_forward_4096, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 (int) sMemSize);
            kernel_4step_ntt_forward_4096<<<n_poly, threadsPerBlock, sMemSize, stream>>>(
                    output, input,
                    tw_root_2n1_.get(), tw_root_2n1_shoup_.get(), // twiddle factors for negacyclic convolution
                    tw_root_2n_.get(), tw_root_2n_shoup_.get(), // twiddle factors for correction step
                    q_);
            break;
        default:
            throw std::invalid_argument("Current 4step NTT implementation requires n to be 1024 or 4096");
    }
}

void FourStepNTT::inverse(BasicInteger *output, const BasicInteger *input, size_t threadsPerBlock, size_t n_poly,
                          const cudaStream_t &stream) {
    size_t sMemSize;
    switch (n_) {
        case 1024:
            sMemSize = n1_ * (n2_ + 1) + 2 * n1_;
            kernel_4step_ntt_inverse_1024<<<n_poly, threadsPerBlock, sMemSize, stream>>>(
                    output, input,
                    tw_inv_root_2n_.get(), tw_inv_root_2n_shoup_.get(), // twiddle factors for correction step
                    tw_inv_root_2n1_.get(), tw_inv_root_2n1_shoup_.get(), // twiddle factors for negacyclic convolution
                    inv_n_, inv_n_shoup_, q_);
            break;
        case 2048:
            sMemSize = n1_ * (n2_ + 1) + 2 * n1_ + n2_;
            kernel_4step_ntt_inverse_2048<<<n_poly, threadsPerBlock, sMemSize, stream>>>(
                    output, input,
                    tw_inv_root_n2_.get(), tw_inv_root_n2_shoup_.get(), // twiddle factors for cyclic convolution
                    tw_inv_root_2n_.get(), tw_inv_root_2n_shoup_.get(), // twiddle factors for correction step
                    tw_inv_root_2n1_.get(), tw_inv_root_2n1_shoup_.get(), // twiddle factors for negacyclic convolution
                    inv_n_, inv_n_shoup_, q_);
            break;
        case 4096:
            sMemSize = n1_ * (n2_ + 1) + 2 * n1_;
            cudaFuncSetAttribute(kernel_4step_ntt_inverse_4096, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 (int) sMemSize);
            kernel_4step_ntt_inverse_4096<<<n_poly, threadsPerBlock, sMemSize, stream>>>(
                    output, input,
                    tw_inv_root_2n_.get(), tw_inv_root_2n_shoup_.get(), // twiddle factors for correction step
                    tw_inv_root_2n1_.get(), tw_inv_root_2n1_shoup_.get(), // twiddle factors for negacyclic convolution
                    inv_n_, inv_n_shoup_, q_);
            break;
        default:
            throw std::invalid_argument("Current 4step inverse NTT implementation requires n to be 1024 or 4096");
    }
}

__global__ void kernel_multiply(BasicInteger *output, const BasicInteger *input1, const BasicInteger *input2,
                                BasicInteger mod, BasicInteger mu_lo, BasicInteger mu_hi) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    output[idx] = modMulBarrett(input1[idx], input2[idx], mod, wide_type<BasicInteger>(mu_lo, mu_hi));
}

void FourStepNTT::multiply(BasicInteger *output, const BasicInteger *input1, const BasicInteger *input2,
                           const cudaStream_t &stream) {
    kernel_multiply<<<n_ / 256, 256, 0, stream>>>(
            output, input1, input2, q_, mu_[0], mu_[1]);
}

__global__ void kernel_multiply_and_accumulate(BasicInteger *output,
                                               const BasicInteger *input1, const BasicInteger *input2,
                                               BasicInteger mod, BasicInteger mu_lo, BasicInteger mu_hi) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    auto tmp = modMulBarrett(input1[idx], input2[idx], mod, wide_type<BasicInteger>(mu_lo, mu_hi));
    tmp += output[idx];
    output[idx] = modFast(tmp, mod);
}

void FourStepNTT::multiply_and_accumulate(BasicInteger *output, const BasicInteger *input1, const BasicInteger *input2,
                                          const cudaStream_t &stream) {
    kernel_multiply_and_accumulate<<<n_ / 256, 256, 0, stream>>>(
            output, input1, input2, q_, mu_[0], mu_[1]);
}

__global__ void kernel_multiply_scalar(BasicInteger *output,
                                       const BasicInteger *input, BasicInteger scalar,
                                       BasicInteger mod, BasicInteger mu_lo, BasicInteger mu_hi) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    output[idx] = modMulBarrett(input[idx], scalar, mod, wide_type<BasicInteger>(mu_lo, mu_hi));
}

void FourStepNTT::multiply_scalar(BasicInteger *output, const BasicInteger *input, BasicInteger scalar,
                                  const cudaStream_t &stream) {
    kernel_multiply_scalar<<<n_ / 256, 256, 0, stream>>>(
            output, input, scalar, q_, mu_[0], mu_[1]);
}

__global__ void kernel_add(BasicInteger *output, const BasicInteger *input1, const BasicInteger *input2,
                           BasicInteger mod) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    output[idx] = modFast(input1[idx] + input2[idx], mod);
}

void FourStepNTT::add(BasicInteger *output, const BasicInteger *input1, const BasicInteger *input2,
                      const cudaStream_t &stream) {
    kernel_add<<<n_ / 256, 256, 0, stream>>>(output, input1, input2, q_);
}

/**
  * This struct contains a op and a precomputed quotient: (op << 64) / mod, for a specific mod.
  * When passed to multiply_uint_mod, a faster variant of Barrett reduction will be performed.
  * Operand must be less than mod.
  */
static inline uint64_t compute_shoup(const uint64_t op, const uint64_t mod) {
    // Using __uint128_t to avoid overflow during multiplication
    __uint128_t temp = op;
    temp <<= 64; // multiplying by 2^64
    return temp / mod;
}

static inline uint32_t compute_shoup(const uint32_t op, const uint32_t mod) {
    uint64_t temp = op;
    temp <<= 32; // multiplying by 2^32
    return temp / mod;
}

void FourStepNTT::gen_tw_table(BasicInteger *tw_root, BasicInteger *tw_root_shoup, BasicInteger root,
                               size_t len, size_t log_len, BasicInteger mod, const cudaStream_t &stream) {
    std::vector<BasicInteger> host_tw_root(len);
    std::vector<BasicInteger> host_tw_root_shoup(len);
    host_tw_root[0] = 1;
    host_tw_root_shoup[0] = compute_shoup(1, mod);
    BasicInteger power = root;
    for (size_t i = 1; i < len; i++) {
        host_tw_root[isecfhe::ReverseBits(i, log_len)] = power;
        host_tw_root_shoup[isecfhe::ReverseBits(i, log_len)] = compute_shoup(power, mod);
        power = isecfhe::util::ModMul(power, root, mod);
    }
    cudaMemcpyAsync(tw_root, host_tw_root.data(), len * sizeof(BasicInteger),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(tw_root_shoup, host_tw_root_shoup.data(), len * sizeof(BasicInteger),
                    cudaMemcpyHostToDevice, stream);
}

FourStepNTT::FourStepNTT(size_t n, BasicInteger mod, const cudaStream_t &stream) {
    // set n, n1, n2, log_n, log_n1, log_n2
    n_ = n;
    if (n_ == 1024) {
        n1_ = 32;
        n2_ = 32;
        log_n_ = 10;
        log_n1_ = 5;
        log_n2_ = 5;
    } else if (n_ == 2048) {
        n1_ = 64;
        n2_ = 32;
        log_n_ = 11;
        log_n1_ = 6;
        log_n2_ = 5;
    } else if (n_ == 4096) {
        n1_ = 64;
        n2_ = 64;
        log_n_ = 12;
        log_n1_ = 6;
        log_n2_ = 6;
    } else {
        throw std::invalid_argument("Current 4step NTT implementation requires n to be 1024, 2048, or 4096");
    }

    // set q
    q_ = mod;
    mu_ = (isecfhe::BigInteger<BasicInteger>(1) << (8 * sizeof(BasicInteger) * 2)).DividedBy(q_).first.GetValue();

    // set root_2n and inv_root_2n
    root_2n_ = isecfhe::RootOfUnity(2 * n_, isecfhe::BigInteger<BasicInteger>(q_)).ConvertToInt<BasicInteger>();
    inv_root_2n_ = isecfhe::util::ModInverse(root_2n_, q_);

    // set inv_n and inv_n_shoup
    inv_n_ = isecfhe::util::ModInverse(static_cast<BasicInteger>(n_), q_);
    inv_n_shoup_ = compute_shoup(inv_n_, q_);

    /*******************************************************************************************************************
     * Precompute twiddle factors for negacyclic convolution in 4-step NTT
     ******************************************************************************************************************/

    // root_2n1 = root_2n ^ n2
    BasicInteger root_2n1 = isecfhe::util::ModExp(root_2n_, static_cast<BasicInteger>(n2_), q_);
    BasicInteger inv_root_2n1 = isecfhe::util::ModInverse(root_2n1, q_);

    tw_root_2n1_ = phantom::util::make_cuda_auto_ptr<BasicInteger>(n1_, stream);
    tw_root_2n1_shoup_ = phantom::util::make_cuda_auto_ptr<BasicInteger>(n1_, stream);
    gen_tw_table(tw_root_2n1_.get(), tw_root_2n1_shoup_.get(), root_2n1,
                 n1_, log_n1_, q_, stream);

    tw_inv_root_2n1_ = phantom::util::make_cuda_auto_ptr<BasicInteger>(n1_, stream);
    tw_inv_root_2n1_shoup_ = phantom::util::make_cuda_auto_ptr<BasicInteger>(n1_, stream);
    gen_tw_table(tw_inv_root_2n1_.get(), tw_inv_root_2n1_shoup_.get(), inv_root_2n1,
                 n1_, log_n1_, q_, stream);

    /*******************************************************************************************************************
     * Precompute twiddle factors for cyclic convolution in 4-step NTT
     ******************************************************************************************************************/

    // root_n2 = root_2n ^ (2 * n1)
    BasicInteger root_n2 = isecfhe::util::ModExp(root_2n_, static_cast<BasicInteger>(2 * n1_), q_);
    BasicInteger inv_root_n2 = isecfhe::util::ModInverse(root_n2, q_);

    tw_root_n2_ = phantom::util::make_cuda_auto_ptr<BasicInteger>(n2_ / 2, stream);
    tw_root_n2_shoup_ = phantom::util::make_cuda_auto_ptr<BasicInteger>(n2_ / 2, stream);
    gen_tw_table(tw_root_n2_.get(), tw_root_n2_shoup_.get(), root_n2,
                 n2_ / 2, log_n2_ - 1, q_, stream);

    tw_inv_root_n2_ = phantom::util::make_cuda_auto_ptr<BasicInteger>(n2_ / 2, stream);
    tw_inv_root_n2_shoup_ = phantom::util::make_cuda_auto_ptr<BasicInteger>(n2_ / 2, stream);
    gen_tw_table(tw_inv_root_n2_.get(), tw_inv_root_n2_shoup_.get(), inv_root_n2,
                 n2_ / 2, log_n2_ - 1, q_, stream);

    /*******************************************************************************************************************
     * Precompute twiddle factors for correction step in 4-step NTT
     ******************************************************************************************************************/

    tw_root_2n_ = phantom::util::make_cuda_auto_ptr<BasicInteger>(n_, stream);
    tw_root_2n_shoup_ = phantom::util::make_cuda_auto_ptr<BasicInteger>(n_, stream);
    tw_inv_root_2n_ = phantom::util::make_cuda_auto_ptr<BasicInteger>(n_, stream);
    tw_inv_root_2n_shoup_ = phantom::util::make_cuda_auto_ptr<BasicInteger>(n_, stream);

    std::vector<BasicInteger> host_tw_root_2n(n_);
    std::vector<BasicInteger> host_tw_root_2n_shoup(n_);
    std::vector<BasicInteger> host_tw_inv_root_2n(n_);
    std::vector<BasicInteger> host_tw_inv_root_2n_shoup(n_);
    for (size_t i = 0; i < n2_; i++) {
        for (size_t j = 0; j < n1_; j++) {
            // calculate exponent
            size_t brev_j = isecfhe::ReverseBits(j, log_n1_);
            size_t exp = 2 * brev_j * i + i;

            // root_ij = root_2n ^ exp
            BasicInteger root_ij = isecfhe::util::ModExp(root_2n_, static_cast<BasicInteger>(exp), q_);
            host_tw_root_2n[i * n1_ + j] = root_ij;
            host_tw_root_2n_shoup[i * n1_ + j] = compute_shoup(root_ij, q_);
        }
    }
    for (size_t i = 0; i < n1_; i++) {
        size_t brev_i = isecfhe::ReverseBits(i, log_n1_);
        for (size_t j = 0; j < n2_; j++) {
            // calculate exponent
            size_t exp = 2 * brev_i * j + j;

            // inv_root_ij = inv_root_2n ^ exp
            BasicInteger inv_root_ij = isecfhe::util::ModExp(inv_root_2n_, static_cast<BasicInteger>(exp), q_);
            host_tw_inv_root_2n[i * n2_ + j] = inv_root_ij;
            host_tw_inv_root_2n_shoup[i * n2_ + j] = compute_shoup(inv_root_ij, q_);
        }
    }
    cudaMemcpyAsync(tw_root_2n_.get(), host_tw_root_2n.data(), n_ * sizeof(BasicInteger),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(tw_root_2n_shoup_.get(), host_tw_root_2n_shoup.data(), n_ * sizeof(BasicInteger),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(tw_inv_root_2n_.get(), host_tw_inv_root_2n.data(), n_ * sizeof(BasicInteger),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(tw_inv_root_2n_shoup_.get(), host_tw_inv_root_2n_shoup.data(), n_ * sizeof(BasicInteger),
                    cudaMemcpyHostToDevice, stream);
}

FourStepNTT::FourStepNTT(size_t n, BasicInteger q1, BasicInteger q2, const cudaStream_t &stream) {
    // set n, n1, n2, log_n, log_n1, log_n2
    n_ = n;
    if (n_ == 1024) {
        n1_ = 32;
        n2_ = 32;
        log_n_ = 10;
        log_n1_ = 5;
        log_n2_ = 5;
    } else if (n_ == 2048) {
        n1_ = 64;
        n2_ = 32;
        log_n_ = 11;
        log_n1_ = 6;
        log_n2_ = 5;
    } else if (n_ == 4096) {
        n1_ = 64;
        n2_ = 64;
        log_n_ = 12;
        log_n1_ = 6;
        log_n2_ = 6;
    } else {
        throw std::invalid_argument("Current 4step NTT implementation requires n to be 1024, 2048, or 4096");
    }

    // set q
    BasicInteger q = q1 * q2;
    q_ = q;
    mu_ = (isecfhe::BigInteger<BasicInteger>(1) << (8 * sizeof(BasicInteger) * 2)).DividedBy(q_).first.GetValue();

    // compute root_2n_q1 and root_2n_q2
    auto root_2n_q1 = isecfhe::RootOfUnity(2 * n_, isecfhe::BigInteger<BasicInteger>(q1)).ConvertToInt<BasicInteger>();
    auto root_2n_q2 = isecfhe::RootOfUnity(2 * n_, isecfhe::BigInteger<BasicInteger>(q2)).ConvertToInt<BasicInteger>();

    // root_2n_q = root_2n_q1 * q2 * Integer(1/Zq1(q2)) + root_2n_q2 * q1 * Integer(1/Zq2(q1))
    BasicInteger inv_q1_q2 = isecfhe::util::ModInverse(q1, q2);
    BasicInteger inv_q2_q1 = isecfhe::util::ModInverse(q2, q1);
    BasicInteger root_2n_q = isecfhe::util::ModAddFast(
            isecfhe::util::ModMul(isecfhe::util::ModMul(root_2n_q1, q2, q), inv_q2_q1, q),
            isecfhe::util::ModMul(isecfhe::util::ModMul(root_2n_q2, q1, q), inv_q1_q2, q), q);

    // set root_2n and inv_root_2n
    root_2n_ = root_2n_q;
    inv_root_2n_ = isecfhe::util::ModInverse(root_2n_, q_);

    // set inv_n and inv_n_shoup
    inv_n_ = isecfhe::util::ModInverse(static_cast<BasicInteger>(n_), q_);
    inv_n_shoup_ = compute_shoup(inv_n_, q_);

    /*******************************************************************************************************************
     * Precompute twiddle factors for negacyclic convolution in 4-step NTT
     ******************************************************************************************************************/

    // root_2n1 = root_2n ^ n2
    BasicInteger root_2n1 = isecfhe::util::ModExp(root_2n_, static_cast<BasicInteger>(n2_), q_);
    BasicInteger inv_root_2n1 = isecfhe::util::ModInverse(root_2n1, q_);

    tw_root_2n1_ = phantom::util::make_cuda_auto_ptr<BasicInteger>(n1_, stream);
    tw_root_2n1_shoup_ = phantom::util::make_cuda_auto_ptr<BasicInteger>(n1_, stream);
    gen_tw_table(tw_root_2n1_.get(), tw_root_2n1_shoup_.get(), root_2n1,
                 n1_, log_n1_, q_, stream);

    tw_inv_root_2n1_ = phantom::util::make_cuda_auto_ptr<BasicInteger>(n1_, stream);
    tw_inv_root_2n1_shoup_ = phantom::util::make_cuda_auto_ptr<BasicInteger>(n1_, stream);
    gen_tw_table(tw_inv_root_2n1_.get(), tw_inv_root_2n1_shoup_.get(), inv_root_2n1,
                 n1_, log_n1_, q_, stream);

    /*******************************************************************************************************************
     * Precompute twiddle factors for cyclic convolution in 4-step NTT
     ******************************************************************************************************************/

    // root_n2 = root_2n ^ (2 * n1)
    BasicInteger root_n2 = isecfhe::util::ModExp(root_2n_, static_cast<BasicInteger>(2 * n1_), q_);
    BasicInteger inv_root_n2 = isecfhe::util::ModInverse(root_n2, q_);

    tw_root_n2_ = phantom::util::make_cuda_auto_ptr<BasicInteger>(n2_ / 2, stream);
    tw_root_n2_shoup_ = phantom::util::make_cuda_auto_ptr<BasicInteger>(n2_ / 2, stream);
    gen_tw_table(tw_root_n2_.get(), tw_root_n2_shoup_.get(), root_n2,
                 n2_ / 2, log_n2_ - 1, q_, stream);

    tw_inv_root_n2_ = phantom::util::make_cuda_auto_ptr<BasicInteger>(n2_ / 2, stream);
    tw_inv_root_n2_shoup_ = phantom::util::make_cuda_auto_ptr<BasicInteger>(n2_ / 2, stream);
    gen_tw_table(tw_inv_root_n2_.get(), tw_inv_root_n2_shoup_.get(), inv_root_n2,
                 n2_ / 2, log_n2_ - 1, q_, stream);

    /*******************************************************************************************************************
     * Precompute twiddle factors for correction step in 4-step NTT
     ******************************************************************************************************************/

    tw_root_2n_ = phantom::util::make_cuda_auto_ptr<BasicInteger>(n_, stream);
    tw_root_2n_shoup_ = phantom::util::make_cuda_auto_ptr<BasicInteger>(n_, stream);
    tw_inv_root_2n_ = phantom::util::make_cuda_auto_ptr<BasicInteger>(n_, stream);
    tw_inv_root_2n_shoup_ = phantom::util::make_cuda_auto_ptr<BasicInteger>(n_, stream);

    std::vector<BasicInteger> host_tw_root_2n(n_);
    std::vector<BasicInteger> host_tw_root_2n_shoup(n_);
    std::vector<BasicInteger> host_tw_inv_root_2n(n_);
    std::vector<BasicInteger> host_tw_inv_root_2n_shoup(n_);
    for (size_t i = 0; i < n2_; i++) {
        for (size_t j = 0; j < n1_; j++) {
            // calculate exponent
            size_t brev_j = isecfhe::ReverseBits(j, log_n1_);
            size_t exp = 2 * brev_j * i + i;

            // root_ij = root_2n ^ exp
            BasicInteger root_ij = isecfhe::util::ModExp(root_2n_, static_cast<BasicInteger>(exp), q_);
            host_tw_root_2n[i * n1_ + j] = root_ij;
            host_tw_root_2n_shoup[i * n1_ + j] = compute_shoup(root_ij, q_);
        }
    }
    for (size_t i = 0; i < n1_; i++) {
        size_t brev_i = isecfhe::ReverseBits(i, log_n1_);
        for (size_t j = 0; j < n2_; j++) {
            // calculate exponent
            size_t exp = 2 * brev_i * j + j;

            // inv_root_ij = inv_root_2n ^ exp
            BasicInteger inv_root_ij = isecfhe::util::ModExp(inv_root_2n_, static_cast<BasicInteger>(exp), q_);
            host_tw_inv_root_2n[i * n2_ + j] = inv_root_ij;
            host_tw_inv_root_2n_shoup[i * n2_ + j] = compute_shoup(inv_root_ij, q_);
        }
    }
    cudaMemcpyAsync(tw_root_2n_.get(), host_tw_root_2n.data(), n_ * sizeof(BasicInteger),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(tw_root_2n_shoup_.get(), host_tw_root_2n_shoup.data(), n_ * sizeof(BasicInteger),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(tw_inv_root_2n_.get(), host_tw_inv_root_2n.data(), n_ * sizeof(BasicInteger),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(tw_inv_root_2n_shoup_.get(), host_tw_inv_root_2n_shoup.data(), n_ * sizeof(BasicInteger),
                    cudaMemcpyHostToDevice, stream);
}
