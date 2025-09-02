#pragma once

#include <iomanip>

#include "algmath.h"
#include "cuparams.cuh"
#include "fft.cuh"
#include "keyswitch.cuh"
#include "trgsw.cuh"
#include "trlwe.cuh"
#include "types.h"
#include "util/pointer.cuh"
#include "utils.cuh"

namespace cuTFHEpp {
    template <typename LvlYZ, typename LvlY = LvlYZ::domainP, typename LvlZ = LvlYZ::targetP>
    struct ProgBootstrappingData {
        using Lvl = LvlYZ;

        TFHEpp::TRLWE<LvlZ> *acc;
        TFHEpp::TRLWE<LvlZ> *temp;
        TFHEpp::TLWE<LvlY> *tlwe_from;
        TFHEpp::TRLWEInFD<LvlZ> *trlwefft;
        TFHEpp::Polynomial<LvlZ> *testvector;

        ProgBootstrappingData() : ProgBootstrappingData(1) {}

        ProgBootstrappingData(size_t batch_size) {
            CUDA_CHECK_RETURN(cudaMalloc(&acc, batch_size * sizeof(TFHEpp::TRLWE<LvlZ>)));
            CUDA_CHECK_RETURN(cudaMalloc(&temp, batch_size * sizeof(TFHEpp::TRLWE<LvlZ>)));
            CUDA_CHECK_RETURN(cudaMalloc(&tlwe_from, batch_size * sizeof(TFHEpp::TLWE<LvlY>)));
            CUDA_CHECK_RETURN(cudaMalloc(&trlwefft, batch_size * sizeof(TFHEpp::TRLWEInFD<LvlZ>)));
            CUDA_CHECK_RETURN(cudaMalloc(&testvector, sizeof(TFHEpp::Polynomial<LvlZ>)));
        }

        ~ProgBootstrappingData() {
            CUDA_CHECK_RETURN(cudaFree(acc));
            CUDA_CHECK_RETURN(cudaFree(temp));
            CUDA_CHECK_RETURN(cudaFree(tlwe_from));
            CUDA_CHECK_RETURN(cudaFree(trlwefft));
            CUDA_CHECK_RETURN(cudaFree(testvector));
        }

        template <typename T, typename U = T::Lvl>
        static constexpr inline bool can_cast() {
            return isLvlCover<LvlYZ, U>();
        }
    };

    template <typename LvlYZ, typename LvlY = LvlYZ::domainP, typename LvlZ = LvlYZ::targetP, uint32_t num_out = 1>
    __device__ inline void ProgBlindRotate(
        const Context &context,
        double *shared_mem,
        TFHEpp::TLWE<LvlY> &tlwe,
        TFHEpp::TRLWE<LvlZ> &acc,
        TFHEpp::TRLWE<LvlZ> &temp,
        TFHEpp::TRLWEInFD<LvlZ> &trlwefft,
        TFHEpp::Polynomial<LvlZ> &testvector) {
        const unsigned int tid = blockDim.x * threadIdx.y + threadIdx.x;
        const unsigned int bdim = blockDim.x * blockDim.y;

        TFHEpp::BootstrappingKeyFFT<LvlYZ> &bkfft = *context.get_bk<LvlYZ>();

        constexpr uint32_t bitwidth = TFHEpp::bits_needed<num_out - 1>();
        const uint32_t a0 = 2 * LvlZ::n -
                            ((tlwe[LvlY::k * LvlY::n] >>
                              (std::numeric_limits<typename LvlY::T>::digits - 1 - LvlZ::nbit + bitwidth))
                             << bitwidth);  // 提取相位的高 nbit+1 位, 正向旋转转换为逆向旋转

        for (int i = tid; i < LvlZ::n; i += bdim)
#pragma unroll
            for (int k = 0; k < LvlZ::k; k++)
                acc[k][i] = 0;

        PolynomialMulByXai<LvlZ>(acc[LvlZ::k], testvector, a0);
        __syncthreads();

        for (int index = 0; index < LvlY::k * LvlY::n; index++) {
            constexpr typename LvlY::T roundoffset =
                1ULL << (std::numeric_limits<typename LvlY::T>::digits - 2 -
                         LvlZ::nbit);
            const uint32_t a =
                (tlwe[index] + roundoffset) >>
                (std::numeric_limits<typename LvlY::T>::digits - 1 -
                 LvlZ::nbit);

            if (a == 0)
                continue;
#pragma unroll
            for (int k = 0; k < LvlZ::k + 1; k++)
                PolynomialMulByXaiMinusOne<LvlZ>(temp[k], acc[k], a);

            trgswfftExternalProduct<LvlZ>(context, shared_mem, temp, trlwefft, bkfft[index], a);

            for (int i = tid; i < LvlZ::n; i += bdim)
#pragma unroll
                for (int k = 0; k < LvlZ::k + 1; k++)
                    acc[k][i] += temp[k][i];
            __syncthreads();
        }
    }

    template <typename LvlYZ, typename LvlY = LvlYZ::domainP, typename LvlZ = LvlYZ::targetP>
    __global__ void ProgBootstrappingTLWE2TLWEFFT(
        const Context &context,
        ProgBootstrappingData<LvlYZ> &bs_data,
        TFHEpp::TLWE<LvlZ> *res,
        const size_t batch_size) {
        __shared__ double shared_mem[LvlZ::n];

        const unsigned int bid = gridDim.x * blockIdx.y + blockIdx.x;
        const unsigned int gdim = gridDim.x * gridDim.y;

        for (int n = bid; n < batch_size; n += gdim) {
            TFHEpp::TLWE<LvlY> &tlwe = bs_data.tlwe_from[n];
            TFHEpp::TRLWE<LvlZ> &acc = bs_data.acc[n];
            TFHEpp::TRLWE<LvlZ> &temp = bs_data.temp[n];
            TFHEpp::TRLWEInFD<LvlZ> &trlwefft = bs_data.trlwefft[n];
            TFHEpp::Polynomial<LvlZ> &testvector = *bs_data.testvector;

            ProgBlindRotate<LvlYZ>(context, shared_mem, tlwe, acc, temp, trlwefft, testvector);
            SampleExtractIndex<LvlZ>(res[n], acc, 0);
        }
    }

#define CUTFHEPP_PROG_BOOTSTRAPPING_TLWE2TLWEFFT(Y, Z)                 \
    template __global__ void ProgBootstrappingTLWE2TLWEFFT<Lvl##Y##Z>( \
        const Context &context,                                        \
        ProgBootstrappingData<Lvl##Y##Z> &bs_data,                     \
        TFHEpp::TLWE<Lvl##Z> *res,                                     \
        size_t batch_size);

    EXPLICIT_LVL_UP_EXTERN(CUTFHEPP_PROG_BOOTSTRAPPING_TLWE2TLWEFFT);

    template <typename LvlXY, typename LvlYZ,
              typename LvlX = LvlXY::domainP, typename LvlY = LvlXY::targetP, typename LvlZ = LvlYZ::targetP>
    void ProgBootstrapping(
        const Context &context,
        util::Pointer<ProgBootstrappingData<LvlYZ>> &bs_data,
        typename LvlZ::T *lut,
        TFHEpp::TLWE<LvlZ> *res,
        TFHEpp::TLWE<LvlX> *tlwe,
        const size_t batch_size) {
        static_assert(std::is_same<LvlY, typename LvlYZ::domainP>::value, "invalid LvlY");

        HomADD_plain<LvlX><<<1, BLOCK_DIM>>>(tlwe, tlwe, LvlX::μ >> 1, batch_size);

        IdentityKeySwitch<LvlXY><<<GRID_DIM, BLOCK_DIM>>>(context, bs_data->tlwe_from, tlwe, batch_size);

        lutpolygen<LvlZ><<<1, BLOCK_DIM>>>(*bs_data->testvector, lut);

        constexpr size_t shared_mem_size = SHM_SIZE<LvlZ>;
        ProgBootstrappingTLWE2TLWEFFT<LvlYZ><<<GRID_DIM, BLOCK_DIM, shared_mem_size>>>(context, bs_data.get(), res, batch_size);
    }

#define CUTFHEPP_PROG_BOOTSTRAPPING(X, Y, Z)                      \
    template void ProgBootstrapping<Lvl##X##Y, Lvl##Y##Z>(        \
        const Context &context,                                   \
        util::Pointer<ProgBootstrappingData<Lvl##Y##Z>> &bs_data, \
        typename Lvl##Z::T *lut,                                  \
        TFHEpp::TLWE<Lvl##Z> *res,                                \
        TFHEpp::TLWE<Lvl##X> *tlwe,                               \
        const size_t batch_size);

    EXPLICIT_LVL_DOWN_UP_EXTERN(CUTFHEPP_PROG_BOOTSTRAPPING);

    template <typename LvlX>
    std::vector<typename LvlX::T> GenLUT(typename LvlX::T (*f)(typename LvlX::T m, typename LvlX::T p), typename LvlX::T p) {
        typename LvlX::T scale = static_cast<typename LvlX::T>(LvlX::Δ);

        // check overflow
        assert((algmath::count_bits(scale) + LvlX::plain_modulus_bit) > std::numeric_limits<typename LvlX::T>::digits);

        typename LvlX::T q = LvlX::n;
        typename LvlX::T x{0};

        size_t vecSize = static_cast<size_t>(LvlX::n);
        std::vector<typename LvlX::T> vec(vecSize, scale);
        for (size_t i = 0; i < vec.size(); ++i, x += p) {
            vec[i] *= f(x / q, p);

            // std::cout << "LUT(" << i << ") = " << vec[i]
            //           << ". f: " << f(x / q, p)
            //           << ". x / n: " << x / (LvlX::n) << ". q / p: " << q / p
            //           << ". x: " << x << ". q: " << q << ". p: " << p << std::endl;
        }
        return vec;
    }

#define CUTFHEPP_GenLUT(X)                                              \
    template std::vector<typename Lvl##X::T> GenLUT<Lvl##X>(            \
        typename Lvl##X::T (*)(typename Lvl##X::T, typename Lvl##X::T), \
        typename Lvl##X::T);

    EXPLICIT_LVL_EXTERN(CUTFHEPP_GenLUT);

    template <typename LvlX>
    std::vector<typename LvlX::T> GenDLUT(double (*f)(double m), typename LvlX::T p) {
        typename LvlX::T scale = static_cast<typename LvlX::T>(LvlX::Δ);

        // check overflow
        assert((algmath::count_bits(scale) + LvlX::plain_modulus_bit) > std::numeric_limits<typename LvlX::T>::digits);

        typename LvlX::T q = LvlX::n;
        typename LvlX::T x{0};

        size_t vecSize = static_cast<size_t>(LvlX::n);
        std::vector<typename LvlX::T> vec(vecSize, scale);
        for (size_t i = 0; i < vec.size(); ++i, x += p) {
            vec[i] = static_cast<typename LvlX::T>(std::round(vec[i] * f(static_cast<double>(x / q))));

            // std::cout << "LUT(" << i << ") = " << vec[i]
            //           << ". f: " << f(x / q, p)
            //           << ". x / n: " << x / (LvlX::n) << ". q / p: " << q / p
            //           << ". x: " << x << ". q: " << q << ". p: " << p << std::endl;
        }
        return vec;
    }

#define CUTFHEPP_GenDLUT(X)                                   \
    template std::vector<typename Lvl##X::T> GenDLUT<Lvl##X>( \
        double (*)(double),                                   \
        typename Lvl##X::T);

    EXPLICIT_LVL_EXTERN(CUTFHEPP_GenDLUT);

#define CUTFHEPP_GenLUT(X)                                              \
    template std::vector<typename Lvl##X::T> GenLUT<Lvl##X>(            \
        typename Lvl##X::T (*)(typename Lvl##X::T, typename Lvl##X::T), \
        typename Lvl##X::T);

    EXPLICIT_LVL_EXTERN(CUTFHEPP_GenLUT);

    template <typename LvlX>
    std::vector<typename LvlX::T> GenDLUTP(double (*f)(double m, typename LvlX::T p), typename LvlX::T p) {
        typename LvlX::T scale = static_cast<typename LvlX::T>(LvlX::Δ);

        // check overflow
        assert((algmath::count_bits(scale) + LvlX::plain_modulus_bit) > std::numeric_limits<typename LvlX::T>::digits);

        typename LvlX::T q = LvlX::n;
        typename LvlX::T x{0};

        size_t vecSize = static_cast<size_t>(LvlX::n);
        std::vector<typename LvlX::T> vec(vecSize, scale);
        for (size_t i = 0; i < vec.size(); ++i, x += p) {
            vec[i] = static_cast<typename LvlX::T>(std::round(vec[i] * f(static_cast<double>(x / q), p)));

            // std::cout << "LUT(" << i << ") = " << vec[i]
            //           << ". f: " << f(x / q, p)
            //           << ". x / n: " << x / (LvlX::n) << ". q / p: " << q / p
            //           << ". x: " << x << ". q: " << q << ". p: " << p << std::endl;
        }
        return vec;
    }

#define CUTFHEPP_GenDLUTP(X)                                   \
    template std::vector<typename Lvl##X::T> GenDLUTP<Lvl##X>( \
        double (*)(double, typename Lvl##X::T),                \
        typename Lvl##X::T);

    EXPLICIT_LVL_EXTERN(CUTFHEPP_GenDLUTP);
}  // namespace cuTFHEpp
