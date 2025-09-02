#pragma once
#include <stdint.h>

namespace cuTFHEpp {
    template <typename P>
    __device__ inline void SampleExtractIndex(TFHEpp::TLWE<P> &tlwe,
                                              const TFHEpp::TRLWE<P> &trlwe,
                                              const int index) {
        const unsigned int tid = blockDim.x * threadIdx.y + threadIdx.x;
        const unsigned int bdim = blockDim.x * blockDim.y;

        for (int i = tid; i < P::n; i += bdim) {
#pragma unroll
            for (int k = 0; k < P::k; k++)
                tlwe[k * P::n + i] =
                    (i <= index) ? trlwe[k][index - i] : -trlwe[k][P::n + index - i];
        }

        if (tid == 0)
            tlwe[P::k * P::n] = trlwe[P::k][index];
    }

    //     template <typename lvlx, typename lvlR>
    //     __global__ void SampleExtractIndex(TFHEpp::TLWE<lvlx> &tlwe,
    //                                        const uint64_t *trlwe,
    //                                        const int *indices,
    //                                        const int indices_size,
    //                                        const double rescale) {
    //         const unsigned int tid = blockDim.x * threadIdx.y + threadIdx.x;
    //         const unsigned int bdim = blockDim.x * blockDim.y;

    //         size_t degree = lvlR::n;
    //         auto *trlwe_ptr0 = trlwe;
    //         auto *trlwe_ptr1 = trlwe + degree;

    //         for (int i = tid; i < lvlR::n; i += bdim) {
    //             for (int iter = 0; iter < indices_size; iter++) {
    //                 int index = indices[iter];
    // #pragma unroll
    //                 for (int k = 0; k < lvlR::k; k++) {
    //                     double trlwe0 = static_cast<int64_t>(trlwe_ptr1[k * degree + index - i]) * rescale;  // scale down
    //                     double trlwe1 = static_cast<int64_t>(trlwe_ptr1[k * degree + lvlR::n + index - i]) * rescale;

    //                     tlwe[iter][k * lvlR::n + i] =
    //                         (i <= index) ? static_cast<uint32_t>(trlwe0) : -static_cast<uint32_t>(trlwe1);
    //                 }
    //             }
    //         }

    //         if (tid == 0) {
    //             for (int iter = 0; iter < indices_size; iter++) {
    //                 int index = indices[iter];
    //                 double trlwe0 = static_cast<int64_t>(trlwe_ptr0[iter * degree + index]) * rescale;  // scale down
    //                 tlwe[iter][lvlR::k * lvlR::n] = static_cast<uint32_t>(trlwe0);
    //             }
    //         }
    //     }

}  // namespace cuTFHEpp
