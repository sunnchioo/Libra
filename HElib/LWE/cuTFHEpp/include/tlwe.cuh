#pragma once

namespace cuTFHEpp {
    template <typename X, typename Y>
    constexpr __host__ __device__ inline bool isLvlCoverT() {
        if constexpr (std::is_same_v<X, Lvl0>)
            if constexpr (std::is_same_v<Y, Lvl0>)
                return true;
        if constexpr (std::is_same_v<X, Lvl1>)
            if constexpr (std::is_same_v<Y, Lvl0> || std::is_same_v<Y, Lvl1>)
                return true;
        if constexpr (std::is_same_v<X, Lvl2>)
            if constexpr (std::is_same_v<Y, Lvl0> || std::is_same_v<Y, Lvl1> || std::is_same_v<Y, Lvl2>)
                return true;
        if constexpr (std::is_same_v<X, Lvl01>)
            if constexpr (std::is_same_v<Y, Lvl01>)
                return true;
        if constexpr (std::is_same_v<X, Lvl02>)
            if constexpr (std::is_same_v<Y, Lvl02> || std::is_same_v<Y, Lvl01>)
                return true;

        if constexpr (std::is_same_v<X, Lvl1L>)
            if constexpr (std::is_same_v<Y, Lvl1L>)
                return true;
        return false;
    };

    // 之后添加
    template <typename P>
    struct cuTRLWE {
        using Lvl = P;

        TFHEpp::TRLWE<P> *data;

        cuTRLWE() : cuTRLWE(1) {}

        cuTRLWE(size_t batch_size) {
            CUDA_CHECK_RETURN(cudaMalloc(&data, batch_size * sizeof(TFHEpp::TRLWE<P>)));
        }

        ~cuTRLWE() {
            CUDA_CHECK_RETURN(cudaFree(data));
        }

        cuTRLWE(const cuTRLWE &) = delete;
        cuTRLWE &operator=(const cuTRLWE &) = delete;
        cuTRLWE &operator=(cuTRLWE &&) = delete;
        cuTRLWE(cuTRLWE &&) = delete;

        template <typename Lvl>
        __host__ __device__ inline TFHEpp::TRLWE<Lvl> *get() const {
            static_assert(isLvlCoverT<P, Lvl>());
            return reinterpret_cast<TFHEpp::TRLWE<Lvl> *>(data);
        }

        template <typename T, typename U = T::Lvl>
        static constexpr inline bool can_cast() {
            return isLvlCoverT<P, U>();
        }
    };

    // template <typename P>
    // struct cuTLWE {
    //     using Lvl = P;

    //     TFHEpp::TLWE<P> *data;

    //     cuTLWE() : cuTLWE(1) {}

    //     cuTLWE(size_t batch_size) {
    //         CUDA_CHECK_RETURN(cudaMalloc(&data, batch_size * sizeof(TFHEpp::TLWE<P>)));
    //     }

    //     ~cuTLWE() {
    //         CUDA_CHECK_RETURN(cudaFree(data));
    //     }

    //     cuTLWE(const cuTLWE &) = delete;
    //     cuTLWE &operator=(const cuTLWE &) = delete;
    //     cuTLWE &operator=(cuTLWE &&) = delete;
    //     cuTLWE(cuTLWE &&) = delete;

    //     template <typename Lvl>
    //     __host__ __device__ inline TFHEpp::TLWE<Lvl> *get() const {
    //         static_assert(isLvlCoverT<P, Lvl>());
    //         return reinterpret_cast<TFHEpp::TLWE<Lvl> *>(data);
    //     }

    //     template <typename T, typename U = T::Lvl>
    //     static constexpr inline bool can_cast() {
    //         return isLvlCoverT<P, U>();
    //     }
    // };

    //     template <typename P>
    //     struct cuTLWE {
    //         using Lvl = P;

    //         TFHEpp::TLWE<P> *data;
    //         size_t size = 0;

    //         cuTLWE() : cuTLWE(1) {}
    //         explicit cuTLWE(size_t batch_size) : size(batch_size) {
    //             CUDA_CHECK_RETURN(cudaMalloc(&data, batch_size * sizeof(TFHEpp::TLWE<P>)));
    //             cudaDeviceSynchronize();
    //         }

    //         ~cuTLWE() {
    //             if (data) {
    //                 CUDA_CHECK_RETURN(cudaFree(data));
    //             }
    //         }

    //         cuTLWE(const cuTLWE &) = delete;
    //         cuTLWE &operator=(const cuTLWE &) = delete;

    //         cuTLWE(cuTLWE &&other) noexcept
    //             : data(other.data), size(other.size) {
    //             other.data = nullptr;
    //             other.size = 0;
    //         }

    //         cuTLWE &operator=(cuTLWE &&other) noexcept {
    //             if (this != &other) {
    //                 if (data) {
    //                     CUDA_CHECK_RETURN(cudaFree(data));
    //                 }
    //                 data = other.data;
    //                 size = other.size;
    //                 other.data = nullptr;
    //                 other.size = 0;
    //             }
    //             return *this;
    //         }

    //         __host__ void resize(size_t new_size) {
    //             if (new_size == size)
    //                 return;

    //             TFHEpp::TLWE<P> *new_data = nullptr;
    //             if (new_size > 0) {
    //                 CUDA_CHECK_RETURN(cudaMalloc(&new_data, new_size * sizeof(TFHEpp::TLWE<P>)));

    //                 size_t mov_size = new_size > size ? size : new_size;
    //                 CUDA_CHECK_RETURN(cudaMemcpy(new_data, data, mov_size * sizeof(TFHEpp::TLWE<P>), cudaMemcpyDeviceToDevice));
    //             }

    //             if (data) {
    //                 CUDA_CHECK_RETURN(cudaFree(data));
    //             }

    //             data = new_data;
    //             size = new_size;

    //             cudaDeviceSynchronize();
    //             CUDA_CHECK_ERROR();
    //         }

    //         __host__ __device__ size_t get_size() const { return size; }

    //         __host__ __device__ TFHEpp::TLWE<P> &get_data(size_t idx) {
    // #ifndef __CUDA_ARCH__
    //             if (idx >= size) {
    //                 throw std::out_of_range("cuTLWE index out of range");
    //             }
    // #else
    //             assert(idx < size);
    // #endif
    //             return data[idx];
    //         }

    //         __host__ __device__ TFHEpp::TLWE<P> *data_ptr() const { return data; }

    //         template <typename Lvl>
    //         __host__ __device__ inline TFHEpp::TLWE<Lvl> *get() const {
    //             static_assert(isLvlCoverT<P, Lvl>());
    //             return reinterpret_cast<TFHEpp::TLWE<Lvl> *>(data);
    //         }

    //         template <typename T, typename U = T::Lvl>
    //         static constexpr inline bool can_cast() {
    //             return isLvlCoverT<P, U>();
    //         }
    //     };

    template <typename P>
    struct cuTLWE {
        using Lvl = P;

        TFHEpp::TLWE<P> *data;
        size_t size = 0;
        cudaStream_t stream_ = nullptr;

        cuTLWE() : cuTLWE(1) {}

        explicit cuTLWE(size_t batch_size) : size(batch_size) {
            CUDA_CHECK_RETURN(cudaMalloc(&data, batch_size * sizeof(TFHEpp::TLWE<P>)));
            cudaDeviceSynchronize();
        }

        explicit cuTLWE(size_t batch_size, const cudaStream_t &stream)
            : size(batch_size), stream_(stream) {
            CUDA_CHECK_RETURN(cudaMallocAsync(&data, batch_size * sizeof(TFHEpp::TLWE<P>), stream));
            cudaDeviceSynchronize();
        }

        ~cuTLWE() {
            cudaStreamSynchronize(stream_);

            if (data) {
                CUDA_CHECK_RETURN(cudaFreeAsync(data, stream_));
                data = nullptr;
            }

            // if (stream_) {
            //     cudaStreamDestroy(stream_);
            //     stream_ = 0;
            // }
            stream_ = nullptr;
            size = 0;

            cudaDeviceSynchronize();
        }

        cuTLWE(const cuTLWE &) = delete;
        cuTLWE &operator=(const cuTLWE &) = delete;

        cuTLWE(cuTLWE &&other) noexcept
            : data(other.data), size(other.size), stream_(other.stream_) {
            other.data = nullptr;
            other.size = 0;
            other.stream_ = 0;
        }

        cuTLWE &operator=(cuTLWE &&other) noexcept {
            if (this != &other) {
                this->~cuTLWE();

                data = other.data;
                size = other.size;
                stream_ = other.stream_;

                other.data = nullptr;
                other.size = 0;
                other.stream_ = 0;
            }
            return *this;
        }

        __host__ void set_stream(cudaStream_t new_stream) {
            cudaStreamSynchronize(stream_);
            if (stream_ && stream_ != new_stream) {
                cudaStreamDestroy(stream_);
            }

            stream_ = new_stream;
        }
        __host__ __device__ cudaStream_t get_stream() const {
            return stream_;
        }
        __host__ void sync_stream() {
            cudaStreamSynchronize(stream_);
        }

        __host__ void resize(size_t new_size) {
            if (new_size == size)
                return;
            sync_stream();
            TFHEpp::TLWE<P> *new_data = nullptr;
            if (new_size > 0) {
                CUDA_CHECK_RETURN(cudaMallocAsync(&new_data, new_size * sizeof(TFHEpp::TLWE<P>), stream_));
                size_t mov_size = std::min(size, new_size);

                if (mov_size > 0) {
                    CUDA_CHECK_RETURN(cudaMemcpyAsync(
                        new_data, data,
                        mov_size * sizeof(TFHEpp::TLWE<P>),
                        cudaMemcpyDeviceToDevice,
                        stream_));
                }
            }
            if (data) {
                CUDA_CHECK_RETURN(cudaFreeAsync(data, stream_));
            }
            data = new_data;
            size = new_size;

            sync_stream();
        }

        __host__ __device__ size_t get_size() const { return size; }

        __host__ __device__ TFHEpp::TLWE<P> &get_data(size_t idx) {
#ifndef __CUDA_ARCH__
            if (idx >= size) {
                throw std::out_of_range("cuTLWE index out of range");
            }
#else
            assert(idx < size);
#endif
            return data[idx];
        }

        __host__ __device__ TFHEpp::TLWE<P> *data_ptr() const { return data; }

        template <typename Lvl>
        __host__ __device__ inline TFHEpp::TLWE<Lvl> *get() const {
            static_assert(isLvlCoverT<P, Lvl>());
            return reinterpret_cast<TFHEpp::TLWE<Lvl> *>(data);
        }

        template <typename T, typename U = T::Lvl>
        static constexpr inline bool can_cast() {
            return isLvlCoverT<P, U>();
        }
    };

}  // namespace cuTFHEpp
