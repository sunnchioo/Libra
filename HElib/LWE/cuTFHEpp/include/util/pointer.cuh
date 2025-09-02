#pragma once

#include <stdexcept>

namespace cuTFHEpp::util {
    template <typename T>
    struct Pointer {
        T *h_data_ = nullptr;
        T *d_data_ = nullptr;
        cudaStream_t stream_ = nullptr;

        Pointer() = default;

        template <typename... Args>
        explicit Pointer(const cudaStream_t &stream, Args &&...args)
            : stream_(stream) {
            initialize(std::forward<Args>(args)...);
        }

        template <typename... Args>
        explicit Pointer(Args &&...args) {
            initialize(std::forward<Args>(args)...);
        }

        ~Pointer() {
            free_resources();
        }

        void free_resources() {
            if (d_data_ != nullptr) {
                if (stream_ != nullptr) {
                    cudaStreamSynchronize(stream_);
                }
                CUDA_CHECK_RETURN(cudaFreeAsync(d_data_, stream_));
                d_data_ = nullptr;
            }
            if (h_data_ != nullptr) {
                delete h_data_;
                h_data_ = nullptr;
            }
            stream_ = nullptr;
        }

        Pointer(const Pointer &) = delete;
        Pointer &operator=(const Pointer &) = delete;

        Pointer(Pointer &&other) noexcept
            : h_data_(other.h_data_), d_data_(other.d_data_) {
            other.h_data_ = nullptr;
            other.d_data_ = nullptr;
        }

        Pointer &operator=(Pointer &&other) noexcept {
            if (this != &other) {
                if (d_data_ != nullptr)
                    cudaFree(d_data_);
                delete h_data_;

                h_data_ = other.h_data_;
                d_data_ = other.d_data_;
                other.h_data_ = nullptr;
                other.d_data_ = nullptr;
            }
            return *this;
        }

        template <typename... Args>
        void initialize(Args &&...args) {
            if (h_data_ != nullptr || d_data_ != nullptr) {
                throw std::runtime_error("Pointer already initialized");
            }

            h_data_ = new T(std::forward<Args>(args)...);
            CUDA_CHECK_RETURN(cudaMallocAsync(&d_data_, sizeof(T), stream_));
            CUDA_CHECK_RETURN(cudaMemcpyAsync(d_data_, h_data_, sizeof(T), cudaMemcpyHostToDevice, stream_));
        }

        template <typename U>
        Pointer<U> &safe_cast() {
            static_assert(T::template can_cast<U>());
            return reinterpret_cast<Pointer<U> &>(*this);
        }

        T *operator->() { return h_data_; }
        const T *operator->() const { return h_data_; }
        T *get_host() const { return h_data_; }

        T *get_ptr() const { return d_data_; }
        T &get() const { return *d_data_; }
    };

    // template <typename T>
    // struct Pointer {
    //     T *h_data_ = nullptr;
    //     T *d_data_ = nullptr;

    //     Pointer() = default;

    //     template <typename... Args>
    //     explicit Pointer(Args &&...args) {
    //         initialize(std::forward<Args>(args)...);
    //     }

    //     ~Pointer() {
    //         if (d_data_ != nullptr)
    //             CUDA_CHECK_RETURN(cudaFree(d_data_));
    //         delete h_data_;
    //     }

    //     Pointer(const Pointer &) = delete;
    //     Pointer &operator=(const Pointer &) = delete;

    //     Pointer(Pointer &&other) noexcept
    //         : h_data_(other.h_data_), d_data_(other.d_data_) {
    //         other.h_data_ = nullptr;
    //         other.d_data_ = nullptr;
    //     }

    //     Pointer &operator=(Pointer &&other) noexcept {
    //         if (this != &other) {
    //             if (d_data_ != nullptr)
    //                 cudaFree(d_data_);
    //             delete h_data_;

    //             h_data_ = other.h_data_;
    //             d_data_ = other.d_data_;
    //             other.h_data_ = nullptr;
    //             other.d_data_ = nullptr;
    //         }
    //         return *this;
    //     }

    //     template <typename... Args>
    //     void initialize(Args &&...args) {
    //         if (h_data_ != nullptr || d_data_ != nullptr) {
    //             throw std::runtime_error("Pointer already initialized");
    //         }

    //         h_data_ = new T(std::forward<Args>(args)...);
    //         CUDA_CHECK_RETURN(cudaMalloc(&d_data_, sizeof(T)));
    //         CUDA_CHECK_RETURN(cudaMemcpy(d_data_, h_data_, sizeof(T), cudaMemcpyHostToDevice));
    //     }

    //     template <typename U>
    //     Pointer<U> &safe_cast() {
    //         static_assert(T::template can_cast<U>());
    //         return reinterpret_cast<Pointer<U> &>(*this);
    //     }

    //     T *operator->() { return h_data_; }
    //     const T *operator->() const { return h_data_; }
    //     T *get_host() const { return h_data_; }

    //     T *get_ptr() const { return d_data_; }
    //     T &get() const { return *d_data_; }
    // };

}  // namespace cuTFHEpp::util

// #pragma once

// namespace cuTFHEpp::util {

//     template <typename T>
//     struct Pointer {
//         T *h_data_ = nullptr;
//         T *d_data_ = nullptr;

//         template <typename... Args>
//         Pointer(Args &&...args) {
//             h_data_ = new T(std::forward<Args>(args)...);
//             CUDA_CHECK_RETURN(cudaMalloc(&d_data_, sizeof(T)));
//             CUDA_CHECK_RETURN(cudaMemcpy(d_data_, h_data_, sizeof(T), cudaMemcpyHostToDevice));
//         }

//         ~Pointer() {
//             if (d_data_ != nullptr)
//                 CUDA_CHECK_RETURN(cudaFree(d_data_));
//             d_data_ = nullptr;
//             if (h_data_ != nullptr)
//                 delete h_data_;
//             h_data_ = nullptr;
//         }

//         Pointer(const Pointer &) = delete;
//         Pointer &operator=(const Pointer &) = delete;
//         Pointer &operator=(Pointer &&) = delete;

//         Pointer(Pointer &&other) {
//             h_data_ = other.h_data_;
//             other.h_data_ = nullptr;
//             d_data_ = other.d_data_;
//             other.d_data_ = nullptr;
//         }

//         template <typename U>
//         Pointer<U> &safe_cast() {
//             static_assert(T::template can_cast<U>());
//             return reinterpret_cast<Pointer<U> &>(*this);
//         }

//         T *operator->() {
//             return h_data_;
//         }

//         T &get() const {
//             return *d_data_;
//         }
//     };
// } // namespace cuTFHEpp::util
