#include "phantom.h"
#include "openfhe.h"

#include "compact_ntt.cuh"

using namespace phantom::util;

int test_4step_ntt(size_t n, BasicInteger q, size_t threadsPerBlock, size_t batch_size) {
    cuda_stream_wrapper stream_wrapper;
    const auto &stream = stream_wrapper.get_stream();
    FourStepNTT ntt(n, q, stream);

    std::vector<BasicInteger> h_input(n);
    std::vector<BasicInteger> h_output(batch_size * n);
    auto d_input = make_cuda_auto_ptr<BasicInteger>(batch_size * n, stream);
    auto d_output = make_cuda_auto_ptr<BasicInteger>(batch_size * n, stream);

    for (int i = 0; i < n; i++) {
        h_input[i] = BasicInteger(i);
    }

    for (size_t i = 0; i < batch_size; i++)
        cudaMemcpyAsync(d_input.get() + i * n, h_input.data(), n * sizeof(BasicInteger), cudaMemcpyHostToDevice,
                        stream);

    ntt.forward(d_output.get(), d_input.get(), threadsPerBlock, batch_size, stream);
//    cudaMemcpyAsync(h_output.data(), d_output.get(), n * sizeof(BasicInteger), cudaMemcpyDeviceToHost,
//                    stream);
//    cudaStreamSynchronize(stream);
//
//    std::cout << "[";
//    for (int i = 0; i < n; i++) {
//        std::cout << h_output[i] << ", ";
//    }
//    std::cout << h_output[n - 1] << "]" << std::endl;

    ntt.inverse(d_output.get(), d_output.get(), threadsPerBlock, batch_size, stream);
    cudaMemcpyAsync(h_output.data(), d_output.get(), batch_size * n * sizeof(BasicInteger), cudaMemcpyDeviceToHost,
                    stream);
    cudaStreamSynchronize(stream);

    for (size_t poly_idx = 0; poly_idx < batch_size; poly_idx++) {
        for (int i = 0; i < n; i++) {
            if (h_output[poly_idx * n + i] != h_input[i])
                throw std::invalid_argument("error in 4step NTT");
        }
    }

    std::cout << n << " 4step NTT correct." << std::endl;

    return 0;
}

int test_composite_4step_ntt(size_t n, BasicInteger q1, BasicInteger q2, size_t threadsPerBlock, size_t batch_size) {
    cuda_stream_wrapper stream_wrapper;
    const auto &stream = stream_wrapper.get_stream();
    FourStepNTT ntt(n, q1, q2, stream);

    std::vector<BasicInteger> h_input(n);
    std::vector<BasicInteger> h_output(batch_size * n);
    auto d_input = make_cuda_auto_ptr<BasicInteger>(batch_size * n, stream);
    auto d_output = make_cuda_auto_ptr<BasicInteger>(batch_size * n, stream);

    for (int i = 0; i < n; i++) {
        h_input[i] = BasicInteger(i);
    }

    for (size_t i = 0; i < batch_size; i++)
        cudaMemcpyAsync(d_input.get() + i * n, h_input.data(), n * sizeof(BasicInteger), cudaMemcpyHostToDevice,
                        stream);

    ntt.forward(d_output.get(), d_input.get(), threadsPerBlock, batch_size, stream);
//    cudaMemcpyAsync(h_output.data(), d_output.get(), n * sizeof(BasicInteger), cudaMemcpyDeviceToHost, stream);
//    cudaStreamSynchronize(stream);
//
//    std::cout << "[";
//    for (int i = 0; i < n; i++) {
//        std::cout << h_output[i] << ", ";
//    }
//    std::cout << h_output[n - 1] << "]" << std::endl;

    ntt.inverse(d_output.get(), d_output.get(), threadsPerBlock, batch_size, stream);
    cudaMemcpyAsync(h_output.data(), d_output.get(), batch_size * n * sizeof(BasicInteger), cudaMemcpyDeviceToHost,
                    stream);
    cudaStreamSynchronize(stream);

    for (size_t poly_idx = 0; poly_idx < batch_size; poly_idx++) {
        for (int i = 0; i < n; i++) {
            if (h_output[poly_idx * n + i] != h_input[i])
                throw std::invalid_argument("error in composite 4step NTT");
        }
    }

    std::cout << n << " composite 4step NTT correct." << std::endl;

    return 0;
}

int main() {
    test_4step_ntt(1024, BasicInteger(12289), 128, 1);
    test_4step_ntt(1024, BasicInteger(12289), 256, 2);
    test_4step_ntt(1024, BasicInteger(12289), 512, 4);
    test_4step_ntt(1024, BasicInteger(12289), 1024, 8);

    test_4step_ntt(2048, BasicInteger(12289), 128, 1);
    test_4step_ntt(2048, BasicInteger(12289), 256, 2);
    test_4step_ntt(2048, BasicInteger(12289), 512, 4);
    test_4step_ntt(2048, BasicInteger(12289), 1024, 8);

    test_4step_ntt(4096, BasicInteger(40961), 128, 1);
    test_4step_ntt(4096, BasicInteger(40961), 256, 2);
    test_4step_ntt(4096, BasicInteger(40961), 512, 4);
    test_4step_ntt(4096, BasicInteger(40961), 1024, 8);

    test_composite_4step_ntt(1024, BasicInteger(12289), BasicInteger(18433), 128, 1);
    test_composite_4step_ntt(1024, BasicInteger(12289), BasicInteger(18433), 256, 2);
    test_composite_4step_ntt(1024, BasicInteger(12289), BasicInteger(18433), 512, 4);
    test_composite_4step_ntt(1024, BasicInteger(12289), BasicInteger(18433), 1024, 8);
    return 0;
}
