#include <nvbench/nvbench.cuh>

#include "compact_ntt.cuh"

using namespace phantom::util;

void ntt_forward_bench(nvbench::state &state) {
    const size_t n = state.get_int64("Dim");
    const size_t threads_per_block = state.get_int64("TPB");
    const size_t batch_size = state.get_int64("Batch");
    BasicInteger q;
    if (n == 1024 || n == 2048) {
        q = BasicInteger(12289);
    } else if (n == 4096) {
        q = BasicInteger(40961);
    } else {
        throw std::invalid_argument("Invalid dimension");
    }

    state.collect_cupti_metrics();

    // Provide throughput information:
    state.add_element_count(batch_size * n, "NumElements");
    state.add_global_memory_reads<nvbench::int64_t>(batch_size * n, "ReadDataSize");
    state.add_global_memory_writes<nvbench::int64_t>(batch_size * n, "WriteDataSize");

    cuda_stream_wrapper stream_wrapper;
    const auto &stream = stream_wrapper.get_stream();
    FourStepNTT ntt(n, q, stream);

    std::vector<BasicInteger> h_input(n);
    std::vector<BasicInteger> h_output(n);
    auto d_input = make_cuda_auto_ptr<BasicInteger>(batch_size * n, stream);
    auto d_output = make_cuda_auto_ptr<BasicInteger>(batch_size * n, stream);

    for (int i = 0; i < n; i++)
        h_input[i] = BasicInteger(i);

    for (size_t i = 0; i < batch_size; i++)
        cudaMemcpyAsync(d_input.get() + i * n, h_input.data(), n * sizeof(BasicInteger), cudaMemcpyHostToDevice,
                        stream);

    state.set_cuda_stream(nvbench::make_cuda_stream_view(stream));
    state.exec([&ntt, &d_output, &d_input, &threads_per_block, &batch_size, &stream](nvbench::launch &launch) {
                   ntt.forward(d_output.get(), d_input.get(), threads_per_block, batch_size, stream);
               }
    );
}

NVBENCH_BENCH(ntt_forward_bench)
        .add_int64_axis("TPB", {64, 128, 256, 512, 1024})
        .add_int64_axis("Batch", {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768})
        .add_int64_axis("Dim", {1024, 2048, 4096})
        .set_timeout(1); // Limit to one second per measurement.

void ntt_inverse_bench(nvbench::state &state) {
    const size_t n = state.get_int64("Dim");
    const size_t threads_per_block = state.get_int64("TPB");
    const size_t batch_size = state.get_int64("Batch");
    BasicInteger q;
    if (n == 1024 || n == 2048) {
        q = BasicInteger(12289);
    } else if (n == 4096) {
        q = BasicInteger(40961);
    } else {
        throw std::invalid_argument("Invalid dimension");
    }

    state.collect_cupti_metrics();

    // Provide throughput information:
    state.add_element_count(batch_size * n, "NumElements");
    state.add_global_memory_reads<nvbench::int64_t>(batch_size * n, "ReadDataSize");
    state.add_global_memory_writes<nvbench::int64_t>(batch_size * n, "WriteDataSize");

    cuda_stream_wrapper stream_wrapper;
    const auto &stream = stream_wrapper.get_stream();
    FourStepNTT ntt(n, q, stream);

    std::vector<BasicInteger> h_input(n);
    std::vector<BasicInteger> h_output(n);
    auto d_input = make_cuda_auto_ptr<BasicInteger>(batch_size * n, stream);
    auto d_output = make_cuda_auto_ptr<BasicInteger>(batch_size * n, stream);

    for (int i = 0; i < n; i++)
        h_input[i] = BasicInteger(i);

    for (size_t i = 0; i < batch_size; i++)
        cudaMemcpyAsync(d_input.get() + i * n, h_input.data(), n * sizeof(BasicInteger), cudaMemcpyHostToDevice,
                        stream);

    ntt.forward(d_output.get(), d_input.get(), threads_per_block, batch_size, stream);

    state.set_cuda_stream(nvbench::make_cuda_stream_view(stream));
    state.exec([&ntt, &d_output, &d_input, &threads_per_block, &batch_size, &stream](nvbench::launch &launch) {
                   ntt.inverse(d_output.get(), d_output.get(), threads_per_block, batch_size, stream);
               }
    );
}

NVBENCH_BENCH(ntt_inverse_bench)
        .add_int64_axis("TPB", {64, 128, 256, 512, 1024})
        .add_int64_axis("Batch", {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768})
        .add_int64_axis("Dim", {1024, 2048, 4096})
        .set_timeout(1); // Limit to one second per measurement.
