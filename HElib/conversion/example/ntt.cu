#include <random>

#include "bootstrapping/Bootstrapper.cuh"
#include "phantom.h"

using namespace std;
using namespace phantom;
using namespace phantom::arith;

void random_real(vector<uint64_t> &vec, size_t size) {
    // random_device rn;
    // mt19937_64 rnd(rn());
    std::mt19937_64 rnd(42);
    thread_local std::uniform_int_distribution<uint64_t> distribution(0, 1024);

    vec.reserve(size);

    for (size_t i = 0; i < size; i++) {
        vec[i] = distribution(rnd);
    }
}

int main() {
    /////////////// length /////////////////
    long logN = 10;  // 16, 14
    // long loge = 10;

    long logn = logN - 1;
    long sparse_slots = (1 << logn);  // 256

    /////////////// length /////////////////

    int logp = 46;
    int logq = 51;
    int log_special_prime = 51;

    int secret_key_hamming_weight = 64;
    // int secret_key_hamming_weight = 0;

    // 15 + 14 + 1 = 30 (整除 alpha=6)
    int remaining_level = 3;  // 15
    int boot_level = 0;       // >= subsum 1 + coefftoslot 2 + ModReduction 9 + slottocoeff 2
    // int total_level = remaining_level + boot_level;
    int special_prime_len = 2;  // 整除才可

    vector<int> coeff_bit_vec;
    coeff_bit_vec.push_back(logq);
    for (int i = 0; i < remaining_level; i++) {
        coeff_bit_vec.push_back(logp);
    }
    for (int i = 0; i < boot_level; i++) {
        coeff_bit_vec.push_back(logq);
    }
    for (int i = 0; i < special_prime_len; i++) {
        coeff_bit_vec.push_back(log_special_prime);
    }

    std::cout << "Setting Parameters..." << endl;
    phantom::EncryptionParameters parms(scheme_type::ckks);
    size_t poly_modulus_degree = (size_t)(1 << logN);
    double scale = pow(2.0, 46);

    const auto &coeffmodulus = phantom::arith::CoeffModulus::Create(poly_modulus_degree, coeff_bit_vec);
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(coeffmodulus);
    parms.set_secret_key_hamming_weight(secret_key_hamming_weight);
    parms.set_sparse_slots(sparse_slots);
    parms.set_special_modulus_size(special_prime_len);

    PhantomContext context(parms);

    PhantomSecretKey secret_key(context);
    PhantomPublicKey public_key = secret_key.gen_publickey(context);
    PhantomRelinKey relin_keys = secret_key.gen_relinkey(context);
    PhantomGaloisKey galois_keys;

    PhantomCKKSEncoder encoder(context);

    CKKSEvaluator ckks_evaluator(&context, &public_key, &secret_key, &encoder, &relin_keys, &galois_keys, scale);

    size_t slot_count = encoder.slot_count();

    vector<uint64_t> input(poly_modulus_degree, 0);
    random_real(input, poly_modulus_degree);

    auto start = system_clock::now();

    const auto &s = phantom::util::global_variables::default_stream->get_stream();

    uint64_t *d_ci;
    cudaMallocAsync(&d_ci, poly_modulus_degree * sizeof(uint64_t), s);
    cudaMemcpyAsync(d_ci, input.data(), poly_modulus_degree * sizeof(uint64_t), cudaMemcpyHostToDevice, s);

    size_t print_size = 1024;
    std::cout << "non ntt: " << std::endl;
    for (size_t i = 0; i < print_size; i++) {
        std::cout << input[i] << " ";
    }
    std::cout << std::endl;

    // nwt_2d_radix8_forward_inplace(d_ci, context.gpu_rns_tables(), 1, 0, s);
    auto &ntt_tables = context.gpu_rns_tables();
    fnwt_1d_opt(d_ci, ntt_tables.twiddle(), ntt_tables.twiddle_shoup(), ntt_tables.modulus(), poly_modulus_degree, 1, 0, s);
    cudaDeviceSynchronize();

    uint64_t *h_ci = new uint64_t[poly_modulus_degree];
    cudaMemcpyAsync(h_ci, d_ci, poly_modulus_degree * sizeof(uint64_t), cudaMemcpyDeviceToHost, s);
    std::cout << "ntt: " << std::endl;
    for (size_t i = 0; i < print_size; i++) {
        std::cout << h_ci[i] << " ";
    }
    std::cout << std::endl;

    cudaDeviceSynchronize();
    // nwt_2d_radix8_backward_inplace(d_ci, context.gpu_rns_tables(), 1, 0, s);
    std::vector<uint64_t> degree_inv = {0};
    try_invert_uint_mod(poly_modulus_degree, coeffmodulus[0], degree_inv[0]);
    std::vector<uint64_t> degree_inv_shoup = {0};
    degree_inv_shoup[0] = compute_shoup(degree_inv[0], coeffmodulus[0].value());

    uint64_t *d_degree_inv;
    cudaMallocAsync(&d_degree_inv, sizeof(uint64_t), s);
    cudaMemcpyAsync(d_degree_inv, degree_inv.data(), sizeof(uint64_t), cudaMemcpyHostToDevice, s);
    uint64_t *d_degree_inv_shoup;
    cudaMallocAsync(&d_degree_inv_shoup, sizeof(uint64_t), s);
    cudaMemcpyAsync(d_degree_inv_shoup, degree_inv_shoup.data(), sizeof(uint64_t), cudaMemcpyHostToDevice, s);
    // inwt_1d_opt(d_ci, ntt_tables.itwiddle(), ntt_tables.itwiddle_shoup(), ntt_tables.modulus(), d_degree_inv, d_degree_inv_shoup, poly_modulus_degree, 1, 0, s);
    inwt_1d_opt(d_ci, ntt_tables.itwiddle(), ntt_tables.itwiddle_shoup(), ntt_tables.modulus(), ntt_tables.n_inv_mod_q(), ntt_tables.n_inv_mod_q_shoup(), poly_modulus_degree, 1, 0, s);
    cudaDeviceSynchronize();

    cudaMemcpyAsync(h_ci, d_ci, poly_modulus_degree * sizeof(uint64_t), cudaMemcpyDeviceToHost, s);
    std::cout << "non ntt: " << std::endl;
    for (size_t i = 0; i < print_size; i++) {
        std::cout << h_ci[i] << " ";
    }
    std::cout << std::endl;

    duration<double> sec = system_clock::now() - start;
}

// int main() {
//     size_t log_max_batch = 16;
//     for (int i = log_max_batch; i >= 0; i = i - 2) {
//         std::cout << "CKKSBoot length: " << i << endl;
//         CKKSBoot(i);
//     }
// }