#include <random>

#include "bootstrapping/Bootstrapper.cuh"
#include "phantom.h"

using namespace std;
using namespace phantom;

void random_real(vector<complex<double>> &vec, size_t size) {
    // random_device rn;
    // mt19937_64 rnd(rn());
    std::mt19937_64 rnd(42);
    thread_local std::uniform_real_distribution<double> distribution(-1, 1);

    vec.reserve(size);

    for (size_t i = 0; i < size; i++) {
        vec[i].real(distribution(rnd));
        vec[i].imag(0);
    }
}

int main() {
    long boundary_K = 25;
    long deg = 59;
    long scale_factor = 2;
    long inverse_deg = 1;

    // The following parameters have been adjusted to satisfy the memory constraints of an A100 GPU
    // long logN = 16; // 16 -> 15  // full only even(14)
    // long loge = 10;

    // long logn = 15; // 14 -> 13
    // long sparse_slots = (1 << logn);

    /////////////// length /////////////////
    long logN = 16;  // 16, 14
    long loge = 10;

    long logn = 15;
    long sparse_slots = (1 << logn);  // 256

    /////////////// length /////////////////

    int logp = 46;
    int logq = 51;
    int log_special_prime = 51;

    int secret_key_hamming_weight = 192;
    // int secret_key_hamming_weight = 0;

    // 15 + 14 + 1 = 30 (整除 alpha=6)
    int remaining_level = 29;  // 15
    int boot_level = 0;        // >= subsum 1 + coefftoslot 2 + ModReduction 9 + slottocoeff 2
    int total_level = remaining_level + boot_level;
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

    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(phantom::arith::CoeffModulus::Create(poly_modulus_degree, coeff_bit_vec));
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

    Bootstrapper bootstrapper(
        loge,
        logn,
        logN - 1,
        total_level,
        scale,
        boundary_K,
        deg,
        scale_factor,
        inverse_deg,
        &ckks_evaluator);

    std::cout << "Generating Optimal Minimax Polynomials..." << endl;
    bootstrapper.prepare_mod_polynomial();

    std::cout << "Adding Bootstrapping Keys..." << endl;
    vector<int> gal_steps_vector;
    gal_steps_vector.push_back(0);  // 16个
    for (int i = 0; i < logN - 1; i++) {
        gal_steps_vector.push_back((1 << i));
        gal_steps_vector.push_back(-(1 << i));
    }
    bootstrapper.addLeftRotKeys_Linear_to_vector_3(gal_steps_vector);  // push back bsgs steps

    ckks_evaluator.decryptor.create_galois_keys_from_steps(gal_steps_vector, *(ckks_evaluator.galois_keys));
    // cudaDeviceSynchronize();
    // CHECK_CUDA_LAST_ERROR();
    std::cout << "Galois key size: " << ckks_evaluator.galois_keys->get_relin_keys_size() << std::endl;
    std::cout << "Galois key generated from steps vector." << endl;

    bootstrapper.slot_vec.push_back(logn);

    // std::cout << "Generating Linear Transformation Coefficients..." << endl;
    // bootstrapper.generate_LT_coefficient_3();
    // bootstrapper.generate_LT_coefficient_ori_3();

    vector<complex<double>> sparse(sparse_slots, complex<double>(0, 0));
    vector<complex<double>> input(slot_count, complex<double>(0, 0));

    random_real(sparse, sparse_slots);

    PhantomPlaintext plain;
    PhantomCiphertext cipher;

    // Create input cipher
    // std::random_device rdv;
    // std::uniform_real_distribution<double> uniform(-8., 8.);
    for (size_t i = 0; i < slot_count; i++) {
        input[i] = sparse[i % sparse_slots];
        // input[i] = 0.9;
    }

    // Decrypt input cipher to obtain the original input

    auto start = system_clock::now();

    // ckks_evaluator.evaluator.multiply_vector_inplace_reduced_error(cipher, input1);
    // ckks_evaluator.evaluator.rescale_to_next_inplace(cipher);
    // ckks_evaluator.print_decrypted_ct(cipher, 10);

    PhantomCiphertext rtn;
    // bootstrapper.bootstrap_3(rtn, cipher);

    const auto &s = phantom::util::global_variables::default_stream->get_stream();

    int totlen2 = (1 << logn) - 1;

    vector<vector<complex<double>>> matrix(8, input);

    std::cout << "mvm " << endl;

    {
        CUDATimer timer("mvm", s);
        timer.start();
        bootstrapper.rotated_bsgs_linear_transform(rtn, cipher, totlen2, 1, logn, matrix);
        timer.stop();
    }

    duration<double> sec = system_clock::now() - start;
    std::cout << "Bootstrapping took: " << sec.count() * 1000 << " ms" << endl;
    std::cout << "Return cipher level: " << rtn.coeff_modulus_size() << endl;
}

// int main() {
//     size_t log_max_batch = 16;
//     for (int i = log_max_batch; i >= 0; i = i - 2) {
//         std::cout << "CKKSBoot length: " << i << endl;
//         CKKSBoot(i);
//     }
// }