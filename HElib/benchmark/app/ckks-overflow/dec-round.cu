#include "../utils/utils.h"
#include "bootstrapping/Bootstrapper.cuh"
#include "ckks_evaluator.cuh"
#include "fileio.h"
#include "phantom.h"

#include <algorithm>
#include <random>
#include <vector>

using CUDATimer = phantom::util::CUDATimer;

int main() {
    // pre compute
    long boundary_K = 25;
    long deg = 59;
    long scale_factor = 2;
    long inverse_deg = 1;

    /////////////// length /////////////////
    long logN = 16; // 16, 14
    long loge = 10;

    long logn = 15;
    long sparse_slots = (1 << logn); // 256

    int logp = 57;
    int logq = 60;
    int log_special_prime = 60;

    int secret_key_hamming_weight = 192;

    int remaining_level = 3;
    int boot_level = 14;                            // >= subsum 1 + coefftoslot 2 + ModReduction 9 + slottocoeff 2
    int total_level = remaining_level + boot_level; // 38
    int special_prime_len = 2;

    vector<int> coeff_bit_vec;
    coeff_bit_vec.push_back(logq); // 39
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
    double scale = pow(2.0, logp);

    parms.set_poly_modulus_degree(poly_modulus_degree);
    std::vector<phantom::arith::Modulus> coeff_modulus = phantom::arith::CoeffModulus::Create(poly_modulus_degree, coeff_bit_vec);
    parms.set_coeff_modulus(coeff_modulus);
    parms.set_secret_key_hamming_weight(secret_key_hamming_weight);
    parms.set_sparse_slots(sparse_slots);
    parms.set_special_modulus_size(special_prime_len);

    for (size_t imodulus = 0; imodulus < coeff_modulus.size(); imodulus++) {
        std::cout << "coeff_modulus[" << imodulus << "]: " << coeff_modulus[imodulus].value() << std::endl;
    }

    PhantomContext context(parms);

    PhantomSecretKey secret_key(context);
    PhantomPublicKey public_key = secret_key.gen_publickey(context);
    PhantomRelinKey relin_keys = secret_key.gen_relinkey(context);
    PhantomGaloisKey galois_keys;

    PhantomCKKSEncoder encoder(context);

    CKKSEvaluator ckks_evaluator(&context, &public_key, &secret_key, &encoder, &relin_keys, &galois_keys, scale);

    size_t slot_count = encoder.slot_count();

    Bootstrapper bootstrapper(loge, logn, logN - 1, total_level, scale, boundary_K, deg, scale_factor, inverse_deg, &ckks_evaluator);

    std::cout << "Generating Optimal Minimax Polynomials..." << endl;
    bootstrapper.prepare_mod_polynomial();

    std::cout << "Adding Bootstrapping Keys..." << endl;
    vector<int> gal_steps_vector;
    gal_steps_vector.push_back(0); // 16个
    for (int i = 0; i < logN - 1; i++) {
        gal_steps_vector.push_back((1 << i));
        gal_steps_vector.push_back(-(1 << i));
    }
    bootstrapper.addLeftRotKeys_Linear_to_vector_3(gal_steps_vector); // push back bsgs steps

    ckks_evaluator.decryptor.create_galois_keys_from_steps(gal_steps_vector, *(ckks_evaluator.galois_keys));

    std::cout << "Galois key size: " << ckks_evaluator.galois_keys->get_relin_keys_size() << std::endl;
    std::cout << "Galois key generated from steps vector." << endl;

    bootstrapper.slot_vec.push_back(logn);

    std::cout << "Generating Linear Transformation Coefficients..." << endl;
    bootstrapper.generate_LT_coefficient_3();
    std::cout << "pre compute done." << std::endl
              << std::endl;

    // set input data

    std::vector<double> input(slot_count, 33.0); // input: (dim, count) count = points*centers 一个维度一个密文
    std::vector<double> input1(slot_count, 2.0);
    std::vector<double> input2(slot_count, 0.03125);

    PhantomCiphertext cipher, cipher1, cipher2, bootrtn;
    PhantomPlaintext plain;

    ckks_evaluator.encoder.encode(input, boot_level + 1, scale, plain);
    ckks_evaluator.encryptor.encrypt(plain, cipher);

    ckks_evaluator.encoder.encode(input1, boot_level + 1, scale, plain);
    ckks_evaluator.encryptor.encrypt(plain, cipher1);

    ckks_evaluator.encoder.encode(input2, boot_level + 1, scale, plain);
    ckks_evaluator.encryptor.encrypt(plain, cipher2);

    // auto s = phantom::util::global_variables::default_stream->get_stream();

    // for (int i = 0; i < remaining_level; i++) {
    ckks_evaluator.evaluator.mod_switch_to_next_inplace(cipher);
    // }

    ckks_evaluator.evaluator.multiply_inplace_reduced_error(cipher, cipher1, *(ckks_evaluator.relin_keys));
    ckks_evaluator.evaluator.rescale_to_next_inplace(cipher);
    ckks_evaluator.print_decrypted_ct(cipher, 10);

    for (int i = 0; i < 1; i++) {
        ckks_evaluator.evaluator.mod_switch_to_next_inplace(cipher);
    }

    ckks_evaluator.evaluator.right_shift_inplace(cipher, 1);
    // cipher.scale() *= 4;
    ckks_evaluator.print_decrypted_ct(cipher, 10);

    bootstrapper.bootstrap_3(bootrtn, cipher);
    ckks_evaluator.print_decrypted_ct(bootrtn, 10);

    // ckks_evaluator.evaluator.multiply_inplace_reduced_error(bootrtn, cipher1, *(ckks_evaluator.relin_keys));
    // ckks_evaluator.evaluator.rescale_to_next_inplace(bootrtn);
    // ckks_evaluator.print_decrypted_ct(bootrtn, 10);

    // ckks_evaluator.evaluator.add_const_inplace(bootrtn, 32.0);

    // ckks_evaluator.print_decrypted_ct(bootrtn, 10);

    // ckks_evaluator.evaluator.multiply_inplace_reduced_error(bootrtn, cipher2, *(ckks_evaluator.relin_keys));
    // ckks_evaluator.evaluator.rescale_to_next_inplace(bootrtn);

    // ckks_evaluator.print_decrypted_ct(bootrtn, 10);
}
