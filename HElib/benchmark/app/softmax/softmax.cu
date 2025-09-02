#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "argmax.cuh"
#include "ckks_evaluator.cuh"
#include "gelu.cuh"
#include "layer_norm.cuh"
#include "matrix_mul.cuh"
#include "phantom.h"
#include "softmax.cuh"
#include "utils.cuh"

using namespace std;
using namespace phantom;
using namespace phantom::arith;
using namespace phantom::util;
using namespace rlwe;

using CUDATimer = phantom::util::CUDATimer;

int main() {
    // size_t N = 1ULL << 16;
    // double SCALE = pow(2.0, 40);
    // vector<int> COEFF_MODULI = {58, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 58};

    // EncryptionParameters params(scheme_type::ckks);

    // params.set_poly_modulus_degree(N);
    // params.set_coeff_modulus(CoeffModulus::Create(N, COEFF_MODULI));

    // PhantomContext context(params);

    // PhantomSecretKey secret_key(context);
    // PhantomPublicKey public_key = secret_key.gen_publickey(context);
    // PhantomRelinKey relin_keys = secret_key.gen_relinkey(context);
    // PhantomGaloisKey galois_keys = secret_key.create_galois_keys(context);

    // PhantomCKKSEncoder encoder(context);

    // CKKSEvaluator ckks_evaluator(&context, &public_key, &secret_key, &encoder, &relin_keys, &galois_keys, SCALE);

    // pre compute
    long boundary_K = 25;
    long deg = 59;
    long scale_factor = 2;
    long inverse_deg = 1;

    /////////////// length /////////////////
    long logN = 16;  // 16, 14
    long loge = 10;

    long logn = 15;
    long sparse_slots = (1 << logn);  // 256

    int logp = 56;
    int logq = 61;
    int log_special_prime = 61;

    int secret_key_hamming_weight = 192;

    // (41,7)(39, 6)-->comp(3,3) or comp(4,4); (25,4)-->comp(2,2)
    int remaining_level = 21;
    int boot_level = 14;                             // >= subsum 1 + coefftoslot 2 + ModReduction 9 + slottocoeff 2
    int total_level = remaining_level + boot_level;  // 38
    int special_prime_len = 4;

    vector<int> coeff_bit_vec;
    coeff_bit_vec.push_back(logq);  // 39
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

    Bootstrapper bootstrapper(loge, logn, logN - 1, total_level, scale, boundary_K, deg, scale_factor, inverse_deg, &ckks_evaluator);

    std::cout << "Generating Optimal Minimax Polynomials..." << endl;
    bootstrapper.prepare_mod_polynomial();

    std::cout << "Adding galois Keys..." << endl;
    vector<int> gal_steps_vector;
    gal_steps_vector.push_back(0);
    for (int i = 0; i < logN - 1; i++) {
        gal_steps_vector.push_back((1 << i));
        gal_steps_vector.push_back(-(1 << i));
    }
    bootstrapper.addLeftRotKeys_Linear_to_vector_3(gal_steps_vector);  // push back bsgs steps

    ckks_evaluator.decryptor.create_galois_keys_from_steps(gal_steps_vector, *(ckks_evaluator.galois_keys));

    std::cout << "Galois key size: " << ckks_evaluator.galois_keys->get_relin_keys_size() << std::endl;
    std::cout << "Galois key generated from steps vector." << endl;

    bootstrapper.slot_vec.push_back(logn);

    std::cout << "Generating Linear Transformation Coefficients..." << endl;
    bootstrapper.generate_LT_coefficient_3();
    std::cout << "pre compute done." << std::endl
              << std::endl;

    PhantomPlaintext plain_input;
    PhantomCiphertext cipher_input;
    PhantomCiphertext cipher_output;

    // Softmax

    SoftmaxEvaluator softmax_evaluator(ckks_evaluator, bootstrapper);

    double num;
    vector<double> input, softmax_calibration;
    ifstream input_file("/mnt/data2/home/syt/data/Libra/boot/data/input/softmax_input_128_128.txt");  // 128个
    while (input_file >> num) {
        input.push_back(num);
    }
    input_file.close();

    ifstream calibration_file("/mnt/data2/home/syt/data/Libra/boot/data/calibration/softmax_calibration_128_128.txt");
    while (calibration_file >> num) {
        softmax_calibration.push_back(num);
    }
    calibration_file.close();

    ckks_evaluator.encoder.encode(input, boot_level + 1, scale, plain_input);
    ckks_evaluator.encryptor.encrypt(plain_input, cipher_input);

    auto timer = Timer();
    softmax_evaluator.softmax_scaled(cipher_input, cipher_output, 128 * 8);
    timer.stop();

    cout << "[Softmax] 128 x 128 takes: " << timer.duration() << " milliseconds" << endl;
    // cout << "Mean Absolute Error: " << ckks_evaluator.calculate_MAE(softmax_calibration, cipher_output, 128) << endl;
}
