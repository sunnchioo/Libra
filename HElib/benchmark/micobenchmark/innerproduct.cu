#include <string.h>

#include <algorithm>
#include <random>
#include <vector>

#include "bootstrapping/Bootstrapper.cuh"
#include "ckks_evaluator.cuh"
#include "fileio.h"
#include "phantom.h"

using CUDATimer = phantom::util::CUDATimer;

void print_matrix(std::vector<std::vector<double>> &matrix, std::string str) {
    std::cout << str << " : " << std::endl;
    for (size_t i = 0; i < matrix.size(); i++) {
        std::cout << "line " << i << " : ";
        for (size_t j = 0; j < 10; j++) {
            std::cout << matrix[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void InnerProdct(CKKSEvaluator &ckks, std::vector<PhantomCiphertext> &x) {
    PhantomCiphertext inner_temp;

    ckks.evaluator.multiply_inplace_reduced_error(x[0], x[1], *(ckks.relin_keys));
    ckks.evaluator.rescale_to_next_inplace(x[0]);

    // inner sum
    for (size_t idim = 0; idim < 6; idim++) {
        ckks.evaluator.rotate_vector(x[0], 1 << idim, *(ckks.galois_keys), inner_temp);
        ckks.evaluator.add(x[0], inner_temp, x[0]);
    }
}

void ct_rotate(CKKSEvaluator &ckks, PhantomCiphertext &x) {
    auto s = phantom::util::global_variables::default_stream->get_stream();

    for (size_t idim = 0; idim < 6; idim++) {
        {
            CUDATimer timer("rotate " + std::to_string(idim), s);
            timer.start();
            ckks.evaluator.rotate_vector(x, 1 << idim, *(ckks.galois_keys), x);
            timer.stop();
        }
    }
}

void many_rotate(CKKSEvaluator &ckks, PhantomCiphertext &x) {
    std::vector<int> steps0 = {1 << 0, 1 << 1};
    std::vector<int> steps1 = {1 << 0, 1 << 1, 1 << 2};
    std::vector<int> steps2 = {1 << 0, 1 << 1, 1 << 2, 1 << 3};
    std::vector<int> steps3 = {1 << 0, 1 << 1, 1 << 2, 1 << 3, 1 << 4};
    std::vector<int> steps4 = {1 << 0, 1 << 1, 1 << 2, 1 << 3, 1 << 4, 1 << 5};

    auto s = phantom::util::global_variables::default_stream->get_stream();
    {
        CUDATimer timer("rotate 0", s);
        timer.start();
        ckks.evaluator.many_rotate(x, *(ckks.galois_keys), steps0);
        timer.stop();
    }
    {
        CUDATimer timer("rotate 1", s);
        timer.start();
        ckks.evaluator.many_rotate(x, *(ckks.galois_keys), steps1);
        timer.stop();
    }
    {
        CUDATimer timer("rotate 2", s);
        timer.start();
        ckks.evaluator.many_rotate(x, *(ckks.galois_keys), steps2);
        timer.stop();
    }
    {
        CUDATimer timer("rotate 3", s);
        timer.start();
        ckks.evaluator.many_rotate(x, *(ckks.galois_keys), steps3);
        timer.stop();
    }
    {
        CUDATimer timer("rotate 4", s);
        timer.start();
        ckks.evaluator.many_rotate(x, *(ckks.galois_keys), steps4);
        timer.stop();
    }
}

int main() {
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

    int remaining_level = 29;
    int boot_level = 0;                              // >= subsum 1 + coefftoslot 2 + ModReduction 9 + slottocoeff 2
    int total_level = remaining_level + boot_level;  // 38
    int special_prime_len = 2;

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

    std::cout << "Adding galois Keys..." << endl;
    vector<int> gal_steps_vector;
    gal_steps_vector.push_back(0);
    for (int i = 0; i < logN - 1; i++) {
        gal_steps_vector.push_back((1 << i));
        gal_steps_vector.push_back(-(1 << i));
    }

    ckks_evaluator.decryptor.create_galois_keys_from_steps(gal_steps_vector, *(ckks_evaluator.galois_keys));

    std::cout << "Galois key size: " << ckks_evaluator.galois_keys->get_relin_keys_size() << std::endl;
    std::cout << "Galois key generated from steps vector." << endl;

    // Create points and centers cipher
    PhantomPlaintext plain;
    // std::vector<std::vector<PhantomCiphertext>> distance_matrix_cipher(centers, std::vector<PhantomCiphertext>(points));
    std::vector<PhantomCiphertext> x(2);

    std::vector<double> input(slot_count, 0.1);  // input: 一个输入一个密文，维度为一个密文里

    for (size_t i = 0; i < 2; i++) {
        ckks_evaluator.encoder.encode(input, scale, plain);
        ckks_evaluator.encryptor.encrypt(plain, x[i]);
    }

    // auto s = phantom::util::global_variables::default_stream->get_stream();

    // {
    //     CUDATimer timer("Euclidean distance", s);
    //     timer.start();
    //     InnerProdct(ckks_evaluator, x);  // 2 level
    //     timer.stop();
    // }

    // ct_rotate(ckks_evaluator, x[0]);

    many_rotate(ckks_evaluator, x[0]);
}