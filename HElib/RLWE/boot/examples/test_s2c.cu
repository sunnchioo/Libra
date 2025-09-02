#include <random>

#include "bootstrapping/Bootstrapper.cuh"
#include "file.h"
#include "phantom.h"

using namespace std;
using namespace phantom;

void random_real(vector<double> &vec, size_t size) {
    // random_device rn;
    // mt19937_64 rnd(rn());
    mt19937_64 rnd(42);

    thread_local std::uniform_real_distribution<double> distribution(-1, 1);

    vec.reserve(size);

    for (size_t i = 0; i < size; i++) {
        vec[i] = distribution(rnd);
    }
}

int main() {
    long boundary_K = 25;
    long deg = 59;
    long scale_factor = 2;
    long inverse_deg = 1;

    // The following parameters have been adjusted to satisfy the memory constraints of an A100 GPU
    long logN = 16;  // 16 -> 15  // full only even(14)
    long loge = 10;

    long logn = 8;  // 14 -> 13
    long sparse_slots = (1 << logn);

    int logq = 51;
    int logp = 46;
    int log_special_prime = 51;

    int secret_key_hamming_weight = 192;

    int remaining_level = 9;
    int boot_level = 0;  // >= subsum 1 + coefftoslot 2 + ModReduction 9 + slottocoeff 2
    int total_level = remaining_level + boot_level;
    int special_prime_len = 2;

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
    gal_steps_vector.push_back(0);
    for (int i = 0; i < logN - 1; i++) {
        gal_steps_vector.push_back((1 << i));
        gal_steps_vector.push_back(-(1 << i));
    }
    bootstrapper.addLeftRotKeys_Linear_to_vector_3(gal_steps_vector);

    ckks_evaluator.decryptor.create_galois_keys_from_steps(gal_steps_vector, *(ckks_evaluator.galois_keys));
    std::cout << "Galois key generated from steps vector." << endl;

    bootstrapper.slot_vec.push_back(logn);

    std::cout << "Generating Linear Transformation Coefficients..." << endl;
    bootstrapper.generate_LT_coefficient_ori_3();  // No scaling coefficients

    // for (size_t i = 0; i < bootstrapper.orig_coeffvec[0].size(); i++) {
    //     std::string filename = "/mnt/data2/home/syt/data/Libra/RLWE/test/origcoeff/orig_coeffvec_" + std::to_string(i);
    //     FileIO::saveComplexVector2D(bootstrapper.orig_coeffvec[0][i], filename);
    // }

    vector<double> sparse(sparse_slots, 0.0);
    vector<double> input(slot_count, 0.0);
    vector<double> before(slot_count, 0.0);
    vector<complex<double>> after(slot_count, 0.0);

    random_real(sparse, sparse_slots);

    PhantomPlaintext plain;
    PhantomCiphertext cipher;

    // Create input cipher
    for (size_t i = 0; i < slot_count; i++) {
        input[i] = sparse[i % sparse_slots];
        // input[i] = 0.9;
    }

    std::cout << "input: ";
    for (size_t i = 0; i < 10; i++) {
        std::cout << input[i] << " ";
    }
    std::cout << "... ";
    for (size_t i = sparse_slots - 10; i < sparse_slots; i++) {
        std::cout << input[i] << " ";
    }
    std::cout << std::endl;

    ckks_evaluator.encoder.encode(input, scale, plain);

    // ckks_evaluator.print_decoded_pt(plain, 10);

    ckks_evaluator.encryptor.encrypt(plain, cipher);

    // ckks_evaluator.print_decrypted_ct(cipher, 10);

    // Mod switch to the lowest level
    // for (int i = 0; i < total_level; i++) {
    //     ckks_evaluator.evaluator.mod_switch_to_next_inplace(cipher);
    // }

    // Decrypt input cipher to obtain the original input
    ckks_evaluator.decryptor.decrypt(cipher, plain);
    ckks_evaluator.encoder.decode(plain, before);
    std::cout << "before: ";
    for (size_t i = 0; i < 10; i++) {
        std::cout << before[i] << " ";
    }
    std::cout << std::endl;

    auto start = system_clock::now();

    PhantomCiphertext rtn;

    // full 只能支持full编码，因为是 fft 的逆运算
    // bootstrapper.slottocoeff_full_3(rtn, cipher);
    // ckks_evaluator.print_decrypted_ct(rtn, true, 128, "After slottocoeff");
    // ckks_evaluator.print_decrypted_nodecode_ct(rtn, 10, 10, "After slottocoeff nondecode");
    // bootstrapper.coefftoslot_full_3(rtn, rtn);  // rtn = 4 * cipher

    // bootstrapper.coefftoslot_full_3(rtn, cipher);
    // ckks_evaluator.print_decrypted_ct(rtn, true, 10, "After coefftoslot_full_3");
    // bootstrapper.slottocoeff_full_3(rtn, rtn);

    // sparse 在稀疏模式下是还原不过去的
    // PhantomCiphertext rot;
    // for (auto i = logn; i < logN - 1; i++) {
    //     ckks_evaluator.evaluator.rotate_vector(cipher, (1 << i), *(ckks_evaluator.galois_keys), rot);
    //     ckks_evaluator.evaluator.add_inplace(cipher, rot);
    // }

    bootstrapper.slottocoeff_sparse_3(rtn, cipher);
    ckks_evaluator.print_decrypted_ct(rtn, true, 10, 10, "After slottocoeff");
    // ckks_evaluator.print_decrypted_nodecode_ct(rtn, sparse_slots, "After slottocoeff nondecode");
    bootstrapper.coefftoslot_sparse_3(rtn, rtn);

    duration<double> sec = system_clock::now() - start;
    std::cout << "Bootstrapping took: " << sec.count() * 1000 << " ms" << endl;
    std::cout << "Return cipher level: " << rtn.coeff_modulus_size() << endl;

    ckks_evaluator.print_decrypted_ct(rtn, true, 10, 10, "After");

    // ckks_evaluator.decryptor.decrypt(rtn, plain);
    // ckks_evaluator.encoder.decode(plain, after);

    // cout << "after: ";
    // for (long i = 0; i < 10; i++) {
    //     std::cout << after[i] << " ";
    // }
    // cout << std::endl;

    // vector<uint64_t> print_plain(slot_count, 0.0);

    // cudaMemcpy(print_plain.data(), plain.data(0), slot_count * sizeof(uint64_t), cudaMemcpyDeviceToHost);

    // double mean_err = 0;
    // for (long i = 0; i < sparse_slots; i++) {
    //     if (i < 10)
    //         std::cout << before[i] << " <----> " << after[i] << endl;
    //     mean_err += abs(before[i] - after[i]);
    // }
    // mean_err /= sparse_slots;
    // std::cout << "Mean absolute error: " << mean_err << endl;

    // c2s: ct -> ct0, ct1
}
