// ckks single eval
#include <vector>

#include "ckks_evaluator.cuh"
#include "fileio.h"
#include "gelu.cuh"
#include "phantom.h"
#include "utils.cuh"

using namespace std;
using namespace phantom;
using namespace phantom::arith;
using namespace phantom::util;
using namespace rlwe;

void random_real(vector<double> &vec, size_t size) {
    random_device rn;
    mt19937_64 rnd(rn());
    thread_local std::uniform_real_distribution<double> distribution(-2, 2);

    vec.reserve(size);

    for (size_t i = 0; i < size; i++) {
        vec[i] = distribution(rnd);
    }
}

void ckks_comp(PhantomCiphertext &cipher0, PhantomCiphertext &cipher1, CKKSEvaluator &ckks_evaluator) {
    vector<double> after;

    PhantomCiphertext rtn;
    PhantomPlaintext delta;

    std::cout << "begin cipher level: " << cipher0.coeff_modulus_size() << endl;

    auto start = system_clock::now();
    ckks_evaluator.evaluator.sub(cipher0, cipher1, rtn);
    ckks_evaluator.print_decrypted_ct(rtn, 10, "sub");

    // vector<double> input(ckks_evaluator.encoder.slot_count(), 0.0);

    // auto text_data = FileIO<double>::LoadText("/mnt/data2/home/syt/data/fhec/benchmark/moti/single/data/points.txt");

    // for (size_t i = 0; i < text_data.size(); i++) {
    //     input[i] = text_data[i];
    // }

    PhantomPlaintext plain;

    // ckks_evaluator.encoder.encode(input, cipher0.scale(), plain);
    // ckks_evaluator.encryptor.encrypt(plain, rtn);

    ckks_evaluator.encoder.encode(ckks_evaluator.init_vec_with_value(1.0 / 8.5), rtn.params_id(), rtn.scale(), delta);

    std::cout << "first chain index: " << rtn.chain_index() << std::endl;
    ckks_evaluator.evaluator.multiply_plain_inplace(rtn, delta);
    ckks_evaluator.evaluator.rescale_to_next_inplace(rtn);
    std::cout << "second chain index: " << rtn.chain_index() << std::endl;

    rtn = ckks_evaluator.sgn_eval(rtn, 2, 2);
    duration<double> sec = system_clock::now() - start;
    std::cout << "comp took: " << sec.count() * 1000 << " ms" << endl;
    std::cout << "Return cipher level: " << rtn.coeff_modulus_size() << endl;

    ckks_evaluator.decryptor.decrypt(rtn, plain);
    ckks_evaluator.encoder.decode(plain, after);
    std::cout << "after slot: " << after.size() << std::endl;

    std::vector<double> save(after.begin(), after.begin() + 100);
    FileIO<double>::SaveText("/mnt/data2/home/syt/data/fhec/benchmark/moti/single/data/res.txt", save);

    std::cout << "compare result: " << std::endl;  // >0:0.5 | <0:-0.5
    for (size_t i = 0; i < 10; i++) {
        std::cout << after[i] << " ";
    }
    std::cout << std::endl;

    double err = 0.;
    for (size_t i = 0; i < after.size(); i++) {
        err += std::abs(std::abs(after[i]) - 0.5);
    }
    std::cout << "error: " << err / after.size() << " size: " << after.size() << std::endl;
}

void ckks_add(PhantomCiphertext &cipher0, PhantomCiphertext &cipher1, CKKSEvaluator &ckks_evaluator, vector<double> &input0, vector<double> &input1) {
    vector<double> after;
    PhantomPlaintext plain0;

    PhantomCiphertext rtn;
    ckks_evaluator.evaluator.add(cipher0, cipher1, rtn);

    ckks_evaluator.decryptor.decrypt(rtn, plain0);
    ckks_evaluator.encoder.decode(plain0, after);

    std::cout << "size: " << after.size() << std::endl;

    std::cout << "res: " << after[0] << " " << after[1] << std::endl;

    double err = 0.;
    for (size_t i = 0; i < after.size(); i++) {
        err += std::abs((input0[i] + input1[i]) - after[i]);
    }
    std::cout << "error: " << err / after.size() << std::endl;
}

void ckks_pmult(PhantomCiphertext &cipher0, PhantomPlaintext &plain1, CKKSEvaluator &ckks_evaluator, vector<double> &input0, vector<double> &input1) {
    vector<double> after(ckks_evaluator.encoder.slot_count(), 0.0);
    PhantomPlaintext plain0;

    PhantomCiphertext rtn;
    ckks_evaluator.evaluator.multiply_plain(cipher0, plain1, rtn);

    ckks_evaluator.decryptor.decrypt(rtn, plain0);
    ckks_evaluator.encoder.decode(plain0, after);

    double err = 0.;
    for (size_t i = 0; i < after.size(); i++) {
        err += std::abs((input0[i] * input1[i]) - after[i]);
    }
    std::cout << "error: " << err / after.size() << std::endl;
}

void ckks_cmult(PhantomCiphertext &cipher0, PhantomCiphertext &cipher1, PhantomCiphertext &cipher2, CKKSEvaluator &ckks_evaluator, vector<double> &input0, vector<double> &input1) {
    vector<double> after(ckks_evaluator.encoder.slot_count(), 0.0);
    PhantomPlaintext plain0;

    PhantomCiphertext rtn, rtn1;
    ckks_evaluator.evaluator.multiply_reduced_error(cipher0, cipher1, *(ckks_evaluator.relin_keys), rtn);
    ckks_evaluator.evaluator.rescale_to_next_inplace(rtn);
    // ckks_evaluator.evaluator.multiply_reduced_error(rtn, cipher2, *(ckks_evaluator.relin_keys), rtn1);
    // ckks_evaluator.evaluator.rescale_to_next_inplace(rtn1);

    ckks_evaluator.decryptor.decrypt(rtn, plain0);
    ckks_evaluator.encoder.decode(plain0, after);

    std::cout << "res: " << after[0] << " " << after[1] << std::endl;

    double err = 0.;
    for (size_t i = 0; i < after.size(); i++) {
        err += std::abs((input0[i] * input1[i]) - after[i]);
    }
    std::cout << "error: " << err / after.size() << std::endl;
}

void ckks_rescale(PhantomCiphertext &cipher0, CKKSEvaluator &ckks_evaluator, vector<double> &input0) {
    vector<double> after(ckks_evaluator.encoder.slot_count(), 0.0);
    PhantomPlaintext plain0;

    // ckks_evaluator.evaluator.rescale_to_next_inplace(cipher0); // rescale 是用在乘法之后，直接使用肯定不对
    // ckks_evaluator.evaluator.mod_switch_scale_to_next(cipher0);
    ckks_evaluator.evaluator.mod_switch_to_next_inplace(cipher0);  // 只是单纯的丢弃最后一个level

    ckks_evaluator.decryptor.decrypt(cipher0, plain0);
    ckks_evaluator.encoder.decode(plain0, after);

    for (size_t i = 0; i < 10; i++) {
        std::cout << " " << after[i] << std::endl;
    }

    double err = 0.;
    for (size_t i = 0; i < after.size(); i++) {
        err += std::abs((input0[i]) - after[i]);
    }
    std::cout << "error: " << err / after.size() << std::endl;
}

void ckks_rotate(PhantomCiphertext &cipher0, CKKSEvaluator &ckks_evaluator, vector<double> &input0) {
    vector<double> after(ckks_evaluator.encoder.slot_count(), 0.0);
    PhantomPlaintext plain0;

    PhantomCiphertext rtn;

    ckks_evaluator.evaluator.rotate_vector(cipher0, 1, *(ckks_evaluator.galois_keys), rtn);

    ckks_evaluator.decryptor.decrypt(rtn, plain0);
    ckks_evaluator.encoder.decode(plain0, after);

    for (size_t i = 0; i < 10; i++) {
        std::cout << " " << after[i] << std::endl;
    }

    double err = 0.;
    for (size_t i = 0; i < after.size(); i++) {
        err += std::abs((input0[(i + 1) % after.size()]) - after[i]);
    }
    std::cout << "error: " << err / after.size() << std::endl;
}

void ckks_eval(int n) {
    // The following parameters have been adjusted to satisfy the memory constraints of an A100 GPU
    long logN = n;  // full only even(14)

    long logn = logN - 1;  // 256
    long sparse_slots = (1 << logn);

    int logp = 46;
    int log_special_prime = 51;

    int secret_key_hamming_weight = 192;

    int remaining_level = 24;
    int special_prime_len = 2;

    vector<int> coeff_bit_vec;
    coeff_bit_vec.push_back(51);  // 最大值是预定义的
    for (int i = 0; i < remaining_level; i++) {
        coeff_bit_vec.push_back(logp);
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

    vector<int> gal_steps_vector;
    gal_steps_vector.push_back(1);
    // for (int i = 0; i < logN - 1; i++) {
    //     gal_steps_vector.push_back((1 << i));
    // }
    ckks_evaluator.decryptor.create_galois_keys_from_steps(gal_steps_vector, *(ckks_evaluator.galois_keys));

    size_t slot_count = encoder.slot_count();

    vector<double> sparse0(sparse_slots, 0.0);
    vector<double> sparse1(sparse_slots, 0.0);

    vector<double> input0(slot_count, 0.0);
    vector<double> input1(slot_count, 0.0);
    vector<double> input2(slot_count, 0.0);

    vector<double> before(slot_count, 0.0);
    // vector<double> after(slot_count, 0.0);

    random_real(sparse0, sparse_slots);
    random_real(sparse1, sparse_slots);

    PhantomPlaintext plain0;
    PhantomPlaintext plain1;
    PhantomPlaintext plain2;

    PhantomCiphertext cipher0;
    PhantomCiphertext cipher1, cipher2;

    // Create input cipher
    std::cout << "slot_count: " << slot_count << std::endl;
    // for (size_t i = 0; i < slot_count; i++) {
    // input0[i] = sparse0[i % sparse_slots];
    // input1[i] = sparse1[i % sparse_slots];
    // input0[i] = 15.55;
    // input1[i] = 32.0;
    // input2[i] = 0.125;

    // input0[i] = 16383.0; // 明文空间为(-2^p, 2^p) p = bit(q_0) - bit(scale)
    // input1[i] = 1.0;
    // input2[i] = 0.125;
    // }

    input0[0] = 0;
    input0[1] = 1;
    input0[2] = 2;
    input0[3] = 1;
    input0[4] = 0;
    input0[5] = 1;
    input0[6] = 2;
    input0[7] = 1;

    input1[0] = 1;
    input1[1] = 0;
    input1[2] = 1;
    input1[3] = 2;
    input1[4] = 1;
    input1[5] = 0;
    input1[6] = 0;
    input1[7] = 0;

    ckks_evaluator.encoder.encode(input0, scale, plain0);
    ckks_evaluator.encryptor.encrypt(plain0, cipher0);
    ckks_evaluator.encoder.encode(input1, scale, plain1);
    ckks_evaluator.encryptor.encrypt(plain1, cipher1);

    ckks_evaluator.encoder.encode(input2, scale, plain2);
    ckks_evaluator.encryptor.encrypt(plain2, cipher2);

    ckks_evaluator.print_decrypted_ct(cipher0, 10);
    ckks_evaluator.print_decrypted_ct(cipher1, 10);

    // exit(0);
    // ckks compare
    ckks_comp(cipher0, cipher1, ckks_evaluator);
    // ckks_add(cipher0, cipher1, ckks_evaluator, input0, input1);
    // ckks_pmult(cipher0, plain1, ckks_evaluator, input0, input1);
    // ckks_cmult(cipher0, cipher1, cipher2, ckks_evaluator, input0, input1);
    // ckks_cmult(cipher0, cipher2, ckks_evaluator, input0, input1);

    // ckks_rescale(cipher0, ckks_evaluator, input0);
    // ckks_rotate(cipher0, ckks_evaluator, input0);
}

int main() {
    ckks_eval(16);

    return 0;
}