#ifndef TARGET_FLYHE_TEMPLATES_H_
#define TARGET_FLYHE_TEMPLATES_H_

#include <string_view>

namespace mlir {
    namespace libra {
        namespace flyhe_cuda {

            constexpr const char *kAddCall = "ckks_evaluator.evaluator.add";
            constexpr const char *kMultCall = "ckks_evaluator.evaluator.multiply";

            constexpr std::string_view kCudaPrelude = R"cpp(
                /***** out begin *****/
#include <vector>
#include "ckks_evaluator.cuh"
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
                /***** out end *****/

                int main() {
                    /***** head begin *****/
                    long logN = 16;

                    long logn = logN - 1;
                    long sparse_slots = (1 << logn);

                    int logp = 46;
                    int log_special_prime = 51;

                    int secret_key_hamming_weight = 192;

                    int remaining_level = 4;
                    int special_prime_len = 2;

                    vector<int> coeff_bit_vec;
                    coeff_bit_vec.push_back(51);
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
                    PhantomRelinKey relin_keys;
                    PhantomGaloisKey galois_keys;

                    PhantomCKKSEncoder encoder(context);

                    CKKSEvaluator ckks_evaluator(&context, &public_key, &secret_key, &encoder, &relin_keys, &galois_keys, scale);

                    size_t slot_count = encoder.slot_count();

                    vector<double> sparse0(sparse_slots, 0.0);
                    vector<double> sparse1(sparse_slots, 0.0);

                    vector<double> input0(slot_count, 0.0);
                    vector<double> input1(slot_count, 0.0);

                    vector<double> after(slot_count, 0.0);

                    random_real(sparse0, sparse_slots);
                    random_real(sparse1, sparse_slots);

                    PhantomPlaintext plain0;
                    PhantomPlaintext plain1;

                    PhantomCiphertext cipher0;
                    PhantomCiphertext cipher1;
                    PhantomCiphertext rtn;

                    std::cout << "slot_count: " << slot_count << std::endl;
                    for (size_t i = 0; i < slot_count; i++) {
                        input0[i] = sparse0[i % sparse_slots];
                        input1[i] = sparse1[i % sparse_slots];
                    }

                    ckks_evaluator.encoder.encode(input0, scale, plain0);
                    ckks_evaluator.encryptor.encrypt(plain0, cipher0);
                    ckks_evaluator.encoder.encode(input1, scale, plain1);
                    ckks_evaluator.encryptor.encrypt(plain1, cipher1);
                    /***** head end *****/
            )cpp";

            constexpr std::string_view kCudaTail = R"cpp(
                /***** tail start *****/
                ckks_evaluator.print_decrypted_ct(rtn, 10);
                return 0;
                /***** tail end *****/
                }
            )cpp";

        }  // namespace tfhe_rust
    }  // namespace heir
}  // namespace mlir

#endif  // TARGET_FLYHE_TEMPLATES_H_
