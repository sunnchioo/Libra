#pragma once

#include <complex>
#include <memory>

#include "bootstrapping/Bootstrapper.cuh"
#include "ckks_evaluator.cuh"
#include "phantom.h"

namespace rlwe {
    using namespace std;
    using namespace phantom;

    // struct CKKSConfig {
    //     long logN = 16;
    //     long logn = 15;
    //     long sparse_slots = 1 << logn;

    //     int logq = 51;
    //     int logp = 46;
    //     int log_special_prime = 51;

    //     double scale = pow(2.0, logp);

    //     int secret_key_hamming_weight = 192;

    //     int remaining_level = 15;
    //     int boot_level = 14;

    //     int special_prime_len = 4;
    //     int special_prime_len_with_boot = 6;

    //     long boundary_K = 25;
    //     long deg = 59;
    //     long scale_factor = 2;
    //     long inverse_deg = 1;
    //     long loge = 10;
    // };

    struct CKKSConfig {  // extract
        long logN = 16;
        long logn = 15;
        long sparse_slots = 1 << logn;

        int logq = 51;
        int logp = 46;
        int log_special_prime = 51;

        double scale = pow(2.0, logp);

        int secret_key_hamming_weight = 192;

        int remaining_level = 5;
        int boot_level = 0;

        int special_prime_len = 2;
        int special_prime_len_with_boot = 2;

        long boundary_K = 25;
        long deg = 59;
        long scale_factor = 2;
        long inverse_deg = 1;
        long loge = 10;
    };

    class trlwevaluator {
    public:
        CKKSConfig config;
        bool flagCKKS = false;
        bool flagBoot = true;

        unique_ptr<CKKSEvaluator> ckks;
        unique_ptr<Bootstrapper> ckksboot;

        trlwevaluator(scheme_type scheme = scheme_type::none) {
            if (scheme == scheme_type::ckks) {
                flagCKKS = true;
                initCKKS(flagBoot);
            }
        }

        // trlwevaluator(scheme_type scheme = scheme_type::none, const CKKSConfig &config = {}) : config(config) {
        //     if (scheme == scheme_type::ckks) {
        //         flagCKKS = true;
        //         initCKKS(flagBoot);
        //     }
        // }

        ~trlwevaluator() = default;

        trlwevaluator(const trlwevaluator &) = delete;
        trlwevaluator &operator=(const trlwevaluator &) = delete;

        const PhantomSecretKey &secret_key() const {
            if (!secret_key_) {
                throw std::logic_error("Secret key not initialized");
            }
            return *secret_key_;
        }

    private:
        unique_ptr<EncryptionParameters> parms_;
        unique_ptr<PhantomContext> context_;
        unique_ptr<PhantomSecretKey> secret_key_;
        unique_ptr<PhantomPublicKey> public_key_;
        unique_ptr<PhantomRelinKey> relin_keys_;
        unique_ptr<PhantomGaloisKey> galois_keys_;
        unique_ptr<PhantomCKKSEncoder> encoder_;

        void initCKKS(bool isboot) {
            // const CKKSConfig config;

            int boot_level = isboot ? config.boot_level : 0;
            int special_prime_len = isboot ? config.special_prime_len_with_boot : config.special_prime_len;
            int total_level = config.remaining_level + boot_level;

            vector<int> coeff_bit_vec;
            coeff_bit_vec.push_back(config.logq);
            for (int i = 0; i < config.remaining_level; i++) {
                coeff_bit_vec.push_back(config.logp);
            }
            for (int i = 0; i < boot_level; i++) {
                coeff_bit_vec.push_back(config.logq);
            }
            for (int i = 0; i < special_prime_len; i++) {
                coeff_bit_vec.push_back(config.log_special_prime);
            }

            std::cout << "Setting Parameters..." << endl;
            parms_ = make_unique<EncryptionParameters>(scheme_type::ckks);

            size_t poly_modulus_degree = 1UL << config.logN;
            auto coeff_modulus = phantom::arith::CoeffModulus::Create(poly_modulus_degree, coeff_bit_vec);
            std::cout << "modulus: ";
            for (size_t i = 0; i < coeff_modulus.size(); i++) {
                std::cout << coeff_modulus[i].value() << " ";
            }
            std::cout << std::endl;

            parms_->set_poly_modulus_degree(poly_modulus_degree);
            parms_->set_coeff_modulus(coeff_modulus);
            parms_->set_secret_key_hamming_weight(config.secret_key_hamming_weight);
            parms_->set_sparse_slots(config.sparse_slots);
            parms_->set_special_modulus_size(special_prime_len);

            context_ = make_unique<PhantomContext>(*parms_);
            secret_key_ = make_unique<PhantomSecretKey>(*context_);
            public_key_ = make_unique<PhantomPublicKey>(secret_key_->gen_publickey(*context_));
            relin_keys_ = make_unique<PhantomRelinKey>(secret_key_->gen_relinkey(*context_));
            galois_keys_ = make_unique<PhantomGaloisKey>();
            encoder_ = make_unique<PhantomCKKSEncoder>(*context_);

            ckks = make_unique<CKKSEvaluator>(context_.get(), public_key_.get(), secret_key_.get(), encoder_.get(),
                                              relin_keys_.get(), galois_keys_.get(), config.scale);

            if (isboot) {
                vector<int> gal_steps_vector;
                gal_steps_vector.push_back(0);  // 16个
                for (int i = 0; i < (config.logN - 1); i++) {
                    gal_steps_vector.push_back((1 << i));
                    gal_steps_vector.push_back(-(1 << i));
                }

                ckksboot = make_unique<Bootstrapper>(config.loge, config.logn, config.logN - 1, total_level,
                                                     config.scale, config.boundary_K, config.deg, config.scale_factor,
                                                     config.inverse_deg, ckks.get());

                std::cout << "Generating Optimal Minimax Polynomials..." << endl;
                ckksboot->prepare_mod_polynomial();

                std::cout << "Adding Bootstrapping Keys..." << endl;
                ckksboot->addLeftRotKeys_Linear_to_vector_3(gal_steps_vector);

                ckks->decryptor.create_galois_keys_from_steps(gal_steps_vector, *(ckks->galois_keys));
                // std::cout << "Galois key size: " << ckks->galois_keys->get_relin_keys_size() << std::endl;
                std::cout << "Galois key generated from steps vector." << endl;

                ckksboot->slot_vec.push_back(config.logn);

                std::cout << "Generating Linear Transformation Coefficients..." << endl;
                // ckksboot->generate_LT_coefficient_3();
                ckksboot->generate_LT_coefficient_ori_3();
            }
        }
    };

}  // namespace rlwe
