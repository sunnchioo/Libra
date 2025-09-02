#include <algorithm>
#include <chrono>
#include <functional>
#include <random>
#include <vector>

#include "ckks_evaluator.cuh"
#include "cutfhe++.h"
#include "fileio.h"
#include "tlwevaluator.cuh"

using namespace rlwe;
using namespace cuTFHEpp;
using namespace cuTFHEpp::util;
// using namespace conver;

using CUDATimer = phantom::util::CUDATimer;
using namespace std::chrono;

void random_real(std::vector<double> &vec, size_t size) {
    // std::random_device rn;
    // std::mt19937_64 rnd(rn());
    std::mt19937_64 rnd(41);
    thread_local std::uniform_real_distribution<double> distribution(-8, 8);

    vec.reserve(size);

    for (size_t i = 0; i < size; i++) {
        vec[i] = distribution(rnd);
    }
}

template <typename T>
T sign(T m, T p) {
    if (m > (p / 2)) {
        return static_cast<T>(0);
    } else {
        return static_cast<T>(1);
    }
}

template <typename LvlXY, typename LvlYZ,
          typename LvlX = LvlXY::domainP, typename LvlY = LvlXY::targetP, typename LvlZ = LvlYZ::targetP>
void SignBoostrapping(const Pointer<Context> &context, Pointer<cuTLWE<LvlX>> &res, Pointer<cuTLWE<LvlX>> &data, const size_t num_test) {
    static_assert(std::is_same<LvlY, typename LvlYZ::domainP>::value, "Invalid LvlY");

    using P = std::conditional_t<isLvlCover<LvlX, LvlZ>(), LvlX, LvlZ>;

    Pointer<ProgBootstrappingData<LvlYZ>> pbs_data(num_test);

    TFHEpp::TLWE<LvlX> *d_tlwe = data->template get<LvlX>();
    TFHEpp::TLWE<LvlZ> *d_res = res->template get<LvlZ>();

    auto lut = GenLUT<LvlX>(sign<typename LvlX::T>, LvlX::plain_modulus);

    typename LvlZ::T *d_lut;
    CUDA_CHECK_RETURN(cudaMalloc(&d_lut, lut.size() * sizeof(typename LvlZ::T)));
    CUDA_CHECK_RETURN(cudaMemcpy(d_lut, lut.data(), lut.size() * sizeof(typename LvlZ::T), cudaMemcpyHostToDevice));

    {  // sgn
        CUDATimer timer("SignBoostrapping", 0);
        timer.start();
        ProgBootstrapping<LvlXY, LvlYZ>(context.get(), pbs_data, d_lut, d_res, d_tlwe, num_test);
        timer.stop();
    }
}

void ckks_comp(PhantomCiphertext &rtn, CKKSEvaluator &ckks_evaluator) {
    PhantomPlaintext delta;

    PhantomPlaintext plain;
    ckks_evaluator.encoder.encode(ckks_evaluator.init_vec_with_value(1.0 / 20), rtn.params_id(), rtn.scale(), delta);

    // std::cout << "first chain index: " << rtn.chain_index() << std::endl;
    ckks_evaluator.evaluator.multiply_plain_inplace(rtn, delta);
    ckks_evaluator.evaluator.rescale_to_next_inplace(rtn);
    // std::cout << "second chain index: " << rtn.chain_index() << std::endl;

    // std::cout << "input cipher level: " << rtn.coeff_modulus_size() << endl;
    // rtn = ckks_evaluator.sgn_eval(rtn, 2, 1);
    rtn = ckks_evaluator.sgn_eval(rtn, 2, 2);  //(21,2)
    // rtn = ckks_evaluator.sgn_eval(rtn, 3, 3);  //(29, 3)
    // rtn = ckks_evaluator.sgn_eval(rtn, 4, 4);  //(34, 5)
    // std::cout << "Return cipher level: " << rtn.coeff_modulus_size() << endl;
}

// 除比较外需要4个level
void ckks_comp() {
    // ckks compare
    long logN = 16;

    long logn = logN - 1;
    long sparse_slots = (1 << logn);

    int logq = 61;
    int logp = 56;
    int log_special_prime = 61;

    int secret_key_hamming_weight = 192;

    int remaining_level = 25;
    int special_prime_len = 2;

    std::vector<int> coeff_bit_vec;
    coeff_bit_vec.push_back(logq);
    for (int i = 0; i < remaining_level; i++) {
        coeff_bit_vec.push_back(logp);
    }
    for (int i = 0; i < special_prime_len; i++) {
        coeff_bit_vec.push_back(log_special_prime);
    }
    std::cout << "Setting Parameters..." << std::endl;
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

    std::vector<int> gal_steps_vector;
    gal_steps_vector.push_back(1);
    // for (int i = 0; i < logN - 1; i++) {
    //     gal_steps_vector.push_back((1 << i));
    // }
    ckks_evaluator.decryptor.create_galois_keys_from_steps(gal_steps_vector, *(ckks_evaluator.galois_keys));

    std::vector<double> error{};
    for (size_t iter = 0; iter < 16; iter++) {
        int slot_size = 1 << iter;
        std::cout << "len: " << slot_size << std::endl;
        std::vector<double> sparse0(sparse_slots, 0.0);
        std::vector<double> sparse1(sparse_slots, 0.0);
        random_real(sparse0, slot_size);
        random_real(sparse1, slot_size);

        std::cout << "sparse: " << std::endl;
        for (size_t i = 0; i < 10; i++) {
            std::cout << sparse0[i] << " ";
        }
        std::cout << std::endl;

        PhantomPlaintext plain;
        PhantomCiphertext cipher0, cipher1;

        ckks_evaluator.encoder.encode(sparse0, scale, plain);
        ckks_evaluator.encryptor.encrypt(plain, cipher0);
        ckks_evaluator.encoder.encode(sparse1, scale, plain);
        ckks_evaluator.encryptor.encrypt(plain, cipher1);

        ckks_evaluator.evaluator.sub_inplace_reduced_error(cipher0, cipher1);

        // Create input cipher
        std::cout << "sparse_slots: " << sparse_slots << std::endl;
        auto s = phantom::util::global_variables::default_stream->get_stream();

        {
            std::cout << "ckks comp begin: " << cipher0.coeff_modulus_size() << std::endl;

            CUDATimer timer("sgn", s);
            timer.start();

            // auto start = system_clock::now();
            ckks_comp(cipher0, ckks_evaluator);
            // duration<double> sec = system_clock::now() - start;
            // std::cout << "comp took: " << sec.count() * 1000 << " ms" << endl;

            timer.stop();

            std::cout << "ckks comp end: " << cipher0.coeff_modulus_size() << std::endl;

            ckks_evaluator.decryptor.decrypt(cipher0, plain);
            ckks_evaluator.encoder.decode(plain, sparse0);
            std::cout << "sparse slot: " << sparse0.size() << std::endl;

            std::cout << "compare result: " << std::endl;  // >0:0.5 | <0:-0.5
            for (size_t i = 0; i < 10; i++) {
                std::cout << sparse0[i] << " ";
            }
            std::cout << std::endl;

            double err = 0.;
            for (size_t i = 0; i < slot_size; i++) {
                err += std::abs(std::abs(sparse0[i]) - 0.5);
            }
            std::cout << "error: " << err / slot_size << " size: " << slot_size << std::endl
                      << std::endl;

            error.push_back(err / slot_size);
        }
    }

    double err_sum = std::accumulate(error.begin(), error.end(), 0.0);  // 计算和
    double err_mean = err_sum / error.size();                           // 计算均值
    std::cout << "sum error: " << err_sum << " avg error: " << err_mean << std::endl
              << std::endl;
}

void tlwe_comp() {
    std::cout << "Setting LWE Parameters..." << std::endl;
    using lwe_enc_lvl = Lvl1;
    using lwe_res_lvl = Lvl1;
    // using lwe_enc_lvl = Lvl2;
    // using lwe_res_lvl = Lvl2;

    // int scale_bits = std::numeric_limits<typename lwe_enc_lvl::T>::digits - lwe_enc_lvl::plain_modulus_bit - 1;
    // int scale_bits = 61;
    // double lwe_scale = pow(2.0, scale_bits);
    // std::cout << "LWE scale: " << lwe_scale << " scale bit: " << scale_bits << std::endl;
    double lwe_scale = lwe_enc_lvl::Δ;

    TFHESecretKey sk;
    TFHEEvalKey ek;
    load_keys<BootstrappingKeyFFTLvl01, BootstrappingKeyFFTLvl02,
              KeySwitchingKeyLvl10, KeySwitchingKeyLvl20, KeySwitchingKeyLvl21>(sk, ek);

    tlwevaluator<lwe_enc_lvl> tlwe_evaluator(&sk, &ek, lwe_scale);

    for (int iter = 0; iter < 16; iter++) {
        int batch = 1 << iter;
        std::cout << "LWE batch size: " << batch << std::endl;

        std::vector<double> sample(batch, 0.0);
        random_real(sample, batch);

        std::vector<lwe_enc_lvl::T> msg(batch);
        for (int i = 0; i < batch; i++) {
            msg[i] = static_cast<lwe_enc_lvl::T>(sample[i]);
        }

        size_t len_lwe = msg.size();
        std::vector<TFHEpp::TLWE<lwe_enc_lvl>> h_lwes(len_lwe);
        Pointer<cuTLWE<lwe_enc_lvl>> d_lwes(len_lwe);
        Pointer<cuTLWE<lwe_res_lvl>> d_lwes_res(len_lwe);

        for (size_t i = 0; i < len_lwe; i++) {
            h_lwes[i] = TFHEpp::tlweSymInt32Encrypt<lwe_enc_lvl>(msg[i], lwe_enc_lvl::α, lwe_scale, sk.key.get<lwe_enc_lvl>());
            auto lwe_dec_num = TFHEpp::tlweSymInt32Decrypt<lwe_enc_lvl>(h_lwes[i], lwe_scale, sk.key.get<lwe_enc_lvl>());
            // std::cout << "decrypt: " << lwe_dec_num << " ground: " << msg[i] << std::endl;
        }

        TFHEpp::TLWE<lwe_enc_lvl> *d_dest = d_lwes->template get<lwe_enc_lvl>();
        TFHEpp::TLWE<lwe_enc_lvl> *h_src = h_lwes.data();
        CUDA_CHECK_RETURN(cudaMemcpy(d_dest, h_src, len_lwe * sizeof(TFHEpp::TLWE<lwe_enc_lvl>), cudaMemcpyHostToDevice));

        auto &context = tlwe_evaluator.get_pbscontext();

        auto res = d_lwes_res->template get<lwe_enc_lvl>();
        auto src = d_lwes->template get<lwe_enc_lvl>();

        // { // sgn
        //     CUDATimer timer("SignBoostrapping", 0);
        //     timer.start();
        SignBoostrapping<Lvl10, Lvl01>(context, d_lwes_res, d_lwes, len_lwe);
        // SignBoostrapping<Lvl20, Lvl02>(context, d_lwes_res, d_lwes, len_lwe);
        int print_len = len_lwe > 10 ? 10 : len_lwe;
        tlwe_evaluator.print_culwe_ct_value(res, print_len, "Sign result");
        // timer.stop();
        // }
    }
}

int main() {
    // ckks_comp();
    tlwe_comp();

    return 0;
}