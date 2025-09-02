#include <algorithm>
#include <chrono>
#include <functional>
#include <random>
#include <vector>

#include "bootstrapping/Bootstrapper.cuh"
#include "ckks_evaluator.cuh"
#include "cutfhe++.h"
#include "fileio.h"
#include "tlwevaluator.cuh"

using namespace rlwe;
using namespace cuTFHEpp;
using namespace cuTFHEpp::util;
// using namespace conver;

using CUDATimer = phantom::util::CUDATimer;

void print_value(std::vector<double> &vector, size_t size, std::string str) {
    std::cout << str << " : " << std::endl;
    for (size_t i = 0; i < size; i++) {
        std::cout << vector[i] << " ";
    }
    std::cout << std::endl;
}

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

/**
 * 第一次 sgn 没有问题，第二次出问题，可能是密文误差的原因，解密之后再填进去是可以正常计算的
 * 可以调节 scale 和 q 的大小来解决这个问题，
 */
void Compare(CKKSEvaluator &ckks, PhantomCiphertext &cipher0, PhantomCiphertext &cipher1, std::vector<PhantomCiphertext> &bool_ct, long point_size) {
    // PhantomPlaintext delta;
    std::vector<double> mask(ckks.encoder.slot_count(), 0.0);
    std::fill(mask.data(), mask.data() + point_size, 1.0 / 20);  // 得大一些才行，可能哪里超了吧

    ckks.evaluator.sub(cipher0, cipher1, bool_ct[0]);
    // ckks.print_decrypted_ct(bool_ct[0], 16, "sub");
    // std::cout << "sub scale 0: " << bool_ct[0].scale() << std::endl; // 7.03688e+13

    // ckks.encoder.encode(ckks.init_vec_with_value(1.0 / 8.5), bool_ct[0].params_id(), bool_ct[0].scale(), delta);
    // ckks.evaluator.multiply_plain_inplace(bool_ct[0], delta);
    ckks.evaluator.multiply_vector_inplace_reduced_error(bool_ct[0], mask);

    // ckks.evaluator.multiply_const_inplace(bool_ct[0], 1.0 / 8); 使用这个会溢出

    ckks.evaluator.rescale_to_next_inplace(bool_ct[0]);
    // ckks.evaluator.mod_switch_to_next_inplace(bool_ct[0]);
    // ckks.print_decrypted_ct(bool_ct[0], 16, "sub normal");

    // ckks.re_encrypt(bool_ct[0]);
    // std::cout << "sub scale 1: " << bool_ct[0].scale() << std::endl; // 7.03688e+13
    // auto sgn = ckks.sgn_eval(bool_ct[0], 2, 2);  // 2: 16; 3:25; 4: 32
    auto sgn = ckks.sgn_eval(bool_ct[0], 3, 3);  // 2: 16; 3:25; 4: 32
    // auto sgn = ckks.sgn_eval(bool_ct[0], 4, 4);  // 2: 16; 3:25; 4: 32 根本不想，噪声太大
    // ckks.print_decrypted_ct(sgn, 16, "sgn");

    ckks.evaluator.add_const(sgn, -0.5, bool_ct[0]);
    ckks.evaluator.multiply_const_inplace(bool_ct[0], -1.0);
    ckks.evaluator.rescale_to_next_inplace(bool_ct[0]);

    ckks.evaluator.add_const(sgn, 0.5, bool_ct[1]);
    ckks.evaluator.multiply_const_inplace(bool_ct[1], 1.0);
    ckks.evaluator.rescale_to_next_inplace(bool_ct[1]);
}

void CompareDistanceMultiCenters(CKKSEvaluator &ckks, Bootstrapper &bootstrapper, PhantomCiphertext &point_cipher, long point_size) {
    long depth = static_cast<long>(std::log2(point_size));
    std::vector<std::vector<PhantomCiphertext>> trans_matrix(depth, std::vector<PhantomCiphertext>(3));
    PhantomCiphertext temp, sum;
    std::vector<PhantomCiphertext> bool_temp(2);
    std::vector<PhantomCiphertext> bool_half(3);

    std::vector<std::vector<double>> mask(3, std::vector<double>(ckks.encoder.slot_count(), 0.0));
    std::vector<int> rotate_step(3, 0.0);

    // std::cout << "depth: " << depth << std::endl;
    // point_size should be power of 2
    auto s = phantom::util::global_variables::default_stream->get_stream();

    for (size_t idepth = 0; idepth < depth; idepth++) {
        if (idepth == 0 || idepth == depth - 1) {
            rotate_step[0] = 0;
            rotate_step[1] = 1;
            rotate_step[2] = point_size - 1;
        } else {
            rotate_step[0] = 0;
            rotate_step[1] = (1 << (idepth + 1)) - 1;
            rotate_step[2] = point_size - rotate_step[1];
        }

        // bool matrix
        ckks.evaluator.rotate_vector(point_cipher, (1 << (idepth + 1)) - 1, *(ckks.galois_keys), temp);
        // ckks.print_decrypted_ct(point_cipher, 16, "--dis0");
        // ckks.print_decrypted_ct(temp, 16, "--dis1");
        Compare(ckks, point_cipher, temp, bool_temp, point_size);  // 18 + 3
        // std::cout << "after compare level: " << bool_temp[0].coeff_modulus_size() << " chain: " << bool_temp[0].chain_index() << std::endl;
        // ckks.print_decrypted_ct(bool_temp[0], 16, "bool_temp[0]");
        // ckks.print_decrypted_ct(bool_temp[1], 16, "bool_temp[1]");

        // exit(0);

        // set mask
        std::fill(mask[0].data(), mask[0].data() + point_size, 0.0);  // all
        std::fill(mask[1].data(), mask[1].data() + point_size, 0.0);
        std::fill(mask[2].data(), mask[2].data() + point_size, 0.0);
        for (size_t icenters = 0; icenters < point_size; icenters += (2 * (1 << (idepth + 1)))) {
            // mask[0][icenters] = 1.0;
            // if ((1 << (idepth + 1)) < point_size) {
            //     // mask[0][icenters + 1] = 1.0;
            //     mask[0][icenters + (1 << (idepth + 1))] = 1.0;
            // }

            mask[1][icenters] = 1.0;
            if ((icenters + (1 << (idepth + 1))) < point_size) {
                mask[2][icenters + (1 << (idepth + 1))] = 1.0;
            }
        }

        // trans 0: 上半部分升序，下半部分降序
        ckks.evaluator.multiply_vector_reduced_error(bool_temp[0], mask[1], bool_half[0]);
        ckks.evaluator.rescale_to_next_inplace(bool_half[0]);
        ckks.evaluator.multiply_vector_reduced_error(bool_temp[1], mask[2], bool_half[1]);
        ckks.evaluator.rescale_to_next_inplace(bool_half[1]);
        ckks.evaluator.add(bool_half[0], bool_half[1], bool_half[2]);

        // ckks.print_decrypted_ct(bool_temp[0], 16, "--bool_temp[0]");
        // ckks.print_decrypted_ct(bool_temp[1], 16, "--bool_temp[1]");

        // ckks.print_decrypted_ct(bool_half[0], 16, "--bool masked[0]");
        // ckks.print_decrypted_ct(bool_half[1], 16, "--bool masked[1]");
        // ckks.print_decrypted_ct(bool_half[2], 16, "--bool masked[2]");

        ckks.evaluator.rotate_vector(bool_half[2], -((1 << (idepth + 1)) - 1), *(ckks.galois_keys), trans_matrix[idepth][0]);
        ckks.evaluator.add_inplace_reduced_error(trans_matrix[idepth][0], bool_half[2]);
        // ckks.print_decrypted_ct(trans_matrix[idepth][0], 16, "----trans_matrix 0");

        int trans_bool_step = 0;
        if (idepth == depth - 1) {
            trans_bool_step = -((1 << (idepth + 1)) - 1);

            ckks.evaluator.multiply_vector_reduced_error(bool_temp[0], mask[2], bool_half[0]);
            ckks.evaluator.rescale_to_next_inplace(bool_half[0]);
            ckks.evaluator.multiply_vector_reduced_error(bool_temp[1], mask[1], bool_half[1]);
            ckks.evaluator.rescale_to_next_inplace(bool_half[1]);

            ckks.evaluator.rotate_vector(bool_half[1], trans_bool_step, *(ckks.galois_keys), trans_matrix[idepth][1]);
            trans_matrix[idepth][2] = bool_half[1];

            // ckks.print_decrypted_ct(bool_half[1], 16, "----bool_half[1]");
            // ckks.print_decrypted_ct(bool_half[0], 16, "----bool_half[0]");

            // ckks.print_decrypted_ct(trans_matrix[idepth][1], 16, "----trans_matrix 1");
            // ckks.print_decrypted_ct(trans_matrix[idepth][2], 16, "----trans_matrix 2");
        } else {
            // trans 1: trans 0 的反
            ckks.evaluator.multiply_vector_reduced_error(bool_temp[0], mask[2], bool_half[0]);
            ckks.evaluator.rescale_to_next_inplace(bool_half[0]);
            ckks.evaluator.multiply_vector_reduced_error(bool_temp[1], mask[1], bool_half[1]);
            ckks.evaluator.rescale_to_next_inplace(bool_half[1]);
            ckks.evaluator.add(bool_half[0], bool_half[1], trans_matrix[idepth][1]);

            // ckks.print_decrypted_ct(bool_half[0], 16, "bool_half 1");
            // ckks.print_decrypted_ct(bool_half[1], 16, "bool_half 1");
            // ckks.print_decrypted_ct(trans_matrix[idepth][1], 16, "----trans_matrix 1");

            // trans 2: trans 1 旋转
            int trans_1_step = -1;
            if (idepth > 0 && idepth < depth - 1) {
                trans_1_step = -rotate_step[1];
            }
            ckks.evaluator.rotate_vector(trans_matrix[idepth][1], trans_1_step, *(ckks.galois_keys), trans_matrix[idepth][2]);
            // ckks.print_decrypted_ct(trans_matrix[idepth][2], 16, "----trans_matrix 2");
        }
        // update point_cipher
        // ckks.print_decrypted_ct(point_cipher, 16, "point_cipher  before");
        ckks.evaluator.rotate_vector(point_cipher, rotate_step[0], *(ckks.galois_keys), sum);
        // ckks.print_decrypted_ct(sum, 16, "point_cipher  after");
        ckks.evaluator.multiply_inplace_reduced_error(sum, trans_matrix[idepth][0], *(ckks.relin_keys));
        ckks.evaluator.rescale_to_next_inplace(sum);
        // ckks.print_decrypted_ct(sum, 16, "sum0");

        ckks.evaluator.rotate_vector(point_cipher, rotate_step[1], *(ckks.galois_keys), temp);
        ckks.evaluator.multiply_inplace_reduced_error(temp, trans_matrix[idepth][1], *(ckks.relin_keys));
        ckks.evaluator.rescale_to_next_inplace(temp);
        ckks.evaluator.add_inplace_reduced_error(sum, temp);
        // ckks.print_decrypted_ct(sum, 16, "sum1");

        ckks.evaluator.rotate_vector(point_cipher, rotate_step[2], *(ckks.galois_keys), temp);  // 优化点：旋转消除
        ckks.evaluator.multiply_inplace_reduced_error(temp, trans_matrix[idepth][2], *(ckks.relin_keys));
        ckks.evaluator.rescale_to_next_inplace(temp);
        ckks.evaluator.add_inplace_reduced_error(sum, temp);

        point_cipher = sum;
        // ckks.print_decrypted_ct(point_cipher, 16, "point_cipher");
        // exit(0);
        // std::cout << "----point_cipher  level: " << point_cipher .coeff_modulus_size() << " chain: " << point_cipher .chain_index() << std::endl; // 1

        // std::memset(mask.data(), 1.0, point_size * sizeof(double));
        std::fill(mask[0].begin(), mask[0].begin() + point_size, 1.0);
        // std::fill(mask[1].begin(), mask[1].begin() + point_size, 1.0);
        // std::cout << "mask: " << mask[0] << " " << mask[1] << " " << mask[point_size] << " " << mask[point_size + 1] << std::endl;
        ckks.evaluator.multiply_vector_inplace_reduced_error(point_cipher, mask[0]);
        ckks.evaluator.rescale_to_next_inplace(point_cipher);
        // ckks.print_decrypted_ct(point_cipher, 16, "point_cipher mask");

        ckks.evaluator.rotate_vector(point_cipher, -point_size, *(ckks.galois_keys), temp);
        ckks.evaluator.add_inplace_reduced_error(point_cipher, temp);

        // ckks.print_decrypted_ct(point_cipher, 16, "point_cipher ");
        // std::cout << "----point_cipher  level: " << point_cipher .coeff_modulus_size() << " chain: " << point_cipher .chain_index() << std::endl; // 8

        // boot

        // {
        //     CUDATimer timer("boot", s);
        // timer.start();
        if (idepth < depth - 1) {
            if (point_cipher.coeff_modulus_size() > 1) {
                size_t count = point_cipher.coeff_modulus_size() - 1;
                for (size_t icount = 0; icount < count; icount++) {
                    ckks.evaluator.mod_switch_to_next_inplace(point_cipher);
                }
            }
            bootstrapper.bootstrap_3(temp, point_cipher);
            point_cipher = temp;
            // ckks.print_decrypted_ct(point_cipher , 16, "point_cipher after boot");
        }
        // timer.stop();
        // }
        // exit(0);
    }
}

double print_min_max(std::vector<double> data, size_t len) {
    double min = data[0];
    double max = data[0];
    for (size_t i = 1; i < len; i++) {
        min = min > data[i] ? data[i] : min;
        max = max < data[i] ? data[i] : max;
    }
    std::cout << "min: " << min << " max: " << max << std::endl;

    return min;
}

void rlwe_findmin() {
    // ckks init
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

    // int logq = 61;
    // int logp = 56;  // 必须大于51
    // int log_special_prime = 61;

    int logq = 61;
    int logp = 56;  // 必须大于51
    int log_special_prime = 61;

    int secret_key_hamming_weight = 192;

    // (29,4)or(39, 6)-->comp(3,3) or comp(4,4); (21,3)-->comp(2,2),(23,2)-->comp(2,2)
    int remaining_level = 29;
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
    }
    for (int i = 0; i < logN - 1; i++) {
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

    std::vector<double> error{};
    // std::vector<size_t> log_iter = {14};
    for (size_t i = 1; i < 15; i++) {  // 15
                                       // for (auto i : log_iter) {
        int data_size = 1 << i;
        std::cout << "len: " << data_size << std::endl;
        std::vector<double> data(1 << 15, 0.0);
        random_real(data, data_size);
        // for (size_t idata = 0; idata < 8; idata++) {
        //     data[idata] = 2.0 - 0.5 * idata;
        // }

        // print_value(data, data_size, "data");
        double min = print_min_max(data, data_size);

        // for (size_t i = 8; i < data_size; i++) {
        //     data[i] = data[i % 8];
        // }
        // for (size_t i = 0; i < data_size; i++) {
        //     data[i + data_size] = data[i];
        // }

        PhantomPlaintext plain;
        PhantomCiphertext data_cipher;

        ckks_evaluator.encoder.encode(data, boot_level + 1, scale, plain);
        ckks_evaluator.encryptor.encrypt(plain, data_cipher);
        ckks_evaluator.print_decrypted_ct(data_cipher, 16, "data");

        auto s = phantom::util::global_variables::default_stream->get_stream();
        {
            std::cout << "begin: " << data_cipher.coeff_modulus_size() << std::endl;

            CUDATimer timer("sgn", s);
            timer.start();
            CompareDistanceMultiCenters(ckks_evaluator, bootstrapper, data_cipher, data_size);
            timer.stop();

            std::cout << "after: " << data_cipher.coeff_modulus_size() << std::endl;

            std::cout << "module size: " << remaining_level + 1 << " --> " << data_cipher.coeff_modulus_size() << " consum: " << remaining_level + 1 - data_cipher.coeff_modulus_size() << std::endl;
        }

        double res_min = ckks_evaluator.print_decrypted_ct_min(data_cipher, 16, "min");

        error.push_back(std::abs(min - res_min));

        std::cout << "curr error: " << std::abs(min - res_min) << " size: " << data_size << std::endl
                  << std::endl;
    }

    double err_sum = std::accumulate(error.begin(), error.end(), 0.0);  // 计算和
    double err_mean = err_sum / error.size();                           // 计算均值
    std::cout << "sum error: " << err_sum << " avg error: " << err_mean << std::endl;
}

// template <typename T>
// T halfX(T m, T p) {
//     return static_cast<T>(std::round(0.5 * m)) % p;
// }
template <typename T>
T halfX(T m) {
    return 0.5 * m;
    // return 1;
}
// template <typename T>
// T halfAbsX(T m, T p) {
//     // return static_cast<T>(std::round(0.5 * std::abs(m))) % p;
//     if (m >= (p / 2)) {
//         return static_cast<T>(std::round(0.5 * (p - m))) % p;
//     } else {
//         return static_cast<T>(std::round(0.5 * m)) % p;
//     }
// }
template <typename T, typename Tp>
T halfAbsX(T m, Tp p) {
    if (m >= (p >> 1)) {
        return 0.5 * (p - m);
    } else {
        return 0.5 * m;
    }
}

template <typename LvlXY, typename LvlYZ,
          typename LvlX = LvlXY::domainP, typename LvlY = LvlXY::targetP, typename LvlZ = LvlYZ::targetP>
void HalfAbsBoostrapping(const Pointer<Context> &context, Pointer<cuTLWE<LvlX>> &res, Pointer<cuTLWE<LvlX>> &data, const size_t num_test) {
    static_assert(std::is_same<LvlY, typename LvlYZ::domainP>::value, "Invalid LvlY");

    using P = std::conditional_t<isLvlCover<LvlX, LvlZ>(), LvlX, LvlZ>;

    Pointer<ProgBootstrappingData<LvlYZ>> pbs_data(num_test);

    TFHEpp::TLWE<LvlX> *d_tlwe = data->template get<LvlX>();
    TFHEpp::TLWE<LvlZ> *d_res = res->template get<LvlZ>();

    // auto lut = GenLUT<LvlX>(halfAbsX<typename LvlX::T>, LvlX::plain_modulus);
    auto lut = GenDLUTP<LvlX>(halfAbsX<double, typename LvlX::T>, LvlX::plain_modulus);

    typename LvlZ::T *d_lut;
    CUDA_CHECK_RETURN(cudaMalloc(&d_lut, lut.size() * sizeof(typename LvlZ::T)));
    CUDA_CHECK_RETURN(cudaMemcpy(d_lut, lut.data(), lut.size() * sizeof(typename LvlZ::T), cudaMemcpyHostToDevice));
    ProgBootstrapping<LvlXY, LvlYZ>(context.get(), pbs_data, d_lut, d_res, d_tlwe, num_test);
}

template <typename LvlXY, typename LvlYZ,
          typename LvlX = LvlXY::domainP, typename LvlY = LvlXY::targetP, typename LvlZ = LvlYZ::targetP>
void HalfBoostrapping(const Pointer<Context> &context, Pointer<cuTLWE<LvlX>> &res, Pointer<cuTLWE<LvlX>> &data, const size_t num_test) {
    static_assert(std::is_same<LvlY, typename LvlYZ::domainP>::value, "Invalid LvlY");

    using P = std::conditional_t<isLvlCover<LvlX, LvlZ>(), LvlX, LvlZ>;

    Pointer<ProgBootstrappingData<LvlYZ>> pbs_data(num_test);

    TFHEpp::TLWE<LvlX> *d_tlwe = data->template get<LvlX>();
    TFHEpp::TLWE<LvlZ> *d_res = res->template get<LvlZ>();

    // auto lut = GenLUT<LvlX>(halfX<typename LvlX::T>, LvlX::plain_modulus);
    auto lut = GenDLUT<LvlX>(halfX<double>, LvlX::plain_modulus);

    typename LvlZ::T *d_lut;
    CUDA_CHECK_RETURN(cudaMalloc(&d_lut, lut.size() * sizeof(typename LvlZ::T)));
    CUDA_CHECK_RETURN(cudaMemcpy(d_lut, lut.data(), lut.size() * sizeof(typename LvlZ::T), cudaMemcpyHostToDevice));
    ProgBootstrapping<LvlXY, LvlYZ>(context.get(), pbs_data, d_lut, d_res, d_tlwe, num_test);
}

// min(m0, m1) = 0.5(m0 + m1) - 0.5|m0 - m1|
template <typename P>
void MinOfTwoBatch(tlwevaluator<P> &tlwer, Pointer<cuTLWE<P>> &dtlwe_min, Pointer<cuTLWE<P>> &data, size_t step) {
    Pointer<cuTLWE<P>> dtlwe_tmp0(step);
    Pointer<cuTLWE<P>> dtlwe_tmp1(step);

    auto rtn = dtlwe_min->template get<P>();
    auto res0 = dtlwe_tmp0->template get<P>();
    auto res1 = dtlwe_tmp1->template get<P>();
    auto front = data->template get<P>();
    auto behind = front + step;

    // {
    // CUDATimer timer("add", 0);
    // timer.start();
    tlwer.add(res0, front, behind, step);
    // timer.stop();
    // }
    // tlwer.print_culwe_ct_value_double(res0, step, "add result");

    // {
    // CUDATimer timer("sub", 0);
    // timer.start();
    tlwer.sub(res1, front, behind, step);
    // timer.stop();
    // }
    // tlwer.print_culwe_ct_value(res1, step, "sub result");

    auto &context = tlwer.get_pbscontext();

    // HalfBoostrapping<Lvl10, Lvl01>(context, dtlwe_tmp0, dtlwe_tmp0, step);  // 0.5(m0 + m1)
    HalfBoostrapping<Lvl20, Lvl02>(context, dtlwe_tmp0, dtlwe_tmp0, step);  // 0.5(m0 + m1)
    // tlwer.print_culwe_ct_value_double(res0, step, "half result");

    // HalfAbsBoostrapping<Lvl10, Lvl01>(context, dtlwe_tmp1, dtlwe_tmp1, step);  // 0.5|m0 - m1|
    HalfAbsBoostrapping<Lvl20, Lvl02>(context, dtlwe_tmp1, dtlwe_tmp1, step);  // 0.5|m0 - m1|
    // tlwer.print_culwe_ct_value_double(res1, step, "half abs result");

    tlwer.sub(rtn, res0, res1, step);  // min(m0, m1)
    // tlwer.print_culwe_ct_value_double(rtn, step, "min result");
}

template <typename P>
void Mini(tlwevaluator<P> &tlwer, Pointer<cuTLWE<P>> &lwe_distances, int centers) {
    // lwe_distances: (points, centers)
    Pointer<cuTLWE<P>> dtlwe_min(centers >> 1);
    size_t depth = std::ceil((std::log2(centers)));
    // find min of each point
    for (size_t idepth = 0; idepth < depth; idepth++) {
        std::cout << "idepth: " << idepth << " ------------------ " << std::endl;
        if (idepth == 0) {
            MinOfTwoBatch<P>(tlwer, dtlwe_min, lwe_distances, centers / (1 << (idepth + 1)));
        } else {
            MinOfTwoBatch<P>(tlwer, dtlwe_min, dtlwe_min, centers / (1 << (idepth + 1)));  // min in first
        }
    }

    tlwer.print_culwe_ct_value_double(dtlwe_min->template get<P>(), 1, "min");
}

void lwe_findmin() {
    std::cout << "Setting LWE Parameters..." << endl;
    // using lwe_enc_lvl = Lvl1;
    using lwe_enc_lvl = Lvl2;
    // int scale_bits = std::numeric_limits<typename lwe_enc_lvl::T>::digits - lwe_enc_lvl::plain_modulus_bit - 1;
    // double lwe_scale = pow(2.0, scale_bits);
    double lwe_scale = lwe_enc_lvl::Δ;

    TFHESecretKey sk;
    TFHEEvalKey ek;
    load_keys<BootstrappingKeyFFTLvl01, BootstrappingKeyFFTLvl02,
              KeySwitchingKeyLvl10, KeySwitchingKeyLvl20, KeySwitchingKeyLvl21>(sk, ek);
    tlwevaluator<lwe_enc_lvl> tlwe_evaluator(&sk, &ek, lwe_scale);

    for (size_t i = 8; i < 9; i++) {  // 16
        int centers = 1 << i;
        std::cout << "len: " << centers << std::endl;
        std::vector<TLWELvl2> lwe_distances(centers);

        std::vector<lwe_enc_lvl::T> lwe_distances_vec(centers, 0);
        for (size_t j = 0; j < centers; j++) {
            lwe_distances_vec[j] = (centers - 1 - j) % 8;
        }

        for (size_t j = 0; j < centers; j++) {
            lwe_distances[j] = TFHEpp::tlweSymInt32Encrypt<lwe_enc_lvl>(lwe_distances_vec[j], lwe_enc_lvl::α, lwe_scale, sk.key.get<lwe_enc_lvl>());
            // auto lwe_dec_num = TFHEpp::tlweSymInt32Decrypt<lwe_enc_lvl>(lwe_distances[j], lwe_scale, sk.key.get<lwe_enc_lvl>());
            // std::cout << "decrypt: " << lwe_dec_num << " ground: " << lwe_distances_vec[j] << std::endl;
        }

        Pointer<cuTLWE<lwe_enc_lvl>> d_lwe_distances(centers);

        TFHEpp::TLWE<lwe_enc_lvl> *dest = d_lwe_distances->template get<lwe_enc_lvl>();
        TFHEpp::TLWE<lwe_enc_lvl> *src = lwe_distances.data();
        CUDA_CHECK_RETURN(cudaMemcpy(dest, src, centers * sizeof(TFHEpp::TLWE<lwe_enc_lvl>), cudaMemcpyHostToDevice));

        {
            CUDATimer timer("Find min", 0);
            timer.start();
            Mini<lwe_enc_lvl>(tlwe_evaluator, d_lwe_distances, centers);
            timer.stop();
        }
    }
}

void lwe_add() {
    std::cout << "Setting LWE Parameters..." << endl;
    // using lwe_enc_lvl = Lvl1;
    using lwe_enc_lvl = Lvl2;
    // int scale_bits = std::numeric_limits<typename lwe_enc_lvl::T>::digits - lwe_enc_lvl::plain_modulus_bit - 1;
    // double lwe_scale = pow(2.0, scale_bits);
    double lwe_scale = lwe_enc_lvl::Δ;

    TFHESecretKey sk;
    TFHEEvalKey ek;
    load_keys<BootstrappingKeyFFTLvl01, BootstrappingKeyFFTLvl02,
              KeySwitchingKeyLvl10, KeySwitchingKeyLvl20, KeySwitchingKeyLvl21>(sk, ek);
    tlwevaluator<lwe_enc_lvl> tlwe_evaluator(&sk, &ek, lwe_scale);

    for (size_t i = 0; i < 16; i++) {  // 16
        int centers = 1 << i;
        std::cout << "len: " << centers << std::endl;
        std::vector<TLWELvl2> lwe_distances(centers);

        std::vector<lwe_enc_lvl::T> lwe_distances_vec(centers, 0);
        for (size_t j = 0; j < centers; j++) {
            lwe_distances_vec[j] = (centers - 1 - j) % 8;
        }

        for (size_t j = 0; j < centers; j++) {
            lwe_distances[j] = TFHEpp::tlweSymInt32Encrypt<lwe_enc_lvl>(lwe_distances_vec[j], lwe_enc_lvl::α, lwe_scale, sk.key.get<lwe_enc_lvl>());
            // auto lwe_dec_num = TFHEpp::tlweSymInt32Decrypt<lwe_enc_lvl>(lwe_distances[j], lwe_scale, sk.key.get<lwe_enc_lvl>());
            // std::cout << "decrypt: " << lwe_dec_num << " ground: " << lwe_distances_vec[j] << std::endl;
        }

        Pointer<cuTLWE<lwe_enc_lvl>> d_lwe_distances(centers);

        TFHEpp::TLWE<lwe_enc_lvl> *dest = d_lwe_distances->template get<lwe_enc_lvl>();
        TFHEpp::TLWE<lwe_enc_lvl> *src = lwe_distances.data();
        CUDA_CHECK_RETURN(cudaMemcpy(dest, src, centers * sizeof(TFHEpp::TLWE<lwe_enc_lvl>), cudaMemcpyHostToDevice));

        {
            CUDATimer timer("Add", 0);
            timer.start();
            tlwe_evaluator.add(dest, dest, dest, centers);
            timer.stop();
        }
    }
}

int main() {
    // rlwe_findmin();
    lwe_findmin();

    // lwe_add();

    return 0;
}