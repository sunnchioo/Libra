/**
 * 两种情景：（1）多数据点，低维度。(2)少数据点，高纬度
 */
// 1. 选择 K 个中心点
// 2. K 作为初始化中心
// 3. 计算每个点到 K 个中心点的距离
// 4. 将每个点分配到最近的中心点
// 5. 更新 K 个中心点
// 6. 重复步骤 3-5，直到中心点不再变化
// 7. 返回 K 个中心点

// #include "../utils/utils.h"
#include "bootstrapping/Bootstrapper.cuh"
// #include "ckks_evaluator.cuh"
#include "fileio.h"
#include "phantom.h"

#include <algorithm>
#include <random>
#include <vector>

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

/**
 * (points - centers)^2
 */
void EuclideanDistance(CKKSEvaluator &ckks, std::vector<PhantomCiphertext> &points_ct, std::vector<PhantomCiphertext> &centers_ct,
                       long points, long dim, long centers, std::vector<std::vector<PhantomCiphertext>> &distance_cipher) {
    for (size_t icenters = 0; icenters < centers; icenters++) {
        for (size_t ipoints = 0; ipoints < points; ipoints++) {
            ckks.evaluator.sub(points_ct[ipoints], centers_ct[icenters], distance_cipher[icenters][ipoints]);
            ckks.evaluator.multiply_inplace_reduced_error(distance_cipher[icenters][ipoints], distance_cipher[icenters][ipoints], *(ckks.relin_keys));
            ckks.evaluator.rescale_to_next_inplace(distance_cipher[icenters][ipoints]);
        }
    }

    // std::cout << "euclidean distance. " << std::endl;

    // inner sum
    PhantomCiphertext temp;
    for (size_t icenters = 0; icenters < centers; icenters++) {
        for (size_t ipoints = 0; ipoints < points; ipoints++) {
            for (size_t idim = 0; idim < static_cast<size_t>(std::log2(dim)) + 1; idim++) {
                ckks.evaluator.rotate_vector(distance_cipher[icenters][ipoints], 1 << idim, *(ckks.galois_keys), temp);
                ckks.evaluator.add(distance_cipher[icenters][ipoints], temp, distance_cipher[icenters][ipoints]);
            }
        }
    }
    // std::cout << "inner sum. " << std::endl;

    // mask
    size_t slot_count = ckks.encoder.slot_count();
    std::vector<double> mask(slot_count, 0.0);
    mask[0] = 1.0;
    for (size_t icenters = 0; icenters < centers; icenters++) {
        for (size_t ipoints = 0; ipoints < points; ipoints++) {
            ckks.evaluator.multiply_vector_inplace_reduced_error(distance_cipher[icenters][ipoints], mask); // get first sum
            ckks.evaluator.rescale_to_next_inplace(distance_cipher[icenters][ipoints]);
        }
    }
    // std::cout << "mask. " << std::endl;
    // for (size_t ipoints = 0; ipoints < points; ipoints++) {
    //     ckks.print_decrypted_ct(distance_cipher[0][ipoints], 10, "mask distance 0" + std::to_string(ipoints));
    // }

    // shift
    for (size_t icenters = 0; icenters < centers; icenters++) {
        for (size_t ipoints = 0; ipoints < points; ipoints++) {
            for (size_t idegree = 0; idegree < static_cast<size_t>(std::log2(dim)); idegree++) {
                ckks.evaluator.rotate_vector(distance_cipher[icenters][ipoints], -(1 << idegree), *(ckks.galois_keys), temp);
                ckks.evaluator.add_inplace(distance_cipher[icenters][ipoints], temp);
            }
        }
    }
    // std::cout << "shift. " << std::endl;
    // for (size_t ipoints = 0; ipoints < points; ipoints++) {
    //     ckks.print_decrypted_ct(distance_cipher[0][ipoints], 10, "distance 0" + std::to_string(ipoints));
    // }
    // exit(0);
}

/**
 * compare distance
 * distance: (centers, points)
 * 当 centers == 2 时，直接相减，每个中心对用所有的点，点的密文为 mask(0/1)
 * 当 centers == 3 时，两个相减，再另外一个减之后相加，可得离哪个点近
 * 当 centers == 4 时，
 */
void CompareDistance(CKKSEvaluator &ckks, std::vector<std::vector<PhantomCiphertext>> &distance, long centers, long points, long dim, std::vector<std::vector<PhantomCiphertext>> &bool_ct) {
    // std::vector<double> mask(ckks.encoder.slot_count(), -1.0);

    PhantomPlaintext delta;
    // mask
    std::vector<double> mask(ckks.encoder.slot_count(), 0.0);
    std::fill(mask.begin(), mask.begin() + dim, 1.0);

    if (centers == 2) {
        for (size_t ipoints = 0; ipoints < points; ipoints++) {
            ckks.evaluator.sub(distance[0][ipoints], distance[1][ipoints], bool_ct[0][ipoints]);
        }

        for (size_t ipoints = 0; ipoints < points; ipoints++) {
            ckks.encoder.encode(ckks.init_vec_with_value(1.0 / 8.5), bool_ct[0][ipoints].params_id(), bool_ct[0][ipoints].scale(), delta);
            ckks.evaluator.multiply_plain_inplace(bool_ct[0][ipoints], delta);
            ckks.evaluator.rescale_to_next_inplace(bool_ct[0][ipoints]);
            auto sgn = ckks.sgn_eval(bool_ct[0][ipoints], 2, 2);

            ckks.evaluator.add_const(sgn, -0.5, bool_ct[0][ipoints]);
            ckks.evaluator.multiply_const_inplace(bool_ct[0][ipoints], -1.0);
            ckks.evaluator.rescale_to_next_inplace(bool_ct[0][ipoints]);

            ckks.evaluator.add_const(sgn, 0.5, bool_ct[1][ipoints]);
            ckks.evaluator.multiply_const_inplace(bool_ct[1][ipoints], 1.0);
            ckks.evaluator.rescale_to_next_inplace(bool_ct[1][ipoints]);

            ckks.evaluator.multiply_vector_inplace_reduced_error(bool_ct[0][ipoints], mask);
            ckks.evaluator.multiply_vector_inplace_reduced_error(bool_ct[1][ipoints], mask);
            ckks.evaluator.rescale_to_next_inplace(bool_ct[0][ipoints]);
            ckks.evaluator.rescale_to_next_inplace(bool_ct[1][ipoints]);
        }
    }
}

// center = (points - center) * bool + center
void UpdateCenters(CKKSEvaluator &ckks, std::vector<PhantomCiphertext> &center_cipher, std::vector<PhantomCiphertext> &points_cipher,
                   long centers, long points,
                   std::vector<std::vector<PhantomCiphertext>> &bool_ct) {
    PhantomCiphertext temp, sum;
    for (size_t icenters = 0; icenters < centers; icenters++) {
        for (size_t ipoints = 0; ipoints < points; ipoints++) {
            ckks.evaluator.sub_reduced_error(points_cipher[ipoints], center_cipher[icenters], temp);
            ckks.evaluator.multiply_inplace_reduced_error(temp, bool_ct[icenters][ipoints], *(ckks.relin_keys));
            ckks.evaluator.rescale_to_next_inplace(temp);
            ckks.evaluator.add_inplace_reduced_error(temp, center_cipher[icenters]);

            if (ipoints == 0)
                sum = temp;
            else
                ckks.evaluator.add_inplace(sum, temp);
        }
        ckks.evaluator.multiply_const(sum, 1.0 / points, center_cipher[icenters]);
        ckks.evaluator.rescale_to_next_inplace(center_cipher[icenters]);
    }
}

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

    int logp = 46;
    int logq = 51;
    int log_special_prime = 51;

    int secret_key_hamming_weight = 192;

    int remaining_level = 24;
    int boot_level = 14;                            // >= subsum 1 + coefftoslot 2 + ModReduction 9 + slottocoeff 2
    int total_level = remaining_level + boot_level; // 38
    int special_prime_len = 3;

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
    std::vector<std::vector<double>> data = FileIO<double>::LoadCSV3D("/mnt/data2/home/syt/data/fhec/benchmark/app/data/kmeans/8data2cent.csv"); // (points, dim)
    if (slot_count < data.size()) {
        throw std::logic_error("Error: Get Out Input Length.");
    }
    long points = data.size();
    long dim = data[0].size();
    long centers = 2;

    std::cout << points << " points. " << dim << " dim. " << centers << " centers. " << std::endl;

    /**
     * point    dim0    dim1    dim...
     *     0    0.1     0.1
     *     1    0.2     0.2
     *     2    0.3     0.3
     */
    std::vector<std::vector<double>> input(points, std::vector<double>(slot_count, 0.0));   // input: 一个输入一个密文，维度为一个密文里
    std::vector<std::vector<double>> center(centers, std::vector<double>(slot_count, 0.0)); // center: 一个 centers 一个密文
    for (size_t i = 0; i < points; i++) {
        std::copy(data[i].begin(), data[i].begin() + dim, input[i].begin());
    }
    print_matrix(input, "input");

    // init center
    for (size_t i = 0; i < centers; i++) {
        std::copy(input[i].begin(), input[i].begin() + dim, center[i].begin());
    }

    print_matrix(center, "init center");

    // Create points and centers cipher
    std::vector<std::vector<PhantomCiphertext>> distance_cipher(centers, std::vector<PhantomCiphertext>(points));
    PhantomPlaintext plain;
    std::vector<PhantomCiphertext> points_cipher(points);
    std::vector<PhantomCiphertext> center_cipher(centers), center_new_cipher(centers);

    std::vector<std::vector<PhantomCiphertext>> bool_cipher(centers, std::vector<PhantomCiphertext>(points));

    for (size_t i = 0; i < points; i++) {
        ckks_evaluator.encoder.encode(input[i], boot_level + 1, scale, plain);
        ckks_evaluator.encryptor.encrypt(plain, points_cipher[i]);
    }
    for (size_t i = 0; i < centers; i++) {
        ckks_evaluator.encoder.encode(center[i], boot_level + 1, scale, plain);
        ckks_evaluator.encryptor.encrypt(plain, center_cipher[i]);
    }

    // ckks_evaluator.print_decrypted_ct(points_cipher[1], 10, "points");
    // ckks_evaluator.print_decrypted_ct(center_cipher[1], 10, "center");

    // std::cout << "points cipher scale: " << points_cipher[0].scale() << std::endl;

    // std::cout << "points cipher level: " << points_cipher[0].coeff_modulus_size() << std::endl;
    // std::cout << "center cipher level: " << center_cipher[0].coeff_modulus_size() << std::endl;

    // std::cout << "chain index 0: " << context.get_context_data(points_cipher[0].chain_index()).parms().coeff_modulus().size() << std::endl; // curr level: 25
    // std::cout << "chain index 1: " << context.get_context_data(0).parms().coeff_modulus().size() << std::endl;                              // all
    // std::cout << "chain index 1: " << context.get_context_data(1).parms().coeff_modulus().size() << std::endl;                              // all - special

    auto s = phantom::util::global_variables::default_stream->get_stream();

    int iter_count = 1;
    for (size_t iter = 0; iter < iter_count; iter++) {
        cout << "iter: " << iter << endl;

        // (Euclidean distance)^2
        {
            CUDATimer timer("Euclidean distance", s);
            timer.start();
            EuclideanDistance(ckks_evaluator, points_cipher, center_cipher, points, dim, centers, distance_cipher); // 2 level
            timer.stop();
        }
        for (size_t ipoints = 0; ipoints < points; ipoints++) {
            ckks_evaluator.print_decrypted_ct(distance_cipher[0][ipoints], 10, "distance 0" + std::to_string(ipoints));
            ckks_evaluator.print_decrypted_ct(distance_cipher[1][ipoints], 10, "distance 1" + std::to_string(ipoints));
        }
        // cout << "distence level: " << distance_cipher[0][0].coeff_modulus_size() << endl; // 23
        // cout << "distence chain: " << distance_cipher[0][0].chain_index() << endl;        // 17

        // compare
        {
            CUDATimer timer("CompareDistance", s);
            timer.start();
            CompareDistance(ckks_evaluator, distance_cipher, centers, points, dim, bool_cipher); // 18 level
            timer.stop();
        }
        for (size_t i = 0; i < points; i++) {
            ckks_evaluator.print_decrypted_ct(bool_cipher[0][i], 10, "bool0");
        }
        // cout << "bool_cipher level: " << bool_cipher[0][0].coeff_modulus_size() << endl; // 5
        // cout << "bool_cipher chain: " << bool_cipher[0][0].chain_index() << endl;        // 35

        // update center
        {
            CUDATimer timer("UpdateCenters", s);
            timer.start();
            UpdateCenters(ckks_evaluator, center_cipher, points_cipher, centers, points, bool_cipher); // 2 level
            timer.stop();
        }
        // for (size_t icenters = 0; icenters < centers; icenters++) {
        //     ckks_evaluator.print_decrypted_ct(center_cipher[icenters], 10, "center " + std::to_string(icenters));
        // }
        // cout << "center_cipher level: " << center_cipher[0].coeff_modulus_size() << endl; // 3
        // center 0 : 0.0600291 - 0.0599984 - 2.57949e-10 7.29065e-10 1.20807e-10 8.34692e-11 - 7.82883e-10 2.55898e-11 1.14999e-09 - 2.61322e-10
        // center 1 : 2.16995 - 0.0800007 - 4.33218e-10 2.52346e-10 1.40643e-09 - 7.55927e-10 8.60584e-11 - 3.19693e-10 - 1.41437e-09 6.21378e-10

        // mod down
        {
            CUDATimer timer("Bootstrapper", s);
            timer.start();
            if (center_cipher[0].coeff_modulus_size() > 1) {
                size_t count = center_cipher[0].coeff_modulus_size() - 1;
                for (size_t icenters = 0; icenters < centers; icenters++) {
                    for (size_t icount = 0; icount < count; icount++) {
                        ckks_evaluator.evaluator.mod_switch_to_next_inplace(center_cipher[icenters]);
                    }
                }
            }
            for (size_t icenters = 0; icenters < centers; icenters++) {
                bootstrapper.bootstrap_3(center_new_cipher[icenters], center_cipher[icenters]);
            }
            timer.stop();
        }

        for (size_t icenters = 0; icenters < centers; icenters++) {
            center_cipher[icenters] = center_new_cipher[icenters];
        }
        // for (size_t icenters = 0; icenters < centers; icenters++) {
        //     ckks_evaluator.print_decrypted_ct(center_cipher[icenters], 10, "boot center " + std::to_string(icenters));
        // }
        // cout << "center_cipher level: " << center_cipher[0].coeff_modulus_size() << endl;  // 25
    }
}
