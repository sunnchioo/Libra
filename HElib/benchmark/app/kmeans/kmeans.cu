// 1. 选择 K 个中心点
// 2. K 作为初始化中心
// 3. 计算每个点到 K 个中心点的距离
// 4. 将每个点分配到最近的中心点
// 5. 更新 K 个中心点
// 6. 重复步骤 3-5，直到中心点不再变化
// 7. 返回 K 个中心点

// #include "../utils/utils.h"
#include "bootstrapping/Bootstrapper.cuh"
#include "ckks_evaluator.cuh"
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
void EuclideanDistance(CKKSEvaluator &ckks, std::vector<PhantomCiphertext> &points_ct, std::vector<PhantomCiphertext> &centers_ct, long dim, PhantomCiphertext &distance_cipher) {
    std::vector<PhantomCiphertext> dist_ct(dim);
    for (size_t i = 0; i < dim; i++) {
        ckks.evaluator.sub(points_ct[i], centers_ct[i], dist_ct[i]);
        ckks.evaluator.square(dist_ct[i], dist_ct[i]);
        // std::cout << "square" << std::endl;
        ckks.evaluator.relinearize_inplace(dist_ct[i], *ckks.relin_keys);
        // std::cout << "relinearize_inplace" << std::endl;

        ckks.evaluator.rescale_to_next_inplace(dist_ct[i]);
    }
    for (size_t i = 1; i < dim; i++) {
        ckks.evaluator.add_inplace(dist_ct[0], dist_ct[i]);
    }

    distance_cipher = dist_ct[0];
}

// if centers == 2
void CompareDistance(CKKSEvaluator &ckks, PhantomCiphertext &distance, long centers, long points, std::vector<PhantomCiphertext> &bool_ct) {
    if (ckks.encoder.slot_count() < points * centers) {
        throw std::logic_error("Error: points * centers > slot_count");
    }

    std::vector<PhantomCiphertext> centers_distence(centers);
    PhantomCiphertext sub;
    PhantomPlaintext delta;

    std::vector<double> mask(ckks.encoder.slot_count(), 0.0);
    std::fill(mask.begin(), mask.begin() + points, 1);

    std::vector<double> mask_data = mask;

    ckks.evaluator.multiply_vector_reduced_error(distance, mask, centers_distence[0]);
    // ckks.evaluator.rescale_to_next_inplace(centers_distence[0]);
    for (size_t i = 1; i < centers; i++) {
        std::rotate(mask.begin(), mask.end() - points, mask.end());
        ckks.evaluator.multiply_vector_reduced_error(distance, mask, centers_distence[i]);
        ckks.evaluator.rotate_vector_inplace(centers_distence[i], points, *(ckks.galois_keys));
        // ckks.evaluator.rescale_to_next_inplace(centers_distence[i]);
    }

    // for (size_t i = 0; i < centers; i++) {
    //     ckks.print_decrypted_ct(centers_distence[i], 10, "centers");
    // }

    ckks.evaluator.sub(centers_distence[0], centers_distence[1], sub); // if centers == 2 距离类别2近的点
    ckks.evaluator.rescale_to_next_inplace(sub);
    // ckks.print_decrypted_ct(sub, 10, "sub");
    // std::cout << "sub scale: " << sub.scale() << " ckks scale: " << ckks.scale << std::endl;

    ckks.encoder.encode(ckks.init_vec_with_value(1.0 / 8.5), sub.params_id(), sub.scale(), delta);
    ckks.evaluator.multiply_plain_inplace(sub, delta);
    ckks.evaluator.rescale_to_next_inplace(sub);
    // ckks.print_decrypted_ct(sub, 10, "sub");

    auto sgn = ckks.sgn_eval(sub, 2, 2); // 因为是通过q来管理scale，所以输入需要和初始的scale一致
    // ckks.print_decrypted_ct(sgn, 10, "sgn eval");
    // std::cout << "sub scale: " << sgn.scale() << std::endl;
    // cout << "bool_ct[1] level: " << bool_ct[1].coeff_modulus_size() << endl;

    ckks.evaluator.add_const(sgn, 0.5, bool_ct[1]);
    // ckks.print_decrypted_ct(bool_ct[1], 10, "bool1");
    ckks.evaluator.multiply_vector_reduced_error(bool_ct[1], mask_data, bool_ct[1]);
    ckks.evaluator.rescale_to_next_inplace(bool_ct[1]);
    // cout << "bool_ct[1] level: " << bool_ct[1].coeff_modulus_size() << endl;
    // std::cout << "bool_ct[1] scale: " << bool_ct[1].scale() << std::endl;

    ckks.evaluator.add_const(sgn, -0.5, bool_ct[0]);
    // ckks.print_decrypted_ct(bool_ct[0], 10, "bool0");
    std::fill(mask_data.begin(), mask_data.begin() + points, -1);
    ckks.evaluator.multiply_vector_reduced_error(bool_ct[0], mask_data, bool_ct[0]);
    ckks.evaluator.rescale_to_next_inplace(bool_ct[0]);

    // ckks.print_decrypted_ct(bool_ct[0], 10, "bool0");
    // ckks.print_decrypted_ct(bool_ct[1], 10, "bool1");

    // cout << "bool level: " << bool_ct[0].coeff_modulus_size() << endl;
    // std::cout << "bool_ct[0] scale: " << bool_ct[0].scale() << std::endl;
}

// all in rotate
void UpdateCenters(CKKSEvaluator &ckks, std::vector<PhantomCiphertext> &center_cipher, std::vector<PhantomCiphertext> &points_cipher,
                   long centers, long points, long dim,
                   std::vector<PhantomCiphertext> &bool_ct) {
    // std::cout << "center begin level: " << center_cipher[0].coeff_modulus_size() << std::endl;

    std::vector<std::vector<PhantomCiphertext>> center_points(centers, std::vector<PhantomCiphertext>(dim)); // 每个center对应的points
    PhantomCiphertext temp;
    std::vector<double> mask(ckks.encoder.slot_count(), 0.0);
    std::fill(mask.begin(), mask.begin() + points, 1);

    // points * bool + center * (1-bool) = (points - center) * bool + center
    for (size_t idim = 0; idim < dim; idim++) {
        ckks.evaluator.sub_reduced_error(points_cipher[idim], center_cipher[idim], center_points[0][idim]);
        ckks.evaluator.multiply_inplace_reduced_error(center_points[0][idim], bool_ct[0], *(ckks.relin_keys));
        ckks.evaluator.rescale_to_next_inplace(center_points[0][idim]);
        ckks.evaluator.add_inplace_reduced_error(center_points[0][idim], center_cipher[idim]);
        ckks.evaluator.multiply_vector_inplace_reduced_error(center_points[0][idim], mask);
        ckks.evaluator.rescale_to_next_inplace(center_points[0][idim]);

        ckks.evaluator.rotate_vector(center_cipher[idim], points, *(ckks.galois_keys), temp);
        ckks.evaluator.sub_reduced_error(points_cipher[idim], temp, center_points[1][idim]);
        ckks.evaluator.multiply_inplace_reduced_error(center_points[1][idim], bool_ct[1], *(ckks.relin_keys));
        ckks.evaluator.rescale_to_next_inplace(center_points[1][idim]);
        ckks.evaluator.add_inplace_reduced_error(center_points[1][idim], temp);
        ckks.evaluator.multiply_vector_inplace_reduced_error(center_points[1][idim], mask);
        ckks.evaluator.rescale_to_next_inplace(center_points[1][idim]);

        // ckks.print_decrypted_ct(center_points[0][idim], 10, "data0");
        // ckks.print_decrypted_ct(center_points[1][idim], 10, "data1");

        // avg
        // ckks.evaluator.multiply_const_inplace(center_points[0][idim], 1.0 / points);
        // ckks.evaluator.multiply_const_inplace(center_points[1][idim], 1.0 / points);
        // ckks.evaluator.rescale_to_next_inplace(center_points[0][idim]);
        // ckks.evaluator.rescale_to_next_inplace(center_points[1][idim]);

        // ckks.print_decrypted_ct(center_points[0][idim], 10, "avg0");
        // ckks.print_decrypted_ct(center_points[1][idim], 10, "avg1"); // 0
    }

    // exit(0);

    // inner sum
    // std::cout << "rotate step: " << static_cast<size_t>(std::log2(points)) + 1 << std::endl;
    for (size_t icenters = 0; icenters < centers; icenters++) {
        for (size_t ipoints = 0; ipoints < static_cast<size_t>(std::log2(points)) + 1; ipoints++) {
            for (size_t idim = 0; idim < dim; idim++) {
                ckks.evaluator.rotate_vector(center_points[icenters][idim], 1 << ipoints, *(ckks.galois_keys), temp);
                ckks.evaluator.add(center_points[icenters][idim], temp, center_points[icenters][idim]);
            }
        }
    }
    // std::cout << "center_points scale: " << center_points[0][0].scale() << std::endl;

    // ckks.print_decrypted_ct(center_points[0][0], 10, "center0"); // 都在第一个points里 第一个就是每个组的值
    // ckks.print_decrypted_ct(center_points[1][0], 10, "center1");

    // mask; broadcast; avg.
    std::fill(mask.begin() + 1, mask.begin() + points, 0);
    for (size_t icenters = 0; icenters < centers; icenters++) {
        for (size_t idim = 0; idim < dim; idim++) {
            ckks.evaluator.multiply_vector_inplace_reduced_error(center_points[icenters][idim], mask); // get first sum
            ckks.evaluator.rescale_to_next_inplace(center_points[icenters][idim]);
        }
    }
    // std::cout << "center_points scale: " << center_points[0][0].scale() << std::endl;

    // ckks.print_decrypted_ct(center_points[0][0], 10, "center mask"); // 都在第一个points里 第一个就是每个组的值
    // ckks.print_decrypted_ct(center_points[1][0], 10, "center mask");

    // std::cout << "rotate step: " << static_cast<size_t>(std::log2(points)) + 1 << std::endl;
    for (size_t icenters = 0; icenters < centers; icenters++) {
        for (size_t ipoints = 0; ipoints < static_cast<size_t>(std::log2(points)) + 1; ipoints++) {
            for (size_t idim = 0; idim < dim; idim++) {
                ckks.evaluator.rotate_vector(center_points[icenters][idim], -(1 << ipoints), *(ckks.galois_keys), temp);
                // ckks.print_decrypted_ct(temp, 10, "center broadcast temp"); // 都在第一个points里 第一个就是每个组的值
                ckks.evaluator.add(center_points[icenters][idim], temp, center_points[icenters][idim]);
            }
        }
    }
    // ckks.print_decrypted_ct(center_points[0][0], 10, "center broadcast"); // 都在第一个points里 第一个就是每个组的值
    // ckks.print_decrypted_ct(center_points[1][0], 10, "center broadcast");

    // std::cout << "1 / points: " << 1.0 / points << std::endl;
    std::fill(mask.begin(), mask.begin() + points, 1.0 / points);
    for (size_t icenters = 0; icenters < centers; icenters++) {
        for (size_t idim = 0; idim < dim; idim++) {
            ckks.evaluator.multiply_vector_inplace_reduced_error(center_points[icenters][idim], mask);
            ckks.evaluator.rescale_to_next_inplace(center_points[icenters][idim]);
        }
    }
    // ckks.print_decrypted_ct(center_points[0][0], 10, "center up");
    // ckks.print_decrypted_ct(center_points[1][0], 10, "center up");
    // std::cout << "center_points[1][0] level: " << center_points[1][0].coeff_modulus_size() << std::endl;

    // ckks.evaluator.rotate_vector(center_points[0][0], -5, *(ckks.galois_keys), temp);
    // ckks.evaluator.rotate_vector(center_points[1][0], -5, *(ckks.galois_keys), temp);

    // shift
    for (size_t icenters = 1; icenters < centers; icenters++) {
        for (size_t idim = 0; idim < dim; idim++) {
            // std::cout << "icenters: " << icenters << " idim: " << idim << std::endl;
            ckks.evaluator.rotate_vector(center_points[icenters][idim], -static_cast<int>(points * icenters), *(ckks.galois_keys), temp);
            ckks.evaluator.add_inplace(center_points[0][idim], temp);
        }
    }

    for (size_t idim = 0; idim < dim; idim++) {
        center_cipher[idim] = center_points[0][idim];
    }
    // ckks.print_decrypted_ct(center_cipher[0], 10, "center");
    // std::cout << "center end level: " << center_cipher[0].coeff_modulus_size() << std::endl;
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

    int remaining_level = 20;
    int boot_level = 14;                            // >= subsum 1 + coefftoslot 2 + ModReduction 9 + slottocoeff 2
    int total_level = remaining_level + boot_level; // 34
    int special_prime_len = 7;

    vector<int> coeff_bit_vec;
    coeff_bit_vec.push_back(logq); // 35
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
    std::vector<std::vector<double>> data = FileIO<double>::LoadCSV3D("/mnt/data2/home/syt/data/fhec/benchmark/app/data/kmeans/5data2cent.csv");
    if (slot_count < data.size()) {
        throw std::logic_error("Error: Get Out Input Length.");
    }
    long points = data.size();
    long dim = data[0].size();
    long centers = 2;

    /**
     * point   0   1   2
     * dim0    0.1 0.2 0.3
     * dim1    0.1 0.2 0.3
     */
    std::vector<std::vector<double>> center(dim, std::vector<double>(centers, 0.0));          // center: (dim, count) count = points*centers 一个维度一个密文
    std::vector<std::vector<double>> input(dim, std::vector<double>(slot_count, 0.0));        // input: (dim, count) count = points*centers 一个维度一个密文
    std::vector<std::vector<double>> center_input(dim, std::vector<double>(slot_count, 0.0)); // center_input: (dim, count) count = points*centers 一个维度一个密文
    for (size_t i = 0; i < points; i++) {
        for (size_t j = 0; j < dim; j++) {
            input[j][i] = data[i][j];
        }
    }
    print_matrix(input, "input");

    // repermutate input
    for (size_t i = 0; i < dim; i++) {
        for (size_t j = 1; j < centers; j++) {
            std::copy(input[i].begin(), input[i].begin() + points, input[i].begin() + j * points);
        }
    }
    for (size_t i = 0; i < dim; i++) {
        center[i][0] = input[i][0];
        center[i][1] = input[i][points - 1];
    }
    // repermutate center
    for (size_t i = 0; i < dim; i++) {
        for (size_t j = 0; j < centers; j++) {
            for (size_t k = 0; k < points; k++) {
                center_input[i][j * points + k] = center[i][j];
            }
        }
    }
    print_matrix(input, "repermutate input");
    print_matrix(center_input, "repermutate center");
    // exit(0);

    // Create points and centers cipher
    PhantomCiphertext distance_cipher;
    PhantomPlaintext plain;
    std::vector<PhantomCiphertext> points_cipher(dim);
    std::vector<PhantomCiphertext> center_cipher(dim);
    std::vector<PhantomCiphertext> bool_cipher(centers);
    std::vector<PhantomCiphertext> bool_new_cipher(centers);

    for (size_t i = 0; i < dim; i++) {
        ckks_evaluator.encoder.encode(input[i], boot_level + 1, scale, plain);
        // ckks_evaluator.print_decoded_pt(plain, 10, "input");
        cudaDeviceSynchronize();
        CHECK_CUDA_LAST_ERROR();
        // std::cout << "encode" << std::endl;

        ckks_evaluator.encryptor.encrypt(plain, points_cipher[i]);
        // ckks_evaluator.print_decrypted_ct(points_cipher[i], 10, "points");
        cudaDeviceSynchronize();
        CHECK_CUDA_LAST_ERROR();
        // std::cout << "encrypt" << std::endl;
        // exit(0);

        // ckks_evaluator.print_decrypted_ct(points_cipher[i], 10);

        ckks_evaluator.encoder.encode(center_input[i], boot_level + 1, scale, plain);
        cudaDeviceSynchronize();
        CHECK_CUDA_LAST_ERROR();
        // std::cout << "encode" << std::endl;

        ckks_evaluator.encryptor.encrypt(plain, center_cipher[i]);
        cudaDeviceSynchronize();
        CHECK_CUDA_LAST_ERROR();
        // std::cout << "encrypt" << std::endl;

        // std::cout << "dim: " << i << std::endl;
        // ckks_evaluator.print_decrypted_ct(center_cipher[i], 10);
    }
    ckks_evaluator.print_decrypted_ct(points_cipher[0], 10, "points");
    ckks_evaluator.print_decrypted_ct(center_cipher[0], 10, "center");

    // exit(0);

    // std::cout << "points cipher scale: " << points_cipher[0].scale() << std::endl;

    // std::cout << "points cipher level: " << points_cipher[0].coeff_modulus_size() << std::endl;
    // std::cout << "center cipher level: " << center_cipher[0].coeff_modulus_size() << std::endl;

    // std::cout << "chain index 0: " << context.get_context_data(points_cipher[0].chain_index()).parms().coeff_modulus().size() << std::endl; // all
    // std::cout << "chain index 1: " << context.get_context_data(1).parms().coeff_modulus().size() << std::endl;                              // all - special

    auto s = phantom::util::global_variables::default_stream->get_stream();

    int iter_count = 1;
    for (size_t iter = 0; iter < iter_count; iter++) {
        // cout << "iter: " << iter << endl;

        // (Euclidean distance)^2
        {
            CUDATimer timer("Euclidean distance", s);
            timer.start();
            EuclideanDistance(ckks_evaluator, points_cipher, center_cipher, dim, distance_cipher); // 1 level
            timer.stop();
        }
        // ckks_evaluator.print_decrypted_ct(distance_cipher, 10, "distance");
        cout << "distence level: " << distance_cipher.coeff_modulus_size() << endl; // 20
        // cout << "distence chain: " << distance_cipher.chain_index() << endl;        // 15

        // compare
        {
            CUDATimer timer("CompareDistance", s);
            timer.start();
            CompareDistance(ckks_evaluator, distance_cipher, centers, points, bool_cipher); // 19 level
            timer.stop();
        }
        cout << "bool_cipher level: " << bool_cipher[0].coeff_modulus_size() << endl; // 10
        // cout << "bool_cipher chain: " << bool_cipher[0].chain_index() << endl;          // 29

        {
            CUDATimer timer("Bootstrapper", s);
            timer.start();
            for (size_t icenters = 0; icenters < centers; icenters++) {
                bootstrapper.bootstrap_3(bool_new_cipher[icenters], bool_cipher[icenters]);
            }
            timer.stop();
        }

        // ckks_evaluator.print_decrypted_ct(bool_new_cipher[0], 10, "bool0 new");
        // ckks_evaluator.print_decrypted_ct(bool_new_cipher[1], 10, "bool1 new");

        // update center
        {
            CUDATimer timer("UpdateCenters", s);
            timer.start();
            UpdateCenters(ckks_evaluator, center_cipher, points_cipher, centers, points, dim, bool_new_cipher);
            timer.stop();
        }
        for (size_t idim = 0; idim < dim; idim++) {
            ckks_evaluator.print_decrypted_ct(center_cipher[idim], 10, "center " + std::to_string(idim));
        }
        cout << "center_cipher level: " << center_cipher[0].coeff_modulus_size() << endl; // 6
        // center up : 0.0600291 0.0600291 0.0600291 0.0600291 0.0600291 - 2.15706e-12 - 1.56403e-12 - 1.19212e-12 9.91977e-12 - 9.01348e-13
        // center up : 2.16995 2.16995 2.16995 2.16995 2.16995 - 1.4757e-11 - 9.77152e-12 - 2.47664e-11 - 1.70927e-11 1.23471e-12
    }
}
