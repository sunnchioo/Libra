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

// #include "../../utils/utils.h"
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

/**
 * 第一次 sgn 没有问题，第二次出问题，可能是密文误差的原因，解密之后再填进去是可以正常计算的
 * 可以调节 scale 和 q 的大小来解决这个问题，
 */
void Compare(CKKSEvaluator &ckks, PhantomCiphertext &cipher0, PhantomCiphertext &cipher1, std::vector<PhantomCiphertext> &bool_ct) {
    // PhantomPlaintext delta;
    std::vector<double> mask(ckks.encoder.slot_count(), 0.0);
    std::fill(mask.data(), mask.data() + 4, 1.0 / 8.0);

    ckks.evaluator.sub(cipher0, cipher1, bool_ct[0]);
    // ckks.print_decrypted_ct(bool_ct[0], 10, "sub");
    // std::cout << "sub scale 0: " << bool_ct[0].scale() << std::endl; // 7.03688e+13

    // ckks.encoder.encode(ckks.init_vec_with_value(1.0 / 8.5), bool_ct[0].params_id(), bool_ct[0].scale(), delta);
    // ckks.evaluator.multiply_plain_inplace(bool_ct[0], delta);
    ckks.evaluator.multiply_vector_inplace_reduced_error(bool_ct[0], mask);

    // ckks.evaluator.multiply_const_inplace(bool_ct[0], 1.0 / 8); 使用这个会溢出

    ckks.evaluator.rescale_to_next_inplace(bool_ct[0]);
    // ckks.evaluator.mod_switch_to_next_inplace(bool_ct[0]);
    // ckks.print_decrypted_ct(bool_ct[0], 10, "sub normal");

    // ckks.re_encrypt(bool_ct[0]);
    // std::cout << "sub scale 1: " << bool_ct[0].scale() << std::endl; // 7.03688e+13
    auto sgn = ckks.sgn_eval(bool_ct[0], 2, 2);  // 2: 16; 3:25; 4: 32
    // ckks.print_decrypted_ct(sgn, 10, "sgn");

    ckks.evaluator.add_const(sgn, -0.5, bool_ct[0]);
    ckks.evaluator.multiply_const_inplace(bool_ct[0], -1.0);
    ckks.evaluator.rescale_to_next_inplace(bool_ct[0]);

    ckks.evaluator.add_const(sgn, 0.5, bool_ct[1]);
    ckks.evaluator.multiply_const_inplace(bool_ct[1], 1.0);
    ckks.evaluator.rescale_to_next_inplace(bool_ct[1]);
}

/**
 * (points - centers)^2
 */
void EuclideanDistanceMultiCenters(CKKSEvaluator &ckks, std::vector<PhantomCiphertext> &points_ct, std::vector<PhantomCiphertext> &centers_ct,
                                   long points, long dim, long centers, std::vector<std::vector<PhantomCiphertext>> &distance_cipher, std::vector<PhantomCiphertext> &merge_distance_cipher) {
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
            ckks.evaluator.multiply_vector_inplace_reduced_error(distance_cipher[icenters][ipoints], mask);  // get first sum
            ckks.evaluator.rescale_to_next_inplace(distance_cipher[icenters][ipoints]);
        }
    }
    // std::cout << "mask. " << std::endl;
    // for (size_t ipoints = 0; ipoints < points; ipoints++) {
    //     for (size_t icenters = 0; icenters < centers; icenters++) {
    //         ckks.print_decrypted_ct(distance_cipher[icenters][ipoints], 10, "mask distance " + std::to_string(ipoints) + " " + std::to_string(icenters));
    //     }
    // }

    // merge: same point to defferent center
    for (size_t ipoints = 0; ipoints < points; ipoints++) {
        merge_distance_cipher[ipoints] = distance_cipher[0][ipoints];
        for (size_t icenters = 1; icenters < centers; icenters++) {
            ckks.evaluator.rotate_vector(distance_cipher[icenters][ipoints], -icenters, *(ckks.galois_keys), temp);
            ckks.evaluator.add_inplace(merge_distance_cipher[ipoints], temp);
        }

        // copy
        ckks.evaluator.rotate_vector(merge_distance_cipher[ipoints], -centers, *(ckks.galois_keys), temp);
        ckks.evaluator.add_inplace(merge_distance_cipher[ipoints], temp);
    }

    // std::cout << "merge. " << std::endl;
    // for (size_t ipoints = 0; ipoints < points; ipoints++) {
    //     ckks.print_decrypted_ct(merge_distance_cipher[ipoints], 10, "distance merge " + std::to_string(ipoints));
    // }
    // exit(0);
}

void EuclideanDistanceMultiCenters(CKKSEvaluator &ckks, std::vector<PhantomCiphertext> &points_ct, std::vector<PhantomCiphertext> &centers_ct,
                                   long points, long dim, long centers, std::vector<PhantomCiphertext> &merge_distance_cipher) {
    PhantomCiphertext distance_temp, inner_temp;

    size_t slot_count = ckks.encoder.slot_count();
    std::vector<double> mask(slot_count, 0.0);
    mask[0] = 1.0;

    for (size_t ipoints = 0; ipoints < points; ipoints++) {
        for (size_t icenters = 0; icenters < centers; icenters++) {
            ckks.evaluator.sub(points_ct[ipoints], centers_ct[icenters], distance_temp);
            ckks.evaluator.multiply_inplace_reduced_error(distance_temp, distance_temp, *(ckks.relin_keys));
            ckks.evaluator.rescale_to_next_inplace(distance_temp);

            // inner sum
            for (size_t idim = 0; idim < static_cast<size_t>(std::log2(dim)) + 1; idim++) {
                ckks.evaluator.rotate_vector(distance_temp, 1 << idim, *(ckks.galois_keys), inner_temp);
                ckks.evaluator.add(distance_temp, inner_temp, distance_temp);
            }

            // mask
            ckks.evaluator.multiply_vector_inplace_reduced_error(distance_temp, mask);  // get first sum
            ckks.evaluator.rescale_to_next_inplace(distance_temp);

            // merge: same point to defferent center
            if (icenters == 0) {
                merge_distance_cipher[ipoints] = distance_temp;
            } else {
                ckks.evaluator.rotate_vector_inplace(distance_temp, -icenters, *(ckks.galois_keys));
                ckks.evaluator.add_inplace(merge_distance_cipher[ipoints], distance_temp);
            }
        }

        // copy
        ckks.evaluator.rotate_vector(merge_distance_cipher[ipoints], -centers, *(ckks.galois_keys), distance_temp);
        ckks.evaluator.add_inplace(merge_distance_cipher[ipoints], distance_temp);
    }
}

/**
 * compare distance
 * distance: (centers, points)
 * 当 centers == 2 时，直接相减，每个中心对用所有的点，点的密文为 mask(0/1)
 * 当 centers == 4 时，同态排序
 */
void CompareDistanceMultiCenters(CKKSEvaluator &ckks, Bootstrapper &bootstrapper, std::vector<PhantomCiphertext> &distance, long centers, long points,
                                 std::vector<PhantomCiphertext> &centers_id_ct) {
    long depth = static_cast<long>(std::log2(centers));
    std::vector<std::vector<PhantomCiphertext>> trans_matrix(depth, std::vector<PhantomCiphertext>(3));
    PhantomCiphertext temp, sum;
    std::vector<PhantomCiphertext> bool_temp(2);
    std::vector<PhantomCiphertext> bool_half(3);

    std::vector<std::vector<double>> mask(3, std::vector<double>(ckks.encoder.slot_count(), 0.0));
    std::vector<int> rotate_step(3, 0.0);

    // centers should be power of 2
    for (size_t iterpoints = 0; iterpoints < points; iterpoints++) {
        for (size_t idepth = 0; idepth < depth; idepth++) {
            if (idepth == 0 || idepth == depth - 1) {
                rotate_step[0] = 0;
                rotate_step[1] = 1;
                rotate_step[2] = centers - 1;
            } else {
                rotate_step[0] = 0;
                rotate_step[1] = (1 << idepth) - 1;
                rotate_step[2] = (1 << idepth) + 1;
            }

            // bool matrix
            ckks.evaluator.rotate_vector(distance[iterpoints], (1 << (idepth + 1)) - 1, *(ckks.galois_keys), temp);
            // ckks.print_decrypted_ct(distance[iterpoints], 10, "--dis0");
            // ckks.print_decrypted_ct(temp, 10, "--dis1");
            Compare(ckks, distance[iterpoints], temp, bool_temp);  // 18 + 3
            // std::cout << "after compare level: " << bool_temp[0].coeff_modulus_size() << " chain: " << bool_temp[0].chain_index() << std::endl;
            // ckks.print_decrypted_ct(bool_temp[0], 10, "bool_temp[0]");
            // ckks.print_decrypted_ct(bool_temp[1], 10, "bool_temp[1]");

            // exit(0);

            // set mask
            std::fill(mask[0].data(), mask[0].data() + centers, 0.0);  // all
            std::fill(mask[1].data(), mask[1].data() + centers, 0.0);
            std::fill(mask[2].data(), mask[2].data() + centers, 0.0);
            for (size_t icenters = 0; icenters < centers; icenters += (2 * (1 << (idepth + 1)))) {
                mask[0][icenters] = 1.0;
                if ((1 << (idepth + 1)) < centers) {
                    mask[0][icenters + (1 << (idepth + 1))] = 1.0;
                }

                mask[1][icenters] = 1.0;
                if ((1 << (idepth + 1)) < centers) {
                    mask[2][icenters + (1 << (idepth + 1))] = 1.0;
                }
            }

            // trans 0: 上半部分升序，下半部分降序
            ckks.evaluator.multiply_vector_reduced_error(bool_temp[0], mask[1], bool_half[0]);
            ckks.evaluator.rescale_to_next_inplace(bool_half[0]);
            ckks.evaluator.multiply_vector_reduced_error(bool_temp[1], mask[2], bool_half[1]);
            ckks.evaluator.rescale_to_next_inplace(bool_half[1]);
            ckks.evaluator.add(bool_half[0], bool_half[1], bool_half[2]);
            // ckks.print_decrypted_ct(bool_half[0], 10, "--bool masked[0]");
            // ckks.print_decrypted_ct(bool_half[1], 10, "--bool masked[1]");
            // ckks.print_decrypted_ct(bool_half[2], 10, "--bool masked[2]");

            ckks.evaluator.rotate_vector(bool_half[2], -((1 << (idepth + 1)) - 1), *(ckks.galois_keys), trans_matrix[idepth][0]);
            ckks.evaluator.add_inplace_reduced_error(trans_matrix[idepth][0], bool_half[2]);
            // ckks.print_decrypted_ct(trans_matrix[idepth][0], 10, "----trans_matrix 0");

            int trans_bool_step = 0;
            if (idepth == depth - 1) {
                trans_bool_step = -((1 << (idepth + 1)) - 1);

                ckks.evaluator.multiply_vector_reduced_error(bool_temp[0], mask[2], bool_half[0]);
                ckks.evaluator.rescale_to_next_inplace(bool_half[0]);
                ckks.evaluator.multiply_vector_reduced_error(bool_temp[1], mask[1], bool_half[1]);
                ckks.evaluator.rescale_to_next_inplace(bool_half[1]);

                ckks.evaluator.rotate_vector(bool_half[1], trans_bool_step, *(ckks.galois_keys), trans_matrix[idepth][1]);
                trans_matrix[idepth][2] = bool_half[1];

                // ckks.print_decrypted_ct(trans_matrix[idepth][1], 10, "----trans_matrix 1");
                // ckks.print_decrypted_ct(trans_matrix[idepth][2], 10, "----trans_matrix 2");
            } else {
                // trans 1: trans 0 的反
                ckks.evaluator.multiply_vector_reduced_error(bool_temp[0], mask[2], bool_half[0]);
                ckks.evaluator.rescale_to_next_inplace(bool_half[0]);
                ckks.evaluator.multiply_vector_reduced_error(bool_temp[1], mask[1], bool_half[1]);
                ckks.evaluator.rescale_to_next_inplace(bool_half[1]);
                ckks.evaluator.add(bool_half[0], bool_half[1], trans_matrix[idepth][1]);

                // ckks.print_decrypted_ct(bool_half[0], 10, "bool_half 1");
                // ckks.print_decrypted_ct(bool_half[1], 10, "bool_half 1");
                // ckks.print_decrypted_ct(trans_matrix[idepth][1], 10, "----trans_matrix 1");

                // trans 2: trans 1 旋转
                int trans_1_step = -1;
                if (depth > 0 && depth < depth - 1) {
                    trans_1_step = -rotate_step[1];
                }
                ckks.evaluator.rotate_vector(trans_matrix[idepth][1], trans_1_step, *(ckks.galois_keys), trans_matrix[idepth][2]);
                // ckks.print_decrypted_ct(trans_matrix[idepth][2], 10, "----trans_matrix 2");
            }
            // exit(0);
            // 0
            // ckks.evaluator.multiply_vector_reduced_error(bool_temp[0], mask[0], trans_matrix[idepth][0]);
            // ckks.evaluator.rescale_to_next_inplace(trans_matrix[idepth][0]);
            // ckks.print_decrypted_ct(trans_matrix[idepth][0], 10, "trans_matrix[idepth][0]");

            // ckks.evaluator.rotate_vector(trans_matrix[idepth][0], -((1 << (idepth + 1)) - 1), *(ckks.galois_keys), temp);
            // ckks.evaluator.add_inplace_reduced_error(trans_matrix[idepth][0], temp);
            // ckks.print_decrypted_ct(trans_matrix[idepth][0], 10, "trans_matrix[idepth][0]");

            // 1
            // ckks.evaluator.multiply_vector_reduced_error(bool_temp[1], mask[1], trans_matrix[idepth][1]);
            // ckks.evaluator.rescale_to_next_inplace(trans_matrix[idepth][1]);
            // ckks.print_decrypted_ct(trans_matrix[idepth][1], 10, "trans_matrix[idepth][1]");

            // 2
            // ckks.evaluator.rotate_vector(bool_temp[1], -(1 << idepth), *(ckks.galois_keys), trans_matrix[idepth][2]);
            // ckks.print_decrypted_ct(trans_matrix[idepth][2], 10, "trans_matrix[idepth][2]");

            // std::cout << "----trans_matrix level: " << trans_matrix[idepth][0].coeff_modulus_size() << " chain: " << trans_matrix[idepth][0].chain_index() << std::endl; // 2

            // update distance
            // ckks.print_decrypted_ct(distance[iterpoints], 10, "distance[iterpoints] before");
            ckks.evaluator.rotate_vector(distance[iterpoints], rotate_step[0], *(ckks.galois_keys), sum);
            // ckks.print_decrypted_ct(distance[iterpoints], 10, "distance[iterpoints] after");
            ckks.evaluator.multiply_inplace_reduced_error(sum, trans_matrix[idepth][0], *(ckks.relin_keys));
            ckks.evaluator.rescale_to_next_inplace(sum);
            // ckks.print_decrypted_ct(sum, 10, "sum0");

            ckks.evaluator.rotate_vector(distance[iterpoints], rotate_step[1], *(ckks.galois_keys), temp);
            ckks.evaluator.multiply_inplace_reduced_error(temp, trans_matrix[idepth][1], *(ckks.relin_keys));
            ckks.evaluator.rescale_to_next_inplace(temp);
            ckks.evaluator.add_inplace_reduced_error(sum, temp);
            // ckks.print_decrypted_ct(sum, 10, "sum1");

            ckks.evaluator.rotate_vector(distance[iterpoints], rotate_step[2], *(ckks.galois_keys), temp);  // 优化点：旋转消除
            ckks.evaluator.multiply_inplace_reduced_error(temp, trans_matrix[idepth][2], *(ckks.relin_keys));
            ckks.evaluator.rescale_to_next_inplace(temp);
            ckks.evaluator.add_inplace_reduced_error(sum, temp);

            distance[iterpoints] = sum;
            // ckks.print_decrypted_ct(distance[iterpoints], 10, "distance " + std::to_string(iterpoints) + " " + std::to_string(idepth));
            // exit(0);
            // std::cout << "----distance[iterpoints] level: " << distance[iterpoints].coeff_modulus_size() << " chain: " << distance[iterpoints].chain_index() << std::endl; // 1

            // std::memset(mask.data(), 1.0, centers * sizeof(double));
            std::fill(mask[0].begin(), mask[0].begin() + centers, 1.0);
            // std::fill(mask[1].begin(), mask[1].begin() + centers, 1.0);
            // std::cout << "mask: " << mask[0] << " " << mask[1] << " " << mask[centers] << " " << mask[centers + 1] << std::endl;
            ckks.evaluator.multiply_vector_inplace_reduced_error(distance[iterpoints], mask[0]);
            ckks.evaluator.rescale_to_next_inplace(distance[iterpoints]);
            // ckks.print_decrypted_ct(distance[iterpoints], 10, "distance mask" + std::to_string(iterpoints) + " " + std::to_string(idepth));

            ckks.evaluator.rotate_vector(distance[iterpoints], -centers, *(ckks.galois_keys), temp);
            ckks.evaluator.add_inplace_reduced_error(distance[iterpoints], temp);

            // ckks.print_decrypted_ct(distance[iterpoints], 10, "distance " + std::to_string(iterpoints) + " " + std::to_string(idepth));
            // std::cout << "----distance[iterpoints] level: " << distance[iterpoints].coeff_modulus_size() << " chain: " << distance[iterpoints].chain_index() << std::endl; // 8

            // boot
            if (distance[iterpoints].coeff_modulus_size() > 1) {
                size_t count = distance[iterpoints].coeff_modulus_size() - 1;
                for (size_t icount = 0; icount < count; icount++) {
                    ckks.evaluator.mod_switch_to_next_inplace(distance[iterpoints]);
                }
            }
            bootstrapper.bootstrap_3(temp, distance[iterpoints]);
            distance[iterpoints] = temp;
            // ckks.print_decrypted_ct(distance[iterpoints], 10, "distance after boot");

            // exit(0);
        }

        // ckks.print_decrypted_ct(distance[iterpoints], 10, "distance 0");
        // exit(0);

        // std::cout << "get id..." << std::endl;
        // get centers id
        for (size_t idepth = 0; idepth < depth; idepth++) {
            size_t curr_depth = depth - idepth - 1;

            if (curr_depth == 0 || curr_depth == depth - 1) {
                rotate_step[0] = 0;
                rotate_step[1] = 1;
                rotate_step[2] = centers - 1;
            } else {
                rotate_step[0] = 0;
                rotate_step[1] = (1 << curr_depth) - 1;
                rotate_step[2] = (1 << curr_depth) + 1;
            }

            // 0
            ckks.evaluator.rotate_vector(centers_id_ct[iterpoints], rotate_step[0], *(ckks.galois_keys), sum);
            ckks.evaluator.multiply_inplace_reduced_error(sum, trans_matrix[curr_depth][0], *(ckks.relin_keys));
            ckks.evaluator.rescale_to_next_inplace(sum);
            // ckks.print_decrypted_ct(sum, 10, "----after trans0");
            // 1
            ckks.evaluator.rotate_vector(centers_id_ct[iterpoints], rotate_step[1], *(ckks.galois_keys), temp);
            ckks.evaluator.multiply_inplace_reduced_error(temp, trans_matrix[curr_depth][1], *(ckks.relin_keys));
            ckks.evaluator.rescale_to_next_inplace(temp);
            ckks.evaluator.add_inplace_reduced_error(sum, temp);
            // ckks.print_decrypted_ct(sum, 10, "----after trans1");

            // 2
            ckks.evaluator.rotate_vector(centers_id_ct[iterpoints], rotate_step[2], *(ckks.galois_keys), temp);
            ckks.evaluator.multiply_inplace_reduced_error(temp, trans_matrix[curr_depth][2], *(ckks.relin_keys));
            ckks.evaluator.rescale_to_next_inplace(temp);
            ckks.evaluator.add_inplace_reduced_error(sum, temp);
            // ckks.print_decrypted_ct(sum, 10, "----after trans2");

            // reform centers_id
            ckks.evaluator.rotate_vector(sum, -centers, *(ckks.galois_keys), centers_id_ct[iterpoints]);
            ckks.evaluator.add_inplace_reduced_error(centers_id_ct[iterpoints], sum);
            // centers_id_ct[iterpoints] = sum;
            // std::cout << "----centers_id_ct[iterpoints] level: " << centers_id_ct[iterpoints].coeff_modulus_size() << " chain: " << centers_id_ct[iterpoints].chain_index() << std::endl;
        }
        // ckks.print_decrypted_ct(centers_id_ct[iterpoints], 10, "------center id " + std::to_string(iterpoints));
        // exit(0);
    }
}

// center = (points - center) * bool + center
void UpdateCenters(CKKSEvaluator &ckks, std::vector<PhantomCiphertext> &center_cipher, std::vector<PhantomCiphertext> &points_cipher,
                   long centers, long points, long dim,
                   std::vector<PhantomCiphertext> &centers_id_ct) {
    std::vector<double> mask(ckks.encoder.slot_count(), 0.0);
    PhantomCiphertext bool_flag;
    PhantomCiphertext temp, sum;

    for (size_t icenters = 0; icenters < centers; icenters++) {
        for (size_t ipoints = 0; ipoints < points; ipoints++) {
            // get mask
            std::fill(mask.data(), mask.data() + centers, 0.0);
            mask[icenters] = 1.0;

            // set bool flag
            ckks.evaluator.multiply_vector_reduced_error(centers_id_ct[ipoints], mask, bool_flag);
            ckks.evaluator.rescale_to_next_inplace(bool_flag);

            if (icenters > 0) {
                ckks.evaluator.rotate_vector_inplace(bool_flag, icenters, *(ckks.galois_keys));
            }

            for (size_t icopy = 0; icopy < static_cast<size_t>(std::log2(dim)); icopy++) {
                ckks.evaluator.rotate_vector(bool_flag, -(1 << icopy), *(ckks.galois_keys), temp);
                ckks.evaluator.add_inplace_reduced_error(bool_flag, temp);
            }

            // add
            if (ipoints == 0) {
                ckks.evaluator.sub_reduced_error(points_cipher[ipoints], center_cipher[icenters], temp);
                ckks.evaluator.multiply_reduced_error(bool_flag, temp, *(ckks.relin_keys), sum);
                ckks.evaluator.rescale_to_next_inplace(sum);
                ckks.evaluator.add_inplace_reduced_error(sum, center_cipher[icenters]);
            } else {
                ckks.evaluator.sub_reduced_error(points_cipher[ipoints], center_cipher[icenters], temp);
                ckks.evaluator.multiply_reduced_error(bool_flag, temp, *(ckks.relin_keys), temp);
                ckks.evaluator.rescale_to_next_inplace(temp);
                ckks.evaluator.add_inplace_reduced_error(temp, center_cipher[icenters]);
                ckks.evaluator.add_inplace_reduced_error(sum, temp);
            }
        }

        // update center
        ckks.evaluator.multiply_const_inplace(sum, 1.0 / points);
        ckks.evaluator.rescale_to_next_inplace(sum);
        center_cipher[icenters] = sum;

        // ckks.print_decrypted_ct(center_cipher[icenters], 10, "center_cipher " + std::to_string(icenters));
        // std::cout << "----center_cipher[icenters] level: " << center_cipher[icenters].coeff_modulus_size() << " chain: " << center_cipher[icenters].chain_index() << std::endl;

        // exit(0);
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

    // (41,7)(39, 6)-->comp(3,3) or comp(4,4); (25,4)-->comp(2,2)
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

    // set input data
    std::vector<std::vector<double>> data = FileIO<double>::LoadCSV3D("/mnt/data2/home/syt/data/Libra/benchmark/app/data/kmeans/16data2cent.csv");  // (points, dim)
    if (slot_count < data.size()) {
        throw std::logic_error("Error: Get Out Input Length.");
    }
    long points = data.size();
    long dim = data[0].size();
    long centers = 2;

    // long points = 32;
    // long dim = 4096;
    // long centers = 1;

    std::cout << points << " points. " << dim << " dim. " << centers << " centers. " << std::endl;

    /**
     * point    dim0    dim1    dim...
     *     0    0.1     0.1
     *     1    0.2     0.2
     *     2    0.3     0.3
     */
    std::vector<std::vector<double>> input(points, std::vector<double>(slot_count, 0.0));    // input: 一个输入一个密文，维度为一个密文里
    std::vector<std::vector<double>> center(centers, std::vector<double>(slot_count, 0.1));  // center: 一个 centers 一个密文
    for (size_t i = 0; i < points; i++) {
        std::copy(data[i].begin(), data[i].begin() + dim, input[i].begin());
    }
    print_matrix(input, "input");

    // init center
    if (centers == 2) {
        std::copy(input[0].begin(), input[0].begin() + dim, center[0].begin());
        std::copy(input[points - 1].begin(), input[points - 1].begin() + dim, center[1].begin());
    } else {
        for (size_t i = 0; i < centers; i++) {
            std::copy(input[i].begin(), input[i].begin() + dim, center[i].begin());
        }
    }
    print_matrix(center, "init center");

    // Create points and centers cipher
    PhantomPlaintext plain;
    // std::vector<std::vector<PhantomCiphertext>> distance_matrix_cipher(centers, std::vector<PhantomCiphertext>(points));
    std::vector<PhantomCiphertext> distance_cipher(points);
    std::vector<PhantomCiphertext> points_cipher(points);
    std::vector<PhantomCiphertext> center_cipher(centers), center_new_cipher(centers), centers_id_cipher(points);

    std::vector<PhantomCiphertext> bool_cipher(points);

    for (size_t i = 0; i < points; i++) {
        ckks_evaluator.encoder.encode(input[i], boot_level + 1, scale, plain);
        ckks_evaluator.encryptor.encrypt(plain, points_cipher[i]);
    }
    for (size_t i = 0; i < centers; i++) {
        ckks_evaluator.encoder.encode(center[i], boot_level + 1, scale, plain);
        ckks_evaluator.encryptor.encrypt(plain, center_cipher[i]);
    }

    std::vector<double> center_id(slot_count, 0.0);
    center_id[0] = 1.0;
    center_id[centers] = 1.0;

    for (size_t ipoints = 0; ipoints < points; ipoints++) {
        ckks_evaluator.encoder.encode(center_id, boot_level + 1, scale, plain);
        ckks_evaluator.encryptor.encrypt(plain, centers_id_cipher[ipoints]);
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

    float total_time = 0.0;
    int iter_count = 1;
    for (size_t iter = 0; iter < iter_count; iter++) {
        cout << "iter: " << iter << endl;

        cout << "begin level: " << points_cipher[0].coeff_modulus_size() << endl;  //

        // (Euclidean distance)^2
        {
            CUDATimer timer("Euclidean distance", s);
            timer.start();
            // EuclideanDistance(ckks_evaluator, points_cipher, center_cipher, points, dim, centers, distance_matrix_cipher); // 2 level
            // EuclideanDistanceMultiCenters(ckks_evaluator, points_cipher, center_cipher, points, dim, centers, distance_matrix_cipher, distance_cipher); // 2 level
            EuclideanDistanceMultiCenters(ckks_evaluator, points_cipher, center_cipher, points, dim, centers, distance_cipher);  // 2 level
            timer.stop();
            total_time += timer.get_mean_time();
        }
        // for (size_t ipoints = 0; ipoints < points; ipoints++) {
        //     ckks_evaluator.print_decrypted_ct(distance_cipher[ipoints], 10, "distance 0" + std::to_string(ipoints));
        // }
        cout << "distence level: " << distance_cipher[0].coeff_modulus_size() << endl;  //
        cout << "distence chain: " << distance_cipher[0].chain_index() << endl;         //
        // exit(0);

        // compare
        {
            CUDATimer timer("CompareDistance", s);
            timer.start();
            CompareDistanceMultiCenters(ckks_evaluator, bootstrapper, distance_cipher, centers, points, centers_id_cipher);  // 18 level
            timer.stop();
            total_time += timer.get_mean_time();
        }
        // for (size_t i = 0; i < points; i++) {
        //     ckks_evaluator.print_decrypted_ct(bool_cipher[0][i], 10, "bool0");
        // }
        cout << "bool_cipher level: " << centers_id_cipher[0].coeff_modulus_size() << endl;  //
        cout << "bool_cipher chain: " << centers_id_cipher[0].chain_index() << endl;         //

        // exit(0);
        // update center
        {
            CUDATimer timer("UpdateCenters", s);
            timer.start();
            UpdateCenters(ckks_evaluator, center_cipher, points_cipher, centers, points, dim, centers_id_cipher);  // 2 level
            timer.stop();
            total_time += timer.get_mean_time();
        }
        // for (size_t icenters = 0; icenters < centers; icenters++) {
        //     ckks_evaluator.print_decrypted_ct(center_cipher[icenters], 10, "center " + std::to_string(icenters));
        // }
        cout << "center_cipher level: " << center_cipher[0].coeff_modulus_size() << endl;  // 3
        // center 0 : 0.0600291 - 0.0599984 - 2.57949e-10 7.29065e-10 1.20807e-10 8.34692e-11 - 7.82883e-10 2.55898e-11 1.14999e-09 - 2.61322e-10
        // center 1 : 2.16995 - 0.0800007 - 4.33218e-10 2.52346e-10 1.40643e-09 - 7.55927e-10 8.60584e-11 - 3.19693e-10 - 1.41437e-09 6.21378e-10

        // boot
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
            total_time += timer.get_mean_time();
        }
        for (size_t icenters = 0; icenters < centers; icenters++) {
            center_cipher[icenters] = center_new_cipher[icenters];
        }
        for (size_t icenters = 0; icenters < centers; icenters++) {
            ckks_evaluator.print_decrypted_ct(center_cipher[icenters], 10, "boot center " + std::to_string(icenters));
        }
        cout << "center_cipher level: " << center_cipher[0].coeff_modulus_size() << endl;  // 25
    }
    std::cout << "sum time: " << total_time / iter_count << std::endl;
}
