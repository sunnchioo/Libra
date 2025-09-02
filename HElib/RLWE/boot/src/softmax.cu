#include <iostream>

#include "softmax.cuh"

using namespace rlwe;

/**
 * 第一次 sgn 没有问题，第二次出问题，可能是密文误差的原因，解密之后再填进去是可以正常计算的
 * 可以调节 scale 和 q 的大小来解决这个问题，
 */
void SoftmaxEvaluator::compare(PhantomCiphertext &cipher0, PhantomCiphertext &cipher1, std::vector<PhantomCiphertext> &bool_ct) {
    // PhantomPlaintext delta;
    std::vector<double> mask(ckks->encoder.slot_count(), 0.0);
    std::fill(mask.data(), mask.data() + 4, 1.0 / 8.0);

    ckks->evaluator.sub(cipher0, cipher1, bool_ct[0]);
    // ckks->print_decrypted_ct(bool_ct[0], 10, "sub");
    // std::cout << "sub scale 0: " << bool_ct[0].scale() << std::endl; // 7.03688e+13

    // ckks->encoder.encode(ckks->init_vec_with_value(1.0 / 8.5), bool_ct[0].params_id(), bool_ct[0].scale(), delta);
    // ckks->evaluator.multiply_plain_inplace(bool_ct[0], delta);
    ckks->evaluator.multiply_vector_inplace_reduced_error(bool_ct[0], mask);

    // ckks->evaluator.multiply_const_inplace(bool_ct[0], 1.0 / 8); 使用这个会溢出

    ckks->evaluator.rescale_to_next_inplace(bool_ct[0]);
    // ckks->evaluator.mod_switch_to_next_inplace(bool_ct[0]);
    // ckks->print_decrypted_ct(bool_ct[0], 10, "sub normal");

    // ckks->re_encrypt(bool_ct[0]);
    // std::cout << "sub scale 1: " << bool_ct[0].scale() << std::endl; // 7.03688e+13
    auto sgn = ckks->sgn_eval(bool_ct[0], 2, 2); // 2: 16; 3:25; 4: 32
    // ckks->print_decrypted_ct(sgn, 10, "sgn");

    ckks->evaluator.add_const(sgn, -0.5, bool_ct[0]);
    ckks->evaluator.multiply_const_inplace(bool_ct[0], -1.0);
    ckks->evaluator.rescale_to_next_inplace(bool_ct[0]);

    ckks->evaluator.add_const(sgn, 0.5, bool_ct[1]);
    ckks->evaluator.multiply_const_inplace(bool_ct[1], 1.0);
    ckks->evaluator.rescale_to_next_inplace(bool_ct[1]);
}

void SoftmaxEvaluator::findmax(PhantomCiphertext &input, long points, PhantomCiphertext &res) {
    long depth = static_cast<long>(std::log2(points));

    std::vector<std::vector<PhantomCiphertext>> trans_matrix(depth, std::vector<PhantomCiphertext>(3));

    PhantomCiphertext temp, sum;
    std::vector<PhantomCiphertext> bool_temp(2);
    std::vector<PhantomCiphertext> bool_half(3);

    std::vector<std::vector<double>> mask(3, std::vector<double>(ckks->encoder.slot_count(), 0.0));
    std::vector<int> rotate_step(3, 0);

    for (size_t idepth = 0; idepth < depth; idepth++) {
        if (idepth == 0 || idepth == depth - 1) {
            rotate_step[0] = 0;
            rotate_step[1] = 1;
            rotate_step[2] = points - 1;
        } else {
            rotate_step[0] = 0;
            rotate_step[1] = (1 << idepth) - 1;
            rotate_step[2] = (1 << idepth) + 1;
        }

        // bool matrix
        ckks->evaluator.rotate_vector(input, (1 << (idepth + 1)) - 1, *(ckks->galois_keys), temp);
        // ckks->print_decrypted_ct(input, 10, "--dis0");
        // ckks->print_decrypted_ct(temp, 10, "--dis1");
        compare(input, temp, bool_temp); // 18 + 3
        // std::cout << "after compare level: " << bool_temp[0].coeff_modulus_size() << " chain: " << bool_temp[0].chain_index() << std::endl;
        // ckks->print_decrypted_ct(bool_temp[0], 10, "bool_temp[0]");
        // ckks->print_decrypted_ct(bool_temp[1], 10, "bool_temp[1]");

        // exit(0);

        // set mask
        std::fill(mask[0].data(), mask[0].data() + points, 0.0); // all
        std::fill(mask[1].data(), mask[1].data() + points, 0.0);
        std::fill(mask[2].data(), mask[2].data() + points, 0.0);
        for (size_t icenters = 0; icenters < points; icenters += (2 * (1 << (idepth + 1)))) {
            mask[0][icenters] = 1.0;
            if ((1 << (idepth + 1)) < points) {
                mask[0][icenters + (1 << (idepth + 1))] = 1.0;
            }

            mask[1][icenters] = 1.0;
            if ((1 << (idepth + 1)) < points) {
                mask[2][icenters + (1 << (idepth + 1))] = 1.0;
            }
        }

        // trans 0: 上半部分升序，下半部分降序
        ckks->evaluator.multiply_vector_reduced_error(bool_temp[0], mask[1], bool_half[0]);
        ckks->evaluator.rescale_to_next_inplace(bool_half[0]);
        ckks->evaluator.multiply_vector_reduced_error(bool_temp[1], mask[2], bool_half[1]);
        ckks->evaluator.rescale_to_next_inplace(bool_half[1]);
        ckks->evaluator.add(bool_half[0], bool_half[1], bool_half[2]);
        // ckks->print_decrypted_ct(bool_half[0], 10, "--bool masked[0]");
        // ckks->print_decrypted_ct(bool_half[1], 10, "--bool masked[1]");
        // ckks->print_decrypted_ct(bool_half[2], 10, "--bool masked[2]");

        ckks->evaluator.rotate_vector(bool_half[2], -((1 << (idepth + 1)) - 1), *(ckks->galois_keys), trans_matrix[idepth][0]);
        ckks->evaluator.add_inplace_reduced_error(trans_matrix[idepth][0], bool_half[2]);
        // ckks->print_decrypted_ct(trans_matrix[idepth][0], 10, "----trans_matrix 0");

        int trans_bool_step = 0;
        if (idepth == depth - 1) {
            trans_bool_step = -((1 << (idepth + 1)) - 1);

            ckks->evaluator.multiply_vector_reduced_error(bool_temp[0], mask[2], bool_half[0]);
            ckks->evaluator.rescale_to_next_inplace(bool_half[0]);
            ckks->evaluator.multiply_vector_reduced_error(bool_temp[1], mask[1], bool_half[1]);
            ckks->evaluator.rescale_to_next_inplace(bool_half[1]);

            ckks->evaluator.rotate_vector(bool_half[1], trans_bool_step, *(ckks->galois_keys), trans_matrix[idepth][1]);
            trans_matrix[idepth][2] = bool_half[1];

            // ckks->print_decrypted_ct(trans_matrix[idepth][1], 10, "----trans_matrix 1");
            // ckks->print_decrypted_ct(trans_matrix[idepth][2], 10, "----trans_matrix 2");
        } else {
            // trans 1: trans 0 的反
            ckks->evaluator.multiply_vector_reduced_error(bool_temp[0], mask[2], bool_half[0]);
            ckks->evaluator.rescale_to_next_inplace(bool_half[0]);
            ckks->evaluator.multiply_vector_reduced_error(bool_temp[1], mask[1], bool_half[1]);
            ckks->evaluator.rescale_to_next_inplace(bool_half[1]);
            ckks->evaluator.add(bool_half[0], bool_half[1], trans_matrix[idepth][1]);

            // ckks->print_decrypted_ct(bool_half[0], 10, "bool_half 1");
            // ckks->print_decrypted_ct(bool_half[1], 10, "bool_half 1");
            // ckks->print_decrypted_ct(trans_matrix[idepth][1], 10, "----trans_matrix 1");

            // trans 2: trans 1 旋转
            int trans_1_step = -1;
            if (depth > 0 && depth < depth - 1) {
                trans_1_step = -rotate_step[1];
            }
            ckks->evaluator.rotate_vector(trans_matrix[idepth][1], trans_1_step, *(ckks->galois_keys), trans_matrix[idepth][2]);
            // ckks->print_decrypted_ct(trans_matrix[idepth][2], 10, "----trans_matrix 2");
        }
        // exit(0);
        // 0
        // ckks->evaluator.multiply_vector_reduced_error(bool_temp[0], mask[0], trans_matrix[idepth][0]);
        // ckks->evaluator.rescale_to_next_inplace(trans_matrix[idepth][0]);
        // ckks->print_decrypted_ct(trans_matrix[idepth][0], 10, "trans_matrix[idepth][0]");

        // ckks->evaluator.rotate_vector(trans_matrix[idepth][0], -((1 << (idepth + 1)) - 1), *(ckks->galois_keys), temp);
        // ckks->evaluator.add_inplace_reduced_error(trans_matrix[idepth][0], temp);
        // ckks->print_decrypted_ct(trans_matrix[idepth][0], 10, "trans_matrix[idepth][0]");

        // 1
        // ckks->evaluator.multiply_vector_reduced_error(bool_temp[1], mask[1], trans_matrix[idepth][1]);
        // ckks->evaluator.rescale_to_next_inplace(trans_matrix[idepth][1]);
        // ckks->print_decrypted_ct(trans_matrix[idepth][1], 10, "trans_matrix[idepth][1]");

        // 2
        // ckks->evaluator.rotate_vector(bool_temp[1], -(1 << idepth), *(ckks->galois_keys), trans_matrix[idepth][2]);
        // ckks->print_decrypted_ct(trans_matrix[idepth][2], 10, "trans_matrix[idepth][2]");

        // std::cout << "----trans_matrix level: " << trans_matrix[idepth][0].coeff_modulus_size() << " chain: " << trans_matrix[idepth][0].chain_index() << std::endl; // 2

        // update distance
        // ckks->print_decrypted_ct(input, 10, "input before");
        ckks->evaluator.rotate_vector(input, rotate_step[0], *(ckks->galois_keys), sum);
        // ckks->print_decrypted_ct(input, 10, "input after");
        ckks->evaluator.multiply_inplace_reduced_error(sum, trans_matrix[idepth][0], *(ckks->relin_keys));
        ckks->evaluator.rescale_to_next_inplace(sum);
        // ckks->print_decrypted_ct(sum, 10, "sum0");

        ckks->evaluator.rotate_vector(input, rotate_step[1], *(ckks->galois_keys), temp);
        ckks->evaluator.multiply_inplace_reduced_error(temp, trans_matrix[idepth][1], *(ckks->relin_keys));
        ckks->evaluator.rescale_to_next_inplace(temp);
        ckks->evaluator.add_inplace_reduced_error(sum, temp);
        // ckks->print_decrypted_ct(sum, 10, "sum1");

        ckks->evaluator.rotate_vector(input, rotate_step[2], *(ckks->galois_keys), temp); // 优化点：旋转消除
        ckks->evaluator.multiply_inplace_reduced_error(temp, trans_matrix[idepth][2], *(ckks->relin_keys));
        ckks->evaluator.rescale_to_next_inplace(temp);
        ckks->evaluator.add_inplace_reduced_error(sum, temp);

        input = sum;
        // ckks->print_decrypted_ct(input, 10, "distance " + std::to_string(iterpoints) + " " + std::to_string(idepth));
        // exit(0);
        // std::cout << "----input level: " << input.coeff_modulus_size() << " chain: " << input.chain_index() << std::endl; // 1

        // std::memset(mask.data(), 1.0, points * sizeof(double));
        std::fill(mask[0].begin(), mask[0].begin() + points, 1.0);
        // std::fill(mask[1].begin(), mask[1].begin() + points, 1.0);
        // std::cout << "mask: " << mask[0] << " " << mask[1] << " " << mask[points] << " " << mask[points + 1] << std::endl;
        ckks->evaluator.multiply_vector_inplace_reduced_error(input, mask[0]);
        ckks->evaluator.rescale_to_next_inplace(input);
        // ckks->print_decrypted_ct(input, 10, "distance mask" + std::to_string(iterpoints) + " " + std::to_string(idepth));

        ckks->evaluator.rotate_vector(input, -points, *(ckks->galois_keys), temp);
        ckks->evaluator.add_inplace_reduced_error(input, temp);

        // ckks->print_decrypted_ct(input, 10, "distance " + std::to_string(iterpoints) + " " + std::to_string(idepth));
        // std::cout << "----input level: " << input.coeff_modulus_size() << " chain: " << input.chain_index() << std::endl; // 8

        // boot
        if (input.coeff_modulus_size() > 1) {
            size_t count = input.coeff_modulus_size() - 1;
            for (size_t icount = 0; icount < count; icount++) {
                ckks->evaluator.mod_switch_to_next_inplace(input);
            }
        }
        bootstrapper->bootstrap_3(temp, input);
        input = temp;
        // ckks->print_decrypted_ct(input, 10, "distance after boot");

        // exit(0);
    }
    // std::cout << "----done input level: " << input.coeff_modulus_size() << " chain: " << input.chain_index() << std::endl; // 8

    // get max
    std::fill(mask[0].data(), mask[0].data() + points, 0.0);
    mask[0][0] = 1.0; // 只保留第一个元素

    ckks->evaluator.multiply_vector_reduced_error(input, mask[0], res);
    ckks->evaluator.rescale_to_next_inplace(res);

    for (size_t imax = 0; imax < depth; imax++) {
        ckks->evaluator.rotate_vector(res, (1 << imax), *(ckks->galois_keys), temp);
        ckks->evaluator.add_inplace_reduced_error(res, temp);
    }
}

void SoftmaxEvaluator::softmax_scaled(PhantomCiphertext &x, PhantomCiphertext &res, int len) {
    cout << "Moduli before SoftMax: " << x.coeff_modulus_size() << endl;

    PhantomCiphertext tmp, exp_x, max;
    int log_step = log2(len);

    { // max
        CUDATimer timer("max", 0);
        timer.start();

        findmax(x, len, max);
        ckks->evaluator.sub_inplace_reduced_error(x, max); // x = x - max

        timer.stop();
    }

    {
        CUDATimer timer("exp", 0);
        timer.start();

        ckks->evaluator.rotate_vector(x, -len, *ckks->galois_keys, tmp);
        ckks->evaluator.add_inplace(x, tmp); // 可以保证前 len 个都为 sum(exp(x_i))

        exp_x = ckks->exp(x);

        timer.stop();
    }

    tmp = exp_x;

    {
        CUDATimer timer("sum", 0);
        timer.start();

        for (int i = 0; i < log_step; ++i) {
            ckks->evaluator.rotate_vector(tmp, pow(2, i), *ckks->galois_keys, res);
            ckks->evaluator.add_inplace(res, tmp);
            tmp = res;
        }

        timer.stop();
    }

    // Normalize res/delta to [0, 1]
    PhantomPlaintext delta;

    {
        CUDATimer timer("inverse", 0);
        timer.start();

        ckks->encoder.encode(0.01, res.params_id(), res.scale(), delta);
        ckks->evaluator.multiply_plain_inplace(res, delta);
        ckks->evaluator.rescale_to_next_inplace(res);

        res = ckks->inverse(res);

        // Recover 1/res

        ckks->encoder.encode(0.01, res.params_id(), res.scale(), delta);
        ckks->evaluator.multiply_plain_inplace(res, delta);
        ckks->evaluator.rescale_to_next_inplace(res);

        timer.stop();
    }

    {
        CUDATimer timer("multiply", 0);
        timer.start();

        ckks->evaluator.mod_switch_to_inplace(exp_x, res.params_id());
        ckks->evaluator.multiply(res, exp_x, res);
        ckks->evaluator.relinearize_inplace(res, *ckks->relin_keys);
        ckks->evaluator.rescale_to_next_inplace(res);

        timer.stop();
    }
    cout << "Moduli left after SoftMax: " << res.coeff_modulus_size() << endl;
}

void SoftmaxEvaluator::softmax(PhantomCiphertext &x, PhantomCiphertext &res, int len) {
    cout << "Moduli before SoftMax: " << x.coeff_modulus_size() << endl;

    PhantomCiphertext tmp, exp_x;
    int log_step = log2(len);

    {
        CUDATimer timer("exp", 0);
        timer.start();

        ckks->evaluator.rotate_vector(x, -len, *ckks->galois_keys, tmp);
        ckks->evaluator.add_inplace(x, tmp); // 可以保证前 len 个都为 sum(exp(x_i))

        exp_x = ckks->exp(x);

        timer.stop();
    }

    tmp = exp_x;

    {
        CUDATimer timer("sum", 0);
        timer.start();

        for (int i = 0; i < log_step; ++i) {
            ckks->evaluator.rotate_vector(tmp, pow(2, i), *ckks->galois_keys, res);
            ckks->evaluator.add_inplace(res, tmp);
            tmp = res;
        }

        timer.stop();
    }

    // Normalize res/delta to [0, 1]
    PhantomPlaintext delta;

    {
        CUDATimer timer("inverse", 0);
        timer.start();

        ckks->encoder.encode(0.01, res.params_id(), res.scale(), delta);
        ckks->evaluator.multiply_plain_inplace(res, delta);
        ckks->evaluator.rescale_to_next_inplace(res);

        res = ckks->inverse(res);

        // Recover 1/res

        ckks->encoder.encode(0.01, res.params_id(), res.scale(), delta);
        ckks->evaluator.multiply_plain_inplace(res, delta);
        ckks->evaluator.rescale_to_next_inplace(res);

        timer.stop();
    }

    {
        CUDATimer timer("multiply", 0);
        timer.start();

        ckks->evaluator.mod_switch_to_inplace(exp_x, res.params_id());
        ckks->evaluator.multiply(res, exp_x, res);
        ckks->evaluator.relinearize_inplace(res, *ckks->relin_keys);
        ckks->evaluator.rescale_to_next_inplace(res);

        timer.stop();
    }
    cout << "Moduli left after SoftMax: " << res.coeff_modulus_size() << endl;
}
