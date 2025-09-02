#include <algorithm>
#include <functional>
#include <random>
#include <vector>

#include "ckks_evaluator.cuh"
#include "conversion.cuh"
#include "cutfhe++.h"
#include "fileio.h"
#include "phantom.h"
#include "tlwevaluator.cuh"

// #include "extract.cuh"
#include "repack.h"

using namespace rlwe;
using namespace cuTFHEpp;
using namespace cuTFHEpp::util;
using namespace conver;

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

void LinearTransform(
    const PhantomContext &context,
    PhantomCiphertext &result,
    std::vector<std::vector<double>> &matrix,
    double scale,
    LTPreKey &eval_key,
    PhantomCKKSEncoder &encoder,
    PhantomGaloisKey &galois_keys) {
    size_t rows = matrix.size();
    size_t columns = matrix.front().size();
    size_t slot_counts = encoder.slot_count();
    if (columns > slot_counts) {
        throw std::invalid_argument("Convert LWE ciphers out of size.");
    }

    // BSGS Parameters
    size_t max_len = std::max(rows, columns);
    size_t min_len = std::min(rows, columns);
    size_t g_tilde = CeilSqrt(min_len);
    size_t b_tilde = CeilDiv(min_len, g_tilde);

    // Baby-Step
    if (eval_key.rotated_keys.size() < g_tilde) {
        // std::cout << "LWEToRLWEKeyGen Error" << std::endl;
        eval_key.rotated_keys.resize(g_tilde);
        eval_key.rotated_keys[0] = eval_key.key;
        for (size_t i = 1; i < g_tilde; i++) {
            eval_key.rotated_keys[i - 1] = eval_key.rotated_keys[i];
            rotate_vector_inplace(context, eval_key.rotated_keys[i - 1], 1, galois_keys);
        }
    }

    // Giant-Step
    std::vector<double> diag(max_len);
    PhantomPlaintext plain;
    plain.chain_index() = eval_key.rotated_keys[0].chain_index();
    for (size_t b = 0; b < b_tilde && g_tilde * b < min_len; b++) {
        PhantomCiphertext temp, sum;
        for (size_t g = 0; g < g_tilde && b * g_tilde + g < min_len; g++) {
            // Get diagonal
            size_t j = b * g_tilde + g;
            for (size_t r = 0; r < max_len; r++) {
                diag[r] = matrix[r % rows][(r + j) % columns];
            }
            std::rotate(diag.rbegin(), diag.rbegin() + b * g_tilde, diag.rend());
            pack_encode(context, diag, scale, plain, encoder);
            if (g == 0) {
                sum = eval_key.rotated_keys[g];
                multiply_plain_inplace(context, sum, plain);
            } else {
                temp = eval_key.rotated_keys[g];
                multiply_plain_inplace(context, temp, plain);
                add_inplace(context, sum, temp);
            }
        }
        if (b == 0) {
            result = sum;
        } else {
            rotate_vector_inplace(context, sum, b * g_tilde, galois_keys);
            add_inplace(context, result, sum);
        }
    }

    if (rows < columns) {
        size_t gama = std::log2(columns / rows);
        for (size_t j = 0; j < gama; j++) {
            PhantomCiphertext temp = result;
            rotate_vector_inplace(context, temp, (1U << j) * rows, galois_keys);
            add_inplace(context, result, temp);
        }
    }
}

int main() {
    // tfhe init
    std::cout << "Setting LWE Parameters..." << endl;
    using lwe_enc_lvl = Lvl1;
    int scale_bits = 26;
    double lwe_scale = pow(2.0, scale_bits);
    // double lwe_scale = lwe_enc_lvl::Δ;

    TFHESecretKey sk;  // 定义时已经初始化
    TFHEEvalKey ek;
    load_keys<BootstrappingKeyFFTLvl01, BootstrappingKeyFFTLvl02,
              KeySwitchingKeyLvl10, KeySwitchingKeyLvl20, KeySwitchingKeyLvl21>(sk, ek);

    size_t dim = 64;
    PhantomRLWE rlwer(dim);
    rlwer.genLWE2RLWEGaloisKeys();
    PhantomCiphertext results;

    {
        CUDATimer timer("mvm", 0);
        PhantomCiphertext results;

        // 1. Preprocess LWE Matrix
        std::vector<std::vector<double>> A(dim);
        for (size_t i = 0; i < dim; i++) {
            A[i] = std::vector<double>(dim);
        }
        double rescale = std::pow(2., rlwer.modulus_bits - rlwer.modq_bits);

        // 2. Linear Transform A * s
        // std::cout << "Linear Transform A * s" << std::endl;
        LTPreKey eval_key = rlwer.genPreKey(sk, Lvl1::n);

        timer.start();
        LinearTransform(*rlwer.context, results, A, 1.0, eval_key, *rlwer.ckks_encoder, *rlwer.galois_keys);
        rescale_to_next_inplace(*rlwer.context, results);  // 1 level
        timer.stop();
    }

    rlwer.print_decrypted_ct(results, 10, "Repack results");
    std::cout << "conversion level: " << results.coeff_modulus_size() << " chain: " << results.chain_index() << std::endl;  // 13 12
}