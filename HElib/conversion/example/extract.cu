#include <algorithm>
#include <functional>
#include <random>
#include <vector>

#include "conversion.cuh"
#include "tlwevaluator.cuh"
#include "trlwevaluator.cuh"

using namespace rlwe;
using namespace cuTFHEpp;
using namespace cuTFHEpp::util;
using namespace phantom;

void random_real(std::vector<double> &vec, size_t size) {
    // std::random_device rn;
    // std::mt19937_64 rnd(rn());
    std::mt19937_64 rnd(42);
    thread_local std::uniform_real_distribution<double> distribution(-1, 1);

    vec.reserve(size);

    for (size_t i = 0; i < size; i++) {
        vec[i] = distribution(rnd);
    }
}

int main() {
    std::cout << "Setting RLWE..." << endl;
    trlwevaluator trlwer(scheme_type::ckks);

    const auto &s = phantom::util::global_variables::default_stream->get_stream();

    std::cout << "Setting LWE..." << endl;
    using lwe_enc_lvl = Lvl1L;
    // double lwe_scale = lwe_enc_lvl::Δ;
    double lwe_scale = std::pow(2.0, 46);

    TFHESecretKey lwe_sk;
    TFHEEvalKey lwe_ek;
    load_keys<BootstrappingKeyFFTLvl01, BootstrappingKeyFFTLvl02,
              KeySwitchingKeyLvl10, KeySwitchingKeyLvl20, KeySwitchingKeyLvl21>(lwe_sk, lwe_ek);
    tlwevaluator<lwe_enc_lvl> tlwer(&lwe_sk, &lwe_ek, lwe_scale);

    Pointer<cuTLWE<lwe_enc_lvl>> lwes(s, 1);  // 256

    std::cout << "Setting conver..." << endl;
    conver::GPUDecomposedLWEKSwitchKey extractKey;

    auto &modulus = trlwer.ckks->context->key_context_data().parms().coeff_modulus();
    std::vector<phantom::arith::Modulus> lwe_modulus{modulus[0], modulus[1]};  // only use first
    conver::LWEParams parms(scheme_type::ckks);
    parms.set_poly_modulus_degree(TFHEpp::lvl1param::n);
    parms.set_coeff_modulus(lwe_modulus);
    auto lwe_context = make_unique<conver::LWEContext>(parms);

    auto &rlwe_sk = trlwer.secret_key();
    std::cout << "Gen Extract Key" << std::endl;
    GenExtractKey(trlwer, lwe_context.get(), extractKey, lwe_sk, rlwe_sk);

    std::cout << "Setting input..." << endl;
    // size_t sparse_slots_size = trlwer.ckks->sparse_slots;
    size_t slot_size = trlwer.ckks->slot_count;
    // std::cout << "sparse_slots_size: " << sparse_slots_size << " slot_size: " << slot_size << std::endl;

    // std::vector<double> sparse(4, 0);
    std::vector<double> sparse = {0., 1., 2., 3.};
    std::vector<double> input(slot_size, 0);

    // random_real(sparse, sparse.size());

    for (size_t i = 0; i < input.size(); i++) {
        input[i] = sparse[i % sparse.size()];
    }
    trlwer.ckks->print_vector(input, 10, "input");

    double ckks_scale = trlwer.ckks->scale;
    PhantomPlaintext plain;
    PhantomCiphertext cipher;
    trlwer.ckks->encoder.encode(input, ckks_scale, plain);
    trlwer.ckks->encryptor.encrypt(plain, cipher);

    trlwer.ckks->print_decrypted_ct(cipher, 10, "decrypted cipher");
    // std::vector<size_t> extract_indices(sparse_slots_size, 0);
    // std::vector<size_t> extract_indices(256, 0);

    std::vector<size_t> extract_indices = {0};
    const size_t logN = TFHEpp::lvlRparam::nbits;
    for (size_t i = 0; i < extract_indices.size(); ++i) {
        extract_indices[i] = phantom::arith::reverse_bits(extract_indices[i], logN - 1);  // 这是这个是因为成的是快速变换矩阵，bit会反转
    }

    std::cout << "extract_indices: ";
    for (int i = 0; i < extract_indices.size(); ++i) {
        std::cout << i << " -- " << extract_indices[i] << std::endl;
    }
    std::cout << std::endl;
    cudaDeviceSynchronize();

    {
        CUDATimer timer("Extract", s);
        timer.start();
        conver::extract<lwe_enc_lvl>(trlwer, lwe_context.get(), cipher, lwes, extract_indices, extractKey);
        timer.stop();
    }
    std::cout << "ectract end." << std::endl;

    // tlwer.print_culwe_ct_value(lwes->template get<lwe_enc_lvl>(), lwe_scale, 1, "ectract result");
    cudaDeviceSynchronize();
    tlwer.tlweSymIntDecryptCudaMod(lwes->template get<lwe_enc_lvl>(), lwe_scale, lwe_modulus[0], 1);
}