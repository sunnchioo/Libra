
/***** out begin *****/
#include <vector>
#include "ckks_evaluator.cuh"
#include "phantom.h"
#include "utils.cuh"

#include <algorithm>
#include <functional>
#include <random>
#include <vector>

#include "conversion.cuh"
#include "tlwevaluator.cuh"
#include "trlwevaluator.cuh"

using namespace std;
using namespace phantom;
using namespace phantom::arith;
using namespace phantom::util;
using namespace rlwe;

using namespace rlwe;
using namespace cuTFHEpp;
using namespace cuTFHEpp::util;

template <typename T, typename Tp>
T sign(T m, Tp p) {
    if (m > (p >> 1)) {
        return 0;
    } else {
        return 1;
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

    auto lut = GenDLUTP<LvlX>(sign<double, typename LvlX::T>, LvlX::plain_modulus);

    typename LvlZ::T *d_lut;
    CUDA_CHECK_RETURN(cudaMalloc(&d_lut, lut.size() * sizeof(typename LvlZ::T)));
    CUDA_CHECK_RETURN(cudaMemcpy(d_lut, lut.data(), lut.size() * sizeof(typename LvlZ::T), cudaMemcpyHostToDevice));
    ProgBootstrapping<LvlXY, LvlYZ>(context.get(), pbs_data, d_lut, d_res, d_tlwe, num_test);
}

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

    /***** RLWE BEGIN *****/
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
    /***** RLWE END *****/

    /***** LWE BEGIN *****/
    std::cout << "Setting LWE..." << endl;
    using lwe_enc_lvl = Lvl1L;
    // double lwe_scale = lwe_enc_lvl::Δ;
    double lwe_scale = std::pow(2.0, 46);

    TFHESecretKey lwe_sk;
    TFHEEvalKey lwe_ek;
    load_keys<BootstrappingKeyFFTLvl01, BootstrappingKeyFFTLvl02,
              KeySwitchingKeyLvl10, KeySwitchingKeyLvl20, KeySwitchingKeyLvl21>(lwe_sk, lwe_ek);
    tlwevaluator<lwe_enc_lvl> tlwer(&lwe_sk, &lwe_ek, lwe_scale);

    // Pointer<cuTLWE<lwe_enc_lvl>> lwes(s, 1);  // 256

    std::cout << "Setting conver..." << endl;
    conver::GPUDecomposedLWEKSwitchKey extractKey;

    auto &modulus = ckks_evaluator.ckks->context->key_context_data().parms().coeff_modulus();
    std::vector<phantom::arith::Modulus> lwe_modulus{modulus[0], modulus[1]};  // only use first
    conver::LWEParams parms(scheme_type::ckks);
    parms.set_poly_modulus_degree(TFHEpp::lvl1param::n);
    parms.set_coeff_modulus(lwe_modulus);
    auto lwe_context = make_unique<conver::LWEContext>(parms);

    auto &rlwe_sk = ckks_evaluator.secret_key();
    std::cout << "Gen Extract Key" << std::endl;
    GenExtractKey(ckks_evaluator, lwe_context.get(), extractKey, lwe_sk, rlwe_sk);
    /***** LWE END *****/

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
    ckks_evaluator.evaluator.add(SIMDCipher0, SIMDCipher1, SIMDCipher2);
    ckks_evaluator.evaluator.cmlut(SIMDCipher2, SIMDCipher2, SIMDCipher2);
    std::vector<size_t> extract_indices = {0};
    int lwes_len = extract_indices.size();
    const size_t logN = TFHEpp::lvlRparam::nbits;
    for (size_t i = 0; i < extract_indices.size(); ++i) {
        extract_indices[i] = phantom::arith::reverse_bits(extract_indices[i], logN - 1);
    }
    Pointer<cuTLWE<lwe_enc_lvl>> lwes(s, 1);
    conver::extract<lwe_enc_lvl>(ckks_evaluator, lwe_context.get(), SIMDCipher2, lwes, extract_indices, extractKey);
    SignBoostrapping<Lvl10, Lvl01>(context, lwes, lwes, lwes_len);

    // TODO: unhandled op: flyhe.simd_store

    /***** tail start *****/
    tlwe_evaluator.print_culwe_ct_value_double_err(lwes, lwes_len, "result", sparse);
    ckks_evaluator.print_decrypted_ct(rtn, 10);
    ? return 0;
    /***** tail end *****/
}
