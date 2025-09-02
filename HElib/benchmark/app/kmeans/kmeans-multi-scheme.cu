// #include "../utils/utils.h"
#include <algorithm>
#include <functional>
#include <random>
#include <vector>

#include "ckks_evaluator.cuh"
#include "conversion.cuh"
#include "cutfhe++.h"
#include "extract.cuh"
#include "fileio.h"
#include "phantom.h"
#include "repack.h"
#include "tlwevaluator.cuh"

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

template <typename T>
T mapNonPositiveTo1(T m, T p) {
    return (m < (p / 2)) ? 0 : 1;
}

template <typename T>
T halfX(T m, T p) {
    return static_cast<T>(std::round(0.5 * m)) % p;
}
template <typename T>
T halfAbsX(T m, T p) {
    // return static_cast<T>(std::round(0.5 * std::abs(m))) % p;
    if (m > (p / 2)) {
        return static_cast<T>(std::round(0.5 * (p - m))) % p;
    } else {
        return static_cast<T>(std::round(0.5 * m)) % p;
    }
}

// center = sum(center * bool) * 1/n
void UpdateCenters(CKKSEvaluator &ckks, std::vector<PhantomCiphertext> &center_cipher, std::vector<PhantomCiphertext> &points_cipher,
                   long centers, long points, long dim,
                   std::vector<PhantomCiphertext> &centers_id_ct, PhantomCiphertext &id_packed_ct) {
    PhantomCiphertext bool_flag;
    PhantomCiphertext temp, sum;
    std::vector<double> mask(ckks.encoder.slot_count(), 0.0);
    std::fill(mask.data(), mask.data() + centers, 1.0);
    for (size_t ipoints = 0; ipoints < points; ipoints++) {
        ckks.evaluator.rotate_vector(id_packed_ct, ipoints * centers, *(ckks.galois_keys), centers_id_ct[ipoints]);
        ckks.evaluator.multiply_vector_inplace_reduced_error(centers_id_ct[ipoints], mask);
        ckks.evaluator.rescale_to_next_inplace(centers_id_ct[ipoints]);
    }

    // std::cout << "count cipher" << std::endl;

    std::fill(mask.data(), mask.data() + centers, 0.0);
    mask[0] = 1.0;
    std::vector<PhantomCiphertext> counter_cipher(centers);
    for (size_t icenters = 0; icenters < centers; icenters++) {
        ckks.evaluator.rotate_vector(id_packed_ct, points * centers + icenters, *(ckks.galois_keys), counter_cipher[icenters]);
        ckks.evaluator.multiply_vector_inplace_reduced_error(counter_cipher[icenters], mask);
        ckks.evaluator.rescale_to_next_inplace(counter_cipher[icenters]);

        for (size_t icopy = 0; icopy < static_cast<size_t>(std::log2(dim)); icopy++) {
            ckks.evaluator.rotate_vector(counter_cipher[icenters], -(1 << icopy), *(ckks.galois_keys), temp);
            ckks.evaluator.add_inplace_reduced_error(counter_cipher[icenters], temp);
        }
    }

    // std::cout << "centers id ct" << std::endl;

    for (size_t icenters = 0; icenters < centers; icenters++) {
        for (size_t ipoints = 0; ipoints < points; ipoints++) {
            // get mask
            std::fill(mask.data(), mask.data() + centers, 0.0);
            mask[icenters] = 1.0;

            // set bool flag
            ckks.evaluator.multiply_vector_reduced_error(centers_id_ct[ipoints], mask, bool_flag);
            ckks.evaluator.rescale_to_next_inplace(bool_flag);

            // std::cout << "set bool flag" << std::endl;

            if (icenters > 0) {
                ckks.evaluator.rotate_vector_inplace(bool_flag, icenters, *(ckks.galois_keys));
            }

            for (size_t icopy = 0; icopy < static_cast<size_t>(std::log2(dim)); icopy++) {
                ckks.evaluator.rotate_vector(bool_flag, -(1 << icopy), *(ckks.galois_keys), temp);
                ckks.evaluator.add_inplace_reduced_error(bool_flag, temp);
            }

            // std::cout << "bool flag" << std::endl;

            // add
            if (ipoints == 0) {
                // std::cout << "multiply_reduced_error" << std::endl;

                ckks.evaluator.multiply_reduced_error(points_cipher[ipoints], bool_flag, *(ckks.relin_keys), sum);
                // std::cout << "rescale_to_next_inplace" << std::endl;

                ckks.evaluator.rescale_to_next_inplace(sum);
            } else {
                ckks.evaluator.multiply_reduced_error(points_cipher[ipoints], bool_flag, *(ckks.relin_keys), temp);
                ckks.evaluator.rescale_to_next_inplace(temp);
                ckks.evaluator.add_inplace_reduced_error(sum, temp);
            }

            // std::cout << "add" << std::endl;
        }

        // std::cout << "update" << std::endl;

        // update center
        ckks.evaluator.multiply_inplace_reduced_error(sum, counter_cipher[icenters], *(ckks.relin_keys));
        ckks.evaluator.rescale_to_next_inplace(sum);
        center_cipher[icenters] = sum;

        // ckks.print_decrypted_ct(center_cipher[icenters], 10, "center_cipher " + std::to_string(icenters));
        // std::cout << "----center_cipher[icenters] level: " << center_cipher[icenters].coeff_modulus_size() << " chain: " << center_cipher[icenters].chain_index() << std::endl;

        // exit(0);
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

    auto lut = GenLUT<LvlX>(halfAbsX<typename LvlX::T>, LvlX::plain_modulus);

    typename LvlZ::T *d_lut;
    CUDA_CHECK_RETURN(cudaMalloc(&d_lut, lut.size() * sizeof(typename LvlZ::T)));
    CUDA_CHECK_RETURN(cudaMemcpy(d_lut, lut.data(), lut.size() * sizeof(typename LvlZ::T), cudaMemcpyHostToDevice));
    ProgBootstrapping<LvlXY, LvlYZ>(context.get(), pbs_data, d_lut, d_res, d_tlwe, num_test);
}

template <typename LvlXY, typename LvlYZ,
          typename LvlX = LvlXY::domainP, typename LvlY = LvlXY::targetP, typename LvlZ = LvlYZ::targetP>
void IndexBoostrapping(const Pointer<Context> &context, Pointer<cuTLWE<LvlX>> &res, Pointer<cuTLWE<LvlX>> &data, const size_t num_test) {
    static_assert(std::is_same<LvlY, typename LvlYZ::domainP>::value, "Invalid LvlY");

    using P = std::conditional_t<isLvlCover<LvlX, LvlZ>(), LvlX, LvlZ>;

    Pointer<ProgBootstrappingData<LvlYZ>> pbs_data(num_test);

    TFHEpp::TLWE<LvlX> *d_tlwe = data->template get<LvlX>();
    TFHEpp::TLWE<LvlZ> *d_res = res->template get<LvlZ>();

    auto lut = GenLUT<LvlX>(mapNonPositiveTo1<typename LvlX::T>, LvlX::plain_modulus);

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

    auto lut = GenLUT<LvlX>(halfX<typename LvlX::T>, LvlX::plain_modulus);

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

    tlwer.add(res0, front, behind, step);
    // tlwer.print_culwe_ct_value(res0, step, "add result");

    tlwer.sub(res1, front, behind, step);
    // tlwer.print_culwe_ct_value(res1, step, "sub result");

    auto &context = tlwer.get_pbscontext();

    HalfBoostrapping<Lvl10, Lvl01>(context, dtlwe_tmp0, dtlwe_tmp0, step);  // 0.5(m0 + m1)
    // tlwer.print_culwe_ct_value(res0, step, "half result");

    HalfAbsBoostrapping<Lvl10, Lvl01>(context, dtlwe_tmp1, dtlwe_tmp1, step);  // 0.5|m0 - m1|
    // tlwer.print_culwe_ct_value(res1, step, "half abs result");

    tlwer.sub(rtn, res0, res1, step);  // min(m0, m1)
    // tlwer.print_culwe_ct_value(rtn, step, "min result");
}

template <typename P>
void MiniIndex(tlwevaluator<P> &tlwer, std::vector<Pointer<cuTLWE<P>>> &lwe_distances, int points, int centers) {
    // lwe_distances: (points, centers)
    Pointer<cuTLWE<P>> dtlwe_min(centers >> 1);
    size_t depth = std::ceil((std::log2(centers)));
    for (size_t ipoints = 0; ipoints < points; ipoints++) {
        // find min of each point
        for (size_t idepth = 0; idepth < depth; idepth++) {
            if (idepth == 0) {
                MinOfTwoBatch<Lvl1>(tlwer, dtlwe_min, lwe_distances[ipoints], centers / (1 << (idepth + 1)));
            } else {
                MinOfTwoBatch<Lvl1>(tlwer, dtlwe_min, dtlwe_min, centers / (1 << (idepth + 1)));  // min in first
            }
        }

        // b_lwe = lut(dis - min)
        auto res = lwe_distances[ipoints]->template get<Lvl1>();
        auto data0 = lwe_distances[ipoints]->template get<Lvl1>();
        auto data1 = dtlwe_min->template get<Lvl1>();

        tlwer.sub_single(res, data0, data1, centers);
        IndexBoostrapping<Lvl10, Lvl01>(tlwer.get_pbscontext(), lwe_distances[ipoints], lwe_distances[ipoints], centers);  // lwe bool

        // sum the min index
        auto count = lwe_distances[points]->template get<Lvl1>();
        tlwer.add(count, count, res, centers);
    }
}

void EuclideanDistanceMultiCenters(CKKSEvaluator &ckks, std::vector<PhantomCiphertext> &points_ct, std::vector<PhantomCiphertext> &centers_ct,
                                   long points, long dim, long centers, PhantomCiphertext &distance_matrix) {
    PhantomCiphertext distance_temp, inner_temp, distance_perpoint;

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
                distance_perpoint = distance_temp;
            } else {
                ckks.evaluator.rotate_vector_inplace(distance_temp, -icenters, *(ckks.galois_keys));
                ckks.evaluator.add_inplace(distance_perpoint, distance_temp);
            }
        }

        // copy, 方便抽取
        if (ipoints == 0) {
            distance_matrix = distance_perpoint;
        } else {
            ckks.evaluator.rotate_vector_inplace(distance_perpoint, -ipoints * centers, *(ckks.galois_keys));
            ckks.evaluator.add_inplace(distance_matrix, distance_perpoint);
        }
    }
}

int main() {
    // kmeans data
    size_t dim_max = 1 << 15;
    std::vector<std::vector<double>> data = FileIO<double>::LoadCSV3D("/mnt/data2/home/syt/data/Libra/benchmark/app/data/kmeans/4data4cent.csv");  // (points, dim)
    if (dim_max < data[0].size()) {
        throw std::logic_error("Error: Get Out Input Length.");
    }
    long points = data.size();
    long dim = data[0].size();
    long centers = 4;

    std::cout << "points: " << points << ", dimP: " << dim << ", centers: " << centers << std::endl;

    // ckks init
    long logN = 16;
    long logn = 15;
    long sparse_slots = (1 << logn);

    int logp = 56;
    int logq = 61;
    int log_special_prime = 61;

    int secret_key_hamming_weight = 192;

    // (41,7)(39, 6)-->comp(3,3) or comp(4,4); (25,4)-->comp(2,2)
    int remaining_level = 17;
    int special_prime_len = 2;

    vector<int> coeff_bit_vec;
    coeff_bit_vec.push_back(logq);
    for (int i = 0; i < remaining_level; i++) {
        coeff_bit_vec.push_back(logp);
    }
    for (int i = 0; i < special_prime_len; i++) {
        coeff_bit_vec.push_back(log_special_prime);
    }

    std::cout << "Setting RLWE Parameters..." << endl;
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

    std::cout << "Adding galois Keys..." << endl;
    vector<int> gal_steps_vector;
    gal_steps_vector.push_back(0);
    for (int i = 0; i < logN - 1; i++) {
        gal_steps_vector.push_back((1 << i));
        gal_steps_vector.push_back(-(1 << i));
    }
    ckks_evaluator.decryptor.create_galois_keys_from_steps(gal_steps_vector, *(ckks_evaluator.galois_keys));

    // tfhe init
    std::cout << "Setting LWE Parameters..." << endl;
    using lwe_enc_lvl = Lvl1;
    // int scale_bits = std::numeric_limits<typename lwe_enc_lvl::T>::digits - lwe_enc_lvl::plain_modulus_bit - 1;
    // double lwe_scale = pow(2.0, scale_bits);
    double lwe_scale = lwe_enc_lvl::Δ;

    TFHESecretKey sk;
    TFHEEvalKey ek;
    load_keys<BootstrappingKeyFFTLvl01, BootstrappingKeyFFTLvl02,
              KeySwitchingKeyLvl10, KeySwitchingKeyLvl20, KeySwitchingKeyLvl21>(sk, ek);
    tlwevaluator<lwe_enc_lvl> tlwe_evaluator(&sk, &ek, lwe_scale);

    std::vector<std::vector<TLWELvl1>> lwe_distances(points + 1, std::vector<TLWELvl1>(centers));  // dis matrix
    std::vector<Pointer<cuTLWE<Lvl1>>> d_lwe_distances;                                            // dis matrix
    d_lwe_distances.reserve(points + 1);

    // kmeans 5,867,507,342  1,572,540,046
    std::cout << "kmeans..." << endl;
    size_t slot_count = encoder.slot_count();
    std::vector<std::vector<double>> input(points, std::vector<double>(slot_count, 0.0));    // input: 一个输入一个密文，维度为一个密文里
    std::vector<std::vector<double>> center(centers, std::vector<double>(slot_count, 0.0));  // center: 一个 centers 一个密文
    for (size_t i = 0; i < points; i++) {
        std::copy(data[i].begin(), data[i].begin() + dim, input[i].begin());
    }
    // print_matrix(input, "input");

    // init center
    if (centers == 2) {
        std::copy(input[0].begin(), input[0].begin() + dim, center[0].begin());
        std::copy(input[points - 1].begin(), input[points - 1].begin() + dim, center[1].begin());
    } else {
        for (size_t i = 0; i < centers; i++) {
            std::copy(input[i].begin(), input[i].begin() + dim, center[i].begin());
        }
    }
    // print_matrix(center, "init center");

    // encrypt data
    PhantomPlaintext plain;
    std::vector<PhantomCiphertext> points_cipher(points), center_cipher(centers), centers_id_cipher(points);
    PhantomCiphertext distance_matrix_cipher;

    for (size_t i = 0; i < points; i++) {
        ckks_evaluator.encoder.encode(input[i], scale, plain);
        ckks_evaluator.encryptor.encrypt(plain, points_cipher[i]);
    }
    for (size_t i = 0; i < centers; i++) {
        ckks_evaluator.encoder.encode(center[i], scale, plain);
        ckks_evaluator.encryptor.encrypt(plain, center_cipher[i]);
    }

    auto s = phantom::util::global_variables::default_stream->get_stream();

    float total_time = 0.0;
    int iter_count = 1;

    std::cout << "iter count: " << iter_count << std::endl;
    for (size_t iter = 0; iter < iter_count; iter++) {
        // (Euclidean distance)^2
        {
            CUDATimer timer("Euclidean distance", s);
            timer.start();
            EuclideanDistanceMultiCenters(ckks_evaluator, points_cipher, center_cipher, points, dim, centers, distance_matrix_cipher);  // 2 level
            timer.stop();
            total_time += timer.get_mean_time();
        }
        // ckks_evaluator.print_decrypted_ct(distance_matrix_cipher, points, centers, "distance matrix");

        // extract, 抽取 ipoints * centers
        {
            CUDATimer timer("Extract", s);
            timer.start();
            RLWEToLWEs(context, distance_matrix_cipher, lwe_distances);
            timer.stop();
        }

        // find min-index per point
        {
            // encrypt lwe distances
            // vector<vector<lwe_enc_lvl::T>> lwe_distances_vec = {{0, 1}, {1, 0}, {2, 1}, {2, 1}, {0, 1}, {1, 0}, {2, 1}, {2, 1}, {0, 0}};
            // vector<vector<lwe_enc_lvl::T>> lwe_distances_vec = {{0, 1, 1, 2}, {1, 0, 2, 1}, {2, 1, 0, 1}, {2, 1, 1, 0}, {0, 1, 1, 2}, {1, 0, 2, 1}, {2, 1, 0, 1}, {2, 1, 1, 0}, {0, 1, 1, 2}, {1, 0, 2, 1}, {2, 1, 0, 1}, {2, 1, 1, 0}, {0, 1, 1, 2}, {1, 0, 2, 1}, {2, 1, 0, 1}, {2, 1, 1, 0}, {0, 0, 0, 0}};

            vector<vector<lwe_enc_lvl::T>> lwe_distances_vec(points + 1, vector<lwe_enc_lvl::T>(centers, 0));
            for (size_t i = 0; i < points; i++) {
                for (size_t j = 0; j < centers; j++) {
                    lwe_distances_vec[i][j] = j;
                }
            }

            for (size_t i = 0; i < points + 1; i++) {
                for (size_t j = 0; j < centers; j++) {
                    lwe_distances[i][j] = TFHEpp::tlweSymInt32Encrypt<lwe_enc_lvl>(lwe_distances_vec[i][j], lwe_enc_lvl::α, lwe_scale, sk.key.get<lwe_enc_lvl>());
                    // auto lwe_dec_num = TFHEpp::tlweSymInt32Decrypt<lwe_enc_lvl>(lwe_distances[i][j], lwe_scale, sk.key.get<lwe_enc_lvl>());
                    // std::cout << "point: " << i << " center: " << j << " decrypt: " << lwe_dec_num << " ground: " << lwe_distances_vec[i][j] << std::endl;
                }
                d_lwe_distances.emplace_back(centers);

                TFHEpp::TLWE<lwe_enc_lvl> *dest = d_lwe_distances[i]->template get<lwe_enc_lvl>();
                TFHEpp::TLWE<lwe_enc_lvl> *src = lwe_distances[i].data();
                CUDA_CHECK_RETURN(cudaMemcpy(dest, src, centers * sizeof(TFHEpp::TLWE<lwe_enc_lvl>), cudaMemcpyHostToDevice));
            }
            CUDATimer timer("Find min-index", s);

            timer.start();
            MiniIndex<Lvl1>(tlwe_evaluator, d_lwe_distances, points, centers);
            timer.stop();
        }

        // exit(0);

        // repack (all to one)
        PhantomRLWE rlwer(1);
        rlwer.genLWE2RLWEGaloisKeys((points + 1) * centers);
        PhantomCiphertext results;

        std::vector<TLWELvl1> h_lwes((points + 1) * centers);
        for (size_t i = 0; i < points + 1; i++) {
            auto h_lwe = h_lwes.data() + i * centers;
            auto d_lwe = d_lwe_distances[i]->template get<lwe_enc_lvl>();
            CUDA_CHECK_RETURN(cudaMemcpy(h_lwe, d_lwe, sizeof(TFHEpp::TLWE<Lvl1>) * centers, cudaMemcpyDeviceToHost));
        }
        // std::vector<TLWELvl1> h_lwes(128);
        // std::vector<uint32_t> msg(h_lwes.size(), 1);
        // for (size_t i = 0; i < msg.size(); i++) {
        //     h_lwes[i] = TFHEpp::tlweSymInt32Encrypt<lwe_enc_lvl>(msg[i], lwe_enc_lvl::α, lwe_scale, sk.key.get<lwe_enc_lvl>());
        // }
        {
            CUDATimer timer("Repack", 0);
            timer.start();
            conver::repack(results, h_lwes, rlwer, sk);  // 11 levels
            timer.stop();
        }
        std::cout << "conversion level: " << results.coeff_modulus_size() << " chain: " << results.chain_index() << std::endl;

        // update center
        {
            std::vector<double> res(slot_count, 0.0);
            std::fill(res.begin(), res.begin() + points * centers, 1.0);
            ckks_evaluator.encoder.encode(res, scale, plain);
            ckks_evaluator.encryptor.encrypt(plain, results);
            std::cout << "res level: " << results.coeff_modulus_size() << " chain: " << results.chain_index() << std::endl;

            CUDATimer timer("update center", 0);
            timer.start();
            UpdateCenters(ckks_evaluator, center_cipher, points_cipher, centers, points, dim, centers_id_cipher, results);
            timer.stop();
            std::cout << "update level: " << center_cipher[0].coeff_modulus_size() << " chain: " << center_cipher[0].chain_index() << std::endl;
        }
    }
}