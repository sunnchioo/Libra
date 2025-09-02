// multi-scheme

#include <algorithm>
#include <functional>
#include <random>
#include <vector>

#include "ckks_evaluator.cuh"
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
T halfX(T m) {
    return static_cast<T>(0.5 * m);
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
void HalfBoostrapping(const Pointer<Context> &context, Pointer<cuTLWE<LvlX>> &res, Pointer<cuTLWE<LvlX>> &data, const size_t num_test) {
    static_assert(std::is_same<LvlY, typename LvlYZ::domainP>::value, "Invalid LvlY");

    using P = std::conditional_t<isLvlCover<LvlX, LvlZ>(), LvlX, LvlZ>;

    Pointer<ProgBootstrappingData<LvlYZ>> pbs_data(num_test);

    TFHEpp::TLWE<LvlX> *d_tlwe = data->template get<LvlX>();
    TFHEpp::TLWE<LvlZ> *d_res = res->template get<LvlZ>();

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

    tlwer.add(res0, front, behind, step);
    tlwer.print_culwe_ct_value_double(res0, step, "add result");

    tlwer.sub(res1, front, behind, step);
    tlwer.print_culwe_ct_value_double(res1, step, "sub result");

    auto &context = tlwer.get_pbscontext();

    HalfBoostrapping<Lvl10, Lvl01>(context, dtlwe_tmp0, dtlwe_tmp0, step);  // 0.5(m0 + m1)
    tlwer.print_culwe_ct_value_double(res0, step, "half result");

    HalfAbsBoostrapping<Lvl10, Lvl01>(context, dtlwe_tmp1, dtlwe_tmp1, step);  // 0.5|m0 - m1|
    tlwer.print_culwe_ct_value_double(res1, step, "half abs result");

    tlwer.sub(rtn, res0, res1, step);  // min(m0, m1)
    tlwer.print_culwe_ct_value_double(rtn, step, "min result");
}

template <typename P>
void Mini(tlwevaluator<P> &tlwer, std::vector<Pointer<cuTLWE<P>>> &lwe_distances, int centers) {
    // lwe_distances: (points, centers)
    Pointer<cuTLWE<P>> dtlwe_min(centers >> 1);
    size_t depth = std::ceil((std::log2(centers)));
    // find min of each point
    for (size_t idepth = 0; idepth < depth; idepth++) {
        if (idepth == 0) {
            MinOfTwoBatch<Lvl1>(tlwer, dtlwe_min, lwe_distances[0], centers / (1 << (idepth + 1)));
        } else {
            MinOfTwoBatch<Lvl1>(tlwer, dtlwe_min, dtlwe_min, centers / (1 << (idepth + 1)));  // min in first
        }
    }

    auto rtn = dtlwe_min->template get<P>();
    // tlwer.print_culwe_ct_value(rtn, centers / (1 << (idepth + 1)), "min result");
    tlwer.print_culwe_ct_value_double(rtn, 1, "min result");
}

void findmin() {
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

    std::vector<Pointer<cuTLWE<Lvl1>>> d_lwe_distances;
    d_lwe_distances.reserve(1);

    for (size_t i = 1; i < 2; i++) {
        int centers = 1 << i;
        std::vector<TLWELvl1> lwe_distances(centers);

        std::vector<lwe_enc_lvl::T> lwe_distances_vec(centers, 0);
        for (size_t j = 0; j < centers; j++) {
            lwe_distances_vec[j] = j % 8;
        }

        for (size_t j = 0; j < centers; j++) {
            lwe_distances[j] = TFHEpp::tlweSymInt32Encrypt<lwe_enc_lvl>(lwe_distances_vec[j], lwe_enc_lvl::α, lwe_scale, sk.key.get<lwe_enc_lvl>());
            auto lwe_dec_num = TFHEpp::tlweSymIntDecryptDouble<lwe_enc_lvl>(lwe_distances[j], lwe_scale, sk.key.get<lwe_enc_lvl>());
            std::cout << " decrypt: " << lwe_dec_num << " ground: " << lwe_distances_vec[j] << std::endl;
        }
        d_lwe_distances.emplace_back(centers);

        // cudaDeviceSynchronize();
        TFHEpp::TLWE<lwe_enc_lvl> *dest = d_lwe_distances[0]->template get<lwe_enc_lvl>();
        TFHEpp::TLWE<lwe_enc_lvl> *src = lwe_distances.data();
        CUDA_CHECK_RETURN(cudaMemcpy(dest, src, centers * sizeof(TFHEpp::TLWE<lwe_enc_lvl>), cudaMemcpyHostToDevice));
        // cudaError_t error = cudaGetLastError();
        // printf("CUDA error: %s\n", cudaGetErrorString(error));
        cudaDeviceSynchronize();

        {
            CUDATimer timer("Find min-index", 0);
            timer.start();
            Mini<Lvl1>(tlwe_evaluator, d_lwe_distances, centers);
            timer.stop();
        }
    }
}

int main() {
    findmin();
}