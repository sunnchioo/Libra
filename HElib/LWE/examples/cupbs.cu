#include <algorithm>
#include <functional>
#include <random>
#include <vector>

#include "cutfhe++.h"
#include "phantom.h"
#include "tlwevaluator.cuh"

using namespace cuTFHEpp;
using namespace cuTFHEpp::util;

using CUDATimer = phantom::util::CUDATimer;

template <typename T>
T mapNonPositiveTo1(T m, T p) {
    return (m < (p / 2)) ? 0 : 1;
}

template <typename T, typename Tp>
T sign(T m, Tp p) {
    if (m > (p >> 1)) {
        return 0;
    } else {
        return 1;
    }
}

template <typename T>
T X(T m, T p) {
    return static_cast<T>(m) % p;
    // return 1;
}

template <typename T>
T halfX(T m) {
    return 0.5 * m;
    // return 1;
}

template <typename T>
T halfAbsX(T m, T p) {
    // return static_cast<T>(std::round(0.5 * std::abs(m))) % p;
    if (m > (p / 2)) {
        return static_cast<T>(std::floor(0.5 * (p - m))) % p;
    } else {
        return static_cast<T>(std::floor(0.5 * m)) % p;
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

template <typename LvlXY, typename LvlYZ,
          typename LvlX = LvlXY::domainP, typename LvlY = LvlXY::targetP, typename LvlZ = LvlYZ::targetP>
void HalfBoostrapping(const Pointer<Context> &context, Pointer<cuTLWE<LvlZ>> &res, Pointer<cuTLWE<LvlX>> &data, const size_t num_test) {
    static_assert(std::is_same<LvlY, typename LvlYZ::domainP>::value, "Invalid LvlY");

    using P = std::conditional_t<isLvlCover<LvlX, LvlZ>(), LvlX, LvlZ>;

    Pointer<ProgBootstrappingData<LvlYZ>> pbs_data(num_test);

    TFHEpp::TLWE<LvlX> *d_tlwe = data->template get<LvlX>();
    TFHEpp::TLWE<LvlZ> *d_res = res->template get<LvlZ>();

    auto lut = GenDLUT<LvlX>(halfX<double>, LvlX::plain_modulus);

    // for (size_t i = 0; i < lut.size(); i++) {
    //     std::cout << "lut[" << i << "] = " << lut[i] << std::endl;
    // }

    typename LvlZ::T *d_lut;
    CUDA_CHECK_RETURN(cudaMalloc(&d_lut, lut.size() * sizeof(typename LvlZ::T)));
    CUDA_CHECK_RETURN(cudaMemcpy(d_lut, lut.data(), lut.size() * sizeof(typename LvlZ::T), cudaMemcpyHostToDevice));
    ProgBootstrapping<LvlXY, LvlYZ>(context.get(), pbs_data, d_lut, d_res, d_tlwe, num_test);
}

template <typename LvlXY, typename LvlYZ,
          typename LvlX = LvlXY::domainP, typename LvlY = LvlXY::targetP, typename LvlZ = LvlYZ::targetP>
void HalfAbsBoostrapping(const Pointer<Context> &context, Pointer<cuTLWE<LvlZ>> &res, Pointer<cuTLWE<LvlX>> &data, const size_t num_test) {
    static_assert(std::is_same<LvlY, typename LvlYZ::domainP>::value, "Invalid LvlY");

    using P = std::conditional_t<isLvlCover<LvlX, LvlZ>(), LvlX, LvlZ>;

    Pointer<ProgBootstrappingData<LvlYZ>> pbs_data(num_test);

    TFHEpp::TLWE<LvlX> *d_tlwe = data->template get<LvlX>();
    TFHEpp::TLWE<LvlZ> *d_res = res->template get<LvlZ>();

    auto lut = GenLUT<LvlX>(halfAbsX<typename LvlX::T>, LvlX::plain_modulus);

    // for (size_t i = 0; i < lut.size(); i++) {
    //     std::cout << "lut[" << i << "] = " << lut[i] << std::endl;
    // }

    typename LvlZ::T *d_lut;
    CUDA_CHECK_RETURN(cudaMalloc(&d_lut, lut.size() * sizeof(typename LvlZ::T)));
    CUDA_CHECK_RETURN(cudaMemcpy(d_lut, lut.data(), lut.size() * sizeof(typename LvlZ::T), cudaMemcpyHostToDevice));
    ProgBootstrapping<LvlXY, LvlYZ>(context.get(), pbs_data, d_lut, d_res, d_tlwe, num_test);
}

template <typename LvlXY, typename LvlYZ,
          typename LvlX = LvlXY::domainP, typename LvlY = LvlXY::targetP, typename LvlZ = LvlYZ::targetP>
void XBoostrapping(const Pointer<Context> &context, Pointer<cuTLWE<LvlZ>> &res, Pointer<cuTLWE<LvlX>> &data, const size_t num_test) {
    static_assert(std::is_same<LvlY, typename LvlYZ::domainP>::value, "Invalid LvlY");

    using P = std::conditional_t<isLvlCover<LvlX, LvlZ>(), LvlX, LvlZ>;

    Pointer<ProgBootstrappingData<LvlYZ>> pbs_data(num_test);

    TFHEpp::TLWE<LvlX> *d_tlwe = data->template get<LvlX>();
    TFHEpp::TLWE<LvlZ> *d_res = res->template get<LvlZ>();

    auto lut = GenLUT<LvlX>(X<typename LvlX::T>, LvlX::plain_modulus);

    // for (size_t i = 0; i < lut.size(); i++) {
    //     std::cout << "lut[" << i << "] = " << lut[i] << std::endl;
    // }

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

void random_real(std::vector<double> &vec, size_t size) {
    std::random_device rn;
    std::mt19937_64 rnd(rn());
    // std::mt19937_64 rnd(41);
    thread_local std::uniform_real_distribution<double> distribution(0, 15);

    vec.reserve(size);

    for (size_t i = 0; i < size; i++) {
        vec[i] = distribution(rnd);
    }
}

int main() {
    std::cout << "Setting LWE Parameters..." << std::endl;
    // using lwe_enc_lvl = Lvl1;
    // using lwe_res_lvl = Lvl1;
    using lwe_enc_lvl = Lvl2;
    using lwe_res_lvl = Lvl2;

    // int scale_bits = std::numeric_limits<typename lwe_enc_lvl::T>::digits - lwe_enc_lvl::plain_modulus_bit - 1;
    // int scale_bits = 61;
    // double lwe_scale = pow(2.0, scale_bits);
    // std::cout << "LWE scale: " << lwe_scale << " scale bit: " << scale_bits << std::endl;
    double lwe_scale = lwe_enc_lvl::Δ;

    TFHESecretKey sk;
    TFHEEvalKey ek;
    load_keys<BootstrappingKeyFFTLvl01, BootstrappingKeyFFTLvl02,
              KeySwitchingKeyLvl10, KeySwitchingKeyLvl20, KeySwitchingKeyLvl21>(sk, ek);

    tlwevaluator<lwe_enc_lvl> tlwe_evaluator(&sk, &ek, lwe_scale);

    // std::vector<lwe_enc_lvl::T> msg = {0, 1, 2, 3, 4, 5, 6, 7};
    // std::vector<lwe_enc_lvl::T> msg = {0, 1, 2, 3};
    // std::vector<lwe_enc_lvl::T> msg = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

    std::vector<lwe_enc_lvl::T> msg(1 << 4, 0);

    std::vector<double> sparse(msg.size(), 0.0);
    random_real(sparse, msg.size());
    for (size_t i = 0; i < msg.size(); i++) {
        sparse[i] = std::round(sparse[i]);
        msg[i] = sparse[i];
    }

    for (int iter = 0; iter < 1; iter++) {
        // std::vector<lwe_enc_lvl::T> msg(1 << iter);
        // std::vector<double> sparse(msg.size(), 1.0);
        // for (size_t i = 0; i < msg.size(); i++) {
        //     msg[i] = 4;
        // }

        size_t len_lwe = msg.size();
        std::vector<TFHEpp::TLWE<lwe_enc_lvl>> h_lwes(len_lwe);
        Pointer<cuTLWE<lwe_enc_lvl>> d_lwes(len_lwe);
        Pointer<cuTLWE<lwe_res_lvl>> d_lwes_res(len_lwe);

        for (size_t i = 0; i < len_lwe; i++) {
            h_lwes[i] = TFHEpp::tlweSymInt32Encrypt<lwe_enc_lvl>(msg[i], lwe_enc_lvl::α, lwe_scale, sk.key.get<lwe_enc_lvl>());
            auto lwe_dec_num = TFHEpp::tlweSymInt32Decrypt<lwe_enc_lvl>(h_lwes[i], lwe_scale, sk.key.get<lwe_enc_lvl>());
            // std::cout << "decrypt: " << lwe_dec_num << " ground: " << msg[i] << std::endl;
        }

        TFHEpp::TLWE<lwe_enc_lvl> *d_dest = d_lwes->template get<lwe_enc_lvl>();
        TFHEpp::TLWE<lwe_enc_lvl> *h_src = h_lwes.data();
        CUDA_CHECK_RETURN(cudaMemcpy(d_dest, h_src, len_lwe * sizeof(TFHEpp::TLWE<lwe_enc_lvl>), cudaMemcpyHostToDevice));

        auto &context = tlwe_evaluator.get_pbscontext();

        auto res = d_lwes_res->template get<lwe_enc_lvl>();
        auto src = d_lwes->template get<lwe_enc_lvl>();

        // HalfBoostrapping<Lvl10, Lvl01>(context, d_lwes_res, d_lwes, len_lwe); // x 2>> 1
        // HalfBoostrapping<Lvl20, Lvl02>(context, d_lwes_res, d_lwes, len_lwe); // 0.5(m0 + m1)
        // tlwe_evaluator.print_culwe_ct_value(res, len_lwe, "half result");

        // HalfAbsBoostrapping<Lvl10, Lvl01>(context, d_lwes_res, d_lwes, len_lwe); // 0.5|m0 - m1|
        // tlwe_evaluator.print_culwe_ct_value(res, len_lwe, "half abs result");
        cudaDeviceSynchronize();

        {  // max
            CUDATimer timer("XBoostrapping", 0);
            timer.start();
            // XBoostrapping<Lvl10, Lvl01>(context, d_lwes_res, d_lwes, len_lwe);  // m
            // XBoostrapping<Lvl20, Lvl02>(context, d_lwes_res, d_lwes, len_lwe);  // m

            // HalfBoostrapping<Lvl10, Lvl10>(context, d_lwes_res, d_lwes, len_lwe);  // x 2>> 1
            // HalfBoostrapping<Lvl20, Lvl02>(context, d_lwes_res, d_lwes, len_lwe);  // x 2>> 1

            // SignBoostrapping<Lvl10, Lvl01>(context, d_lwes_res, d_lwes, len_lwe);
            SignBoostrapping<Lvl20, Lvl02>(context, d_lwes_res, d_lwes, len_lwe);
            timer.stop();
        }

        // tlwe_evaluator.print_culwe_ct_value(res, len_lwe, "X result");
        // tlwe_evaluator.print_culwe_ct_value_double(res, len_lwe, "X result");
        tlwe_evaluator.print_culwe_ct_value_double_err(res, len_lwe, "X result", sparse);

        // IndexBoostrapping<Lvl10, Lvl01>(context, d_lwes_res, d_lwes, len_lwe);
        // tlwe_evaluator.print_culwe_ct_value(res, len_lwe, "Index result");

        // SignBoostrapping<Lvl10, Lvl01>(context, d_lwes_res, d_lwes, len_lwe);
        // tlwe_evaluator.print_culwe_ct_value(res, len_lwe, "Sign result");
    }
}