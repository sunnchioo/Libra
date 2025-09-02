#pragma once

#include <cmath>
#include <cuComplex.h>
#include <fstream>

#include "ciphertext.h"
#include "ckks.h"
#include "context.cuh"
#include "evaluate.cuh"
#include "ntt.cuh"
#include "phantom.h"
#include "plaintext.h"
#include "secretkey.h"

#define PI 3.14159265358979323846

namespace bootstraper {
    // [] of matirx
    typedef std::vector<std::vector<size_t>> MatrixIndex;
    typedef std::vector<std::vector<std::vector<cuDoubleComplex>>> MatrixList;
    typedef std::vector<std::vector<PhantomPlaintext>> MatrixLTPtxt;
    typedef uint64_t MatrixType;

    typedef PhantomPlaintext Plaintext;
    typedef PhantomCiphertext Ciphertext;

    class ckksbootstraper {
    private:
        double scale;

        // linear transforme param
        size_t N;
        size_t radix, depth; // radix: default = 2

        // n = l·t, l and t near n^1/2, l(1<<(std::log(n)>>1)), t(n/l)
        size_t n, l, t;
        uint64_t root_n;
        uint64_t lt_prime;

        size_t lt_N, lt_n;

        std::vector<std::vector<cuDoubleComplex>> omiga;

        MatrixIndex host_S2CDFTMatrix_index;
        MatrixIndex host_C2SDFTMatrix_index;
        MatrixList host_S2CDFTMatrix;
        MatrixList host_C2SDFTMatrix;

        MatrixLTPtxt lt_m0;
        MatrixLTPtxt lt_m1;

        // compent
        PhantomCKKSEncoder *encoder;
        PhantomSecretKey *secret_key;
        PhantomPublicKey *public_key;
        PhantomRelinKey *relin_key;
        PhantomGaloisKey *galois_key;

        // chebyshev params
        std::vector<std::vector<double>> chebyshev_coeffs; // load
        size_t heaplen = 0;

        size_t eval_mod_level;
        size_t chebyshev_degree, bg_m, bg_l; // Default = 30

        double bound_a, bound_b;
        size_t K, K_new, double_angle_r, k;
        std::vector<uint32_t> pow2;
        std::vector<std::vector<double>> chebyshev_qi_coeffs;

    public:
        ckksbootstraper(const PhantomContext &context, const double &scale,
                        PhantomCKKSEncoder &encoder, PhantomSecretKey &secret_key, PhantomRelinKey &relin_key, PhantomGaloisKey &galois_key);
        ~ckksbootstraper();

        auto square(const PhantomContext &context, PhantomCiphertext &dest, PhantomCiphertext &ct);

        auto double_angle_formula_scaled(const PhantomContext &context, Ciphertext &cipher, double scale_coeff);
        void read_heap_from_file(std::ifstream &in);
        void homomorphic_poly_evaluation(const PhantomContext &context, PhantomCiphertext &rtn, PhantomCiphertext &cipher);
        void modular_reduction(const PhantomContext &context, PhantomCiphertext &rtn, PhantomCiphertext &cipher);
        void initchebyshev_new();
        void initchebyshev();
        void initLinerTransMatrix_new(const PhantomContext &context);
        void initLinerTransMatrix(const PhantomContext &context);
        void initLinerTransSlot2CoeffMatrix(const PhantomContext &context);
        void initLinerTransCoeff2SlotMatrix(const PhantomContext &context);

        void loadSlot2CoeffMatrix();
        void loadCoeff2SlotMatrix();

        auto multiply_const_rescale(const PhantomContext &context, const PhantomCiphertext &encrypted, double constant);
        auto sub_ct(const PhantomContext &context, const PhantomCiphertext &encrypted1, const PhantomCiphertext &encrypted2);
        auto add_ct(const PhantomContext &context, const PhantomCiphertext &encrypted1, const PhantomCiphertext &encrypted2);
        void rotate_const_right(std::vector<cuDoubleComplex> &M, size_t rot);
        auto multiply_ct(const PhantomContext &context, const PhantomCiphertext &encrypted1, const PhantomCiphertext &encrypted2);
        auto multiply_plain(const PhantomContext &context, const PhantomCiphertext &encrypted1, PhantomPlaintext &plain);
        void check(const PhantomContext &context, const PhantomCiphertext &encrypted, const PhantomPlaintext &plain);
        void check(const PhantomContext &context, const PhantomCiphertext &encrypted, const PhantomPlaintext &plain, const PhantomCiphertext &result);
        void check(const PhantomContext &context, const PhantomCiphertext &encrypted1, const PhantomCiphertext &encrypted2, const PhantomCiphertext &result);
        void check(const PhantomContext &context, const PhantomCiphertext &result);

        // This instantiates a boot circuit evaluating:
        //
        // 0) User defined circuit in the slots domain
        // 1) SlotsToCoeffs: Homomorphic Decoding
        // 2) User defined circuit in the coeffs domain
        // 3) ScaleDown: Scale the ciphertext to q0/|m|
        // 4) ModUp: Raise modulus from q0 to qL
        // 5) CoeffsToSlots: Homomorphic encoding
        // 6) EvalMod (and to back to 0): Homomorphic modular reduction
        auto EvalMod_new(const PhantomContext &context, PhantomCiphertext &ct);
        void EvalMod(const PhantomContext &context, PhantomCiphertext &ct);
        auto ModRasing(const PhantomContext &context, PhantomCiphertext &ct);
        auto ScaleDown(const PhantomContext &context, PhantomCiphertext &ct);
        auto SlotsToCoeffs(const PhantomContext &context, PhantomCiphertext &ct);
        void MultiplyByDiagMatrixBSGS_inplace(const PhantomContext &context, PhantomCiphertext &ct, size_t index, PhantomCiphertext &opout, MatrixLTPtxt &lt_m, bool is_c2s);
        void MultiplyByDiagMatrixBSGS(const PhantomContext &context, PhantomCiphertext &ct, size_t index, PhantomCiphertext &opout, MatrixLTPtxt &lt_m, bool is_c2s);
        auto CoeffsToSlots(const PhantomContext &context, PhantomCiphertext &ct, PhantomCiphertext &ct0, PhantomCiphertext &ct1);
        void ckksbootstrapping_inplace(const PhantomContext &context, PhantomCiphertext &ct);
        inline auto ckksbootstrapping(const PhantomContext &context, PhantomCiphertext &encrypted) {
            ckksbootstrapping_inplace(context, encrypted);
        }
    };

} // namespace bootstraper