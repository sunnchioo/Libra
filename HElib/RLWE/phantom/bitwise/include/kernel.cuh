#include "openfhe.h"

constexpr size_t ThreadsPerBlock = 128;

/**
 * The signed digit decomposition which takes an RLWE ciphertext input and outputs a vector of its digits
 * @param output decomposed digits of the input element
 * @param input the input RLWE ciphertext
 * @param Q modulus for the RingGSW/RingLWE scheme
 * @param baseG gadget base used in bootstrapping
 * @param N number of coefficients in RGSW ciphertext
 */
__global__ void kernel_SignedDigitDecompose(BasicInteger *output, const BasicInteger *input,
                                            BasicInteger Q, uint32_t baseG, size_t N);
__global__ void kernel_SignedDigitDecompose_opt(BasicInteger *output, const BasicInteger *input,
                                                BasicInteger Q, uint32_t baseG, size_t N);

__global__ void kernel_SignedDigitDecompose_fuse_1024(BasicInteger *g_dct, const BasicInteger *g_acc,
                                                      BasicInteger Q, uint32_t baseG,
                                                      const BasicInteger *tw_root_2n1,
                                                      const BasicInteger *tw_root_2n1_shoup,
                                                      const BasicInteger *tw_root_2n,
                                                      const BasicInteger *tw_root_2n_shoup);

__global__ void kernel_EvalAccCoreDM(BasicInteger *acc, const BasicInteger *dct, const BasicInteger *RingGSWACCKey,
                                     size_t N, BasicInteger mod, BasicInteger mu0, BasicInteger mu1, uint32_t digitsG2);

__global__ void kernel_EvalAccCoreDM_batch(BasicInteger *acc, const BasicInteger *dct, BasicInteger **acc_keys,
                                           size_t N, BasicInteger mod, BasicInteger mu0, BasicInteger mu1,
                                           uint32_t digitsG2);

__global__ void kernel_EvalAccCoreDM_1024_batch_fuse(
    BasicInteger *acc, const BasicInteger *dct, BasicInteger **acc_keys,
    size_t N, BasicInteger mod, BasicInteger mu0, BasicInteger mu1,
    uint32_t digitsG2,
    const BasicInteger *tw_inv_root_2n,
    const BasicInteger *tw_inv_root_2n_shoup,
    const BasicInteger *tw_inv_root_2n1,
    const BasicInteger *tw_inv_root_2n1_shoup,
    BasicInteger inv_n, BasicInteger inv_n_shoup,
    int IsCompositeNTT, BasicInteger P);

__global__ void kernel_EvalAccCoreCGGI(BasicInteger *d_acc, const BasicInteger *d_dct,
                                       const BasicInteger *d_ACCKey0, const BasicInteger *d_ACCKey1,
                                       const BasicInteger *d_monic_polys, size_t N,
                                       BasicInteger mod, BasicInteger mu0, BasicInteger mu1,
                                       uint32_t digitsG2, size_t indexPos, size_t indexNeg);

__global__ void kernel_EvalAccCoreCGGI_binary(BasicInteger *acc, const BasicInteger *dct,
                                              const BasicInteger *d_ACCKey, const BasicInteger *monic_polys, size_t N,
                                              BasicInteger mod, BasicInteger mu0, BasicInteger mu1,
                                              uint32_t digitsG2, size_t indexPos);

__global__ void kernel_EvalAccCoreCGGI_batch(BasicInteger *acc, const BasicInteger *dct,
                                             const BasicInteger *acc_key0, const BasicInteger *acc_key1,
                                             const BasicInteger *monic_polys, size_t N,
                                             BasicInteger mod, BasicInteger mu0, BasicInteger mu1, uint32_t digitsG2,
                                             const uint32_t *indexPos_batch, const uint32_t *indexNeg_batch);

__global__ void kernel_EvalAccCoreCGGI_binary_batch(BasicInteger *acc, const BasicInteger *dct,
                                                    const BasicInteger *acc_key,
                                                    const BasicInteger *monic_polys, size_t N,
                                                    BasicInteger mod, BasicInteger mu0, BasicInteger mu1,
                                                    uint32_t digitsG2,
                                                    const uint32_t *indexPos_batch);

__global__ void kernel_EvalAccCoreCGGI_1024_batch_fuse(
    BasicInteger *acc, const BasicInteger *dct,
    const BasicInteger *acc_key0, const BasicInteger *acc_key1,
    const BasicInteger *monic_polys,
    size_t N,
    BasicInteger mod, BasicInteger mu0, BasicInteger mu1,
    uint32_t digitsG2,
    const uint32_t *indexPos_batch,
    const uint32_t *indexNeg_batch,
    const BasicInteger *tw_inv_root_2n,
    const BasicInteger *tw_inv_root_2n_shoup,
    const BasicInteger *tw_inv_root_2n1,
    const BasicInteger *tw_inv_root_2n1_shoup,
    BasicInteger inv_n, BasicInteger inv_n_shoup,
    int IsCompositeNTT, BasicInteger P, BasicInteger Q);

__global__ void kernel_EvalAccCoreCGGI_1024_binary_batch_fuse(
    BasicInteger *acc, const BasicInteger *dct,
    const BasicInteger *acc_key,
    const BasicInteger *monic_polys,
    size_t N,
    BasicInteger mod, BasicInteger mu0, BasicInteger mu1,
    uint32_t digitsG2,
    const uint32_t *indexPos_batch,
    const BasicInteger *tw_inv_root_2n,
    const BasicInteger *tw_inv_root_2n_shoup,
    const BasicInteger *tw_inv_root_2n1,
    const BasicInteger *tw_inv_root_2n1_shoup,
    BasicInteger inv_n, BasicInteger inv_n_shoup,
    int IsCompositeNTT, BasicInteger P, BasicInteger Q);

__global__ void kernel_element_add(BasicInteger *output, const BasicInteger *input1, const BasicInteger *input2,
                                   size_t dim, BasicInteger mod);

__global__ void kernel_automorphism_modSwitch(BasicInteger *output, const BasicInteger *input,
                                              BasicInteger qKS, BasicInteger Q, size_t logN);

__global__ void kernel_scale_by_p(BasicInteger *ct_Q, const BasicInteger *ct_PQ, size_t dim,
                                  BasicInteger Q, BasicInteger PQ);

__global__ void kernel_LWEKeySwitch_modSwitch(BasicInteger *res_ct_A,
                                              const BasicInteger *ct_A, const BasicInteger *ksk_A,
                                              size_t n, size_t N,
                                              BasicInteger qAfter, BasicInteger qKS, BasicInteger log_baseKS,
                                              size_t digitCount);
