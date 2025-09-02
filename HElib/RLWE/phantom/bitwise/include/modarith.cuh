#include <cuda.h>
#include <cuda_runtime.h>
#include <type_traits>

template<typename T>
concept supportedTypes = std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t>;

template<typename T> requires supportedTypes<T>
struct wide_type {
    T lo{};
    T hi{};

    wide_type() = default;

    __host__ __device__ wide_type(const T lo, const T hi) : lo(lo), hi(hi) {}
};

/** 128-bit wide addition
 * @param op1
 * @param op2
 * @return op1 + op2
 */
__device__ inline wide_type<uint64_t> addWide(const wide_type<uint64_t> &op1, const wide_type<uint64_t> &op2) {
    wide_type<uint64_t> res;
    asm ("{\n\t"
         "add.cc.u64 %0, %2, %4;\n\t"
         "addc.u64   %1, %3, %5;\n\t"
         "}"
            : "=l"(res.lo), "=l"(res.hi)
            : "l"(op1.lo), "l"(op1.hi), "l"(op2.lo), "l"(op2.hi));
    return res;
}

/** 64-bit wide addition
 * @param op1
 * @param op2
 * @return op1 + op2
 */
__device__ inline wide_type<uint32_t> addWide(const wide_type<uint32_t> &op1, const wide_type<uint32_t> &op2) {
    wide_type<uint32_t> res;
    asm ("{\n\t"
         "add.cc.u32 %0, %2, %4;\n\t"
         "addc.u32   %1, %3, %5;\n\t"
         "}"
            : "=r"(res.lo), "=r"(res.hi)
            : "r"(op1.lo), "r"(op1.hi), "r"(op2.lo), "r"(op2.hi));
    return res;
}

/** 64-bit multiplication
 * @param op1
 * @param op2
 * @return high 64-bit of op1 * op2
 */
__device__ inline uint64_t mulHi(const uint64_t op1, const uint64_t op2) {
    return __umul64hi(op1, op2);
}

/** 32-bit multiplication
 * @param op1
 * @param op2
 * @return high 32-bit of op1 * op2
 */
__device__ inline uint32_t mulHi(const uint32_t op1, const uint32_t op2) {
    return __umulhi(op1, op2);
}

/** wide multiplication
 * @param op1
 * @param op2
 * @return op1 * op2
 */
template<typename T>
requires supportedTypes<T>
__device__ inline wide_type<T> mulWide(const T op1, const T op2) {
    wide_type<T> res;
    res.lo = op1 * op2;
    res.hi = mulHi(op1, op2);
    return res;
}

/**
 * fast modular reduction
 * conditionally subtracts modulus if op is in [0, 2 * mod]
 * @param op in [0, 2 * mod]
 * @param mod
 * @return reduced op in [0, mod]
 */
template<typename T>
requires supportedTypes<T>
__host__ __device__ inline T modFast(const T op, const T mod) {
    const T tmp = op - mod;
    constexpr size_t shift_bits = 8 * sizeof(T) - 1;
    return tmp + (tmp >> shift_bits) * mod;
}

/** Barrett reduction
 * @param op
 * @param mod
 * @param mu
 * @return op % mod in [0, mod]
 */
template<typename T>
requires supportedTypes<T>
__device__ inline T modBarrett(const T op, const T mod, const T mu) {
    T s = mulHi(op, mu);
    T res = op - s * mod;
    return modFast(res, mod);
}

/**
 * Barrett reduction for double-width operand and modulus
 * @param op
 * @param mod
 * @param mu Barrett constant
 * @return reduced op in [0, mod]
 */
__device__ inline uint64_t
modWideBarrett(const wide_type<uint64_t> op, const uint64_t mod, const wide_type<uint64_t> mu) {
    uint64_t res;

    asm(
            "{\n\t"
            " .reg .u64 tmp;\n\t"
            // Multiply input and const_ratio
            // Round 1
            " mul.hi.u64 tmp, %1, %3;\n\t"
            " mad.lo.cc.u64 tmp, %1, %4, tmp;\n\t"
            " madc.hi.u64 %0, %1, %4, 0;\n\t"
            // Round 2
            " mad.lo.cc.u64 tmp, %2, %3, tmp;\n\t"
            " madc.hi.u64 %0, %2, %3, %0;\n\t"
            // This is all we care about
            " mad.lo.u64 %0, %2, %4, %0;\n\t"
            // Barrett subtraction
            " mul.lo.u64 %0, %0, %5;\n\t"
            " sub.u64 %0, %1, %0;\n\t"
            "}"
            : "=l"(res)
            : "l"(op.lo), "l"(op.hi), "l"(mu.lo), "l"(mu.hi), "l"(mod));

    return modFast(res, mod);
}

/**
 * Barrett reduction for double-width operand and modulus
 * @param op
 * @param mod
 * @param mu Barrett constant
 * @return reduced op in [0, mod]
 */
__device__ inline uint32_t
modWideBarrett(const wide_type<uint32_t> op, const uint32_t mod, const wide_type<uint32_t> mu) {
    uint32_t res;

    asm(
            "{\n\t"
            " .reg .u32 tmp;\n\t"
            // Multiply input and const_ratio
            // Round 1
            " mul.hi.u32 tmp, %1, %3;\n\t"
            " mad.lo.cc.u32 tmp, %1, %4, tmp;\n\t"
            " madc.hi.u32 %0, %1, %4, 0;\n\t"
            // Round 2
            " mad.lo.cc.u32 tmp, %2, %3, tmp;\n\t"
            " madc.hi.u32 %0, %2, %3, %0;\n\t"
            // This is all we care about
            " mad.lo.u32 %0, %2, %4, %0;\n\t"
            // Barrett subtraction
            " mul.lo.u32 %0, %0, %5;\n\t"
            " sub.u32 %0, %1, %0;\n\t"
            "}"
            : "=r"(res)
            : "r"(op.lo), "r"(op.hi), "r"(mu.lo), "r"(mu.hi), "r"(mod));
    return modFast(res, mod);
}

/**
 * fast modular addition
 * conditionally subtracts modulus if op1 + op2 is in [0, 2 * mod]
 * @param op1 in [0, mod]
 * @param op2 in [0, mod]
 * @param mod
 * @return reduced op1 + op2 in [0, mod]
 */
template<typename T>
requires supportedTypes<T>
__host__ __device__ inline T modAdd(const T op1, const T op2, const T mod) {
    const T tmp = op1 + op2 - mod;
    constexpr size_t shift_bits = 8 * sizeof(T) - 1;
    return tmp + (tmp >> shift_bits) * mod;
}

/**
 * Modular multiplication using wide Barrett reduction
 * @param op1
 * @param op2
 * @param mod
 * @param mu Barrett constant
 * @return result of (op1 * op2) % mod in [0, mod]
 */
template<typename T>
requires supportedTypes<T>
__device__ inline T modMulBarrett(const T op1, const T op2, const T mod, const wide_type<T> mu) {
    const wide_type<T> product = mulWide(op1, op2);
    return modWideBarrett(product, mod, mu);
}

/** Modular multiplication using Shoup's method
 * @brief a * b % mod, Shoup's implementation
 * @param op1
 * @param op2
 * @param op2_shoup shoup pre-computation of b
 * @param mod
 * @return result of (op1 * op2) % mod in [0, 2 * mod)
 */
template<typename T>
requires supportedTypes<T>
__device__ inline T modMulShoupLazy(const T op1, const T op2, const T op2_shoup, const T mod) {
    T hi = mulHi(op1, op2_shoup);
    return op1 * op2 - hi * mod;
}

/** Modular multiplication using Shoup's method
 * @brief a * b % mod, Shoup's implementation
 * @param op1
 * @param op2
 * @param op2_shoup shoup pre-computation of b
 * @param mod
 * @return a * b % mod, range [0, mod)
 */
template<typename T>
requires supportedTypes<T>
__device__ inline T modMulShoup(const T op1, const T op2, const T op2_shoup, const T mod) {
    T hi = mulHi(op1, op2_shoup);
    T res = op1 * op2 - hi * mod;
    return modFast(res, mod);
}
