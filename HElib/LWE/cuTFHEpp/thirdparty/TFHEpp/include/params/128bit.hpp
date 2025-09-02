#pragma once

#include <cmath>
#include <cstdint>
#include <limits>

struct lvlRparam {
    static constexpr std::uint32_t nbits = 16;
    static constexpr std::uint32_t n = 1 << nbits;  // dimension
    static constexpr std::uint32_t k = 1;
    using T = uint64_t;  // RLWE representation
};

struct lvl1Lparam {
    static constexpr std::uint32_t nbit = 10;      // dimension must be a power of 2 for ease of polynomial// multiplication.
    static constexpr std::uint32_t n = 1 << nbit;  // dimension
    static constexpr std::uint32_t k = 1;          // (k+1) rows of Gadget matrix
    static constexpr std::uint32_t l = 3;
    static constexpr std::uint32_t Bgbit = 6;
    static constexpr std::uint32_t Bg = 1 << Bgbit;
    static const inline double α = std::pow(2.0, -25);  // fresh noise
    // static const inline double α = 0.0;
    using T = uint64_t;  // Torus representation
    using sT = int64_t;  // Torus representation

    static constexpr T μ = 1U << 25;  // μ = Δ
    // static constexpr uint32_t plain_modulus = 2;
    static constexpr uint32_t plain_modulus_bit = 5;
    static constexpr uint32_t plain_modulus = 1 << plain_modulus_bit;  // 4
    static constexpr double Δ =
        static_cast<double>(1ULL << (std::numeric_limits<T>::digits - plain_modulus_bit - 1));  // 留出有 1 bit
};  //  自举中的密钥切换（Key Switching） 或 功能性自举（Functional Bootstrapping）

// 0,1时为中间编码
// bit encode: 0 -> 0/q, 1 -> 1/2 = (q/2) / q
// 编码空间为 [-μ, μ]， μ = q/2，将 0,1 编码为 -q/4, q/4, 中心编码
struct lvl0param {
    static constexpr std::uint32_t n = 672;  // dimension
    static constexpr std::uint32_t k = 1;
    static const inline double α = std::pow(2.0, -16);                  // fresh noise
    using T = uint32_t;                                                 // Torus representation
    static constexpr T μ = 1U << (std::numeric_limits<T>::digits - 2);  // 1 << 29
    // static constexpr uint32_t plain_modulus = 2;
    static constexpr uint32_t plain_modulus_bit = 1;
    static constexpr uint32_t plain_modulus = 1 << plain_modulus_bit;
    static constexpr double Δ =
        static_cast<double>(1ULL << std::numeric_limits<T>::digits) /
        plain_modulus;  // 1 << 30
};  // 原始 LWE 加密，支持快速加密和解密，但噪声较高

struct lvl1param {
    static constexpr std::uint32_t nbit = 10;      // dimension must be a power of 2 for ease of polynomial// multiplication.
    static constexpr std::uint32_t n = 1 << nbit;  // dimension
    static constexpr std::uint32_t k = 1;          // (k+1) rows of Gadget matrix

    //
    // static constexpr std::uint32_t l = 3;
    // static constexpr std::uint32_t Bgbit = 6;

    //
    // static constexpr std::uint32_t l = 3;
    // static constexpr std::uint32_t Bgbit = 9;

    //
    // static constexpr std::uint32_t l = 4;  // 这样噪声低一些
    // static constexpr std::uint32_t Bgbit = 7;

    // 不需低噪声就可以，10^{-2}
    static constexpr std::uint32_t l = 5;
    static constexpr std::uint32_t Bgbit = 6;

    static constexpr std::uint32_t Bg = 1 << Bgbit;
    static const inline double α = std::pow(2.0, -25);  // fresh noise
    // static const inline double α = std::pow(2.0, -52);  // fresh noise
    // static const inline double α = 0.0;
    using T = uint32_t;  // Torus representation
    using sT = int32_t;  // Torus representation
    // using T = uint64_t;               // Torus representation
    static constexpr T μ = 1U << 26;  // μ = Δ, 31-plain_modulus_bit
    // static constexpr uint32_t plain_modulus = 2;
    static constexpr uint32_t plain_modulus_bit = 5;
    static constexpr uint32_t plain_modulus = 1 << plain_modulus_bit;  // 4
    static constexpr double Δ =
        static_cast<double>(1ULL << (std::numeric_limits<T>::digits - plain_modulus_bit - 1));  // 留出有 1 bit
};  //  自举中的密钥切换（Key Switching） 或 功能性自举（Functional Bootstrapping）

// 线性编码
struct lvl2param {
    static const std::uint32_t nbit = 11;          // dimension must be a power of 2 for
                                                   // ease of polynomial multiplication.
    static constexpr std::uint32_t n = 1 << nbit;  // dimension
    static constexpr std::uint32_t k = 1;

    // static constexpr std::uint32_t l = 4;
    // static constexpr std::uint32_t Bgbit = 9;

    // static constexpr std::uint32_t l = 5;
    // static constexpr std::uint32_t Bgbit = 7; // less noise, but more time

    static constexpr std::uint32_t l = 3;
    static constexpr std::uint32_t Bgbit = 8;

    static constexpr std::uint32_t Bg = 1 << Bgbit;
    static const inline double α = std::pow(2.0, -52);  // fresh noise
    using T = uint64_t;                                 // Torus representation
    using sT = int64_t;                                 // Torus representation
    static constexpr T μ = 1ULL << 58;                  // μ = Δ, 63-plain_modulus_bit
    // static constexpr T μ = 1ULL << 37; // μ = Δ
    // static constexpr uint32_t plain_modulus = 8;
    static constexpr uint32_t plain_modulus_bit = 5;
    // static constexpr uint32_t plain_modulus_bit = 26;
    static constexpr uint32_t plain_modulus = 1 << plain_modulus_bit;
    static constexpr double Δ = μ;
    // static constexpr double Δ =
    //     static_cast<double>(__uint128_t(1) << std::numeric_limits<T>::digits) /
    //     plain_modulus; // 31 bit
    // static constexpr double Δ =
    // static_cast<double>(1ULL << (std::numeric_limits<T>::digits - plain_modulus_bit)); // 61 bit
};  // 高精度自举 或 复杂运算（如多比特明文操作）

// Key Switching parameters
struct lvl10param {
    static constexpr std::uint32_t t = 2;  // number of addition in keyswitching
    static constexpr std::uint32_t basebit =
        10;                                       // how many bit should be encrypted in keyswitching key
    static const inline double α = lvl0param::α;  // key noise
    using domainP = lvl1param;
    using targetP = lvl0param;
};

struct lvl11param {
    static constexpr std::uint32_t t = 6;  // number of addition in keyswitching
    static constexpr std::uint32_t basebit =
        4;                                        // how many bit should be encrypted in keyswitching key
    static const inline double α = lvl1param::α;  // key noise
    using domainP = lvl1param;
    using targetP = lvl1param;
};

struct lvl20param {
    static constexpr std::uint32_t t = 2;  // number of addition in keyswitching
    static constexpr std::uint32_t basebit =
        10;                                       // how many bit should be encrypted in keyswitching key
    static const inline double α = lvl0param::α;  // key noise
    using domainP = lvl2param;
    using targetP = lvl0param;
};

struct lvl21param {
    static constexpr std::uint32_t t = 10;  // number of addition in
                                            // keyswitching
    static constexpr std::uint32_t basebit =
        3;                                        // how many bit should be encrypted in keyswitching key
    static const inline double α = lvl1param::α;  // key noise
    using domainP = lvl2param;
    using targetP = lvl1param;
};

struct lvl22param {
    static constexpr std::uint32_t t = 8;  // number of addition in keyswitching
    static constexpr std::uint32_t basebit =
        4;                                        // how many bit should be encrypted in keyswitching key
    static const inline double α = lvl2param::α;  // key noise
    using domainP = lvl2param;
    using targetP = lvl2param;
};
