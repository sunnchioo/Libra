#pragma once

#include <iostream>
#include <set>
#include <vector>
#include <cmath>

#include "arith/big_integer.h"
#include "arith/big_integer_modop.h"
#include "arith/prng/pseudorandomnumbergenerator.h"

namespace isecfhe {

    /**
     * @description: Generate a random big integer between 0 and uper bound.
     * @param upperBound The upper bound of the generated random integer.
     * @return Randomly generated big integer.
     * TODO: Currently using system RNG generator which can not generate big integer, need better one.
     */
    template<typename IntType>
    IntType RNG(const IntType &upperBound) {
        constexpr uint32_t chunk_min{0};
        constexpr uint32_t chunk_max{std::numeric_limits<uint32_t>::max()};
        constexpr uint32_t chunk_width{std::numeric_limits<uint32_t>::digits};
        static std::uniform_int_distribution<uint32_t> distribution(chunk_min, chunk_max);

        uint32_t chunksPerValue{(upperBound.GetMSB() - 1) / chunk_width};
        uint32_t shiftChunk{chunksPerValue * chunk_width};
        std::uniform_int_distribution<uint32_t>::param_type bound(chunk_min,
                                                                  (upperBound >> shiftChunk).ConvertToLimb());
        while (true) {
            IntType result{};
            for (uint32_t i{0}, shift{0}; i < chunksPerValue; ++i, shift += chunk_width)
                result += IntType{distribution(PseudoRandomNumberGenerator::GetPRNG())} << shift;
            result += IntType{distribution(PseudoRandomNumberGenerator::GetPRNG(), bound)} << shiftChunk;
            if (result < upperBound)
                return result;
        }
    }

    /*
    TODO: Understands the Miller rabin primality test algorithm, then normalize parameter comments
      A witness function used for the Miller-Rabin Primality test.
      Inputs: a is a randomly generated witness between 2 and p-1,
      p is the number to be tested for primality,
      s and d satisfy p-1 = ((2^s) * d), d is odd.
      Output: true if p is composite,
      false if p is likely prime
    */
    template<typename IntType>
    static bool WitnessFunction(const IntType &a, const IntType &d, uint32_t s, const IntType &p) {
        IntType mod = util::ModExp(a, d, p);
        bool prevMod = false;
        for (uint32_t i = 0; i < s; i++) {
            prevMod = (mod != IntType(1) && mod != p - IntType(1));
            mod = util::ModMul(mod, mod, p);
            if (mod == IntType(1) && prevMod) {
                return true;
            }
        }
        return (mod != IntType(1));
    }

    /**
     * A helper function to RootOfUnity function. This finds a generator for a given
     * prime q. Input: BigInteger q which is a prime. Output: A generator of prime q
     */
    template<typename IntType>
    static IntType FindGenerator(const IntType &q) {
        std::set<IntType> primeFactors;

        IntType qm1 = q - IntType(1);
        IntType qm2 = q - IntType(2);

        PrimeFactorize<IntType>(qm1, primeFactors);

        bool generatorFound = false;
        IntType gen;
        while (!generatorFound) {
            uint32_t count = 0;
            gen = RNG(qm2) + IntType(1);

            for (auto it = primeFactors.begin(); it != primeFactors.end(); ++it) {
                IntType t = qm1 / (*it);
                if (util::ModExp(gen, t, q) == IntType(1)) break;
                else
                    count++;
            }
            if (count == primeFactors.size()) generatorFound = true;
        }
        return gen;
    }

    /**
     * The Miller-Rabin Primality Test
     * 
     * @param p The candidate prime to test.
     * @param iterCount Number of iterations used for primality testing (default = 100.
     * @return False if evidence of non-primality is found. True is no evidence of non-primality is found.
     */
    template<typename IntType>
    bool IsPrime(const IntType &p, const uint32_t iterCount = 100) {
        if (p < IntType(2) || p == IntType(2) || p == IntType(3) || p == IntType(5)) return true;
        if (p % 2 == IntType(0)) return false;

        IntType d = p - IntType(1);
        uint32_t s = 0;
        while (d % 2 == IntType(0)) {
            d = d / IntType(2);
            s++;
        }
        bool composite = true;
        for (uint32_t i = 0; i < iterCount; ++i) {
            IntType a = RNG(p - IntType(3)) + IntType(2);
            composite = (WitnessFunction(a, d, s, p));
            if (composite) break;
        }
        return (!composite);
    }


    /**
     * Return greatest common divisor of two big integers.
     * 
     * @param a One integer to find greatest common divisor of.
     * @param b Another integer to find greatest common divisor of. 
     * @return The greatest common divisor of a and b.
     */
    template<typename IntType>
    IntType GCD(const IntType &a, const IntType &b) {
        IntType m_a = a;
        IntType m_b = b;
        IntType m_t;

        while (m_b != IntType(0)) {
            m_t = m_b;
            m_b = m_a % m_b;
            m_a = m_t;
        }
        return m_a;
    }

    template<typename IntType>
    static const IntType PollardRho(const IntType &n) {
        // check divisibility by 2
        if (n % 2 == IntType(0)) return IntType(2);

        IntType divisor(1);

        IntType c = RNG(n);
        IntType x = RNG(n);
        IntType xx(x);

        do {
            x = (x * x + c) % n;
            xx = (xx * xx + c) % n;
            xx = (xx * xx + c) % n;

            divisor = GCD((x > xx) ? x - xx : xx - x, n);
        } while (divisor == IntType(1));
        return divisor;
    }

    /**
     * Recursively factorizes to find the distinct primefactors of a number.
     * Side effect: the value of number is destroyed.
     * 
     * @param n The value to factorize. [note the value of n is destroyed]
     * @param primeFactors Set of factors found.
     */
    template<typename IntType>
    void PrimeFactorize(IntType n, std::set<IntType> &primeFactors) {

        if (n == IntType(0) || n == IntType(1)) return;

        if (MillerRabinPrimalityTest(n)) {
            primeFactors.insert(n);
            return;
        }
        IntType divisor(PollardRho(n));
        PrimeFactorize(divisor, primeFactors);
        n /= divisor;
        PrimeFactorize(n, primeFactors);
    }

    /**
     * 寻找最小的质数满足 mod m = 1, 其长度至少为nBits
     * @tparam IntType
     * @param nBits
     * @param m
     * @return
     */
    template<typename IntType>
    IntType FirstPrime(uint64_t nBits, uint64_t m) {
        IntType M(m);
        IntType q(IntType(1) << nBits);
        IntType r(util::Mod(q, M));
        IntType qNew(q + IntType(1) - r);
        IntType qNew1((q - 1) / m * m + 1);

        if (r > IntType(0))
            qNew += M;
        while (!MillerRabinPrimalityTest(qNew)) {
            if ((qNew += M) < q)
                FHE_THROW(ParamException, std::string(__func__) +": overflow growing candidate");
        }
        return qNew;
    }

    /**
     * 寻找小于q且距离q最近的质数满足 mod m = 1
     * @tparam IntType
     * @param q
     * @param m
     * @return
     */
    template<typename IntType>
    IntType PreviousPrime(const IntType &q, uint64_t m) {
        IntType M(m), qNew(q - M);
        while (!MillerRabinPrimalityTest(qNew)) {
            if ((qNew -= M) > q)
                FHE_THROW(ParamException, std::string(__func__) +": overflow shrinking candidate");
        }
        return qNew;
    }

    /**
     * 获取count个的比特数为bit_size的素数，满足mod factor = 1
     *
     * @tparam IntType
     * @param factor
     * @param bit_size 素数的比特数
     * @param count 素数的数量
     * @return
     */
    template<typename IntType>
    std::vector<IntType> GetPrimes(IntType factor, int bit_size, size_t count) {
        std::vector<IntType> destination;

        // Start with (2^bit_size - 1) / factor * factor + 1
        IntType value = ((IntType(1) << bit_size) - 1) / factor * factor + 1;

        IntType lower_bound = IntType(1) << (bit_size - 1);
        while (count > 0 && value > lower_bound) {
            if (is_prime(value)) {
                destination.emplace_back(value);
                count--;
            }
            value -= factor;
        }
        if (count > 0) {
            FHE_THROW(ParamException, "failed to find enough qualifying primes");
        }
        return destination;
    }

    /**
     * 判断value是否为素数
     * @tparam IntType
     * @param value
     * @param num_rounds 默认最大循环次数为40
     * @return
     */
    template<typename IntType>
    bool is_prime(const IntType &value, std::size_t num_rounds = 40) {
        // First check the simplest cases.
        if (value < 2) {
            return false;
        }
        if (2 == value) {
            return true;
        }
        if (0 == (value & 0x1)) {
            return false;
        }
        if (3 == value) {
            return true;
        }
        if (0 == (value % 3)) {
            return false;
        }
        if (5 == value) {
            return true;
        }
        if (0 == (value % 5)) {
            return false;
        }
        if (7 == value) {
            return true;
        }
        if (0 == (value % 7)) {
            return false;
        }
        if (11 == value) {
            return true;
        }
        if (0 == (value % 11)) {
            return false;
        }
        if (13 == value) {
            return true;
        }
        if (0 == (value % 13)) {
            return false;
        }

        // Second, Miller-Rabin test.
        // Find r and odd d that satisfy value = 2^r * d + 1.
        IntType d = value - 1;
        uint64_t r = 0;
        while (0 == (d & 0x1)) {
            d >>= 1;
            r++;
        }
        if (r == 0) {
            return false;
        }

        // 1) Pick a = 2, check a^(value - 1).
        // 2) Pick a randomly from [3, value - 1], check a^(value - 1).
        // 3) Repeat 2) for another num_rounds - 2 times.
        std::random_device rand;

        std::uniform_int_distribution<unsigned long long> dist(3, value.ConvertToLimb() - 1);
        for (size_t i = 0; i < num_rounds; i++) {
            IntType a = i ? dist(rand) : 2;
            IntType x = util::ModExp(a, d, value);
            if (x == 1 || x == value - 1) {
                continue;
            }
            uint64_t count = 0;
            do {
                x = util::ModMul(x, x, value);
                count++;
            } while (x != value - 1 && count < r - 1);
            if (x != value - 1) {
                return false;
            }
        }
        return true;
    }

    /**
     * 寻找大于q且距离q最近的质数满足 mod m = 1
     * @tparam IntType
     * @param q
     * @param m
     * @return
     */
    template<typename IntType>
    IntType NextPrime(const IntType &q, uint64_t m) {
        IntType M(m), qNew(q + M);
        while (!MillerRabinPrimalityTest(qNew)) {
            if ((qNew += M) < q)
                FHE_THROW(ParamException, std::string(__func__) +": overflow growing candidate");
        }
        return qNew;
    }

    template<typename IntType>
    IntType LastPrime(uint32_t nBits, uint64_t m) {
        if (nBits > 60)
            FHE_THROW(ParamException, std::string(__func__) +": Requested bit length " + std::to_string(nBits) +
                                                            " exceeds maximum allowed length 60");

        IntType M(m);
        IntType q(IntType(1) << nBits);
        IntType r(util::Mod(q, M));
        IntType qNew(q + IntType(1) - r);
        if (r < IntType(2))
            qNew -= M;
        while (!MillerRabinPrimalityTest(qNew)) {
            if ((qNew -= M) > q)
                FHE_THROW(ParamException, std::string(__func__) +": overflow shrinking candidate");
        }

        if (qNew.GetMSB() != nBits)
            FHE_THROW(ParamException,
                      std::string(__func__) +": Requested " + std::to_string(nBits) + " bits, but returned " +
                                            std::to_string(qNew.GetMSB()) + ". Please adjust parameters.");

        return qNew;
    }

    /**
     * Get the residue classes in Z*_n
     * 
     * @param n The input number.
     * @return Vector of residue classes x under Z_n such that gcd(x, n) == 1.
     */
    template<typename IntType>
    std::vector<IntType> GetTotientList(const IntType &n) {
        std::vector<IntType> result;
        IntType one(1);
        for (IntType i = IntType(1); i < n; i = i + IntType(1)) {
            if (GCD(i, n) == one) { result.push_back(i); }
        }
        return result;
    }

    /**
     * Returns the totient value: phi of a number n.
     * 
     * @param n The input number.
     * @return Phi of n which is the number of integers m coprime to n such that 1 <= m <= n.
     */
    // TODO: Replace with an native integer version in nbtheory2?
    uint64_t GetTotient(const uint64_t n) {
        std::set<BigInt> factors;
        BigInt enn(n);
        PrimeFactorize(enn, factors);

        BigInt primeProd(1);
        BigInt numerator(1);

        for (auto &r: factors) {
            numerator = numerator * (r - 1);
            primeProd = primeProd * r;
        }

        return ((enn / primeProd) * numerator).ConvertToLimb();
    }

    /**
     * Finds roots of unity for given input.  Assumes the the input is a power of two.
     * 
     * @param m Number which is cyclotomic
     * @param modulo Modulo which is used to find generator.
     * @return A root of unity;
     */
    template<typename IntType>
    IntType RootOfUnity(uint32_t m, const IntType &modulo) {
        IntType M(m);
        if ((modulo - IntType(1)) % m != IntType(0)) {
            // TODO: Throw an exception?
            return IntType();
        }
        IntType result;
        IntType gen = FindGenerator(modulo);
        IntType mid = (modulo - IntType(1)) / M;
        result = util::ModExp(gen, mid, modulo);
        if (result == IntType(1)) { result = RootOfUnity(m, modulo); }

        /**
          * At this point, result contains a primitive root of unity. However,
          * we want to return the minimum root of unity, to avoid different
          * crypto contexts having different roots of unity for the same
          * cyclotomic order and moduli. Therefore, we are going to cycle over
          * all primitive roots of unity and select the smallest one (minRU).
          *
          * To cycle over all primitive roots of unity, we raise the root of
          * unity in result to all the powers that are co-prime to the
          * cyclotomic order. In power-of-two cyclotomics, this will be the
          * set of all odd powers, but here we use a more general routine
          * to support arbitrary cyclotomics.
          *
          */
        IntType x = result % modulo;
        IntType minRU(x);
        IntType curPowIdx(1);
        std::vector<IntType> coprimes = GetTotientList<IntType>(m);
        for (uint32_t i = 0; i < coprimes.size(); i++) {
            auto nextPowIdx = coprimes[i];
            IntType diffPow(nextPowIdx - curPowIdx);
            for (IntType j(0); j < diffPow; j += IntType(1)) { x = (x * result) % modulo; }
            if (x < minRU && x != IntType(1)) { minRU = x; }
            curPowIdx = nextPowIdx;
        }
        return minRU;
    }


    template<typename IntType>
    bool MillerRabinPrimalityTest(const IntType &p, const uint32_t niter = 100) {
        static const IntType ZERO(0);
        static const IntType TWO(2);
        static const IntType THREE(3);
        static const IntType FIVE(5);

        if (p == TWO || p == THREE || p == FIVE)
            return true;
        if (p < TWO || (util::Mod(p, TWO) == ZERO))
            return false;

        IntType d(p - IntType(1));
        uint32_t s(0);
        while (util::Mod(d, TWO) == ZERO) {
            // d.DividedByEq(TWO);
            d = d >> 1;
            ++s;
        }
        for (uint32_t i = 0; i < niter; ++i) {
            if (WitnessFunction(util::ModAdd(RNG(p - THREE), TWO, p), d, s, p))
                return false;
        }
        return true;
    }

    template<typename IntType>
    IntType FindGeneratorCyclic(const IntType &q) {
        IntType phi_q(GetTotient(q));
        IntType phi_q_m1(GetTotient(q));
        std::set<IntType> primeFactors;
        PrimeFactorize<IntType>(phi_q, primeFactors);
        uint32_t cnt;
        IntType gen;
        do {
            cnt = 0;
            gen = RNG(phi_q_m1) + IntType(1);  // gen is random in [1, phi(q)]

            // Generator must lie in the group!
            if (GCD<IntType>(gen, q) != IntType(1))
                continue;

            // Order of a generator cannot divide any co-factor
            for (auto it = primeFactors.begin(); it != primeFactors.end(); ++it, ++cnt) {
                if (util::ModExp(gen, phi_q / (*it), q) == ONE)
                    break;
            }
        } while (cnt != primeFactors.size());
        return gen;
    }

    // TODO: Understand and rewrite these helper functions below
    /**
     * Method to reverse bits of num and return an unsigned int, for all bits up to
     * an including the designated most significant bit.
     *
     * @param input an unsigned int
     * @param msb the most significant bit.  All larger bits are disregarded.
     *
     * @return an unsigned integer that represents the reversed bits.
     */

    // precomputed reverse of a byte

    inline static unsigned char reverse_byte(unsigned char x) {
        static const unsigned char table[] = {
                0x00, 0x80, 0x40, 0xc0, 0x20, 0xa0, 0x60, 0xe0, 0x10, 0x90, 0x50, 0xd0, 0x30, 0xb0,
                0x70, 0xf0, 0x08, 0x88, 0x48, 0xc8, 0x28, 0xa8, 0x68, 0xe8, 0x18, 0x98, 0x58, 0xd8,
                0x38, 0xb8, 0x78, 0xf8, 0x04, 0x84, 0x44, 0xc4, 0x24, 0xa4, 0x64, 0xe4, 0x14, 0x94,
                0x54, 0xd4, 0x34, 0xb4, 0x74, 0xf4, 0x0c, 0x8c, 0x4c, 0xcc, 0x2c, 0xac, 0x6c, 0xec,
                0x1c, 0x9c, 0x5c, 0xdc, 0x3c, 0xbc, 0x7c, 0xfc, 0x02, 0x82, 0x42, 0xc2, 0x22, 0xa2,
                0x62, 0xe2, 0x12, 0x92, 0x52, 0xd2, 0x32, 0xb2, 0x72, 0xf2, 0x0a, 0x8a, 0x4a, 0xca,
                0x2a, 0xaa, 0x6a, 0xea, 0x1a, 0x9a, 0x5a, 0xda, 0x3a, 0xba, 0x7a, 0xfa, 0x06, 0x86,
                0x46, 0xc6, 0x26, 0xa6, 0x66, 0xe6, 0x16, 0x96, 0x56, 0xd6, 0x36, 0xb6, 0x76, 0xf6,
                0x0e, 0x8e, 0x4e, 0xce, 0x2e, 0xae, 0x6e, 0xee, 0x1e, 0x9e, 0x5e, 0xde, 0x3e, 0xbe,
                0x7e, 0xfe, 0x01, 0x81, 0x41, 0xc1, 0x21, 0xa1, 0x61, 0xe1, 0x11, 0x91, 0x51, 0xd1,
                0x31, 0xb1, 0x71, 0xf1, 0x09, 0x89, 0x49, 0xc9, 0x29, 0xa9, 0x69, 0xe9, 0x19, 0x99,
                0x59, 0xd9, 0x39, 0xb9, 0x79, 0xf9, 0x05, 0x85, 0x45, 0xc5, 0x25, 0xa5, 0x65, 0xe5,
                0x15, 0x95, 0x55, 0xd5, 0x35, 0xb5, 0x75, 0xf5, 0x0d, 0x8d, 0x4d, 0xcd, 0x2d, 0xad,
                0x6d, 0xed, 0x1d, 0x9d, 0x5d, 0xdd, 0x3d, 0xbd, 0x7d, 0xfd, 0x03, 0x83, 0x43, 0xc3,
                0x23, 0xa3, 0x63, 0xe3, 0x13, 0x93, 0x53, 0xd3, 0x33, 0xb3, 0x73, 0xf3, 0x0b, 0x8b,
                0x4b, 0xcb, 0x2b, 0xab, 0x6b, 0xeb, 0x1b, 0x9b, 0x5b, 0xdb, 0x3b, 0xbb, 0x7b, 0xfb,
                0x07, 0x87, 0x47, 0xc7, 0x27, 0xa7, 0x67, 0xe7, 0x17, 0x97, 0x57, 0xd7, 0x37, 0xb7,
                0x77, 0xf7, 0x0f, 0x8f, 0x4f, 0xcf, 0x2f, 0xaf, 0x6f, 0xef, 0x1f, 0x9f, 0x5f, 0xdf,
                0x3f, 0xbf, 0x7f, 0xff,
        };
        return table[x];
    }

    static int shift_trick[] = {0, 7, 6, 5, 4, 3, 2, 1};

    /* Function to reverse bits of num */
    inline uint32_t ReverseBits(uint32_t num, uint32_t msb) {
        uint32_t msbb = (msb >> 3) + (msb & 0x7 ? 1 : 0);

        switch (msbb) {
            case 1:
                return (reverse_byte((num) & 0xff) >> shift_trick[msb & 0x7]);

            case 2:
                return (reverse_byte((num) & 0xff) << 8 | reverse_byte((num >> 8) & 0xff)) >>
                                                                                           shift_trick[msb & 0x7];

            case 3:
                return (reverse_byte((num) & 0xff) << 16 | reverse_byte((num >> 8) & 0xff) << 8 |
                        reverse_byte((num >> 16) & 0xff)) >>
                                                          shift_trick[msb & 0x7];

            case 4:
                return (reverse_byte((num) & 0xff) << 24 | reverse_byte((num >> 8) & 0xff) << 16 |
                        reverse_byte((num >> 16) & 0xff) << 8 | reverse_byte((num >> 24) & 0xff)) >>
                                                                                                  shift_trick[msb &
                                                                                                              0x7];
            default:
                return -1;
                // OPENFHE_THROW(math_error, "msbb value not handled:" +
                // std::to_string(msbb));
        }
    }

    inline uint32_t GetMSB(uint64_t x) {
        if (x == 0) return 0;

        // hardware instructions for finding MSB are used are used;
#if defined(_MSC_VER)
        // a wrapper for VC++
        unsigned long msb;// NOLINT
        _BitScanReverse64(&msb, x);
        return msb + 1;
#else
        // a wrapper for GCC
        return 64 - (sizeof(unsigned long) == 8 ? __builtin_clzl(x) : __builtin_clzll(x));// NOLINT
#endif
    }

    inline uint32_t GetMSB(uint32_t x) { return GetMSB((uint64_t) x); }

    /**
     * Get MSB of an unsigned 64 bit integer.
     *
     * @param x the input to find MSB of.
     *
     * @return the index of the MSB bit location.
     */
    inline uint32_t GetMSB64(uint64_t x) { return GetMSB(x); }


    inline void PrecomputeAutoMap(uint32_t n, uint32_t k, std::vector<uint32_t> *precomp) {
        uint32_t m = n << 1;  // cyclOrder
        uint32_t logm = std::round(log2(m));
        uint32_t logn = std::round(log2(n));
        for (uint32_t j = 0; j < n; j++) {
            uint32_t jTmp = ((j << 1) + 1);
            uint32_t idx = ((jTmp * k) - (((jTmp * k) >> logm) << logm)) >> 1;
            uint32_t jrev = ReverseBits(j, logn);
            uint32_t idxrev = ReverseBits(idx, logn);
            (*precomp)[jrev] = idxrev;
        }
    }

    inline uint32_t ModInverse(uint32_t a, uint32_t b) {
        // usint b0 = b;
        uint32_t t, q;
        uint32_t x0 = 0, x1 = 1;
        if (b == 1)
            return 1;
        while (a > 1) {
            q = a / b;
            t = b, b = a % b, a = t;
            t = x0, x0 = x1 - q * x0, x1 = t;
        }
        // if (x1 < 0) x1 += b;
        // TODO: x1 is never < 0

        return x1;
    }
}// namespace isecfhe
