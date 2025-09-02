#pragma once

#include "big_integer.h"
#include "modulus.h"

namespace isecfhe {

    namespace util {

        template<typename IntegerType>
        IntegerType Mod(const IntegerType &operand, const IntegerType &modulus) {
            return operand % modulus;
        }


        template<typename IntegerType>
        IntegerType ModBarrett(const IntegerType &operand,
                               const Modulus <IntegerType> &modulus) {
            IntegerType estimated = operand * (modulus.GetConstRatio());
            estimated >>= 128;
            estimated *= modulus.GetValue();
            IntegerType ret = operand - estimated;
            return ret < modulus.GetValue() ? ret : ret - modulus.GetValue();
        }

        template<typename NativeInt>
        BigInteger<NativeInt> ModBarrett(const BigInteger<NativeInt> &operand,
                                         const Modulus <NativeInt> &modulus) {
            BigInteger<NativeInt> estimated = operand * (modulus.GetConstRatio());
            estimated >>= 128;
            estimated *= modulus.GetValue();
            BigInteger<NativeInt> ret = operand - estimated;
            return ret < modulus.GetValue() ? ret : ret - modulus.GetValue();
        }

//        inline uint64_t ModBarrett64(uint64_t &operand, const Modulus<uint64_t> &modulus) {
//            uint64_t modulusValue = modulus.GetValue().getValueOfIndex(0);
//            uint64_t ration1 = modulus.GetConstRatio().getValueOfIndex(1);
//            __int128 estimated = static_cast<__int128>(operand) * static_cast<__int128>(ration1);
//            estimated >>= 64;
//
//            estimated *= modulusValue;
//            uint64_t ret = operand - estimated;
//            return ret < modulusValue ? ret : ret - modulusValue;
//        }

        template<typename IntegerType>
        IntegerType ModAdd(const IntegerType &operand1,
                           const IntegerType &operand2,
                           const IntegerType &modulus) {
            IntegerType a(operand1);
            IntegerType b(operand2);
            if (a >= modulus) { a = Mod(a, modulus); }
            if (b >= modulus) { b = Mod(b, modulus); }
            a += b;
            return Mod(a, modulus);
        }


        template<typename IntegerType = limbtype>
        IntegerType ModAddFast(const IntegerType &a, const IntegerType &b, const IntegerType &modulus) {
            IntegerType res = a + b;
            if (res >= modulus)
                res -= modulus;
            return res;
        }


        template<typename IntegerType>
        IntegerType ModIncrement(const IntegerType &operand,
                                 const IntegerType &modulus) {
            IntegerType a(1);
            return ModAdd(operand, a, modulus);
        }

        template<typename IntegerType>
        IntegerType ModSub(const IntegerType &operand1, const IntegerType &operand2,
                           const IntegerType &modulusValue) {
            if (operand1 == operand2) {
                return IntegerType(0);
            }

            IntegerType a(operand1);
            IntegerType b(operand2);
            if (a >= modulusValue) { a = Mod(a, modulusValue); }
            if (b >= modulusValue) { b = Mod(b, modulusValue); }

            bool negative = a < b;
            IntegerType difference = negative ? b - a : a - b;

            if (negative) {
                return modulusValue - difference;
            } else {
                return difference;
            }

        }


        /**
         * a/b 均小于 modulus
         * @tparam IntegerType
         * @param a
         * @param b
         * @param modulus
         * @return
         */
        template<typename IntegerType = limbtype>
        IntegerType ModSubFast(const IntegerType &a, const IntegerType &b, const IntegerType &modulus) {
            if (a < b)
                return a + modulus - b;
            return a - b;
        }

        template<typename IntegerType, typename std::enable_if<std::is_integral_v<IntegerType>, bool>::type = true>
        IntegerType ModMul(const IntegerType &a,
                           const IntegerType &b,
                           const IntegerType &modulus) {
            auto av{a};
            auto bv{b};
            if (a >= modulus)
                av = a % modulus;
            if (b >= modulus)
                bv = b % modulus;
            DNativeInt rv{static_cast<DNativeInt>(av) * bv};
            DNativeInt dmv{modulus};
            if (rv >= dmv)
                rv %= dmv;
            return rv;

        }


        template<typename NativeInt>
        BigInteger<NativeInt> ModMul(const BigInteger<NativeInt> &operand,
                                     const BigInteger<NativeInt> &another,
                                     const BigInteger<NativeInt> modulus) {
            BigInteger<NativeInt> mulRes = operand.Mul(another);
            return Mod(mulRes, modulus);
        }


        template<typename NativeInt>
        BigInteger<NativeInt> ModMul(const BigInteger<NativeInt> &operand,
                                     const BigInteger<NativeInt> &another,
                                     const uint64_t modulus) {
            BigInteger<NativeInt> mulRes = operand.Mul(another);
            return Mod(mulRes, BigInteger<NativeInt>(modulus));
        }


        static void MultD(limbtype a, limbtype b, typeD &res) {
            if constexpr (std::is_same_v<limbtype, uint32_t>) {
                uint64_t c{static_cast<uint64_t>(a) * b};
                res.hi = static_cast<uint32_t>(c >> 32);
                res.lo = static_cast<uint32_t>(c);
            }

            if constexpr (std::is_same_v<limbtype, uint64_t>) {
#if defined(HAVE_INT128)
                // includes defined(__x86_64__), defined(__powerpc64__), defined(__riscv), defined(__s390__)
                uint128_t c{static_cast<uint128_t>(a) * b};
                res.hi = static_cast<uint64_t>(c >> 64);
                res.lo = static_cast<uint64_t>(c);
#endif
            }
        }


        /**
  * Barrett modulus multiplication that assumes the operands are < modulus.
  *
  * @param &b is the scalar to multiply.
  * @param &modulus is the modulus to perform operations with.
  * @param &mu is the Barrett value.
  * @return is the result of the modulus multiplication operation.
  */
        /* Source: http://homes.esat.kuleuven.be/~fvercaut/papers/bar_mont.pdf
        @article{knezevicspeeding,
        title={Speeding Up Barrett and Montgomery Modular Multiplications},
        author={Knezevic, Miroslav and Vercauteren, Frederik and Verbauwhede,
        Ingrid}
        }
        We use the Generalized Barrett modular reduction algorithm described in
        Algorithm 2 of the Source. The algorithm was originally proposed in J.-F.
        Dhem. Modified version of the Barrett algorithm. Technical report, 1994
        and described in more detail in the PhD thesis of the author published at
        http://users.belgacom.net/dhem/these/these_public.pdf (Section 2.2.4).
        We take \alpha equal to n + 3. So in our case, \mu = 2^(n + \alpha) =
        2^(2*n + 3). Generally speaking, the value of \alpha should be \ge \gamma
        + 1, where \gamma + n is the number of digits in the dividend. We use the
        upper bound of dividend assuming that none of the dividends will be larger
        than 2^(2*n + 3). The value of \mu is computed by NativeVector::ComputeMu.
        */
        template<typename IntegerType = limbtype>
        IntegerType
        ModMulFast(const IntegerType &a, const IntegerType &b, const IntegerType &modulus, const IntegerType &mu) {
            int64_t n = util::GetMSBForLimb(modulus) - 2;
            typeD tmp;
            MultD(a, b, tmp);
            auto rv = tmp.Convert2DNativeInt();
            MultD(tmp.RShiftD(n), mu, tmp);
            rv -= DNativeInt(modulus) * (tmp.Convert2DNativeInt() >> (n + 7));
            IntegerType r(rv);
            if (r >= modulus)
                r -= modulus;
            return r;
        }


        template<typename IntegerType = limbtype>
        IntegerType ModMulFast(const IntegerType &a, const IntegerType &b, const IntegerType &modulus) {
            DNativeInt rv{static_cast<DNativeInt>(a) * b};
            DNativeInt dmv{modulus};
            if (rv >= dmv)
                rv %= dmv;
            return rv;
        }


        template<typename NativeInt>
        BigInteger<NativeInt> ComputeMu(BigInteger<NativeInt> val) {
            return (BigInteger<NativeInt>(1) << (2 * val.GetMSB() + 3)).DividedBy(val);
        }

        template<typename NativeInt = limbtype>
        NativeInt ComputeMu(NativeInt val) {
            auto &&tmp{DNativeInt{1} << (2 * util::GetMSBForLimb(val) + 3)};
            return tmp / DNativeInt(val);
        }

        template<typename IntegerType, typename std::enable_if<std::is_integral_v<IntegerType>, bool>::type = true>
        IntegerType ModExp(const IntegerType &operand,
                           const IntegerType &exponent,
                           const IntegerType &modulus) {

            __int128 t{operand};
            __int128 p{exponent};
            __int128 m{modulus};
            __int128 r{1};
            if (p & 0x1) {
                r = r * t;
                if (r >= m)
                    r = r % m;
            }
            while (p >>= 1) {
                t = t * t;
                if (t >= m)
                    t = t % m;
                if (p & 0x1) {
                    r = r * t;
                    if (r >= m)
                        r = r % m;
                }
            }
            return r;
        }


        template<typename NativeInt>
        BigInteger<NativeInt> ModExp(const BigInteger<NativeInt> &operand,
                                     const BigInteger<NativeInt> &exponent,
                                     const BigInteger<NativeInt> &modulus) {
            BigInteger<NativeInt> mid = Mod(operand, modulus);
            BigInteger<NativeInt> product(1);
            BigInteger<NativeInt> exp(exponent);

            if (exponent == 0) { return 1; }

            if (operand == 1) { return operand; }
            if (exponent == 1) { return Mod(operand, modulus); }
            while (true) {
                if (exp.getValueOfIndex(0) % 2 == 1) { product = product * mid; }
                if (product >= modulus) { product = Mod(product, modulus); }
                exp = exp >> 1;
                if (exp == 0) { break; }
                mid = mid * mid;
                mid = Mod(mid, modulus);
            }
            return product;
        }


        template<typename NativeInt>
        BigInteger<NativeInt>
        ExtendedEuclideanAlgorithm(const BigInteger<NativeInt> &a, const BigInteger<NativeInt> &b,
                                   BigInteger<NativeInt> &x, BigInteger<NativeInt> &y) {
            if (a == 0) {
                x = 0;
                y = 1;
                return b;
            }

            BigInteger<NativeInt> x1, y1;
            auto tmp = b.DividedBy(a);
            BigInteger<NativeInt> gcd = ExtendedEuclideanAlgorithm(tmp.second, a, x1, y1);

            x = y1 - tmp.first * x1;
            y = x1;

            // Handle negative coefficients
            if (a < 0) { x = -x; }
            if (b < 0) { y = -y; }

            return gcd;
        }

        template<typename IntegerType>
        IntegerType
        ExtendedEuclideanAlgorithm(const IntegerType &a, const IntegerType &b,
                                   IntegerType &x, IntegerType &y) {
            if (a == 0) {
                x = 0;
                y = 1;
                return b;
            }

            IntegerType x1, y1;
//            auto tmp = b.DividedBy(a);
            IntegerType gcd = ExtendedEuclideanAlgorithm(b % a, a, x1, y1);
//            if (b % a != tmp.second) {
//                FHEDebug("b.DividedBy(a) quotientIn = "<< tmp.first  << " , remainderIn="  << tmp.second);
//                FHEDebug("b=" << b <<" a=" << a<< "  b%a = "<< b % a);
//            }

            x = y1 - b / a * x1;
            y = x1;

            // Handle negative coefficients
            if (a < 0) { x = -x; }
            if (b < 0) { y = -y; }

            return gcd;
        }


        template<typename IntegerType, typename std::enable_if<std::is_integral_v<IntegerType>, bool>::type = true>
        IntegerType ModInverse(const IntegerType &value, const IntegerType &mod) {
            SignedNativeInt modulus(mod);
            SignedNativeInt a(value % mod);
            if (a == 0) {
                std::string msg = std::to_string(value) + " does not have a ModInverse using " + std::to_string(mod);
                FHE_THROW(ParamException, msg);
            }
            if (modulus == 1)
                return IntegerType();

            SignedNativeInt y{0};
            SignedNativeInt x{1};
            while (a > 1) {
                auto t = modulus;
                auto q = a / t;
                modulus = a % t;
                a = t;
                t = y;
                y = x - q * y;
                x = t;
            }
            if (x < 0)
                x += mod;
            return x;
        }

        template<typename NativeInt>
        BigInteger<NativeInt> ModInverse(const BigInteger<NativeInt> &operand,
                                         const BigInteger<NativeInt> &modulus) {
            if (operand == 0) {
                FHE_THROW(ParamException, "operand is 0");
            }

            // Step 1: Compute gcd(a, b) and coefficients x, y such that ax + by = gcd(a, b)
            BigInteger<NativeInt> a = operand;
            BigInteger<NativeInt> b = modulus;
            BigInteger<NativeInt> x, y;
            BigInteger<NativeInt> gcd = ExtendedEuclideanAlgorithm(a, b, x, y);

            // Step 2: If gcd(a, b) is not 1, then a has no inverse mod b
            if (gcd != 1) {
                FHE_THROW(isecfhe::MathException, "Modular inverse does not exist");
            }

            // Step 3: Compute a^-1 mod b using coefficient x
            BigInteger<NativeInt> inverse = x % b;// inverse = x mod b
//            FHEDebug("inverse:{} x:{} b:{}", inverse, x, b);

            if (inverse < 0) {
                inverse += (b < 0 ? -b : b);// inverse = inverse + |b| if inverse < 0
            }
            if (modulus < 0) {
                inverse = -inverse;// if modulus is negative, inverse should be negative
            }
            if (operand < 0) {
                inverse = -inverse;// if operand is negative, inverse should be negative
            }
            return inverse;
        }


        template<typename NativeInt>
        BigInteger<NativeInt> ModInverse(const BigInteger<NativeInt> &operand,
                                         const NativeInt &modulus) {
            return ModInverse(operand, BigInteger<NativeInt>(modulus));
        }


        template<typename NativeInt>
        BigInteger<NativeInt> Gcd(const BigInteger<NativeInt> &a, const BigInteger<NativeInt> &b) {
            if (a == 0 || b == 0) { return a != 0 ? a : b; }

            BigInteger<NativeInt> absA = a.Abs();
            BigInteger<NativeInt> absB = b.Abs();
            while (absB != 0) {
                BigInteger<NativeInt> tmp = absA % absB;
                absA = absB;
                absB = tmp;
            }

            return absA;
        }

        template<typename NativeInt>
        void ExtendedGcd(const BigInteger<NativeInt> &a, const BigInteger<NativeInt> &b,
                         BigInteger<NativeInt> &x, BigInteger<NativeInt> &y) {
            if (b == 0) {
                x = 1;
                y = 0;
                return;
            }

            BigInteger<NativeInt> absA = a.Abs();
            BigInteger<NativeInt> absB = b.Abs();
            BigInteger<NativeInt> s = 0;
            BigInteger<NativeInt> oldS = 1;
            BigInteger<NativeInt> t = 1;
            BigInteger<NativeInt> oldT = 0;
            BigInteger<NativeInt> r = absB;
            BigInteger<NativeInt> oldR = absA;

            while (r != 0) {
                BigInteger<NativeInt> quotient = oldR.DividedBy(r).first;

                BigInteger<NativeInt> tmp = r;
                r = oldR - quotient * r;
                oldR = tmp;

                tmp = s;
                s = oldS - quotient * s;
                oldS = tmp;

                tmp = t;
                t = oldT - quotient * t;
                oldT = tmp;
            }

            if (a < 0) {
                x = oldS < 0 ? oldS + b : oldS;
            } else {
                x = oldS;
            }
            if (b < 0) {
                y = oldT < 0 ? oldT + a : oldT;
            } else {
                y = oldT;
            }
        }


        template<typename IntegerType>
        IntegerType ModNegate(const IntegerType &operand,
                              const Modulus <IntegerType> &modulus) {
            // Calculate modulus
            IntegerType mod = modulus.GetValue();
            bool is_neg_modulus = mod.getSign();

            // Calculate the modular negation of the absolute value of the operand
            IntegerType absOperand = operand.Abs();
            IntegerType negation =
                    (is_neg_modulus ? -absOperand : mod - absOperand) % mod;

            // If the operand was negative, return the negation
            if (operand < 0) { return negation; }

            // If the operand was non-negative, return the negation or 0, whichever is smaller
            IntegerType result = (negation == 0 ? 0 : negation);

            // Make sure the result is non-negative and less than the modulus
            if (result < 0 || result >= mod) {
                FHE_THROW(isecfhe::MathException, "Modular negation is not possible");
            }

            return result;
        }


        template<typename NativeInt>
        BigInteger<NativeInt> ModNegate(const BigInteger<NativeInt> &operand,
                                        const Modulus <NativeInt> &modulus) {
            // Calculate modulus
            BigInteger<NativeInt> mod = modulus.GetValue();
            bool is_neg_modulus = mod.getSign();

            // Calculate the modular negation of the absolute value of the operand
            BigInteger<NativeInt> absOperand = operand.Abs();
            BigInteger<NativeInt> negation =
                    (is_neg_modulus ? -absOperand : mod - absOperand) % mod;

            // If the operand was negative, return the negation
            if (operand < 0) { return negation; }

            // If the operand was non-negative, return the negation or 0, whichever is smaller
            BigInteger<NativeInt> result = (negation == 0 ? 0 : negation);

            // Make sure the result is non-negative and less than the modulus
            if (result < 0 || result >= mod) {
                FHE_THROW(isecfhe::MathException, "Modular negation is not possible");
            }

            return result;
        }

    }// namespace util
}// namespace isecfhe
