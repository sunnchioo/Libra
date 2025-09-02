#pragma once

#include "exception.h"
#include <cstdint>
#include <iosfwd>
#include <set>

namespace isecfhe {
    template<typename IntType>
    void PrimeFactorize(IntType n, std::set<IntType> &primeFactors);

    namespace util {

        inline int nlz64(uint64_t x) {
            int n;
            if (x == 0) { return (64); }
            n = 0;
            if (x <= 0x00000000FFFFFFFF) {
                n = n + 32;
                x = x << 32;
            }
            if (x <= 0x0000FFFFFFFFFFFF) {
                n = n + 16;
                x = x << 16;
            }
            if (x <= 0x00FFFFFFFFFFFFFF) {
                n = n + 8;
                x = x << 8;
            }
            if (x <= 0x0FFFFFFFFFFFFFFF) {
                n = n + 4;
                x = x << 4;
            }
            if (x <= 0x3FFFFFFFFFFFFFFF) {
                n = n + 2;
                x = x << 2;
            }
            if (x <= 0x7FFFFFFFFFFFFFFF) { n = n + 1; }
            return n;
        }

        inline int nlz32(uint32_t x) {
            int n;

            if (x == 0) { return (32); }
            n = 0;
            if (x <= 0x0000FFFF) {
                n = n + 16;
                x = x << 16;
            }
            if (x <= 0x00FFFFFF) {
                n = n + 8;
                x = x << 8;
            }
            if (x <= 0x0FFFFFFF) {
                n = n + 4;
                x = x << 4;
            }
            if (x <= 0x3FFFFFFF) {
                n = n + 2;
                x = x << 2;
            }
            if (x <= 0x7FFFFFFF) { n = n + 1; }
            return n;
        }


        template<typename NativeInt, std::enable_if_t<
                std::is_same_v<NativeInt, uint64_t> || std::is_same_v<NativeInt, uint32_t>, bool> = true>
        inline int nlz(NativeInt x) {
            if (typeid(x) == typeid(uint64_t)) {
                return nlz64(x);
            } else if (typeid(x) == typeid(uint32_t)) {
                return nlz32(x);
            } else {
                FHE_THROW(isecfhe::TypeException, "not support native int type");
            }
        }

        template<typename NativeInt, std::enable_if_t<
                std::is_same_v<NativeInt, uint64_t> || std::is_same_v<NativeInt, uint32_t>, bool> = true>
        uint32_t GetMSBForLimb(NativeInt x) {
            uint64_t y = ((uint64_t) x);
            if (y == 0) {
                return 0;
            } else {
                return 64 - (sizeof(unsigned long) == 8 ? __builtin_clzl(y) : __builtin_clzll(y));
            }
        }


        /**
         * Determines if a number is a power of 2.
         *
         * @param Input to test if it is a power of 2.
         * @return is true if the unsigned int is a power of 2.
         */
        template<typename T, std::enable_if_t<std::is_integral_v<T>, bool> = true>
        inline constexpr bool IsPowerOfTwo(T Input) {
            return Input && !(Input & (Input - 1));
        }
    }// namespace util
};   // namespace isecfhe
