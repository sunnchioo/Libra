#pragma once

#include "big_integer.h"

namespace isecfhe {

    template<typename IntegerType>
    class Modulus {

    public:
        Modulus() {}

        Modulus(IntegerType val) { set_value(val); }


        template<typename T = IntegerType, std::enable_if_t<std::is_integral_v<T>, bool> = true>
        Modulus(std::string strVal) {
            set_value(IntegerType(std::stoi((strVal))));
        }

        Modulus(std::string strVal) {
            set_value(IntegerType(strVal));
        }

        Modulus(const Modulus &modulus) {
            value = modulus.value;
            is_prime = modulus.is_prime;
            is_empty = modulus.is_empty;
            const_ratio = modulus.const_ratio;
        }

        inline const IntegerType GetConstRatio() const { return const_ratio; };

        inline const IntegerType GetValue() const { return value; };

        inline const bool IsEmpty() const { return is_empty; };

        Modulus &operator=(const Modulus &val) {
            if (this != &val) { this->set_value(val.value); }
            return *this;
        }

        Modulus &operator=(Modulus &&val) noexcept {
            if (this != &val) { this->set_value(val.value); }
            return *this;
        }

    private:
        IntegerType value;
        bool is_prime = false;
        bool is_empty = false;
        IntegerType const_ratio;

        static const IntegerType numerator = IntegerType(1) << 128;

        void set_value(IntegerType input) {
            if (input == 0UL) {
                value = 0;
                const_ratio = IntegerType(0);
            } else {
                value = input;
                const_ratio = numerator / input;
                this->is_prime = IsPrime(input);
                this->is_empty = input == 1;
            }
        }
    };
}// namespace isecfhe
