#pragma once

#include "exception.h"
#include "integer_interface.h"
#include "numth_util.h"
#include <cstdint>
#include <iosfwd>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <cmath>
#include <limits>

namespace isecfhe {

    template<typename NativeInt>
    class BigInteger;


    using uint128_t = unsigned __int128;
    using int128_t = __int128;

    template<typename utype>
    struct DataTypes {
        using SignedType = void;
        using DoubleType = void;
        using SignedDoubleType = void;
    };
    template<>
    struct DataTypes<uint32_t> {
        using SignedType = int32_t;
        using DoubleType = uint64_t;
        using SignedDoubleType = int64_t;
    };
    template<>
    struct DataTypes<uint64_t> {
        using SignedType = int64_t;
#if defined(HAVE_INT128)
        using DoubleType       = unsigned __int128;
        using SignedDoubleType = __int128;
#else
        using DoubleType = uint64_t;
        using SignedDoubleType = int64_t;
#endif
    };
#if defined(HAVE_INT128)
    template <>
    struct DataTypes<unsigned __int128> {
        using SignedType       = __int128;
        using DoubleType       = unsigned __int128;;
        using SignedDoubleType = __int128;
    };
#endif


    typedef uint64_t limbtype;

    using SignedNativeInt = typename DataTypes<limbtype>::SignedType;
    using DNativeInt = typename DataTypes<limbtype>::DoubleType;
    using SDNativeInt = typename DataTypes<limbtype>::SignedDoubleType;


    static inline int LimbTypeMaxBits = std::numeric_limits<limbtype>::digits;

    struct typeD {
        limbtype hi{0};
        limbtype lo{0};


        DNativeInt Convert2DNativeInt() {
            return (DNativeInt(hi) << LimbTypeMaxBits) | lo;
        }


        limbtype RShiftD(int64_t shift) {
            return (lo >> shift) | (hi << (LimbTypeMaxBits - shift));
        }
    };


    template<uint32_t N>
    struct Log2 {
        static const uint32_t value = 1 + Log2<N / 2>::value;
    };

    template<>
    struct Log2<2> {
        static const uint32_t value = 1;
    };

    typedef BigInteger<limbtype> BigInt;

    template<typename NativeInt>
    class BigInteger : public IntegerInterface<BigInteger<NativeInt>> {

    public:
        using Slimb_t = typename DataTypes<NativeInt>::SignedType;
        using Dlimb_t = typename DataTypes<NativeInt>::DoubleType;
        using SDlimb_t = typename DataTypes<NativeInt>::SignedDoubleType;


        // CONSTRUCTORS
        explicit BigInteger(const std::string &strValue = "") {
            AssignVal(strValue);
        }

        explicit BigInteger(const char *str) : BigInteger(std::string(str)) {};

        BigInteger(uint64_t val, bool signVal = false) {
            this->sign = signVal;
            if (std::is_same<uint64_t, NativeInt>::value) {
                value.push_back(val);
            } else if (std::is_same<uint32_t, NativeInt>::value) {
                if (val >= (1LL << 32)) {
                    value.push_back(val & m_MaxLimb);
                    value.push_back(val >> 32);
                } else {
                    value.push_back(val);
                }
            }
            RefreshMSB();
        }

        BigInteger(uint32_t val, bool signVal = false) {
            this->sign = signVal;
            value.push_back(val);
            RefreshMSB();
        }

        BigInteger(std::vector<NativeInt> vals, bool sign = false) {
            while (!vals.empty() && !vals.back()) { vals.pop_back(); }
            if (vals.empty()) {
                value.push_back(0);
                return;
            }
            this->sign = sign;
            value = vals;
            m_MSB = (vals.size() - 1) * m_limbBitLength + m_GetMSBForLimb(vals.back());
            if (m_MSB == 0) { sign = false; }
        }

        const std::string ConvertToString() const {
            bool negative = sign;
            if (value.size() == 1) {
                if (value[0] == 0) { negative = false; }
                return std::string(negative ? "-" : "").append(std::to_string(value[0]));
            }

            std::vector<char> decimalArr;
            decimalArr.push_back(0);

            for (int i = value.size() - 1; i >= 0; --i) {
                int maxBitIdx = m_limbBitLength - 1;
                for (int j = maxBitIdx; j >= 0; --j) {
                    char carry = 0;
                    for (auto &m: decimalArr) {
                        m *= 2;
                        m = (char) (m + carry);
                        carry = 0;
                        if (m > 9) {
                            m -= 10;
                            carry = 1;
                        }
                    }

                    if (carry == 1) {
                        decimalArr.push_back(1);
                        carry = 0;
                    }

                    uint64_t bMask = 1;
                    for (int k = 0; k < j; ++k) { bMask <<= 1; }
                    if ((value[i] & bMask) != 0) { decimalArr[0] += 1; }
                }
            }

            std::string printValue;
            for (auto it = decimalArr.rbegin(); it != decimalArr.rend(); ++it) {
                // print type of *it
                // print type of char

                printValue.push_back((char) (*it + '0'));
            }

            return std::string(negative ? "-" : "").append(printValue);
        }

        NativeInt ConvertToLimb() const {
            if (value.size() > 1) {
                FHE_THROW(ConfigException, "value size not 1");
            }
            if (sign) {
                FHE_THROW(ConfigException, "value is negative");
            }
            return value[0];
        }

        template<typename T, typename std::enable_if<std::is_integral<T>::value &&
                                                     std::is_signed_v<T>, bool>::type  = true>
        T ConvertToInt() const {
            if (value.size() > 1) {
                FHE_THROW(ConfigException, "value size not 1");
            }

            if (sign) {
                T res = value[0];
                return -res;
            } else {
                return T(value[0]);
            }
        }

        template<typename T, typename std::enable_if<std::is_integral<T>::value &&
                                                     !std::is_signed_v<T>, bool>::type  = true>
        T ConvertToInt() const {
            if (value.size() > 1) {
                FHE_THROW(ConfigException, "value size not 1");
            }
            if (sign) {
                FHE_THROW(ConfigException, "value is negative");
            }
            return T(value[0]);
        }

        template<typename T, typename std::enable_if<!std::is_same<T, const BigInteger>::value
                                                     && !std::is_same<T, uint64_t>::value,
                bool>::type = true>
        BigInteger &operator=(const T &val) {
            return (*this = BigInteger(val));
        }

        BigInteger &operator=(const uint64_t &val) {
            return (*this = BigInteger(val, false));
        }
//
//        constexpr BigInteger& operator=(const BigInteger& val) noexcept {
//            value = val.value;
//            sign = val.sign;
//            m_MSB = val.m_MSB;
//            return *this;
//        }

//        constexpr BigInteger& operator=(BigInteger&& val) noexcept {
//            value = std::move(val.value);
//            sign = val.sign;
//            m_MSB = val.m_MSB;
//            return *this;
//        }


        /**
         * Constructors from smaller basic types
         *
         * @param val is the initial integer represented as a basic integer type.
         */
        template<typename T, typename std::enable_if<std::is_integral<T>::value &&
                                                     !std::is_same<T, char>::value &&
                                                     !std::is_same<T, uint64_t>::value,
                bool>::type = true>
        BigInteger(T val) {
            if (val < 0) {
                *this = BigInteger(static_cast<uint64_t>(-val), true);
            } else {
                *this = BigInteger(static_cast<uint64_t>(val));
            }
        }// NOLINT



        BigInteger PrepModMulConst(const BigInteger &modulus) const {
            if (modulus == 0)
                FHE_THROW(ParamException, "Divide by zero");
            if (modulus.GetMSB() > m_uintBitLength) {
                FHE_THROW(ParamException, "modulus too big");
            }
            BigInteger &&w(this->LeftShift(m_limbBitLength));
            return w / modulus;
        }


        /**
         * @brief Compare the current BigInteger with another one
         * 
         * @param another is the BigInteger to be compared with. 
         * @return int -1 for strictly less than, 0 for equal to and 1 for strictly greater than conditions.
         */
        int Compare(const BigInteger<NativeInt> &another) const {
            if (!sign && another.sign) { return 1; }
            if (sign && !another.sign) { return -1; }
            int absoluteCompare = AbsoluteCompare(another);
            return sign ? -absoluteCompare : absoluteCompare;
        }

        int AbsoluteCompare(const BigInteger<NativeInt> &another) const {
            if (m_MSB < another.m_MSB) {
                return -1;
            } else if (m_MSB > another.m_MSB) {
                return 1;
            } else {
                if (value.size() != another.value.size()) {
                    FHE_THROW(MathException, "value size not match");
                }
                for (int i = value.size() - 1; i >= 0; i--) {
                    if (value[i] > another.value[i]) {
                        return 1;
                    } else if (value[i] < another.value[i]) {
                        return -1;
                    }
                }
                return 0;
            }
        }

        BigInteger<NativeInt> Abs() const {
            if (value.empty()) {
                return BigInteger<NativeInt>(0);
            } else {
                return BigInteger<NativeInt>(value, false);
            }
        }


        BigInteger<NativeInt> Add(const BigInteger<NativeInt> &num) const {
            int absoluteCompare = AbsoluteCompare(num);
            if (!sign && num.sign) {
                if (AbsoluteCompare(num) == 0) {
                    return BigInteger<NativeInt>();
                } else if (absoluteCompare > 0) {
                    return SubWithSameSign(num);
                } else {
                    return num.SubWithSameSign(*this, true);
                }
            }
            if (sign && !num.sign) {
                if (AbsoluteCompare(num) == 0) {
                    return BigInteger<NativeInt>();
                } else if (absoluteCompare > 0) {
                    return SubWithSameSign(num, sign);
                } else {
                    return num.SubWithSameSign(*this);
                }
            }
            return AddWithSameSign(num);
        }

        BigInteger<NativeInt> &AddEq(const BigInteger<NativeInt> &num) {
            int absoluteCompare = AbsoluteCompare(num);
            if (sign == false && num.sign == true) {
                if (AbsoluteCompare(num) == 0) {
                    value.clear();
                    value.push_back(0);
                } else if (absoluteCompare > 0) {
                    // AssignObj(SubWithSameSign(num));
                    *this = SubWithSameSign(num);
                } else {
                    // AssignObj(num.SubWithSameSign(*this, true));
                    *this = num.SubWithSameSign(*this, true);
                }
            } else if (sign == true && num.sign == false) {
                if (AbsoluteCompare(num) == 0) {
                    value.clear();
                    value.push_back(0);
                } else if (absoluteCompare > 0) {
                    // AssignObj(SubWithSameSign(num, sign));
                    *this = SubWithSameSign(num, sign);
                } else {
                    // AssignObj(num.SubWithSameSign(*this));
                    *this = num.SubWithSameSign(*this);
                }
            } else {
                // AssignObj(AddWithSameSign(num));
                *this = AddWithSameSign(num);
            }
            return *this;
        }

        BigInteger<NativeInt> Sub(const BigInteger<NativeInt> &num) const {
            int absoluteCompare = AbsoluteCompare(num);
            if (sign != num.sign) {
                return AddWithSameSign(num, sign);
            } else {
                if (AbsoluteCompare(num) == 0) {
                    return BigInteger<NativeInt>();
                } else if (absoluteCompare > 0) {
                    return SubWithSameSign(num);
                } else {
                    return num.SubWithSameSign(*this, true);
                }
            }
        }

        BigInteger<NativeInt> &SubEq(const BigInteger<NativeInt> &num) {
            int absoluteCompare = AbsoluteCompare(num);
            if (sign != num.sign) {
                // AssignObj(AddWithSameSign(num, sign));
                *this = AddWithSameSign(num, sign);
            } else {
                if (AbsoluteCompare(num) == 0) {
                    value.clear();
                    value.push_back(0);
                } else if (absoluteCompare > 0) {
                    // AssignObj(SubWithSameSign(num));
                    *this = SubWithSameSign(num);
                } else {
                    // AssignObj(num.SubWithSameSign(*this, true));
                    *this = num.SubWithSameSign(*this, true);
                }
            }
            return *this;
        }

        BigInteger<NativeInt> operator-() const { return BigInteger(this->value, !this->sign); }

        std::size_t length() const { return value.size(); }

        std::vector<NativeInt> getValue() const { return value; }

        NativeInt getValueOfIndex(uint32_t index) const { return value[index]; }

        bool getSign() const { return sign; }

        BigInteger<NativeInt> Mul(const BigInteger<NativeInt> &b) const {
            std::vector<NativeInt> values;
            for (int i = 0; i < value.size(); ++i) {
                for (int j = 0; j < b.value.size(); ++j) {
                    NativeInt temp_result[2];
                    MultiplyWithKaratsuba(value[i], b.value[j], temp_result);
                    uint8_t carry = 0;
                    NativeInt sum;
                    if (i + j + 1 > values.size()) {
                        values.push_back(temp_result[0]);
                    } else {
                        carry = addWithCarry(temp_result[0], values[i + j], carry, &sum);
                        values[i + j] = sum;
                        temp_result[1] += carry;
                        carry = 0;
                    }

                    if (i + j + 2 > values.size()) {
                        values.push_back(temp_result[1]);
                    } else {
                        carry = addWithCarry(temp_result[1], values[i + j + 1], carry, &sum);
                        values[i + j + 1] = sum;
                        uint8_t currentIdx = i + j + 2;
                        while (carry) {
                            if (currentIdx >= values.size()) {
                                values.push_back(carry);
                            } else {
                                carry = addWithCarry(0, values[currentIdx], carry, &sum);
                                values[currentIdx] = sum;
                            }
                        }
                    }
                }
            }

            return BigInteger(values, sign ^ b.sign);
        }

        BigInteger<NativeInt> &MulEq(const BigInteger<NativeInt> &b) {
            return *this = this->Mul(b);
        }

        std::pair<BigInteger<NativeInt>, BigInteger<NativeInt>>
        DividedBy(const BigInteger<NativeInt> &denominator) const {
            bool finalSign = sign xor denominator.sign;
            BigInteger quotientIn(0U, finalSign);
            BigInteger remainderIn(0U, finalSign);
            if (this->AbsoluteCompare(denominator) < 0) {
                quotientIn = BigInteger(0U, finalSign);
                remainderIn = BigInteger(value, finalSign);
            } else {
                Divide(quotientIn, remainderIn, *this, denominator);
            }
            return std::make_pair(quotientIn, remainderIn);
        }

        BigInteger<NativeInt> &DividedByEq(const BigInteger<NativeInt> &b) {
            std::pair<BigInteger<NativeInt>, BigInteger<NativeInt>> resultPair = this->DividedBy(b);
            return *this = resultPair.first;
        }

        // Mod
        BigInteger<NativeInt> Mod(const BigInteger<NativeInt> &modulus) const {
            if (GetMSB() < modulus.GetMSB() ||
                GetMSB() == modulus.GetMSB() && AbsoluteCompare(modulus) < 0) {
                if (getSign()) {
                    return *this + modulus;
                } else {
                    return BigInteger<NativeInt>(GetValue());
                }
            }

            if (GetMSB() == modulus.GetMSB() && AbsoluteCompare(modulus) == 0) {
                return BigInteger<NativeInt>();
            }

            // use simple masking operation if modulus is 2
            if (modulus.GetMSB() == 2 && modulus.GetValue()[0] == 2) {
                if (GetValue()[0] % 2 == 0) {
                    return BigInteger<NativeInt>();
                } else {
                    return BigInteger<NativeInt>("1");
                }
            }
            auto f = DividedBy(modulus);
            if (getSign()) {
                return f.second + modulus;
            } else {
                return f.second;
            }
        }

        BigInteger<NativeInt> &ModEq(const BigInteger<NativeInt> &modulus) {
            return *this = this->Mod(modulus);
        }

        BigInteger<NativeInt> operator%(const BigInteger<NativeInt> &modulus) const {
            return this->Mod(modulus);
        }

        const BigInteger<NativeInt> &operator%=(const BigInteger<NativeInt> &modulus) {
            return *this = *this % modulus;
        }

        BigInteger<NativeInt> ModMul(const BigInteger<NativeInt> &b,
                                     const BigInteger<NativeInt> &modulus) const {
            BigInteger a(*this);
            BigInteger ans = 0;
            size_t nSize = a.value.size();
            size_t bSize = b.value.size();
            BigInteger tmpans;
            ans.value.reserve(nSize + bSize);
            tmpans.value.reserve(nSize + bSize);

            for (size_t i = 0; i < bSize; ++i) {
                tmpans.value.clear();// make sure there are no limbs to start.
                Dlimb_t limbb(b.value[i]);
                Dlimb_t temp = 0;
                NativeInt ofl = 0;
                uint32_t ix = 0;
                while (ix < i) {
                    tmpans.value.push_back(0);// equivalent of << shift
                    ++ix;
                }

                for (auto itr: a.value) {
                    temp = ((Dlimb_t) itr * (Dlimb_t) limbb) + ofl;
                    tmpans.value.push_back((NativeInt) temp);
                    ofl = temp >> a.m_limbBitLength;
                }
                // check if there is any final overflow
                if (ofl) { tmpans.value.push_back(ofl); }
                // tmpans.m_state = INITIALIZED;
                // tmpans.SetMSB();
                tmpans.RefreshMSB();
                ans += tmpans;
                ans %= modulus;
            }
            return ans;
        }

        // Bit Operation
        BigInteger<NativeInt> And(const BigInteger<NativeInt> another) const {
            std::vector<NativeInt> vals;
            for (auto i = 0; i < value.size() && i < another.value.size(); ++i) {
                vals.push_back(value[i] & another.value[i]);
            }

            return BigInteger(vals, !(!sign & !another.sign));
        }

        BigInteger<NativeInt> &AndEq(const BigInteger<NativeInt> another) {
            return *this = this->And(another);
        }

        BigInteger<NativeInt> operator&(const BigInteger<NativeInt> &another) const {
            return And(another);
        }

        BigInteger<NativeInt> operator&=(const BigInteger<NativeInt> &another) {
            return *this = And(another);
        }

        BigInteger<NativeInt> Or(const BigInteger<NativeInt> another) const {
            std::vector<NativeInt> vals;
            for (auto i = 0; (i < value.size() || i < another.value.size()); ++i) {
                if (i >= value.size()) {
                    vals.push_back(another.value[i]);
                } else if (i >= another.value.size()) {
                    vals.push_back(value[i]);
                } else {
                    vals.push_back(value[i] | another.value[i]);
                }
            }

            return BigInteger(vals, sign | another.sign);
        }

        BigInteger<NativeInt> &OrEq(const BigInteger<NativeInt> another) {
            return *this = this->Or(another);
        }

        BigInteger<NativeInt> operator|(const BigInteger<NativeInt> &another) const {
            return Or(another);
        }

        const BigInteger<NativeInt> operator|=(const BigInteger<NativeInt> &another) {
            return *this = Or(another);
        }

        BigInteger<NativeInt> Not() const {
            std::vector<NativeInt> vals;
            for (auto val: value) { vals.push_back(~val); }
            return BigInteger(vals, sign);
        }

        BigInteger<NativeInt> &NotEq() {
            return *this = this->Not();
        }

        BigInteger<NativeInt> operator~() const { return Not(); }

        BigInteger<NativeInt> Xor(const BigInteger<NativeInt> another) const {
            std::vector<NativeInt> vals;
            for (auto i = 0; (i < value.size() || i < another.value.size()); ++i) {
                if (i >= value.size()) {
                    vals.push_back(another.value[i]);
                } else if (i >= another.value.size()) {
                    vals.push_back(value[i]);
                } else {
                    vals.push_back(value[i] ^ another.value[i]);
                }
            }

            return BigInteger(vals, sign ^ another.sign);
        }

        const BigInteger<NativeInt> &XorEq(const BigInteger<NativeInt> another) {
            return *this = Xor(another);
        }

        BigInteger<NativeInt> operator^(const BigInteger<NativeInt> &another) const {
            return Xor(another);
        }

        const BigInteger<NativeInt> operator^=(const BigInteger<NativeInt> &another) {
            return *this = Xor(another);
        }

        BigInteger<NativeInt> Negate() const {
            std::vector<NativeInt> vals;
            auto carry = 0;
            for (auto val: value) {
                NativeInt temp_val;
                carry = addWithCarry(~val, 1, carry, &temp_val);
                vals.push_back(temp_val);
            }
            if (carry) { vals.push_back(carry); }
            return BigInteger(vals, sign);
        }

        BigInteger<NativeInt> &NegateEq() {
            return *this = this->Negate();
        }

        BigInteger<NativeInt> LeftShift(uint16_t shift) const {
            if (this->m_MSB == 0) { return BigInteger(); }
            BigInteger ans(*this);
            // compute the number of whole limb shifts
            uint32_t shiftByLimb = shift >> m_log2LimbBitLength;
            // ans.value.reserve(shiftByLimb+this->value.size());
            // compute the remaining number of bits to shift
            NativeInt remainingShift = (shift & (m_limbBitLength - 1));
            // first shift by the # remainingShift bits
            if (remainingShift != 0) {
                for (size_t i = ceilIntByUInt(m_MSB) - 1; i > 0; i--) {
                    ans.value[i] = (ans.value[i] << remainingShift) |
                                   ans.value[i - 1] >> (m_limbBitLength - remainingShift);
                }
                ans.value[0] = ans.value[0] << remainingShift;
            }
            if (shiftByLimb != 0) {
                int currentSize = ans.value.size();
                ans.value.resize(currentSize + shiftByLimb);// allocate more storage
                for (int i = currentSize - 1; i >= 0; i--) {// shift limbs required # of indicies
                    ans.value[i + shiftByLimb] = ans.value[i];
                }
                for (int i = shiftByLimb - 1; i >= 0; i--) { ans.value[i] = 0; }
            }
            ans.m_MSB += remainingShift + shiftByLimb * m_limbBitLength;
            return ans;
        }

        BigInteger<NativeInt> &LeftShiftEq(uint16_t shift) {
            return *this = this->LeftShift(shift);
        }

        BigInteger<NativeInt> operator<<(uint16_t shift) const { return LeftShift(shift); }

        const BigInteger<NativeInt> &operator<<=(uint16_t shift) {
            return *this = this->LeftShift(shift);
        }

        BigInteger<NativeInt> RightShift(uint16_t shift) const {
            if (this->m_MSB == 0 || this->m_MSB <= shift) { return BigInteger(); }

            BigInteger ans(*this);
            uint16_t shiftByLimb = shift >> m_log2LimbBitLength;
            NativeInt remainingShift = (shift & (m_limbBitLength - 1));
            NativeInt negativeShift = m_limbBitLength - remainingShift;
            if (shiftByLimb != 0) {
                for (size_t i = shiftByLimb; i < ans.value.size(); ++i) {
                    ans.value[i - shiftByLimb] = ans.value[i];
                }
                for (uint32_t i = 0; i < shiftByLimb; ++i) { ans.value.pop_back(); }
            }

            // remainderShift bit shifts
            if (remainingShift != 0) {
                for (uint32_t i = 0; i < ans.value.size() - 1; i++) {
                    ans.value[i] = (ans.value[i] >> remainingShift) | ans.value[i + 1] << negativeShift;
                }
                ans.value[ans.value.size() - 1] = ans.value[ans.value.size() - 1] >> remainingShift;
            }
            while (!ans.value.back()) { ans.value.pop_back(); }
            ans.RefreshMSB();
            return ans;
        }

        BigInteger<NativeInt> &RightShiftEq(uint16_t shift) {
            return *this = this->RightShift(shift);
        }

        BigInteger<NativeInt> operator>>(uint16_t shift) const { return RightShift(shift); }

        const BigInteger<NativeInt> &operator>>=(uint16_t shift) {
            return *this = this->RightShift(shift);
        }

        inline BigInteger Exp(uint32_t p) {
            if (p == 0) { return BigInteger(1); }
            if (p == 1) { return *this; }
            BigInteger tmp = Exp(p / 2);
            if (p % 2 == 0) {
                return tmp * tmp;
            } else {
                return tmp * tmp * (*this);
            }
        }

        uint32_t GetMSB() const { return m_MSB; }

        std::vector<NativeInt> GetValue() const { return value; }

        bool GetSign() { return sign; }

        /**
         * Gets the bit at the specified index.
         *
         * @param index is the index of the bit to get.
         * @return resulting bit.
         */
        uint8_t GetBitAtIndex(uint32_t index) const {
            if (index > m_MSB) return 0;
            uint32_t i = MSB2NLimbs(index) - 1;
            // uint32_t i = index / m_limbBitLength;
            NativeInt x = value[i];
            uint32_t remain = index % m_limbBitLength == 0 ? m_limbBitLength : index % m_limbBitLength;
            NativeInt bMask = 1;
            bMask <<= (remain - 1);
            return x & bMask ? 1 : 0;
        }

        /**
         * Get the number of digits using a specific base.
         * Warning: only power-of-2 bases are currently supported.
         * Example: for number 83, index 3 and base 4 we have:
         *
         *                         index:1,2,3,4
         * 83 --base 4 decomposition--> (3,0,1,1) --at index 3--> 1
         *
         * The return number is 1.
         *
         * @param index is the location to return value from in the specific base.
         * @param base is the base with which to determine length in.
         * @return the length of the representation in a specific base.
         */
        uint32_t GetDigitAtIndexForBase(uint32_t index, uint32_t base) const {
            uint32_t digitLength = ceil(log2(base));
            uint32_t digit = 0;
            uint32_t newIndex = 1 + (index - 1) * digitLength;
            for (uint32_t i = 1; i < base; i = i * 2) {
                digit += GetBitAtIndex(newIndex) * i;
                newIndex++;
            }

            return digit;
        }

        template<typename T, std::enable_if_t<std::is_integral_v<T>, bool> = true>
        BigInteger<NativeInt> &operator+=(const T &rhs) {
            return this->AddEq(BigInteger<NativeInt>(rhs));
        }

        template<typename T, std::enable_if_t<std::is_integral_v<T>, bool> = true>
        BigInteger<NativeInt> &operator-=(const T &rhs) {
            return this->SubEq(BigInteger<NativeInt>(rhs));
        }

        template<typename T, std::enable_if_t<std::is_integral_v<T>, bool> = true>
        BigInteger<NativeInt> &operator*=(const T &rhs) {
            return this->MulEq(BigInteger<NativeInt>(rhs));
        }

    protected:
        std::vector<NativeInt> value;
        bool sign = false;
        static const uint32_t m_limbBitLength{sizeof(NativeInt) * 8};
        uint32_t m_MSB = 0;
        static constexpr NativeInt m_MaxLimb = std::numeric_limits<NativeInt>::max();
        static constexpr uint32_t m_log2LimbBitLength{Log2<m_limbBitLength>::value};
        static constexpr uint32_t m_uintBitLength{std::numeric_limits<NativeInt>::digits};

    private:
        void AssignVal(const std::string &str) {
            // clear the current value of value;
            value.clear();
            std::string v = str;
            if (!str.empty() && str[0] == '-') {
                sign = true;
                v = str.substr(1);
            }
            v.erase(0, v.find_first_not_of('0'));
            v.erase(0, v.find_first_not_of(' '));
            if (v.empty()) {
                v = "0";// set to one zero
                value.push_back(0);
                return;
            }
            const size_t arr_size = v.length();
            std::unique_ptr<u_int8_t[]> dec_arr = std::make_unique<u_int8_t[]>(arr_size);
            // std::array<uint8_t, arr_size> didi;
            // std::unique_ptr<std::array<uint8_t, arr_size>> dec_arr = std::unique_ptr<std::array<uint8_t, arr_size>>();
            for (size_t i = 0; i < arr_size; i++)// store the string to decimal array
                dec_arr[i] = (uint8_t) stoi(v.substr(i, 1));

            size_t zero_ptr = 0;
            // index of highest non-zero number in decimal number
            // define  bit register array
            std::unique_ptr<u_int8_t[]> bit_arr = std::make_unique<u_int8_t[]>(m_limbBitLength);
            // std::unique_ptr<std::array<uint8_t, m_limbBitLength>> bit_arr = std::unique_ptr<std::array<uint8_t, m_limbBitLength>>;
            int cnt = m_limbBitLength - 1;
            // cnt is a pointer to the bit position in bit_arr, when bit_arr is complete it
            // is ready to be transfered to Value
            while (zero_ptr != arr_size) {
                bit_arr[cnt] = dec_arr[arr_size - 1] % 2;
                // start divide by 2 in the DecValue array
                for (size_t i = zero_ptr; i < arr_size - 1; i++) {
                    dec_arr[i + 1] = (dec_arr[i] % 2) * 10 + dec_arr[i + 1];
                    dec_arr[i] >>= 1;
                }
                dec_arr[arr_size - 1] >>= 1;
                // division ends here
                cnt--;
                if (cnt == -1) {// cnt = -1 indicates bit_arr is ready for transfer
                    cnt = m_limbBitLength - 1;
                    value.push_back(UintInBinaryToDecimal(bit_arr.get()));
                }
                if (dec_arr[zero_ptr] == 0) {
                    zero_ptr++;// division makes Most significant digit zero, hence we increment
                    // zero_ptr to next value
                }
                if (zero_ptr == arr_size && dec_arr[arr_size - 1] == 0) {
                    const NativeInt temp = UintInBinaryToDecimal(bit_arr.get());
                    if (temp != 0) {
                        value.push_back(temp);
                    }
                    RefreshMSB();
                }
            }
        }

        // BigInteger<NativeInt> &AssignObj(const BigInteger<NativeInt> &other);
        static uint32_t ceilIntByUInt(const NativeInt Number) {
            // mask to perform bitwise AND
            static NativeInt mask = m_limbBitLength - 1;

            if (!Number) { return 1; }

            if ((Number & mask) != 0) {
                return (Number >> m_log2LimbBitLength) + 1;
            } else {
                return Number >> m_log2LimbBitLength;
            }
        }

        void RefreshMSB() {
            m_MSB = (value.size() - 1) * m_limbBitLength + m_GetMSBForLimb(value.back());
        }

        uint32_t m_GetMSBForLimb(NativeInt x) {
            uint64_t y = ((uint64_t) x);
            if (y == 0) {
                return 0;
            } else {
                return 64 - (sizeof(unsigned long) == 8 ? __builtin_clzl(y) : __builtin_clzll(y));
            }
        }

        void NormalizeLimbs() {
            for (uint32_t i = this->value.size() - 1; i >= 1; i--) {
                if (!this->value.back()) {
                    this->value.pop_back();
                } else {
                    break;
                }
            }

            RefreshMSB();
        }

        NativeInt UintInBinaryToDecimal(uint8_t *a) {
            NativeInt Val = 0;
            NativeInt one = 1;
            for (int i = m_limbBitLength - 1; i >= 0; i--) {
                Val += one * *(a + i);
                one <<= 1;
                *(a + i) = 0;
            }
            return Val;
        }

        static uint8_t addWithCarry(NativeInt operand1, NativeInt operand2, uint8_t carry,
                                    NativeInt *result) {
            operand1 += operand2;
            *result = operand1 + carry;
            return (operand1 < operand2) || (~operand1 < carry);
        }

        static uint8_t subWithBorrow(NativeInt operand1, NativeInt operand2, uint8_t borrow,
                                     NativeInt *result) {
            auto diff = operand1 - operand2;
            *result = diff - (borrow != 0);
            return (diff > operand1) || (diff < borrow);
        }

        BigInteger<NativeInt> AddWithSameSign(const BigInteger<NativeInt> &num,
                                              bool sign = false) const {
            std::vector<NativeInt> resultVectors;

            uint8_t carry = 0;
            NativeInt currentLimb;
            int i = 0;
            while (i < value.size() && i < num.value.size()) {
                carry = addWithCarry(value[i], num.value[i], carry, &currentLimb);
                resultVectors.push_back(currentLimb);
                ++i;
            }
            while (i < value.size()) {
                carry = addWithCarry(value[i], 0, carry, &currentLimb);
                resultVectors.push_back(currentLimb);
                ++i;
            }
            while (i < num.value.size()) {
                carry = addWithCarry(0, num.value[i], carry, &currentLimb);
                resultVectors.push_back(currentLimb);
                ++i;
            }

            return BigInteger(resultVectors, sign);
        }

        BigInteger<NativeInt> SubWithSameSign(const BigInteger<NativeInt> &num,
                                              bool signVal = false) const {

            std::vector<NativeInt> resultVectors;

            uint8_t borrow = 0;
            NativeInt currentLimb;

            for (int i = 0; i < value.size(); ++i) {
                if (i >= num.value.size()) {
                    borrow = subWithBorrow(value[i], 0, borrow, &currentLimb);
                    resultVectors.push_back(currentLimb);
                } else {
                    borrow = subWithBorrow(value[i], num.value[i], borrow, &currentLimb);
                    resultVectors.push_back(currentLimb);
                }
            }


            return BigInteger(resultVectors, signVal);
        }

        /**
         *
         * @param quotientIn 商
         * @param remainderIn 余数
         * @param uIn
         * @param v
         * @return
         */
        bool Divide(BigInteger &quotientIn, BigInteger &remainderIn, const BigInteger &uIn,
                    const BigInteger &vIn) const {

            std::vector<NativeInt> &quotient = quotientIn.value;
            std::vector<NativeInt> &remainder = remainderIn.value;
            const std::vector<NativeInt> &u = (uIn.value);
            const std::vector<NativeInt> &v = (vIn.value);

            int m = u.size();
            int n = v.size();
            quotient.resize(m - n + 1);

            Dlimb_t qhat;   // Estimated quotient digit.
            Dlimb_t rhat;   // remainder
            Dlimb_t product;// Product of two digits.
            SDlimb_t t, k;
            int s = 0, i = 0, j = 0;

            const auto ffs = (Dlimb_t) m_MaxLimb;     // Number  (2**64)-1.
            const Dlimb_t b = (Dlimb_t) m_MaxLimb + 1;// Number base (2**64).


            if (m < n || n <= 0 || v[n - 1] == 0) {
                return false;// Return if invalid param.
            }

            if (n == 1) {                               // Take care of
                k = 0;                                  // the case of a
                for (j = m - 1; j >= 0; j--) {          // single-digit
                    quotient[j] = (k * b + u[j]) / v[0];// divisor here.
                    k = (k * b + u[j]) - quotient[j] * v[0];
                }
                if (remainder.size() != 0) { remainder[0] = k; }
                remainderIn.NormalizeLimbs();
                quotientIn.NormalizeLimbs();
                return true;
            }

            s = util::nlz(v[n - 1]);

            std::vector<NativeInt> vn(n);
            for (i = n - 1; i > 0; i--) { vn[i] = (v[i] << s) | v[i - 1] >> (m_limbBitLength - s); }
            vn[0] = v[0] << s;

            std::vector<NativeInt> un(m + 1);

            un[m] = u[m - 1] >> (m_limbBitLength - s);
            for (i = m - 1; i > 0; i--) { un[i] = (u[i] << s) | (u[i - 1] >> (m_limbBitLength - s)); }
            un[0] = u[0] << s;

            // Main loop
            for (j = m - n; j >= 0; j--) {
                // Compute estimate qhat of q[j].
                qhat = (un[j + n] * b + un[j + n - 1]) / vn[n - 1];
                rhat = (un[j + n] * b + un[j + n - 1]) - qhat * vn[n - 1];
                while (qhat >= b || qhat * vn[n - 2] > b * rhat + un[j + n - 2]) {
                    qhat = qhat - 1;
                    rhat = rhat + vn[n - 1];
                    if (rhat >= b) { break; }
                }

                // Multiply and subtract.
                k = 0;
                for (i = 0; i < n; i++) {
                    product = qhat * vn[i];
                    t = un[i + j] - k - (product & ffs);
                    un[i + j] = t;
                    k = (product >> m_limbBitLength) - (t >> m_limbBitLength);
                }
                t = un[j + n] - k;
                un[j + n] = t;


                // Store quotient digit.
                quotient[j] = qhat;
                if (t < 0) {                      // If we subtracted too
                    quotient[j] = quotient[j] - 1;// much, add back.
                    k = 0;
                    for (i = 0; i < n; i++) {
                        t = (Dlimb_t) un[i + j] + vn[i] + k;
                        un[i + j] = t;
                        k = t >> m_limbBitLength;
                    }
                    un[j + n] = un[j + n] + k;
                }
            }

            // store remainder
            remainder.resize(n);
            for (i = 0; i < n - 1; i++) {
                remainder[i] = (un[i] >> s) | un[i + 1] << (m_limbBitLength - s);
            }
            remainder[n - 1] = un[n - 1] >> s;
            remainderIn.NormalizeLimbs();
            quotientIn.NormalizeLimbs();
            return true;
        }

        /**
         * @brief Karatsuba 算法计算 NativeInt * NativeInt, 结果为一个两倍于 NativeInt 长度的数
         *
         * (a * 2^n + b) * (c * 2^n + d) = ac*2^2n + (ad + bc)*2^n + bd
         * 这里 n = NativeInt 长度 / 2，下面的计算时由于进位的原因，做了多次平移。
         */
        static void MultiplyWithKaratsuba(NativeInt operand1, NativeInt operand2,
                                          NativeInt *resultTwo) {
            NativeInt mask = 0x0;
            uint32_t halfLimbLength = m_limbBitLength / 2;
            for (int i = 0; i < halfLimbLength; ++i) { mask += (NativeInt) 1 << i; }

            NativeInt a = operand1 >> halfLimbLength;
            NativeInt b = operand1 & mask;
            NativeInt c = operand2 >> halfLimbLength;
            NativeInt d = operand2 & mask;

            NativeInt right = b * d;
            NativeInt middle;
            NativeInt left = a * c + (static_cast<NativeInt>(addWithCarry(a * d, b * c, 0, &middle))
                    << halfLimbLength);
            NativeInt temp_sum = (right >> halfLimbLength) + (middle & mask);

            resultTwo[1] = left + (middle >> halfLimbLength) + (temp_sum >> halfLimbLength);
            resultTwo[0] = (temp_sum << halfLimbLength) | (right & mask);
        }

        /**
         * function to return the ceiling of the input number divided by
         * the number of bits in the limb data type.  DBC this is to
         * determine how many limbs are needed for an input bitsize.
         * @param number is the number to be divided.
         * @return the ceiling of Number/(bits in the limb data type)
         */
        static uint32_t MSB2NLimbs(const uint32_t number) {
            static uint32_t mask = m_limbBitLength - 1;

            if (!number) return 1;
            if ((number & mask) != 0) {
                return (number >> m_log2LimbBitLength) + 1;
            } else {
                return number >> m_log2LimbBitLength;
            }
        }
    };

    template<typename NativeInt>
    std::ostream &operator<<(std::ostream &os, const BigInteger<NativeInt> &a) {
        return os << a.ConvertToString();
    }


    const static BigInt ZERO(0);
    const static BigInt ONE(1);
    const static BigInt TWO(2);
    const static BigInt THREE(3);
    const static BigInt FOUR(4);
    const static BigInt FIVE(5);
}// namespace isecfhe
