#pragma once

#include "exception.h"
#include "modulus.h"
#include "data_config.h"
#include "big_integer_modop.h"

#include <cstdint>
#include <iosfwd>
#include <memory>
#include <string>
#include <vector>

namespace isecfhe {

    enum State {
        INITIALIZED, GARBAGE
    };

    template<typename IntegerType>
    class IntVector {

    public:

        using IntType = IntegerType;

        IntVector() = default;

        IntVector(uint32_t length) : data(length), state(GARBAGE) {
            for (uint32_t i = 0; i < length; i++) { this->data[i] = 0; }
        }

        IntVector(uint32_t length, const IntegerType &modulus)
                : modulus(modulus), data(length, 0), state(INITIALIZED) {};

        IntVector(const IntVector &bigVector) : state(INITIALIZED), data(bigVector.data), modulus(bigVector.modulus) {};

        IntVector(IntVector &&bigVector) noexcept: state(INITIALIZED), data(std::move(bigVector.data)),
                                                   modulus(std::move(bigVector.modulus)) {};

        template<typename T = limbtype>
        IntVector(uint32_t length, const IntegerType &modulus, std::initializer_list<std::string> rhs)
                : state(INITIALIZED) {
            this->data.resize(length);
            this->modulus = modulus;
            uint32_t len = rhs.size();
            for (uint32_t i = 0; i < length; i++) {
                if (i < len) {
                    IntegerType val = std::stoi(*(rhs.begin() + i));
                    this->data[i] = util::Mod(val, this->modulus);

                } else {
                    this->data[i] = 0;
                }
            }
        }


        template<typename T = IntegerType, typename std::enable_if_t<std::is_same_v<T, BigInt>>>
        IntVector(uint32_t length, const IntegerType &modulus, std::initializer_list<std::string> rhs)
                : state(INITIALIZED) {
            this->data.resize(length);
            this->modulus = modulus;
            uint32_t len = rhs.size();
            for (uint32_t i = 0; i < length; i++) {
                if (i < len) {
                    IntegerType val = IntegerType(*(rhs.begin() + i));
                    this->data[i] = util::Mod(val, this->modulus);
                } else {
                    this->data[i] = 0;
                }
            }
        }


        IntVector(uint32_t length, const IntegerType &modulus, std::initializer_list<uint64_t> rhs)
                : state(INITIALIZED) {
            this->data.resize(length);
            this->modulus = modulus;
            uint32_t len = rhs.size();
            for (uint32_t i = 0; i < length; i++) {
                if (i < len) {
                    IntegerType val = IntegerType(*(rhs.begin() + i));
                    this->data[i] = util::Mod(val, this->modulus);
                } else {
                    this->data[i] = IntegerType(0);
                }
            }
        }

        template<typename T = limbtype>
        IntVector(const std::vector<std::string> &s, const T &modulus) {
            this->data.resize(s.size());
            this->modulus = modulus;
            for (uint32_t i = 0; i < s.size(); i++) {
                IntegerType val = std::stoi(s[i]);
                this->data[i] = util::Mod(val, this->modulus);
            }
            state = INITIALIZED;
        }

        template<typename T = IntegerType, typename std::enable_if_t<std::is_same_v<T, BigInt>>>
        IntVector(const std::vector<std::string> &s, const T &modulus) {
            this->data.resize(s.size());
            this->modulus = modulus;
            for (uint32_t i = 0; i < s.size(); i++) {
                auto val = IntegerType(s[i]);
                this->data[i] = util::Mod(val, this->modulus);
            }
            state = INITIALIZED;
        }


        IntVector(const std::vector<IntegerType> &s, const IntegerType &modulus) : state(INITIALIZED) {
            this->data.resize(s.size());
            this->modulus = modulus;
            for (uint32_t i = 0; i < s.size(); i++) {
                auto val = s[i];
                this->data[i] = util::Mod(val, this->modulus);
            }
        }

        template<typename T = limbtype>
        IntVector &operator=(std::initializer_list<std::string> rhs) {
            size_t len = rhs.size();
            if (this->data.size() < len) { this->data.resize(len); }
            for (uint32_t i = 0; i < this->data.size(); i++) {// this loops over each entry
                if (i < len) {
                    this->data[i] = std::stoi(*(rhs.begin() + i));
                } else {
                    this->data[i] = 0;
                }
            }
            if (this->state == INITIALIZED) { this->Mod(); }
            return *this;
        }

        template<typename T = IntegerType, typename std::enable_if_t<std::is_same_v<T, BigInt>>>
        IntVector &operator=(std::initializer_list<std::string> rhs) {
            size_t len = rhs.size();
            if (this->data.size() < len) { this->data.resize(len); }
            for (uint32_t i = 0; i < this->data.size(); i++) {// this loops over each entry
                if (i < len) {
                    this->data[i] = *(rhs.begin() + i);
                } else {
                    this->data[i] = 0;
                }
            }
            if (this->state == INITIALIZED) { this->Mod(); }
            return *this;
        }


        IntVector &operator=(std::initializer_list<int64_t> rhs) {
            size_t len = rhs.size();
            if (this->data.size() < len) { this->data.resize(len); }
            for (uint32_t i = 0; i < this->data.size(); i++) {// this loops over each entry
                if (i < len) {
                    this->data[i] = *(rhs.begin() + i);
                } else {
                    this->data[i] = 0;
                }
            }
            if (this->state == INITIALIZED) { this->Mod(); }
            return *this;
        }

        virtual ~IntVector() {
            this->data.clear();
        }

        bool operator==(const IntVector &rhs) const {
            if (this == &rhs) return true;
            if (data.size() != rhs.data.size() || modulus != rhs.modulus)
                return false;
            for (uint32_t i = 0; i < data.size(); ++i) {
                if (data[i] != rhs.data[i]) return false;
            }
            return true;
        }

        bool operator!=(const IntVector &rhs) const {
            return !(*this == rhs);
        }

        IntVector &operator=(const IntVector &rhs) {
            if (this != &rhs) {
                if (this->data.size() == rhs.data.size()) {
                    for (uint32_t i = 0; i < this->data.size(); i++) { this->data[i] = rhs.data[i]; }
                } else {
                    this->data.resize(rhs.data.size());
                    for (uint32_t i = 0; i < this->data.size(); i++) { this->data[i] = rhs.data[i]; }
                }
                this->modulus = rhs.modulus;
                this->state = INITIALIZED;
            }
            return *this;
        }

        IntVector &operator=(IntVector &&rhs) noexcept {
            if (this != &rhs) {
                this->data.swap(rhs.data);// swap the two IntVector contents,
                if (rhs.data.size() > 0) { rhs.data.clear(); }
                this->modulus = rhs.modulus;
                this->state = INITIALIZED;
            }
            return *this;
        }

        IntegerType *ConvertToIntList() {
            IntegerType *result = new IntegerType[GetLength()];
            for (int i = 0; i < GetLength(); i++) { result[i] = at(i); }
            return result;
        }

        IntegerType &at(size_t i) {
            if (!this->IndexCheck(i)) { FHE_THROW(MathException, "Vector index out of range"); }
            return this->data[i];
        }

        const IntegerType &at(size_t i) const {
            if (!this->IndexCheck(i)) { FHE_THROW(MathException, "Vector index out of range"); }
            return this->data[i];
        }

        IntegerType &operator[](size_t idx) { return (this->data[idx]); }

        const IntegerType &operator[](size_t idx) const { return (this->data[idx]); }

        size_t GetLength() const { return data.size(); }

        void SetModulus(const IntegerType &value) {
            if (this->state == INITIALIZED) {
                FHE_THROW(isecfhe::ConfigException, "modulus already set");
            }
            this->modulus = value;
            this->Mod();
            this->state = INITIALIZED;
        }

        void SwitchModulus(const IntegerType &val) {
            IntegerType newModulus(val);
            IntegerType oldModulus(this->modulus);
            IntegerType n;
            IntegerType oldModulusByTwo(oldModulus >> 1);
            IntegerType diff((oldModulus > newModulus) ? (oldModulus - newModulus)
                                                       : (newModulus - oldModulus));

            int count1 = 0, count2 = 0, count3 = 0, count4 = 0;
            for (uint32_t i = 0; i < this->data.size(); i++) {
                n = this->at(i);
                if (oldModulus < newModulus) {
                    if (n > oldModulusByTwo) {
                        this->data[i] = util::ModAdd(n, diff, val);
                        count1++;
                    } else {
                        this->data[i] = util::Mod(n, val);
                        count2++;
                    }
                } else {
                    if (n > oldModulusByTwo) {
                        this->data[i] = util::ModSub(n, diff, val);
                        count3++;
                    } else {
                        this->data[i] = util::Mod(n, val);
                        count4++;
                    }
                }
            }
            this->state = INITIALIZED;
            this->modulus = val;
        }

        const IntegerType &GetModulus() const {
            return this->modulus;
        }

        IntVector ModAdd(const IntegerType &b) const {
            IntVector<IntegerType> ans(*this);
            return ans.ModAddEq(b);
        }

        const IntVector &ModAddEq(const IntegerType &b) {
            for (uint32_t i = 0; i < this->data.size(); i++) {
                data[i] = util::ModAdd(data[i], b, modulus);
            }
            return *this;
        }

        IntVector ModAddAtIndex(uint32_t i, const IntegerType &b) const {
            if (!this->IndexCheck(i)) { FHE_THROW(MathException, "IntVector index out of range"); }
            IntVector<IntegerType> ans(*this);
            ans.data[i] = util::ModAdd(ans.data[i], b, ans.modulus);
            return ans;
        }

        const IntVector &ModAddAtIndexEq(uint32_t i, const IntegerType &b) {
            return *this = ModAddAtIndex(i, b);
        }

        IntVector ModAdd(const IntVector &b) const {
            IntVector<IntegerType> ans(*this);
            return ans.ModAddEq(b);
        }

        const IntVector &ModAddEq(const IntVector &b) {
            if (this->modulus != b.modulus) {
                FHE_THROW(MathException, "IntVector adding IntVectors of different modulus");
            } else if (this->data.size() != b.data.size()) {
                FHE_THROW(MathException, "IntVector adding IntVectors of different lengths");
            }
            for (uint32_t i = 0; i < this->data.size(); i++) {
                data[i] = util::ModAdd(data[i], b.data[i], modulus);
            }
            return *this;
        }

        IntVector ModSub(const IntegerType &b) const {
            IntVector<IntegerType> ans(*this);
            return ans.ModSubEq(b);
        }

        const IntVector &ModSubEq(const IntegerType &b) {
            for (uint32_t i = 0; i < this->data.size(); i++) {
                data[i] = util::ModSub(data[i], b, modulus);
            }
            return *this;
        }

        IntVector ModSub(const IntVector &b) const {
            IntVector<IntegerType> ans(*this);
            return ans.ModSubEq(b);
        }

        const IntVector &ModSubEq(const IntVector &b) {
            if (this->modulus != b.modulus) {
                FHE_THROW(MathException, "IntVector subtract IntVectors of different modulus");
            } else if (this->data.size() != b.data.size()) {
                FHE_THROW(MathException, "IntVector subtract IntVectors of different lengths");
            }
            for (uint32_t i = 0; i < this->data.size(); i++) {
                data[i] = util::ModSub(data[i], b.data[i], modulus);
            }

            return *this;
        }

        IntVector ModMul(const IntegerType &b) const {
            IntVector<IntegerType> ans(*this);
            return ans.ModMulEq(b);
        }

        const IntVector &ModMulEq(const IntegerType &b) {
            for (uint32_t i = 0; i < this->data.size(); i++) {
                data[i] = util::ModMul(data[i], b, modulus);
            }
            return *this;
        }

        IntVector ModMul(const IntVector &b) const {
            IntVector<IntegerType> ans(*this);
            return ans.ModMulEq(b);
        }

        const IntVector &ModMulEq(const IntVector &b) {
            if (this->modulus != b.modulus) {
                FHE_THROW(MathException, "IntVector multiply IntVectors of different modulus");
            } else if (this->data.size() != b.data.size()) {
                FHE_THROW(MathException, "IntVector multiply IntVectors of different lengths");
            }
            for (uint32_t i = 0; i < this->data.size(); i++) {
                data[i] = util::ModMul(data[i], b.data[i], modulus);
            }
            return *this;
        }

        IntVector ModExp(const IntegerType &b) const {
            IntVector<IntegerType> ans(*this);
            return ans.ModExpEq(b);
        }

        const IntVector &ModExpEq(const IntegerType &b) {
            for (uint32_t i = 0; i < this->data.size(); i++) {
                data[i] = util::ModExp(data[i], b, this->modulus);
            }
            return *this;
        }

        IntVector ModInverse() const {
            IntVector<IntegerType> ans(*this);
            return ans.ModInverseEq();
        }

        const IntVector &ModInverseEq() {
            for (uint32_t i = 0; i < this->data.size(); i++) {
                data[i] = util::ModInverse(data[i], this->modulus);
            }
            return *this;
        }

        IntVector Multiply(const IntVector &b) const;

        const IntVector MultiplyEq(const IntVector &b);

        IntVector MultiplyAndRound(const IntegerType &p, const IntegerType &q) const;

        const IntVector &MultiplyAndRoundEq(const IntegerType &p, const IntegerType &q);

        IntVector DivideAndRound(const IntegerType &q) const;

        const IntVector &DivideAndRoundEq(const IntegerType &q);

        /**
         * Digit vector at a specific index for all entries for a given number base.
         * Example: for vector (83, 1, 45), index 3 and base 4 we have:
         *
         *                           index:1,2,3,4
         * |83|                           |3,0,1,1|                 |1|
         * |1 | --base 4 decomposition--> |1,0,0,0| --at index 3--> |0|
         * |45|                           |1,3,2,0|                 |2|
         *
         * The return vector is (1,0,2)
         *
         * @param index is the index to return the digit from in all entries.
         * @param base is the base to use for the operation.
         * @return is the digit at a specific index for all entries for a given number
         * base
         */
//        Vector<IntegerType> GetDigitAtIndexForBase(uint32_t index, uint32_t base) const;


        const void ConvertToString(std::ostream &out) const {
            auto len = data.size();
            out << " modulus: " << modulus;
            out << " value:[";

            uint32_t printCount = len < 12 ? len : 12;
            for (size_t i = 0; i < printCount; ++i) {
                out << data[i];
                out << ((i == (printCount - 1)) ? "...]" : ",");
            }
        }

    private:
        std::vector<IntegerType> data;
        IntegerType modulus;
        State state;

        bool IndexCheck(size_t index) const { return index < this->data.size(); }

        void Mod() {
            for (uint32_t i = 0; i < this->data.size(); i++) {
                this->data[i] = util::Mod(this->data[i], this->modulus);
            }
        }
    };

}// namespace isecfhe
