#pragma once

#include "exception.h"
#include "integer_interface.h"

namespace isecfhe {
    template<typename NativeInt>
    class NativeInteger : public IntegerInterface<NativeInteger<NativeInt>> {

    public:
        NativeInteger(const std::string &strValue) { AssignVal(strValue); }

        NativeInteger<NativeInt> Add(const NativeInteger<NativeInt> &b) {
            return NativeInteger("0");
        }

        const NativeInteger<NativeInt> &AddEq(const NativeInteger<NativeInt> &b) {
            return NativeInteger("0");
        }

        NativeInteger<NativeInt> Sub(const NativeInteger<NativeInt> &b) const {
            return NativeInteger("0");
        }

        const NativeInteger<NativeInt> &SubEq(const NativeInteger<NativeInt> &b) {
            return NativeInteger("0");
        }

        NativeInteger<NativeInt> Mul(const NativeInteger<NativeInt> &b) const {
            return NativeInteger("0");
        }

        const NativeInteger<NativeInt> &MulEq(const NativeInteger<NativeInt> &b) {
            return NativeInteger("0");
        }

        NativeInteger<NativeInt> DividedBy(const NativeInteger<NativeInt> &b) const {
            return NativeInteger("0");
        }

        const NativeInteger<NativeInt> &DividedByEq(const NativeInteger<NativeInt> &b) {
            return NativeInteger("0");
        }

        const std::string ConvertToString() { return "NativeInteger test"; }


    protected:
        void AssignVal(const std::string &string) {
            std::string str = string;
            if (str[0] == '-') {
                sign = true;
                str = string.substr(1);
            }
            NativeInt test_value = 0;
            value = 0;
            for (size_t i = 0; i < str.length(); i++) {
                int v = str[i] - '0';
                if (v < 0 || v > 9) {
                    FHE_THROW(isecfhe::TypeException, "String contains a non-digit");
                }
                value *= 10;
                value += v;

                if (value < test_value) {
                    FHE_THROW(isecfhe::MathException,
                              str + " is too large to fit in this native "
                                    "integer object");
                }
                test_value = value;
            }
        }


    private:
        NativeInt value;
        bool sign;
    };

}// namespace isecfhe
