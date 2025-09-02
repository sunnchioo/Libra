#pragma once

#include "big_integer.h"
#include "int_vector.h"

namespace isecfhe {

    template <typename IntType>
    class IntVector;


    using BigInt = BigInteger<limbtype>;

    using BigIntVector = IntVector<BigInt>;

    using NativeVector = IntVector<limbtype>;

}
