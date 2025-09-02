#pragma once

#include <cstdint>

namespace algmath {

    inline uint8_t count_bits(uint32_t value) {
        value = (value & 0x55555555) + ((value >> 1) & 0x55555555);
        value = (value & 0x33333333) + ((value >> 2) & 0x33333333);
        value = (value & 0x0f0f0f0f) + ((value >> 4) & 0x0f0f0f0f);
        value = (value & 0x00ff00ff) + ((value >> 8) & 0x00ff00ff);
        value = (value & 0x0000ffff) + ((value >> 16) & 0x0000ffff);

        return value;
    }

} // namespace math
