#pragma once

#include <cstdint>

namespace mlir::libra::simd {

    // 共享的默认 simd level
    // N = 65536, Q_max = 1749, 61, 56 ...
    inline constexpr int64_t DEFAULT_LEVEL = 31;

}  // namespace mlir::libra::simd
