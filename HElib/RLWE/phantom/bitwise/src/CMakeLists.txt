# include_directories(${CMAKE_SOURCE_DIR}/phantom/bitwise/include)

add_library(bitwise SHARED
        binfhe-base-scheme.cu
        binfhe-base-scheme-batch.cu
        binfhecontext.cu
        compact_ntt.cu
        kernel.cu
        kernel-fuse.cu
        rgsw-acc.cu
        rgsw-acc-batch.cu

        arith/prng/blake2b-ref.cpp
        arith/prng/blake2xb-ref.cpp)
