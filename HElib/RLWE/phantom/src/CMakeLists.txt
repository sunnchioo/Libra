include_directories(${CMAKE_SOURCE_DIR}/RLWE/phantom/include) 

add_library(phantomfhe STATIC
        context.cu
        modulus.cu
        batchencoder.cu
        prng.cu
        polymath.cu
        secretkey.cu
        rns.cu
        rns_base.cu
        rns_bconv.cu
        scalingvariant.cu
        evaluate.cu
        fft.cu
        ckks.cu
        galois.cu
        eval_key_switch.cu
        ckks_evaluator.cu

        ntt/fntt_2d.cu
        ntt/ntt_1d.cu
        ntt/ntt_keyswitch_old.cu
        ntt/intt_2d.cu
        ntt/ntt_modup.cu
        ntt/ntt_moddown.cu

        util/blake2b.cu
        util/blake2xb.cu
        util/globals.cu
        util/hash.cu
        util/ntt.cu
        util/numth.cu
        util/rns.cu
        util/uintarith.cu
        util/uintarithmod.cu
        util/uintarithsmallmod.cu
)

target_compile_options(phantomfhe PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--default-stream per-thread>)
target_compile_options(phantomfhe PRIVATE $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx>)
target_compile_features(phantomfhe PUBLIC cxx_std_20 cuda_std_20)
