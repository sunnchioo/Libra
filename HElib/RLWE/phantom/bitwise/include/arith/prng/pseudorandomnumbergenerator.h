#pragma once

#include "blake2engine.h"
#include <chrono>
#include <memory>
#include <random>
#include <thread>

#include "arith/prng/blake2.h"

namespace isecfhe {
    class PseudoRandomNumberGenerator {
    public:
        static Blake2Engine &GetPRNG() {
            // initialization of PRNGs
            if (m_prng == nullptr) {
#pragma omp critical
                {
#if defined(FIXED_SEED)
                    FHEDebug(" ******** ******************************  ********* ");
                    FHEDebug(" ******** 请注意 当前为dedug模式, seed值固定 ********* ");
                    FHEDebug(" ******** ******************************  ********* \n");
                    std::array<uint32_t, 16> seed{};
                    seed[0] = 1;
                    m_prng  = std::make_shared<Blake2Engine>(seed);
#else
                    // A 512-bit seed is generated for each thread (this roughly corresponds
                    // to 256 bits of security). The seed is the sum of a random sample
                    // generated using std::random_device (typically works correctly in
                    // Linux, MacOS X, and MinGW starting with GCC 9.2) and a BLAKE2 sample
                    // seeded from current time stamp, a hash of the current thread, and a
                    // memory location of a heap variable. The BLAKE2 sample is added in
                    // case random_device is deterministic (happens on MinGW with GCC
                    // below 9.2). All future calls to PRNG use the seed generated here.

                    // The code below derives randomness from time, thread id, and a memory
                    // location of a heap variable. This seed is relevant only if the
                    // implementation of random_device is deterministic (as in older
                    // versions of GCC in MinGW)
                    std::array<uint32_t, 16> initKey{};
                    // high-resolution clock typically has a nanosecond tick period
                    // Arguably this may give up to 32 bits of entropy as the clock gets
                    // recycled every 4.3 seconds
                    initKey[0] = std::chrono::high_resolution_clock::now().time_since_epoch().count();
                    // A thread id is often close to being random (on most systems)
                    initKey[1] = std::hash<std::thread::id>{}(std::this_thread::get_id());
                    // On a 64-bit machine, the thread id is 64 bits long
                    // skip on 32-bit arm architectures
#if !defined(__arm__) && !defined(__EMSCRIPTEN__)
                    if (sizeof(size_t) == 8)
                        initKey[2] = (std::hash<std::thread::id>{}(std::this_thread::get_id()) >> 32);
#endif

                    // heap variable; we are going to use the least 32 bits of its memory
                    // location as the counter for BLAKE2 This will increase the entropy of
                    // the BLAKE2 sample
                    void *mem = malloc(1);
                    uint32_t counter = reinterpret_cast<long long>(mem);  // NOLINT
                    free(mem);

                    Blake2Engine gen(initKey, counter);

                    std::uniform_int_distribution<uint32_t> distribution(0);
                    std::array<uint32_t, 16> seed{};
                    for (uint32_t i = 0; i < 16; i++) {
                        seed[i] = distribution(gen);
                    }

                    std::array<uint32_t, 16> rdseed{};
                    size_t attempts = 3;
                    bool rdGenPassed = false;
                    size_t idx = 0;
                    while (!rdGenPassed && idx < attempts) {
                        try {
                            std::random_device genR;
                            for (uint32_t i = 0; i < 16; i++) {
                                // we use the fact that there is no overflow for unsigned integers
                                // (from C++ standard) i.e., arithmetic mod 2^32 is performed. For
                                // the seed to be random, it is sufficient for one of the two
                                // samples below to be random. In almost all practical cases,
                                // distribution(genR) is random. We add distribution(gen) just in
                                // case there is an implementation issue with random_device (as in
                                // older MinGW systems).
                                rdseed[i] = distribution(genR);
                            }
                            rdGenPassed = true;
                        }
                        catch (std::exception &e) {
                        }
                        idx++;
                    }

                    for (uint32_t i = 0; i < 16; i++) {
                        seed[i] += rdseed[i];
                    }

                    m_prng = std::make_shared<Blake2Engine>(seed);
#endif
                }
            }
            count++;
            return *m_prng;
        }

        static inline int count = 0;
    private:
        // shared pointer to a thread-specific PRNG engine
        static inline std::shared_ptr<Blake2Engine> m_prng;
    };

}
