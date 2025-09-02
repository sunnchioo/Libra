#pragma once

#include "bootstrapping/Bootstrapper.cuh"
#include "ckks_evaluator.cuh"
#include "phantom.h"

namespace rlwe {
    using namespace std;
    using namespace phantom;

    using CUDATimer = phantom::util::CUDATimer;

    class SoftmaxEvaluator {
    private:
        CKKSEvaluator *ckks = nullptr;
        Bootstrapper *bootstrapper = nullptr;

    public:
        SoftmaxEvaluator(CKKSEvaluator &ckks) : ckks(&ckks) {}
        SoftmaxEvaluator(CKKSEvaluator &ckks, Bootstrapper &bootstrapper) : ckks(&ckks), bootstrapper(&bootstrapper) {}

        void compare(PhantomCiphertext &cipher0, PhantomCiphertext &cipher1, std::vector<PhantomCiphertext> &bool_ct);
        void findmax(PhantomCiphertext &input, long points, PhantomCiphertext &res);
        void softmax_scaled(PhantomCiphertext &x, PhantomCiphertext &res, int len);
        void softmax(PhantomCiphertext &x, PhantomCiphertext &res, int len);
    };
} // namespace rlwe
