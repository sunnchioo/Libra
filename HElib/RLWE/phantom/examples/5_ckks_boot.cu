#include <algorithm>
#include <chrono>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <mutex>
#include <random>
#include <vector>

#include "bootstraper.h"
#include "example.h"
#include "phantom.h"
#include "util.cuh"

using namespace std;
using namespace phantom;
using namespace phantom::arith;
using namespace phantom::util;

#define EPSINON 0.001

inline bool operator==(const cuDoubleComplex &lhs, const cuDoubleComplex &rhs) {
    return fabs(lhs.x - rhs.x) < EPSINON;
}

inline bool compare_double(const double &lhs, const double &rhs) {
    return fabs(lhs - rhs) < EPSINON;
}

void examples_ckks_boot_inplace(const PhantomContext &context, const double &scale) {

    PhantomCKKSEncoder encoder(context);

    PhantomSecretKey secret_key(context);
    PhantomPublicKey public_key = secret_key.gen_publickey(context);
    PhantomRelinKey relin_key = secret_key.gen_relinkey(context);
    PhantomGaloisKey galois_key = secret_key.create_galois_keys(context);

    vector<cuDoubleComplex> output;
    // input message in [-1, 1]
    vector<cuDoubleComplex> input(encoder.slot_count(), make_cuDoubleComplex(0, 0));
    for (size_t i = 0; i < input.size(); i++) {
        input[i] = make_cuDoubleComplex(0.9, 0);
    }
    std::cout << "input slot count: " << encoder.slot_count() << std::endl;
    for (auto i = 0; i < 10; i++) {
        std::cout << "(" << input[i].x << ", " << input[i].y << ") ";
    }
    std::cout << std::endl;

    PhantomPlaintext plain;
    PhantomCiphertext cipher;
    encoder.encode(context, input, scale, plain);
    public_key.encrypt_asymmetric(context, plain, cipher);

    bootstraper::ckksbootstraper ckksboot(context, scale, encoder, secret_key, relin_key, galois_key);
    ckksboot.ckksbootstrapping(context, cipher);

    secret_key.decrypt(context, cipher, plain);
    encoder.decode(context, plain, output);

    std::cout << "result: ";
    for (auto i = 0; i < 10; i++) {
        std::cout << "(" << output[i].x << ", " << output[i].y << ") ";
        // if (!compare_double(input[i], output[i]))
        //     throw std::logic_error("error in example_ckks_boot");
    }
    std::cout << std::endl;
}

void examples_ckks_boot() {
    EncryptionParameters parms(scheme_type::ckks);

    // size_t N = 1 << 16;
    size_t N = 1 << 15;
    size_t alpha = 1;
    int log_default_scale = 40;
    double scale = pow(2.0, log_default_scale);

    // modulus params
    std::vector<int> modulus_size = {40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                                     40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                                     40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                                     40, 40, 40, 40, 40, // 40, 40, 40, 40, 40,
                                     40, 40, 40, 40, 40, 40, 40, 40, 40, 51};

    parms.set_poly_modulus_degree(N);
    parms.set_special_modulus_size(alpha);
    parms.set_coeff_modulus(CoeffModulus::Create(N, modulus_size));

    // size_t poly_modulus_degree = 1 << 15;
    // size_t alpha = 12;
    // parms.set_poly_modulus_degree(poly_modulus_degree);
    // parms.set_coeff_modulus(CoeffModulus::Create(
    //     poly_modulus_degree,
    //     {50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
    //      50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
    //      50, 50, 50, 50, 50, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60}));
    // parms.set_special_modulus_size(alpha);
    // double scale = pow(2.0, 50);

    PhantomContext context(parms);
    print_parameters(context);

    examples_ckks_boot_inplace(context, scale);
}