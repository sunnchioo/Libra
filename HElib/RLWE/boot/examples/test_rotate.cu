#include <random>

#include "bootstrapping/Bootstrapper.cuh"
#include "phantom.h"

using namespace std;
using namespace phantom;

#define EPSINON 0.001

inline bool operator==(const cuDoubleComplex &lhs, const cuDoubleComplex &rhs) {
    return fabs(lhs.x - rhs.x) < EPSINON;
}

inline bool compare_double(const double &lhs, const double &rhs) {
    return fabs(lhs - rhs) < EPSINON;
}

/*
Helper function: Prints a vector of floating-point values.
*/
inline void print_vector(std::vector<cuDoubleComplex> vec, std::size_t print_size = 4, int prec = 3) {
    /*
    Save the formatting information for std::cout.
    */
    std::ios old_fmt(nullptr);
    old_fmt.copyfmt(std::cout);

    std::size_t slot_count = vec.size();

    std::cout << std::fixed << std::setprecision(prec);
    std::cout << std::endl;
    if (slot_count <= 2 * print_size) {
        std::cout << "    [";
        for (std::size_t i = 0; i < slot_count; i++) {
            std::cout << " " << vec[i].x << " + i * " << vec[i].y << ((i != slot_count - 1) ? "," : " ]\n");
        }
    } else {
        vec.resize(std::max(vec.size(), 2 * print_size));
        std::cout << "    [";
        for (std::size_t i = 0; i < print_size; i++) {
            std::cout << " " << vec[i].x << " + i * " << vec[i].y << ",";
        }
        if (vec.size() > 2 * print_size) {
            std::cout << " ...,";
        }
        for (std::size_t i = slot_count - print_size; i < slot_count; i++) {
            std::cout << " " << vec[i].x << " + i * " << vec[i].y << ((i != slot_count - 1) ? "," : " ]\n");
        }
    }
    std::cout << std::endl;

    /*
    Restore the old std::cout formatting.
    */
    std::cout.copyfmt(old_fmt);
}

void example_ckks_rotation(PhantomContext &context, const double &scale) {
    std::cout << "Example: CKKS HomRot test" << std::endl;

    // KeyGen
    PhantomSecretKey secret_key(context);
    PhantomPublicKey public_key = secret_key.gen_publickey(context);
    PhantomGaloisKey galois_keys = secret_key.create_galois_keys(context);

    int step = 1;

    PhantomCKKSEncoder encoder(context);

    size_t slot_count = encoder.slot_count();
    cout << "Number of slots: " << slot_count << endl;

    vector<cuDoubleComplex> x_msg, result;
    double rand_real, rand_imag;

    size_t x_size = slot_count;
    x_msg.reserve(x_size);
    for (size_t i = 0; i < x_size; i++) {
        rand_real = (double)rand() / RAND_MAX;
        rand_imag = (double)rand() / RAND_MAX;
        x_msg.push_back(make_cuDoubleComplex(rand_real, rand_imag));
    }
    cout << "Message vector: " << endl;
    print_vector(x_msg, 3, 7);

    PhantomPlaintext x_plain;
    PhantomPlaintext x_rot_plain;

    encoder.encode(context, x_msg, scale, x_plain);

    PhantomCiphertext x_cipher;

    public_key.encrypt_asymmetric(context, x_plain, x_cipher);

    cout << "Compute, rot vector x." << endl;

    auto start = system_clock::now();

    rotate_inplace(context, x_cipher, step, galois_keys);

    duration<double> sec = system_clock::now() - start;
    std::cout << "Rotate took: " << sec.count() * 1000 << " ms" << endl;
    std::cout << "Return cipher level: " << x_cipher.coeff_modulus_size() << endl;

    secret_key.decrypt(context, x_cipher, x_rot_plain);

    encoder.decode(context, x_rot_plain, result);
    cout << "Result vector: " << endl;
    print_vector(result, 3, 7);

    bool correctness = true;
    for (size_t i = 0; i < x_size; i++) {
        correctness &= result[i] == x_msg[(i + step) % x_size];
    }
    if (!correctness)
        throw std::logic_error("Homomorphic rotation error");
    result.clear();
    x_msg.clear();

    std::cout << "Example: CKKS HomConj test" << std::endl;

    x_msg.reserve(x_size);
    for (size_t i = 0; i < x_size; i++) {
        rand_real = (double)rand() / RAND_MAX;
        rand_imag = (double)rand() / RAND_MAX;
        x_msg.push_back(make_cuDoubleComplex(rand_real, rand_imag));
    }
    cout << "Message vector: " << endl;
    print_vector(x_msg, 3, 7);

    PhantomPlaintext x_conj_plain;

    encoder.encode(context, x_msg, scale, x_plain);
    public_key.encrypt_asymmetric(context, x_plain, x_cipher);

    cout << "Compute, conjugate vector x." << endl;
    rotate_inplace(context, x_cipher, 0, galois_keys);

    secret_key.decrypt(context, x_cipher, x_conj_plain);

    encoder.decode(context, x_conj_plain, result);
    cout << "Result vector: " << endl;
    print_vector(result, 3, 7);

    correctness = true;
    for (size_t i = 0; i < x_size; i++) {
        correctness &= result[i] == make_cuDoubleComplex(x_msg[i].x, -x_msg[i].y);
    }
    if (!correctness)
        throw std::logic_error("Homomorphic conjugate error");
    result.clear();
    x_msg.clear();
}

int main() {

    phantom::EncryptionParameters parms(scheme_type::ckks);
    int alpha = 6;

    size_t poly_modulus_degree = 1 << 16;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(phantom::arith::CoeffModulus::Create(
        poly_modulus_degree,
        {51, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46,
         51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51,
         51, 51, 51, 51, 51, 51}));
    parms.set_special_modulus_size(alpha);
    // parms.set_special_modulus_size(1);
    double scale = pow(2.0, 46);

    PhantomContext context(parms);

    example_ckks_rotation(context, scale);
}