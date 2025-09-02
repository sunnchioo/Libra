#include <algorithm>
#include <iostream>

#include "cutfhe++.h"
#include "phantom.h"
#include "tlwe.h"

using namespace cuTFHEpp;
using namespace cuTFHEpp::util;
using namespace phantom::arith;

template <typename P>
class tlwevaluator {
    using Lvl = P;
    using TLWELvl = TFHEpp::TLWE<Lvl>;

private:
    TFHESecretKey *sk;
    TFHEEvalKey *ek;

    Pointer<Context> pbscontext;

    double scale_ = 1.0;  // Default scale value
    inline static thread_local std::random_device generator;

public:
    tlwevaluator(TFHESecretKey *sk, TFHEEvalKey *ek, double scale) : sk(sk), ek(ek), scale_(scale) {
        if (sk == nullptr || ek == nullptr) {
            throw std::invalid_argument("Secret key and evaluation key must not be null.");
        }
        pbscontext.initialize(*ek);
    }

    tlwevaluator() {
        TFHESecretKey sk_;
        TFHEEvalKey ek_;
        load_keys<BootstrappingKeyFFTLvl01, BootstrappingKeyFFTLvl02,
                  KeySwitchingKeyLvl10, KeySwitchingKeyLvl20, KeySwitchingKeyLvl21>(sk_, ek_);

        sk = &sk_;
        ek = &ek_;

        scale_ = Lvl::Δ;
    }

    ~tlwevaluator() = default;

    void add(TFHEpp::TLWE<Lvl> *res, const TFHEpp::TLWE<Lvl> *src0, const TFHEpp::TLWE<Lvl> *src1, int batch_size) {
        // std::cout << "LWE ciphertext add 0" << std::endl;
        cuTFHEpp::HomADD<Lvl><<<GRID_DIM, BLOCK_DIM>>>(res, src0, src1, batch_size);
        // std::cout << "LWE ciphertext add 1" << std::endl;
    }

    void sub(TFHEpp::TLWE<Lvl> *res, const TFHEpp::TLWE<Lvl> *src0, const TFHEpp::TLWE<Lvl> *src1, int batch_size) {
        cuTFHEpp::HomSUB<Lvl><<<GRID_DIM, BLOCK_DIM>>>(res, src0, src1, batch_size);
    }

    void sub_single(TFHEpp::TLWE<Lvl> *res, const TFHEpp::TLWE<Lvl> *src0, const TFHEpp::TLWE<Lvl> *src1, int batch_size) {
        cuTFHEpp::HomSUBSingle<Lvl><<<GRID_DIM, BLOCK_DIM>>>(res, src0, src1, batch_size);
    }

    TFHEpp::TLWE<Lvl> tlweSymIntEncrypt(const Lvl::T p, const double α, const double scale, const TFHEpp::Key<Lvl> &key) {
        std::uniform_int_distribution<typename Lvl::T> Torusdist(0, std::numeric_limits<typename Lvl::T>::max());
        TFHEpp::TLWE<P> res = {};
        res[P::k * P::n] =
            TFHEpp::ModularGaussian<P>(static_cast<typename Lvl::T>(p * scale), α);
        for (int k = 0; k < P::k; k++)
            for (int i = 0; i < P::n; i++) {
                res[k * P::n + i] = Torusdist(generator);
                res[P::k * P::n] += res[k * P::n + i] * key[k * P::n + i];
            }
        return res;
    }

    TFHEpp::TLWE<Lvl> tlweSymIntEncrypt(const typename P::T p, const double α, const TFHEpp::Key<P> &key) {
        std::uniform_int_distribution<typename P::T> Torusdist(0, std::numeric_limits<typename P::T>::max());
        TFHEpp::TLWE<P> res = {};
        res[P::k * P::n] =
            TFHEpp::ModularGaussian<P>(static_cast<typename P::T>(p * scale_), α);
        for (int k = 0; k < P::k; k++)
            for (int i = 0; i < P::n; i++) {
                res[k * P::n + i] = Torusdist(generator);
                res[P::k * P::n] += res[k * P::n + i] * key[k * P::n + i];
            }
        return res;
    }

    typename P::T tlweSymIntDecrypt(const TFHEpp::TLWE<P> &c, const double scale, const TFHEpp::Key<P> &key) {
        typename P::T phase = c[P::k * P::n];
        typename P::T plain_modulus = (1ULL << (std::numeric_limits<typename P::T>::digits - 1)) / scale;

        for (int k = 0; k < P::k; k++)
            for (int i = 0; i < P::n; i++)
                phase -= c[k * P::n + i] * key[k * P::n + i];
        typename P::T res =
            static_cast<typename P::T>(std::round(phase / scale)) % plain_modulus;
        return res;
    }

    void tlweSymIntDecryptCudaMod(const TFHEpp::TLWE<P> *c, const double scale, Modulus q0, const TFHEpp::Key<P> &key) {
        TFHEpp::TLWE<Lvl> temp;
        cudaMemcpy(&temp, c, sizeof(TFHEpp::TLWE<Lvl>), cudaMemcpyDeviceToHost);

        typename P::T phase = temp[P::k * P::n];
        typename P::T acc = 0;
        for (int k = 0; k < P::k; k++) {
            acc = dot_product_mod(temp.data() + k * P::n, key.data() + k * P::n, P::n, q0);
        }
        acc = add_uint_mod(phase, acc, q0);

        typename P::sT res = acc;
        if (acc >= (q0.value() >> 1))
            res -= q0.value();

        double plain = static_cast<double>(res / scale);
        std::cout << "plain: " << plain << std::endl;
    }

    void tlweSymIntDecryptCudaMod(const TFHEpp::TLWE<P> *c, const double scale, Modulus q0, int batch_size) {
        const auto &key = sk->key.get<P>();
        for (size_t i = 0; i < batch_size; i++) {
            tlweSymIntDecryptCudaMod(c, scale, q0, key);
        }
    }

    void print_lwe_ct_vec(std::vector<TLWELvl> &lwe_ct, std::string name) {
        std::cout << "LWE ciphertext vector: " << name << std::endl;
        TFHEpp::print_lwe_ct_vec<Lvl>(lwe_ct, scale_, *sk);
    }

    void print_lwe_ct_vec(std::vector<TLWELvl> &lwe_ct, const double scale, std::string name) {
        std::cout << "LWE ciphertext vector: " << name << std::endl;
        TFHEpp::print_lwe_ct_vec<Lvl>(lwe_ct, scale, *sk);
    }

    void print_lwe_ct_vec_double(std::vector<TLWELvl> &lwe_ct, std::string name) {
        std::cout << "LWE ciphertext vector: " << name << std::endl;
        TFHEpp::print_lwe_ct_vec_double<Lvl>(lwe_ct, scale_, *sk);
    }

    void print_lwe_ct_vec_double(std::vector<TLWELvl> &lwe_ct, const double scale, std::string name) {
        std::cout << "LWE ciphertext vector: " << name << std::endl;
        TFHEpp::print_lwe_ct_vec_double<Lvl>(lwe_ct, scale, *sk);
    }

    void print_lwe_ct_vec_double_err(std::vector<TLWELvl> &lwe_ct, const double scale, std::string name, std::vector<double> &res) {
        std::cout << "LWE ciphertext vector: " << name << std::endl;
        TFHEpp::print_lwe_ct_vec_double_err<Lvl>(lwe_ct, scale, *sk, res);
    }

    void print_lwe_ct_value(std::vector<TFHEpp::TLWE<Lvl>> &value, std::string name) {
        std::cout << "LWE value vector: " << name << std::endl;
        for (size_t i = 0; i < value.size(); i++) {
            std::cout << "value " << i << " : ";
            for (size_t j = 0; j < 10; j++) {
                std::cout << value[i][j] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    void print_culwe_ct_value(TFHEpp::TLWE<Lvl> *value, int batch_size, std::string name) {
        // std::cout << "cuLWE value vector: " << name << std::endl;

        std::vector<TFHEpp::TLWE<Lvl>> temp(batch_size);
        cudaMemcpy(temp.data(), value, batch_size * sizeof(TFHEpp::TLWE<Lvl>), cudaMemcpyDeviceToHost);

        print_lwe_ct_vec(temp, name);
    }

    void print_culwe_ct_value(TFHEpp::TLWE<Lvl> *value, const double scale, int batch_size, std::string name) {
        // std::cout << "cuLWE value vector: " << name << std::endl;

        std::vector<TFHEpp::TLWE<Lvl>> temp(batch_size);
        cudaMemcpy(temp.data(), value, batch_size * sizeof(TFHEpp::TLWE<Lvl>), cudaMemcpyDeviceToHost);

        print_lwe_ct_vec(temp, scale, name);
    }

    void print_culwe_ct_value_double(TFHEpp::TLWE<Lvl> *value, int batch_size, std::string name) {
        std::cout << "cuLWE value vector: " << name << std::endl;

        std::vector<TFHEpp::TLWE<Lvl>> temp(batch_size);
        cudaMemcpy(temp.data(), value, batch_size * sizeof(TFHEpp::TLWE<Lvl>), cudaMemcpyDeviceToHost);

        print_lwe_ct_vec_double(temp, scale_, name);
    }

    void print_culwe_ct_value_double_err(TFHEpp::TLWE<Lvl> *value, int batch_size, std::string name, std::vector<double> &g) {
        std::vector<double> res(g.size());

        std::cout << "cuLWE value vector: " << name << std::endl;

        std::vector<TFHEpp::TLWE<Lvl>> temp(batch_size);
        cudaMemcpy(temp.data(), value, batch_size * sizeof(TFHEpp::TLWE<Lvl>), cudaMemcpyDeviceToHost);

        print_lwe_ct_vec_double_err(temp, scale_, name, res);

        double err = 0.;
        for (size_t i = 0; i < g.size(); i++) {
            err += std::abs(1.0 - res[i]);
        }
        std::cout << "error: " << err / g.size() << " size: " << g.size() << std::endl
                  << std::endl;
    }

    void print_culwe_ct_value_double(TFHEpp::TLWE<Lvl> *value, const double scale, int batch_size, std::string name) {
        // std::cout << "cuLWE value vector: " << name << std::endl;

        std::vector<TFHEpp::TLWE<Lvl>> temp(batch_size);
        cudaMemcpy(temp.data(), value, batch_size * sizeof(TFHEpp::TLWE<Lvl>), cudaMemcpyDeviceToHost);

        print_lwe_ct_vec_double(temp, scale, name);
    }

    TFHEEvalKey *get_ek() const noexcept {
        return ek;
    }

    TFHESecretKey *get_sk() const noexcept {
        return sk;
    }

    // Context *get_context() const noexcept {
    //     return pbscontext.get_ptr();
    // }
    const Pointer<Context> &get_pbscontext() const noexcept {
        return pbscontext;
    }

    Context *get_context_host() const noexcept {
        return pbscontext.get_host();
    }
};
