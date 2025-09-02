#pragma once

// This code was modified based on HE3DB: https://github.com/zhouzhangwalker/HE3DB

#include "types.h"

namespace TFHEpp {
    template <class P>
    TFHEpp::TLWE<P> tlweSymInt32Encrypt(const typename P::T p, const double α, const double scale, const TFHEpp::Key<P> &key) {
        std::uniform_int_distribution<typename P::T> Torusdist(0, std::numeric_limits<typename P::T>::max());
        TFHEpp::TLWE<P> res = {};
        res[P::k * P::n] =
            TFHEpp::ModularGaussian<P>(static_cast<typename P::T>(p * scale), α);
        for (int k = 0; k < P::k; k++)
            for (int i = 0; i < P::n; i++) {
                res[k * P::n + i] = Torusdist(generator);
                res[P::k * P::n] += res[k * P::n + i] * key[k * P::n + i];
            }
        return res;
    }

    template <class P>
    typename P::T tlweSymInt32Decrypt(const TFHEpp::TLWE<P> &c, const double scale, const TFHEpp::Key<P> &key) {
        typename P::T phase = c[P::k * P::n];
        typename P::T plain_modulus = (1ULL << (std::numeric_limits<typename P::T>::digits - 1)) / scale;
        // plain_modulus *= 2;
        // typename P::T plain_modulus = P::plain_modulus;

        for (int k = 0; k < P::k; k++)
            for (int i = 0; i < P::n; i++)
                phase -= c[k * P::n + i] * key[k * P::n + i];
        typename P::T res =
            static_cast<typename P::T>(std::round(phase / scale)) % plain_modulus;
        return res;
    }

    template <class P>
    double tlweSymIntDecryptDouble(const TFHEpp::TLWE<P> &c, const double scale, const TFHEpp::Key<P> &key) {
        typename P::T phase = c[P::k * P::n];
        // typename P::T plain_modulus = (1ULL << (std::numeric_limits<typename P::T>::digits - 1)) / scale;
        // plain_modulus *= 2;
        // typename P::T plain_modulus = P::plain_modulus;
        typename P::T half = 1ULL << (std::numeric_limits<typename P::T>::digits - 1);
        // std::cout << "std::numeric_limits<typename P::T>::digits: " << std::numeric_limits<typename P::T>::digits << std::endl; 32

        for (int k = 0; k < P::k; k++)
            for (int i = 0; i < P::n; i++)
                phase -= c[k * P::n + i] * key[k * P::n + i];

        typename P::sT value = phase;
        if (phase >= half) {
            value -= half;
            value -= half;
        }
        double res = value / scale;
        return res;
    }

    template <class P>
    void print_lwe_ct_vec(std::vector<TFHEpp::TLWE<P>> &c, const double scale, const TFHEpp::SecretKey &key) {
        std::vector<typename P::T> res(c.size());
        for (size_t i = 0; i < c.size(); i++) {
            res[i] = tlweSymInt32Decrypt<P>(c[i], scale, key.key.get<P>());
        }

        std::cout << "Decrypted LWE result: ";
        for (size_t i = 0; i < res.size(); i++) {
            std::cout << res[i] << " ";
        }
        std::cout << std::endl;
    }

    template <class P>
    void print_lwe_ct_vec_double(std::vector<TFHEpp::TLWE<P>> &c, const double scale, const TFHEpp::SecretKey &key) {
        std::vector<double> res(c.size());
        for (size_t i = 0; i < c.size(); i++) {
            res[i] = tlweSymIntDecryptDouble<P>(c[i], scale, key.key.get<P>());
        }

        std::cout << "Decrypted LWE double result: ";
        for (size_t i = 0; i < res.size(); i++) {
            std::cout << res[i] << " ";
        }
        std::cout << std::endl;
    }

    template <class P>
    void print_lwe_ct_vec_double_err(std::vector<TFHEpp::TLWE<P>> &c, const double scale, const TFHEpp::SecretKey &key, std::vector<double> &res) {
        for (size_t i = 0; i < c.size(); i++) {
            res[i] = tlweSymIntDecryptDouble<P>(c[i], scale, key.key.get<P>());
        }

        // std::cout << "Decrypted LWE double result: ";
        // for (size_t i = 0; i < res.size(); i++) {
        //     std::cout << res[i] << " ";
        // }
        // std::cout << std::endl;
    }
}  // namespace TFHEpp
