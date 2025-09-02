#pragma once
#include <fstream>

#include "ciphertext.h"
#include "context.cuh"
#include "plaintext.h"
#include "prng.cuh"

class PhantomSecretKey;

class PhantomPublicKey;

class PhantomRelinKey;

class PhantomGaloisKey;

// class PhantomLweKey;

// class PhantomLweKey {

//     friend class PhantomSecretKey;

// private:
//     bool gen_flag_ = false;
//     PhantomCiphertext lwe_sk_;
//     phantom::parms_id_type parms_id_ = phantom::parms_id_zero;

//     // bool gen_flag_ = false;
//     // uint64_t chain_index_ = 0;
//     // uint64_t sk_max_power_ = 0; // the max power of secret key
//     // uint64_t poly_modulus_degree_ = 0;
//     // uint64_t coeff_modulus_size_ = 0;

//     // phantom::util::cuda_auto_ptr<uint64_t> data_rns_;
//     // phantom::util::cuda_auto_ptr<uint64_t> secret_key_array_; // the powers of secret key

// public:
//     PhantomLweKey() = default;

//     PhantomLweKey(const PhantomLweKey &) = delete;

//     PhantomLweKey &operator=(const PhantomLweKey &) = delete;

//     PhantomLweKey(PhantomLweKey &&) = default;

//     PhantomLweKey &operator=(PhantomLweKey &&) = default;

//     ~PhantomLweKey() = default;
// };

class PhantomPublicKey {
    friend class PhantomSecretKey;

private:
    bool gen_flag_ = false;
    PhantomCiphertext pk_;
    phantom::util::cuda_auto_ptr<uint8_t> prng_seed_a_;  // for compress pk

    /** Encrypt zero using the public key, internal function, no modulus switch here.
     * @param[in] context PhantomContext
     * @param[inout] cipher The generated ciphertext
     * @param[in] chain_index The id of the corresponding context data
     * @param[in] is_ntt_form Whether the ciphertext should be in NTT form
     */
    void encrypt_zero_asymmetric_internal_internal(const PhantomContext &context, PhantomCiphertext &cipher,
                                                   size_t chain_index,
                                                   bool is_ntt_form, const cudaStream_t &stream) const;

public:
    /** Encrypt zero using the public key, and perform the model switch is necessary
     * @brief pk [pk0, pk1], ternary variable u, cbd (gauss) noise e0, e1, return [pk0*u+e0, pk1*u+e1]
     * @param[in] context PhantomContext
     * @param[inout] cipher The generated ciphertext
     * @param[in] chain_index The id of the corresponding context data
     */
    void encrypt_zero_asymmetric_internal(const PhantomContext &context, PhantomCiphertext &cipher,
                                          size_t chain_index, const cudaStream_t &stream) const;

public:
    PhantomPublicKey() = default;

    PhantomPublicKey(const PhantomPublicKey &) = delete;

    PhantomPublicKey &operator=(const PhantomPublicKey &) = delete;

    PhantomPublicKey(PhantomPublicKey &&) = default;

    PhantomPublicKey &operator=(PhantomPublicKey &&) = default;

    ~PhantomPublicKey() = default;

    // new add
    const PhantomCiphertext &get_pk() const {
        if (!gen_flag_) {
            throw std::logic_error("Public key not generated yet");
        }
        return pk_;
    }
    void setpublickey(const uint64_t *key, const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream);

    /** asymmetric encryption.
     * @brief: asymmetric encryption requires modulus switching.
     * @param[in] context PhantomContext
     * @param[in] plain The data to be encrypted
     * @param[out] cipher The generated ciphertext
     */
    void encrypt_asymmetric(const PhantomContext &context, const PhantomPlaintext &plain, PhantomCiphertext &cipher,
                            const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream);

    // for python wrapper

    inline PhantomCiphertext encrypt_asymmetric(const PhantomContext &context, const PhantomPlaintext &plain,
                                                const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream) {
        PhantomCiphertext cipher;
        encrypt_asymmetric(context, plain, cipher, stream_wrapper);
        return cipher;
    }

    inline PhantomCiphertext encrypt_zero_asymmetric(const PhantomContext &context,
                                                     const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream) {
        // const auto &s = cudaStreamPerThread;
        const auto &s = stream_wrapper.get_stream();
        PhantomCiphertext cipher;
        encrypt_zero_asymmetric_internal(context, cipher, context.get_first_index(), s);
        return cipher;
    }

    // void save(std::ostream &stream) const {
    //     if (!gen_flag_)
    //         throw std::invalid_argument("PhantomPublicKey has not been generated");
    //     pk_.save(stream);
    // }

    // void load(std::istream &stream) {
    //     pk_.load(stream);
    //     gen_flag_ = true;
    // }
};

/** PhantomRelinKey contains the relinear key in RNS and NTT form
 * gen_flag denotes whether the secret key has been generated.
 */
class PhantomRelinKey {
    friend class PhantomSecretKey;

private:
    size_t pk_num_ = 0;
    bool gen_flag_ = false;
    phantom::parms_id_type parms_id_ = phantom::parms_id_zero;
    std::vector<phantom::util::cuda_auto_ptr<uint64_t>> public_keys_;
    phantom::util::cuda_auto_ptr<uint64_t *> public_keys_ptr_;

public:
    PhantomRelinKey() = default;

    PhantomRelinKey(const PhantomRelinKey &) = delete;

    PhantomRelinKey &operator=(const PhantomRelinKey &) = delete;

    PhantomRelinKey(PhantomRelinKey &&) = default;

    PhantomRelinKey &operator=(PhantomRelinKey &&) = default;

    explicit PhantomRelinKey(const PhantomContext &context) {
        auto &context_data = context.get_context_data(0);
        auto &parms = context_data.parms();
        auto &key_modulus = parms.key_modulus();

        size_t size_P = parms.special_modulus_size();
        size_t size_QP = key_modulus.size();
        size_t size_Q = size_QP - size_P;
        if (size_P != 0) {
            size_t dnum = size_Q / size_P;
            pk_num_ = dnum;
        }
        gen_flag_ = false;
    }

    ~PhantomRelinKey() = default;

    [[nodiscard]] inline auto public_keys_ptr() const {
        return public_keys_ptr_.get();
    }

    [[nodiscard]] inline auto public_keys_ptr_size() const {
        return public_keys_.size();
    }

    // new add
    std::vector<phantom::util::cuda_auto_ptr<uint64_t>> &reline_keys() {
        gen_flag_ = false;
        return public_keys_;
    }

    uint64_t *reline_keys(size_t index) {
        if (index >= public_keys_.size())
            throw std::out_of_range("Index invalid");
        return public_keys_[index].get();
    }

    uint64_t **reline_keys_ptr() {
        if (!gen_flag_)
            rebuild_reline_keys_ptrs();
        return public_keys_ptr_.get();
    }

    void rebuild_reline_keys_ptrs() {
        std::vector<uint64_t *> host_ptrs;
        for (auto &key : public_keys_) {
            host_ptrs.push_back(key.get());
        }
        // public_keys_ptr_.cuda_memcpy_from_host(host_ptrs.data(), host_ptrs.size());
        CHECK_CUDA_ERROR(cudaMemcpy(public_keys_ptr_.get(), host_ptrs.data(), sizeof(uint64_t *) * host_ptrs.size(), cudaMemcpyHostToDevice));
        gen_flag_ = true;
    }
};

/** PhantomGaloisKey stores Galois keys.
 * gen_flag denotes whether the Galois key has been generated.
 */
class PhantomGaloisKey {
    friend class PhantomSecretKey;

private:
    bool gen_flag_ = false;
    phantom::parms_id_type parms_id_ = phantom::parms_id_zero;
    std::vector<PhantomRelinKey> relin_keys_;
    size_t relin_key_num_ = 0;

public:
    PhantomGaloisKey() = default;

    PhantomGaloisKey(const PhantomGaloisKey &) = delete;

    explicit PhantomGaloisKey(const PhantomContext &context) {
    }

    PhantomGaloisKey &operator=(const PhantomGaloisKey &) = delete;

    PhantomGaloisKey(PhantomGaloisKey &&) = default;

    PhantomGaloisKey &operator=(PhantomGaloisKey &&) = default;

    ~PhantomGaloisKey() = default;

    [[nodiscard]] auto &get_relin_keys(size_t index) const {
        return relin_keys_.at(index);
    }

    [[nodiscard]] auto get_relin_keys_size() const {
        return relin_keys_.size();
    }

    [[nodiscard]] auto &relin_key_num() {
        return relin_key_num_;
    }
};

/** PhantomSecretKey contains the secret key in RNS and NTT form
 * gen_flag denotes whether the secret key has been generated.
 */
class PhantomSecretKey {
private:
    bool gen_flag_ = false;
    uint64_t chain_index_ = 0;
    uint64_t sk_max_power_ = 0;  // the max power of secret key
    uint64_t poly_modulus_degree_ = 0;
    uint64_t coeff_modulus_size_ = 0;

    phantom::util::cuda_auto_ptr<uint64_t> data_rns_;
    phantom::util::cuda_auto_ptr<uint64_t> secret_key_array_;          // the powers of secret key
    phantom::util::cuda_auto_ptr<uint64_t> secret_key_non_ntt_array_;  // the powers of secret key

    /** Generate the powers of secret key
     * @param[in] context PhantomContext
     * @param[in] max_power the mox power of secret key
     * @param[out] secret_key_array
     */
    void compute_secret_key_array(const PhantomContext &context, size_t max_power, const cudaStream_t &stream);

    [[nodiscard]] inline auto secret_key_array() const {
        return secret_key_array_.get();
    }
    [[nodiscard]] inline auto secret_key_non_ntt_array() const {
        return secret_key_non_ntt_array_.get();
    }

    /** Generate one public key for this secret key
     * Return PhantomPublicKey
     * @param[in] context PhantomContext
     * @param[inout] relin_key The generated relinear key
     * @throws std::invalid_argument if secret key or relinear key has not been inited
     */
    void generate_one_kswitch_key(const PhantomContext &context, uint64_t *new_key, PhantomRelinKey &relin_key,
                                  const cudaStream_t &stream) const;

    void
    bfv_decrypt(const PhantomContext &context, const PhantomCiphertext &encrypted, PhantomPlaintext &destination,
                const cudaStream_t &stream);

    void
    ckks_decrypt(const PhantomContext &context, const PhantomCiphertext &encrypted, PhantomPlaintext &destination,
                 const cudaStream_t &stream);

    void
    bgv_decrypt(const PhantomContext &context, const PhantomCiphertext &encrypted, PhantomPlaintext &destination,
                const cudaStream_t &stream);

    // new add
    // void
    // lwe_decrypt(const PhantomContext &context, const PhantomCiphertext &encrypted, PhantomPlaintext &destination,
    //             const cudaStream_t &stream);

public:
    void gen_secretkey(const PhantomContext &context, const cudaStream_t &stream);

    explicit inline PhantomSecretKey(const PhantomContext &context) {
        gen_secretkey(context, phantom::util::global_variables::default_stream->get_stream());
    }

    explicit PhantomSecretKey(const phantom::EncryptionParameters &parms) {
        poly_modulus_degree_ = parms.poly_modulus_degree();
        coeff_modulus_size_ = parms.coeff_modulus().size();

        data_rns_ = phantom::util::make_cuda_auto_ptr<uint64_t>(poly_modulus_degree_ * coeff_modulus_size_,
                                                                phantom::util::global_variables::default_stream->get_stream());

        gen_flag_ = false;
    }

    explicit PhantomSecretKey() = default;

    PhantomSecretKey(const PhantomSecretKey &) = delete;

    PhantomSecretKey &operator=(const PhantomSecretKey &) = delete;

    PhantomSecretKey(PhantomSecretKey &&) = default;

    PhantomSecretKey &operator=(PhantomSecretKey &&) = default;

    ~PhantomSecretKey() = default;

    // new add 设置密钥，临时函数
    uint64_t get_degree() { return poly_modulus_degree_; };
    uint64_t get_coeff_modulus_size() { return coeff_modulus_size_; };

    void set_secretkey(const uint64_t *secretkey_host, const cudaStream_t &stream = phantom::util::global_variables::default_stream->get_stream());

    [[nodiscard]] PhantomPublicKey gen_publickey(const PhantomContext &context) const;

    [[nodiscard]] PhantomRelinKey gen_relinkey(const PhantomContext &context);

    void gen_relinkey(const PhantomContext &context, PhantomRelinKey &relin_key, bool save_seed = false);

    [[nodiscard]] PhantomGaloisKey create_galois_keys(const PhantomContext &context) const;

    void create_galois_keys(const PhantomContext &context, PhantomGaloisKey &galois_key, bool save_seed = false) const;

    [[nodiscard]] PhantomGaloisKey create_galois_keys_from_elts(PhantomContext &context, const std::vector<uint32_t> &elts) const;

    [[nodiscard]] PhantomGaloisKey create_galois_keys_from_steps(PhantomContext &context, const std::vector<int> &steps) const;

    [[nodiscard]] inline auto get_secret_key_array() const {
        return secret_key_array_.get();
    }

    [[nodiscard]] inline auto get_secret_key_non_ntt_array() const {
        return secret_key_non_ntt_array_.get();
    }

    [[nodiscard]] inline auto coeff_count() const {
        return sk_max_power_ * poly_modulus_degree_ * coeff_modulus_size_;
    }

    void resize(uint64_t chain_index, uint64_t max_power, uint64_t poly_degree, uint64_t coeff_size, const cudaStream_t &stream = phantom::util::global_variables::default_stream->get_stream()) {
        if (chain_index_ == chain_index &&
            sk_max_power_ >= max_power &&
            poly_modulus_degree_ == poly_degree &&
            coeff_modulus_size_ == coeff_size) {
            return;
        }

        chain_index_ = chain_index;
        sk_max_power_ = max_power;
        poly_modulus_degree_ = poly_degree;
        coeff_modulus_size_ = coeff_size;

        const size_t total_elements = max_power * poly_degree * coeff_size;

        secret_key_array_.reset();
        secret_key_array_ = phantom::util::make_cuda_auto_ptr<uint64_t>(total_elements, stream);

        secret_key_non_ntt_array_.reset();
        secret_key_non_ntt_array_ = phantom::util::make_cuda_auto_ptr<uint64_t>(total_elements, stream);

        if (!secret_key_array_.get()) {
            throw std::bad_alloc();
        }
        if (!secret_key_non_ntt_array_.get()) {
            throw std::bad_alloc();
        }
        // phantom::util::set_zero_uint(secret_key_array_.get(), total_elements);
    }

    /** Encrypt zero using the secret key, the ciphertext is in NTT form
     * @param[in] context PhantomContext
     * @param[inout] cipher The generated ciphertext
     * @param[in] chain_index The index of the context data
     * @param[in] is_ntt_form Whether the ciphertext needs to be in NTT form
     */
    void encrypt_zero_symmetric(const PhantomContext &context, PhantomCiphertext &cipher, const uint8_t *prng_seed_a,
                                size_t chain_index, bool is_ntt_form, const cudaStream_t &stream) const;

    /** Symmetric encryption, the plaintext and ciphertext are in NTT form
     * @param[in] context PhantomContext
     * @param[in] plain The data to be encrypted
     * @param[out] cipher The generated ciphertext
     */
    void encrypt_symmetric(const PhantomContext &context, const PhantomPlaintext &plain, PhantomCiphertext &cipher,
                           const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream) const;

    void encrypt_symmetric(const PhantomContext &context, const PhantomPlaintext &plain, PhantomCiphertext &cipher, bool save_seed = false,
                           const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream) const {
        encrypt_symmetric(context, plain, cipher, stream_wrapper);
    }

    /** decryption
     * @param[in] context PhantomContext
     * @param[in] cipher The ciphertext to be decrypted
     * @param[out] plain The plaintext
     */
    void decrypt(const PhantomContext &context, const PhantomCiphertext &cipher, PhantomPlaintext &plain,
                 const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream);

    // for python wrapper

    [[nodiscard]] inline PhantomCiphertext
    encrypt_symmetric(const PhantomContext &context, const PhantomPlaintext &plain,
                      const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream) const {
        PhantomCiphertext cipher;
        encrypt_symmetric(context, plain, cipher, stream_wrapper);
        return cipher;
    }

    [[nodiscard]] inline PhantomPlaintext
    decrypt(const PhantomContext &context, const PhantomCiphertext &cipher,
            const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream) {
        PhantomPlaintext plain;
        decrypt(context, cipher, plain, stream_wrapper);
        return plain;
    }

    /**
    Computes the invariant noise budget (in bits) of a ciphertext. The
    invariant noise budget measures the amount of room there is for the noise
    to grow while ensuring correct decryptions. This function works only with
    the BFV scheme.
    * @param[in] context PhantomContext
    * @param[in] cipher The ciphertext to be decrypted
    */
    [[nodiscard]] int invariant_noise_budget(const PhantomContext &context, const PhantomCiphertext &cipher,
                                             const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream);

    // Newly added for debugging purposes
    inline void load_secret_key(const PhantomContext &context, std::ifstream &sk_in) {
        std::string line;
        size_t total_line_count = 0;

        auto new_sk_array_data = new uint64_t[context.coeff_mod_size_ * context.poly_degree_];

        while (std::getline(sk_in, line)) {
            uint64_t value = std::stoull(line);
            new_sk_array_data[total_line_count] = value;
            total_line_count++;
        }

        if (total_line_count != context.coeff_mod_size_ * context.poly_degree_) {
            throw std::invalid_argument("Invalid secret key input.");
        }

        cudaMemcpy(secret_key_array_.get(), new_sk_array_data, context.coeff_mod_size_ * context.poly_degree_ * sizeof(uint64_t), cudaMemcpyHostToDevice);
        delete[] new_sk_array_data;
    }

    void save(std::ostream &stream) const {
        if (!gen_flag_)
            throw std::invalid_argument("PhantomSecretKey has not been generated");

        stream.write(reinterpret_cast<const char *>(&sk_max_power_), sizeof(size_t));
        stream.write(reinterpret_cast<const char *>(&poly_modulus_degree_), sizeof(size_t));
        stream.write(reinterpret_cast<const char *>(&coeff_modulus_size_), sizeof(size_t));

        uint64_t *h_data;
        cudaMallocHost(&h_data, poly_modulus_degree_ * coeff_modulus_size_ * sizeof(uint64_t));
        cudaMemcpy(h_data, secret_key_array_.get(), poly_modulus_degree_ * coeff_modulus_size_ * sizeof(uint64_t),
                   cudaMemcpyDeviceToHost);
        stream.write(reinterpret_cast<char *>(h_data), poly_modulus_degree_ * coeff_modulus_size_ * sizeof(uint64_t));
        cudaFreeHost(h_data);
    }

    void load(std::istream &stream) {
        stream.read(reinterpret_cast<char *>(&sk_max_power_), sizeof(size_t));
        stream.read(reinterpret_cast<char *>(&poly_modulus_degree_), sizeof(size_t));
        stream.read(reinterpret_cast<char *>(&coeff_modulus_size_), sizeof(size_t));

        uint64_t *h_data;
        cudaMallocHost(&h_data, poly_modulus_degree_ * coeff_modulus_size_ * sizeof(uint64_t));
        stream.read(reinterpret_cast<char *>(h_data), poly_modulus_degree_ * coeff_modulus_size_ * sizeof(uint64_t));
        secret_key_array_ = phantom::util::make_cuda_auto_ptr<uint64_t>(poly_modulus_degree_ * coeff_modulus_size_,
                                                                        cudaStreamPerThread);
        cudaMemcpyAsync(secret_key_array_.get(), h_data, poly_modulus_degree_ * coeff_modulus_size_ * sizeof(uint64_t),
                        cudaMemcpyHostToDevice, cudaStreamPerThread);

        cudaStreamSynchronize(cudaStreamPerThread);

        // cleanup h_data
        cudaFreeHost(h_data);

        gen_flag_ = true;
    }
};
