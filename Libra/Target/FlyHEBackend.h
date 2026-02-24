#pragma once
#include "BackendDescriptor.h"

namespace mlir {
    namespace libra {
        namespace backend {

            class FlyHEBackend : public BackendDescriptor {
            public:
                // SIMD (CKKS) Wrappers
                std::string getSIMDEncrypt() const override { return "FlyHE_SIMDEncrypt"; }
                std::string getSIMDDecrypt() const override { return "FlyHE_SIMDDecrypt"; }
                std::string getSIMDMultFunc() const override { return "FlyHE_SIMDMult"; }
                std::string getSIMDAddFunc() const override { return "FlyHE_SIMDAdd"; }
                std::string getSIMDSubFunc() const override { return "FlyHE_SIMDSub"; }

                // SISD (TFHE) Wrappers
                std::string getSISDEncrypt() const override { return "FlyHE_SISDEncrypt"; }
                std::string getSISDDecrypt() const override { return "FlyHE_SISDDecrypt"; }
                std::string getSISDAddFunc() const override { return "FlyHE_SISDAdd"; }
                std::string getSISDSubFunc() const override { return "FlyHE_SISDSub"; }
                std::string getSISDMinFunc() const override { return "FlyHE_SISDMin"; }
                std::string getSISDPBSFunc() const override { return "FlyHE_SISDPBS"; }
            };

        }
    }
}