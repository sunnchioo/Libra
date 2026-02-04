#pragma once // <--- 必须添加这行
#include "BackendDescriptor.h"

namespace mlir {
    namespace libra {
        namespace backend {

            class FlyHEBackend : public BackendDescriptor {
            public:
                std::string getSIMDMultFunc() const override { return "FlyHE.SimdMult"; }
                std::string getSIMDAddFunc() const override { return "FlyHE.SimdAdd"; }
                std::string getSISDMinFunc() const override { return "FlyHE.Min"; }

                std::string getSIMDEncrypt() const override { return "FlyHE.SIMDEncrypt"; }
                std::string getSIMDDecrypt() const override { return "FlyHE.SIMDDecrypt"; }
                std::string getSIMDSubFunc() const override { return "FlyHE.SIMDSub"; }

                std::string getSISDEncrypt() const override { return "FlyHE.SISDEncrypt"; }
                std::string getSISDDecrypt() const override { return "FlyHE.SISDDecrypt"; }
                std::string getSISDAddFunc() const override { return "FlyHE.SISDAdd"; }
                std::string getSISDSubFunc() const override { return "FlyHE.SISDSub"; }
                std::string getSISDPBSFunc() const override { return "FlyHE.SISDPBS"; }
            };

        } // namespace backend
    } // namespace libra
} // namespace mlir