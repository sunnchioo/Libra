#include "BackendDescriptor.h"

class FlyHEBackend : public BackendDescriptor {
public:
    std::string getSIMDMultFunc() const override { return "FlyHE_SimdMult"; }
    std::string getSIMDAddFunc() const override { return "FlyHE_SimdAdd"; }
    std::string getSISDMinFunc() const override { return "FlyHE_Min"; }

    std::string getSIMDEncrypt() const override { return "FlyHE.SIMDEncrypt"; }
    std::string getSIMDDecrypt() const override { return "FlyHE.SIMDDecrypt"; }
    std::string getSIMDMultFunc() const override { return "FlyHE.SIMDMult"; }
    std::string getSIMDAddFunc() const override { return "FlyHE.SIMDAdd"; }
    std::string getSIMDSubFunc() const override { return "FlyHE.SIMDSub"; }

    std::string getSISDEncrypt() const override { return "FlyHE.SISDEncrypt"; }
    std::string getSISDDecrypt() const override { return "FlyHE.SISDDecrypt"; }
    std::string getSISDAddFunc() const override { return "FlyHE.SISDAdd"; }
    std::string getSISDSubFunc() const override { return "FlyHE.SISDSub"; }
    std::string getSISDMinFunc() const override { return "FlyHE.SISDMin"; }
    std::string getSISDPBSFunc() const override { return "FlyHE.SISDPBS"; }
};
