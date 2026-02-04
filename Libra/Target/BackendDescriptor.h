#pragma once // <--- 必须添加这行
#include <string>

namespace mlir {
    namespace libra {
        namespace backend {

            class BackendDescriptor {
            public:
                virtual ~BackendDescriptor() = default; // 建议加上虚析构函数
                virtual std::string getSIMDEncrypt() const = 0;
                virtual std::string getSIMDDecrypt() const = 0;
                virtual std::string getSIMDMultFunc() const = 0;
                virtual std::string getSIMDAddFunc() const = 0;
                virtual std::string getSIMDSubFunc() const = 0;

                virtual std::string getSISDEncrypt() const = 0;
                virtual std::string getSISDDecrypt() const = 0;
                virtual std::string getSISDAddFunc() const = 0;
                virtual std::string getSISDSubFunc() const = 0;
                virtual std::string getSISDMinFunc() const = 0;
                virtual std::string getSISDPBSFunc() const = 0;
            };

        } // namespace backend
    } // namespace libra
} // namespace mlir