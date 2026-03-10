// SCFHETypeConverter.h
#ifndef LIBRA_SCFHE_TYPECONVERTER_H
#define LIBRA_SCFHE_TYPECONVERTER_H

#include "mlir/Transforms/DialectConversion.h"

namespace mlir::libra::scfhe {

    class SCFHETypeConverter : public TypeConverter {
    public:
        explicit SCFHETypeConverter(MLIRContext* ctx);
    };

} // namespace mlir::libra::scfhe

#endif // LIBRA_SCFHE_TYPECONVERTER_H