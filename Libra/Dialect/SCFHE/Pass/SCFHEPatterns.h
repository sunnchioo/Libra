// SCFHEPatterns.h
#ifndef LIBRA_SCFHE_PATTERNS_H
#define LIBRA_SCFHE_PATTERNS_H

#include "mlir/Transforms/DialectConversion.h"
#include "SCFHEAnalysis.h"

namespace mlir::libra::scfhe {

    // 注册转换阶段 (Phase 3) 的 Patterns
    void populateSCFHEConversionPatterns(RewritePatternSet& patterns,
                                         TypeConverter& typeConverter,
                                         MLIRContext* ctx,
                                         const ArgAnalysis& analysis);

    // 注册清理阶段 (Phase 5) 的 Patterns
    void populateSCFHECleanupPatterns(RewritePatternSet& patterns, MLIRContext* ctx);

} // namespace mlir::libra::scfhe

#endif // LIBRA_SCFHE_PATTERNS_H