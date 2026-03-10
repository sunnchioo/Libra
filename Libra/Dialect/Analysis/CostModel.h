#ifndef LIBRA_MDSEL_COSTMODEL_H
#define LIBRA_MDSEL_COSTMODEL_H

#include "mlir/IR/Operation.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/JSON.h"

namespace mlir::libra::mdsel {

    enum class Mode { SIMD,
                      SISD };

    // 全局常量定义
    constexpr double INF_COST = 1e15;
    constexpr int MAX_SIMD_LEVEL = 31;
    constexpr int BOOT_LEVEL = 17;
    constexpr int SLIM_BOOT_TRIGGER = 3;

    class CostModel {
    public:
        explicit CostModel(llvm::StringRef jsonFilename);

        double getOpCost(Operation* op, Mode m, int64_t level, int64_t vecCnt) const;
        double getBootCost(int64_t vectorCount) const;
        double getCastCost(Mode from, Mode to, int64_t vectorCount) const;

    private:
        llvm::StringMap<llvm::DenseMap<int64_t, double>> costTable;

        int64_t getLogCount(int64_t count) const;
        double lookup(llvm::StringRef key, int64_t idx) const;
        std::string getOpKey(Operation* op, Mode m) const;
    };

} // namespace mlir::libra::mdsel

#endif // LIBRA_MDSEL_COSTMODEL_H