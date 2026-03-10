#include "CostModel.h"
#include "SIMDOps.h"
#include "SISDOps.h"

#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <cmath>

#define DEBUG_TYPE "mode-select"

using namespace mlir;
using namespace mlir::libra::mdsel;

CostModel::CostModel(StringRef jsonFilename) {
    auto fileOrErr = llvm::MemoryBuffer::getFile(jsonFilename);
    if (auto ec = fileOrErr.getError()) {
        LLVM_DEBUG(llvm::dbgs() << "[ModeSel] Error opening cost file: " << ec.message() << "\n");
        return;
    }
    llvm::Expected<llvm::json::Value> rootOrErr = llvm::json::parse(fileOrErr.get()->getBuffer());
    if (!rootOrErr)
        return;
    auto* obj = rootOrErr->getAsObject();
    if (!obj)
        return;
    auto* costs = obj->getObject("costs");
    if (!costs)
        return;

    for (auto& modePair : *costs) {
        StringRef mode = modePair.first;
        auto* modeObj = modePair.second.getAsObject();
        if (!modeObj)
            continue;
        for (auto& opPair : *modeObj) {
            StringRef opname = opPair.first;
            auto* opObj = opPair.second.getAsObject();
            if (!opObj)
                continue;

            if (auto* lat = opObj->getObject("latency")) {
                std::string baseKey = (mode + "." + opname).str();

                for (auto& kv : *lat) {
                    long long idx = 0;
                    if (!llvm::StringRef(kv.first).getAsInteger(10, idx)) {
                        // Case 1: 值是数字
                        if (auto val = kv.second.getAsNumber()) {
                            costTable[baseKey][idx] = *val;
                        }
                        // Case 2: 值是对象 (reduce_add)
                        else if (auto* nestedObj = kv.second.getAsObject()) {
                            std::string compoundKey = baseKey + "." + std::to_string(idx);
                            for (auto& innerKv : *nestedObj) {
                                long long innerIdx = 0;
                                if (!llvm::StringRef(innerKv.first).getAsInteger(10, innerIdx)) {
                                    if (auto innerVal = innerKv.second.getAsNumber()) {
                                        costTable[compoundKey][innerIdx] = *innerVal;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

int64_t CostModel::getLogCount(int64_t count) const {
    if (count <= 1)
        return 0;
    return static_cast<int64_t>(std::ceil(std::log2(static_cast<double>(count))));
}

double CostModel::lookup(StringRef key, int64_t idx) const {
    auto it = costTable.find(key);

    // 【添加的功能】：如果 cost 表中没有找到相应的键，返回明确的错误并终止程序
    if (it == costTable.end()) {
        llvm::errs() << "\n=======================================================\n"
                     << "[ModeSel] FATAL ERROR: Cost lookup failed!\n"
                     << " -> Cannot find key '" << key << "' in cost_table.json.\n"
                     << " -> Please add this key to your JSON file.\n"
                     << "=======================================================\n";
        exit(1);
    }

    auto jt = it->second.find(idx);
    if (jt == it->second.end()) {
        if (!it->second.empty()) {
            // 可选：找不到具体 index 时给个警告，并使用默认值（防止层数/大小稍微越界导致崩溃）
            // llvm::errs() << "[ModeSel] Warning: Index '" << idx << "' not found for key '" << key << "'. Using fallback.\n";
            return it->second.begin()->second;
        }
        llvm::errs() << "\n[ModeSel] FATAL ERROR: Index '" << idx << "' not found for key '" << key << "'!\n";
        exit(1);
    }
    return jt->second;
}

std::string CostModel::getOpKey(Operation* op, Mode m) const {
    std::string prefix = (m == Mode::SIMD) ? "simd." : "sisd.";
    if (isa<simd::SIMDAddOp, sisd::SISDAddOp>(op))
        return prefix + "add";
    if (isa<simd::SIMDSubOp, sisd::SISDSubOp>(op))
        return prefix + "sub";
    if (isa<simd::SIMDMultOp>(op))
        return prefix + "mult";
    if (isa<simd::SIMDMinOp, sisd::SISDMinOp>(op))
        return prefix + "min";
    if (isa<simd::SIMDEncryptOp, sisd::SISDEncryptOp>(op))
        return prefix + "encrypt";
    if (isa<simd::SIMDDecryptOp, sisd::SISDDecryptOp>(op))
        return prefix + "decrypt";
    if (isa<simd::SIMDDivOp, sisd::SISDDivOp>(op))
        return prefix + "div";
    if (isa<simd::SIMDLoadOp, sisd::SISDLoadOp>(op))
        return prefix + "load";
    if (isa<simd::SIMDStoreOp, sisd::SISDStoreOp>(op))
        return prefix + "store";
    if (isa<simd::SIMDReduceAddOp, sisd::SISDReduceAddOp>(op))
        return prefix + "reduce_add";
    return "";
}

double CostModel::getOpCost(Operation* op, Mode m, int64_t level, int64_t vecCnt) const {
    if (isa<simd::SIMDMultOp>(op) && m == Mode::SISD)
        return INF_COST;

    std::string baseKey = getOpKey(op, m);
    if (baseKey.empty()) {
        llvm::errs() << "[ModeSel] Unknown Op: " << op->getName() << "\n";
        exit(1);
    }

    // === 【核心修复】：对于 reduce_add，成本取决于输入向量的大小，而不是输出(1)的大小 ===
    int64_t effectiveVecCnt = vecCnt;
    if (isa<simd::SIMDReduceAddOp, sisd::SISDReduceAddOp>(op)) {
        if (auto t = dyn_cast<simd::SIMDCipherType>(op->getOperand(0).getType()))
            effectiveVecCnt = t.getPlaintextCount();
        else if (auto t = dyn_cast<sisd::SISDCipherType>(op->getOperand(0).getType()))
            effectiveVecCnt = t.getPlaintextCount();
    }

    // 处理 SIMD reduce_add (二维查表：Level -> LogSize)
    if (baseKey == "simd.reduce_add") {
        std::string dynamicKey = baseKey + "." + std::to_string(level);
        return lookup(dynamicKey, getLogCount(effectiveVecCnt));
    }

    // 处理其他所有算子，包括 sisd.reduce_add (一维查表)
    int64_t lookupIndex = 0;
    if (m == Mode::SIMD) {
        if (isa<simd::SIMDMinOp, sisd::SISDMinOp>(op))
            lookupIndex = getLogCount(effectiveVecCnt);
        else
            lookupIndex = level;
    } else {
        lookupIndex = getLogCount(effectiveVecCnt);
    }

    return lookup(baseKey, lookupIndex);
}

double CostModel::getBootCost(int64_t vectorCount) const {
    return lookup("simd.boot", getLogCount(vectorCount));
}

double CostModel::getCastCost(Mode from, Mode to, int64_t vectorCount) const {
    if (from == to)
        return 0.0;
    std::string key = (from == Mode::SIMD) ? "simd.cast_to_sisd" : "sisd.cast_to_simd";
    return lookup(key, getLogCount(vectorCount));
}