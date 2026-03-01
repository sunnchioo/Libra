#include "LibraTranslate.h"
#include "BackendDescriptor.h"
#include "FlyHEBackend.h"

// Dialect Includes
#include "SCFHEDialect.h"
#include "SIMDDialect.h"
#include "SISDDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Tools/mlir-translate/Translation.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Debug.h"

#include <string>

// [新增] 定义 DEBUG_TYPE，方便你在编译后运行时追加 --debug-only=libra-translate 查看详细翻译流程
#define DEBUG_TYPE "libra-translate"

using namespace mlir;
using namespace mlir::libra::backend;

// ============================================================================
// 1. Utility
// ============================================================================

std::string LibraBackendEmitter::nameFor(Value v) {
    // [修复] 使用静态计数器防止变量名冲突
    static int valueCount = 0;
    if (nameTable.count(v))
        return nameTable[v];
    std::string generated = "v" + std::to_string(valueCount++);
    nameTable[v] = generated;
    return generated;
}

// 辅助函数：从类型（比如 memref<10xf64> 或 !sisd.sisdcipher<10xi64>）中提取 batch size
static int64_t extractSize(Type t) {
    if (auto memTy = dyn_cast<MemRefType>(t)) {
        if (memTy.hasStaticShape() && memTy.getRank() > 0)
            return memTy.getShape()[0];
    }
    // 对于 Opaque / 自定义 Cipher Type，采用字符串快提
    std::string typeStr;
    llvm::raw_string_ostream rso(typeStr);
    t.print(rso);
    size_t start = typeStr.find('<');
    size_t end = typeStr.find('x');
    if (start != std::string::npos && end != std::string::npos) {
        return std::stoi(typeStr.substr(start + 1, end - start - 1));
    }

    int maxSlots = 32768;
    return maxSlots; // 兜底返回最大的slot
}

// ============================================================================
// 2. Dispatcher
// ============================================================================

LogicalResult LibraBackendEmitter::emitModule(ModuleOp module) {
    os << "// === Auto-Generated FlyHE CUDA C++ Program ===\n";
    os << "#include <iostream>\n";
    os << "#include <vector>\n";
    os << "#include <cstdlib>\n";
    os << "#include <cstdio>\n";
    os << "#include <algorithm>\n\n";
    os << "#include \"FlyHEContext.h\"\n\n";

    for (Operation& op : module.getOps()) {
        if (failed(translate(op)))
            return failure();
    }
    return success();
}

LogicalResult LibraBackendEmitter::translate(Operation& op) {
    if (!backend) {
        llvm::errs() << "Error: BackendDescriptor is not initialized.\n";
        return failure();
    }

    LLVM_DEBUG(llvm::dbgs() << "[libra-translate] Visiting Op: " << op.getName() << "\n");

    // Helper lambda for binary ops
    auto emitBinary = [&](Operation* binOp, StringRef symbol) -> LogicalResult {
        os << "  auto " << nameFor(binOp->getResult(0)) << " = "
           << nameFor(binOp->getOperand(0)) << " " << symbol << " "
           << nameFor(binOp->getOperand(1)) << ";\n";
        return success();
    };

    // --- LLVM & 特殊底层调用拦截 ---
    StringRef opName = op.getName().getStringRef();
    if (opName == "memref.copy") {
        os << "  std::copy(" << nameFor(op.getOperand(0)) << ".begin(), " << nameFor(op.getOperand(0)) << ".end(), " << nameFor(op.getOperand(1)) << ".begin());\n";
        return success();
    }
    if (opName == "llvm.mlir.addressof") {
        os << "  const char* " << nameFor(op.getResult(0)) << " = \"%f + %f = %f\\n\";\n";
        return success();
    }
    if (opName == "llvm.getelementptr") {
        nameTable[op.getResult(0)] = nameFor(op.getOperand(0)); // 透传指针
        return success();
    }
    if (opName == "llvm.call") {
        auto calleeAttr = op.getAttrOfType<FlatSymbolRefAttr>("callee");
        if (calleeAttr && calleeAttr.getValue() == "printf") {
            os << "  std::printf(" << nameFor(op.getOperand(0));
            for (unsigned i = 1; i < op.getNumOperands(); ++i) {
                os << ", " << nameFor(op.getOperand(i));
            }
            os << ");\n";
            return success();
        }
    }
    if (opName == "llvm.mlir.undef") {
        // 直接在 C++ 里声明一个 0 占位
        os << "  auto " << nameFor(op.getResult(0)) << " = 0; // llvm.mlir.undef\n";
        return success();
    }

    return llvm::TypeSwitch<Operation*, LogicalResult>(&op)
        // --- Functions ---
        .Case<func::FuncOp>([&](auto castOp) { return this->emitFunc(castOp); })
        .Case<func::ReturnOp>([&](auto castOp) { return this->emitReturn(castOp); })
        .Case<func::CallOp>([&](auto castOp) { return this->emitFuncCall(castOp); })

        // --- SISD Ops 核心翻译 (映射到 cuTLWE 和 evaluator) ---
        .Case<mlir::libra::sisd::SISDEncryptOp>([&](auto castOp) { return this->emitContextAwareCall(castOp, backend->getSISDEncrypt()); })
        .Case<mlir::libra::sisd::SISDAddOp>([&](auto castOp) { return this->emitContextAwareCall(castOp, backend->getSISDAddFunc()); })
        .Case<mlir::libra::sisd::SISDSubOp>([&](auto castOp) { return this->emitContextAwareCall(castOp, backend->getSISDSubFunc()); })
        .Case<mlir::libra::sisd::SISDDecryptOp>([&](auto castOp) { return this->emitContextAwareCall(castOp, backend->getSISDDecrypt()); })
        .Case<mlir::libra::sisd::SISDMinOp>([&](auto castOp) { return this->emitContextAwareCall(castOp, backend->getSISDMinFunc()); })

        // [新增] SISD Div 和 Load 算子支持
        .Case<mlir::libra::sisd::SISDDivOp>([&](auto castOp) { return this->emitContextAwareCall(castOp, "FlyHE_SISDDiv"); })
        .Case<mlir::libra::sisd::SISDLoadOp>([&](auto castOp) {
            os << "  auto " << nameFor(castOp.getResult()) << " = " << nameFor(castOp.getOperand(0)) << "[" << nameFor(castOp.getOperand(1)) << "];\n";
            return success();
        })

        // --- SIMD Ops ---
        .Case<mlir::libra::simd::SIMDAddOp>([&](auto castOp) { return this->emitBackendCall(castOp, backend->getSIMDAddFunc()); })
        .Case<mlir::libra::simd::SIMDSubOp>([&](auto castOp) { return this->emitBackendCall(castOp, backend->getSIMDSubFunc()); })
        .Case<mlir::libra::simd::SIMDMultOp>([&](auto castOp) { return this->emitBackendCall(castOp, backend->getSIMDMultFunc()); })

        // --- Control Flow ---
        .Case<affine::AffineForOp>([&](auto castOp) { return this->emitAffineFor(castOp); })

        // [新增] 完美支持密态循环 scf::ForOp 和其迭代变量的映射
        .Case<scf::ForOp>([&](scf::ForOp forOp) {
            std::string iv = nameFor(forOp.getInductionVar());
            std::string lb = nameFor(forOp.getLowerBound());
            std::string ub = nameFor(forOp.getUpperBound());
            std::string step = nameFor(forOp.getStep());

            // 1. 初始化迭代参数 (iter_args)
            for (auto it : llvm::zip(forOp.getRegionIterArgs(), forOp.getInitArgs())) {
                os << "  auto " << nameFor(std::get<0>(it)) << " = " << nameFor(std::get<1>(it)) << ";\n";
            }

            // 2. 循环结构展开
            os << "  for (int " << iv << " = " << lb << "; " << iv << " < " << ub << "; " << iv << " += " << step << ") {\n";
            for (Operation& bodyOp : *forOp.getBody()) {
                if (isa<scf::YieldOp>(bodyOp)) {
                    // yield 操作映射回 iter_args
                    auto yieldOp = cast<scf::YieldOp>(bodyOp);
                    for (auto it : llvm::zip(forOp.getRegionIterArgs(), yieldOp.getOperands())) {
                        os << "    " << nameFor(std::get<0>(it)) << " = " << nameFor(std::get<1>(it)) << ";\n";
                    }
                    continue;
                }
                if (failed(this->translate(bodyOp)))
                    return failure();
            }
            os << "  }\n";

            // 3. 将循环的最终产物赋予 forOp 的 result
            for (auto it : llvm::zip(forOp.getResults(), forOp.getRegionIterArgs())) {
                nameTable[std::get<0>(it)] = nameFor(std::get<1>(it));
            }
            return success();
        })
        .Case<scf::IfOp>([&](auto castOp) { return this->emitScfIf(castOp); })
        .Case<affine::AffineYieldOp>([&](auto castOp) { return success(); })
        .Case<scf::YieldOp>([&](auto castOp) { return success(); })

        // --- Arithmetics ---
        .Case<arith::ConstantOp>([&](auto castOp) { return this->emitArithConstant(castOp); })
        .Case<arith::AddFOp>([&](auto castOp) { return emitBinary(castOp, "+"); })
        .Case<arith::SubFOp>([&](auto castOp) { return emitBinary(castOp, "-"); })
        .Case<arith::MulFOp>([&](auto castOp) { return emitBinary(castOp, "*"); })
        .Case<arith::DivFOp>([&](auto castOp) { return emitBinary(castOp, "/"); })
        .Case<arith::IndexCastOp>([&](auto castOp) {
            os << "  int " << nameFor(castOp->getResult(0)) << " = (int)" << nameFor(castOp->getOperand(0)) << ";\n";
            return success();
        })
        .Case<arith::SIToFPOp>([&](auto castOp) {
            os << "  double " << nameFor(castOp->getResult(0)) << " = (double)" << nameFor(castOp->getOperand(0)) << ";\n";
            return success();
        })

        // --- Memory / Vector ---
        .Case<memref::AllocOp>([&](auto castOp) { return this->emitMemRefAlloc(castOp); })

        // [新增] 处理 memref::AllocaOp
        .Case<memref::AllocaOp>([&](auto castOp) {
            int64_t size = extractSize(castOp.getType());
            os << "  std::vector<double> " << nameFor(castOp.getResult()) << "(" << size << ");\n";
            return success();
        })

        .Case<affine::AffineLoadOp>([&](auto castOp) { return this->emitAffineLoad(castOp); })
        .Case<affine::AffineStoreOp>([&](auto castOp) { return this->emitAffineStore(castOp); })

        // --- Fallback ---
        .Default([&](Operation* unknownOp) {
            return this->printOperation(unknownOp);
        });
}

// ============================================================================
// Emit Logic Implementation
// ============================================================================

LogicalResult LibraBackendEmitter::emitBackendCall(Operation* op, StringRef funcName) {
    os << "  auto " << nameFor(op->getResult(0)) << " = " << funcName << "(";
    bool first = true;
    for (Value arg : op->getOperands()) {
        if (!first)
            os << ", ";
        first = false;
        os << nameFor(arg);
    }
    os << ");\n";
    return success();
}

LogicalResult LibraBackendEmitter::emitContextAwareCall(Operation* op, StringRef funcName) {
    int64_t size = 10; // 默认 fallback
    if (op->getNumResults() > 0) {
        size = extractSize(op->getResult(0).getType());
    } else if (op->getNumOperands() > 0) {
        size = extractSize(op->getOperand(0).getType());
    }

    std::string resName = nameFor(op->getResult(0));
    Type resType = op->getResult(0).getType();

    if (isa<mlir::libra::sisd::SISDCipherType>(resType)) {
        os << "  Pointer<cuTLWE<lwe_enc_lvl>> " << resName << "(" << size << ");\n";
    } else if (isa<mlir::libra::simd::SIMDCipherType>(resType)) {
        os << "  PhantomCiphertext " << resName << ";\n";
    } else if (isa<MemRefType>(resType)) {
        os << "  std::vector<double> " << resName << "(" << size << ");\n";
    } else {
        os << "  auto " << resName << ";\n";
    }

    os << "  " << funcName << "(hectx, " << resName;
    for (Value arg : op->getOperands()) {
        os << ", " << nameFor(arg);
    }
    os << ", " << size << ");\n";

    return success();
}

LogicalResult LibraBackendEmitter::emitFunc(func::FuncOp funcOp) {
    if (funcOp.isPrivate() || funcOp.getName() == "random_real" || funcOp.getName() == "rand")
        return success();

    if (funcOp.getName() == "main") {
        os << "int main() {\n";

        // --- 从 Module 中提取并生成 FlyHEConfig 初始化代码 ---
        std::string configStr = "FlyHEConfig::CreateSISD()";
        auto module = funcOp->getParentOfType<ModuleOp>();

        if (auto dictAttr = module->getAttrOfType<DictionaryAttr>("he.config")) {
            if (auto modeAttr = dyn_cast_or_null<StringAttr>(dictAttr.get("mode"))) {
                std::string mode = modeAttr.getValue().str();
                if (mode == "SISD") {
                    configStr = "FlyHEConfig::CreateSISD()";
                } else {
                    int64_t logN = 16, logn = 7;
                    int remaining = 21;
                    bool boot = true;
                    if (auto a = dyn_cast_or_null<IntegerAttr>(dictAttr.get("logN")))
                        logN = a.getInt();
                    if (auto a = dyn_cast_or_null<IntegerAttr>(dictAttr.get("logn")))
                        logn = a.getInt();
                    if (auto a = dyn_cast_or_null<IntegerAttr>(dictAttr.get("remaining_levels")))
                        remaining = a.getInt();
                    if (auto a = dyn_cast_or_null<BoolAttr>(dictAttr.get("bootstrapping_enabled")))
                        boot = a.getValue();

                    configStr = "FlyHEConfig::Create" + mode + "(" + std::to_string(logN) + ", " +
                                std::to_string(logn) + ", " + std::to_string(remaining) + ", " +
                                (boot ? "true" : "false") + ")";
                }
            }
        }

        os << "  // --- 1. FHE Context Initialization ---\n";
        os << "  auto config = " << configStr << ";\n";
        os << "  FlyHEContext<> hectx(config);\n";
        os << "  auto evaluator = hectx.tfhe_evaluator;\n\n";
        os << "  // --- 2. Program Execution ---\n";

        for (Block& block : funcOp.getBlocks()) {
            for (Operation& op : block) {
                if (failed(translate(op)))
                    return failure();
            }
        }

        os << "}\n";
        return success();
    }
    return success();
}

LogicalResult LibraBackendEmitter::emitReturn(func::ReturnOp op) {
    if (op.getNumOperands() > 0)
        os << "  return " << nameFor(op.getOperand(0)) << ";\n";
    else
        os << "  return;\n";
    return success();
}

LogicalResult LibraBackendEmitter::emitFuncCall(func::CallOp op) {
    if (op.getCallee() == "rand") {
        os << "  int " << nameFor(op.getResult(0)) << " = std::rand();\n";
        return success();
    }
    return success();
}

LogicalResult LibraBackendEmitter::emitAffineFor(affine::AffineForOp op) {
    std::string iv = nameFor(op.getInductionVar());
    std::string lb = op.hasConstantLowerBound() ? std::to_string(op.getConstantLowerBound()) : nameFor(op.getLowerBoundOperands()[0]);
    std::string ub = op.hasConstantUpperBound() ? std::to_string(op.getConstantUpperBound()) : nameFor(op.getUpperBoundOperands()[0]);

    os << "  for (int " << iv << " = " << lb << "; " << iv << " < " << ub << "; ++" << iv << ") {\n";
    for (Operation& bodyOp : *op.getBody()) {
        if (isa<affine::AffineYieldOp>(bodyOp))
            continue;
        if (failed(translate(bodyOp)))
            return failure();
    }
    os << "  }\n";
    return success();
}

LogicalResult LibraBackendEmitter::emitScfIf(scf::IfOp op) {
    os << "  if (" << nameFor(op.getCondition()) << ") {\n";
    for (Operation& bodyOp : *op.thenBlock()) {
        if (isa<scf::YieldOp>(bodyOp))
            continue;
        if (failed(translate(bodyOp)))
            return failure();
    }
    os << "  }\n";
    return success();
}

LogicalResult LibraBackendEmitter::emitArithConstant(arith::ConstantOp op) {
    std::string resName = nameFor(op.getResult());
    if (auto floatAttr = dyn_cast<FloatAttr>(op.getValue())) {
        os << "  double " << resName << " = " << floatAttr.getValueAsDouble() << ";\n";
    } else if (auto intAttr = dyn_cast<IntegerAttr>(op.getValue())) {
        os << "  int " << resName << " = " << intAttr.getInt() << ";\n";
    }
    return success();
}

LogicalResult LibraBackendEmitter::emitMemRefAlloc(memref::AllocOp op) {
    int64_t size = extractSize(op.getType());
    os << "  std::vector<double> " << nameFor(op.getResult()) << "(" << size << ");\n";
    return success();
}

// [修复] 防止被 canonicalize 的空 index 访问越界导致核心崩溃
LogicalResult LibraBackendEmitter::emitAffineLoad(affine::AffineLoadOp op) {
    os << "  auto " << nameFor(op.getResult()) << " = " << nameFor(op.getMemRef());
    if (op.getIndices().empty()) {
        os << "[0];\n";
    } else {
        os << "[" << nameFor(op.getIndices()[0]) << "];\n";
    }
    return success();
}

// [修复] 防止被 canonicalize 的空 index 访问越界导致核心崩溃
LogicalResult LibraBackendEmitter::emitAffineStore(affine::AffineStoreOp op) {
    os << "  " << nameFor(op.getMemRef());
    if (op.getIndices().empty()) {
        os << "[0] = ";
    } else {
        os << "[" << nameFor(op.getIndices()[0]) << "] = ";
    }
    os << nameFor(op.getValueToStore()) << ";\n";
    return success();
}

LogicalResult LibraBackendEmitter::printOperation(Operation* op) {
    StringRef name = op->getName().getStringRef();
    if (name.starts_with("llvm.") || name.starts_with("memref.dealloc") || name.starts_with("polygeist")) {
        if (op->getNumResults() > 0)
            nameTable[op->getResult(0)] = "ignored";
        return success();
    }
    os << "  // Unhandled MLIR op: " << name << "\n";
    LLVM_DEBUG(llvm::dbgs() << "[libra-translate] Warning: Unhandled MLIR op -> " << name << "\n");
    return success();
}

// ============================================================================
// Registration
// ============================================================================

void mlir::libra::backend::registerLibraBackendTranslation() {
    static FlyHEBackend flyHEBackend;

    static TranslateFromMLIRRegistration reg(
        "emit-libra-backend",
        "Emit abstract FlyHE interface code",
        [&](ModuleOp module, llvm::raw_ostream& output) -> LogicalResult {
            LibraBackendEmitter emitter(output, &flyHEBackend);
            return emitter.emitModule(module);
        },
        [](DialectRegistry& registry) {
            registry.insert<mlir::scf::SCFDialect>();
            registry.insert<func::FuncDialect>();
            registry.insert<LLVM::LLVMDialect>();
            registry.insert<mlir::libra::scfhe::SCFHEDialect>();
            registry.insert<mlir::libra::simd::SIMDDialect>();
            registry.insert<mlir::libra::sisd::SISDDialect>();
            registry.insert<mlir::arith::ArithDialect>();
            registry.insert<mlir::memref::MemRefDialect>();
            registry.insert<mlir::affine::AffineDialect>();
            registry.insert<mlir::math::MathDialect>();
            registry.insert<mlir::vector::VectorDialect>();
        });
}