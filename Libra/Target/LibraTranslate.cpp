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

#include <string>

using namespace mlir;
using namespace mlir::libra::backend;

// ============================================================================
// 1. Utility
// ============================================================================

std::string LibraBackendEmitter::nameFor(Value v) {
    if (nameTable.count(v))
        return nameTable[v];
    std::string generated = "v" + std::to_string(nameTable.size());
    nameTable[v] = generated;
    return generated;
}

// 辅助函数：从类型（比如 memref<10xf64> 或 !sisd.sisdcipher<10xi64>）中提取 batch size
static int64_t extractSize(Type t) {
    if (auto memTy = dyn_cast<MemRefType>(t)) {
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

    // Helper lambda for binary ops
    auto emitBinary = [&](Operation* op, StringRef symbol) -> LogicalResult {
        os << "  auto " << nameFor(op->getResult(0)) << " = "
           << nameFor(op->getOperand(0)) << " " << symbol << " "
           << nameFor(op->getOperand(1)) << ";\n";
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

    return llvm::TypeSwitch<Operation*, LogicalResult>(&op)
        // --- Functions ---
        .Case<func::FuncOp>([&](auto op) { return this->emitFunc(op); })
        .Case<func::ReturnOp>([&](auto op) { return this->emitReturn(op); })
        .Case<func::CallOp>([&](auto op) { return this->emitFuncCall(op); })

        // --- SISD Ops 核心翻译 (映射到 cuTLWE 和 evaluator) ---
        .Case<mlir::libra::sisd::SISDEncryptOp>([&](auto op) {
            return this->emitContextAwareCall(op, backend->getSISDEncrypt());
        })
        .Case<mlir::libra::sisd::SISDAddOp>([&](auto op) {
            return this->emitContextAwareCall(op, backend->getSISDAddFunc());
        })
        .Case<mlir::libra::sisd::SISDSubOp>([&](auto op) {
            return this->emitContextAwareCall(op, backend->getSISDSubFunc());
        })
        .Case<mlir::libra::sisd::SISDDecryptOp>([&](auto op) {
            return this->emitContextAwareCall(op, backend->getSISDDecrypt());
        })
        .Case<mlir::libra::sisd::SISDMinOp>([&](auto op) {
            return this->emitContextAwareCall(op, backend->getSISDMinFunc());
        })

        // --- SIMD Ops (保留) ---
        .Case<mlir::libra::simd::SIMDAddOp>([&](auto op) { return this->emitBackendCall(op, backend->getSIMDAddFunc()); })
        .Case<mlir::libra::simd::SIMDSubOp>([&](auto op) { return this->emitBackendCall(op, backend->getSIMDSubFunc()); })
        .Case<mlir::libra::simd::SIMDMultOp>([&](auto op) { return this->emitBackendCall(op, backend->getSIMDMultFunc()); })

        // --- Control Flow ---
        .Case<affine::AffineForOp>([&](auto op) { return this->emitAffineFor(op); })
        .Case<scf::IfOp>([&](auto op) { return this->emitScfIf(op); })
        .Case<affine::AffineYieldOp>([&](auto op) { return success(); })
        .Case<scf::YieldOp>([&](auto op) { return success(); })

        // --- Arithmetics ---
        .Case<arith::ConstantOp>([&](auto op) { return this->emitArithConstant(op); })
        .Case<arith::AddFOp>([&](auto op) { return emitBinary(op, "+"); })
        .Case<arith::SubFOp>([&](auto op) { return emitBinary(op, "-"); })
        .Case<arith::MulFOp>([&](auto op) { return emitBinary(op, "*"); })
        .Case<arith::DivFOp>([&](auto op) { return emitBinary(op, "/"); })
        .Case<arith::IndexCastOp>([&](auto op) {
            os << "  int " << nameFor(op->getResult(0)) << " = (int)" << nameFor(op->getOperand(0)) << ";\n";
            return success();
        })
        .Case<arith::SIToFPOp>([&](auto op) {
            os << "  double " << nameFor(op->getResult(0)) << " = (double)" << nameFor(op->getOperand(0)) << ";\n";
            return success();
        })

        // --- Memory / Vector ---
        .Case<memref::AllocOp>([&](auto op) { return this->emitMemRefAlloc(op); })
        .Case<affine::AffineLoadOp>([&](auto op) { return this->emitAffineLoad(op); })
        .Case<affine::AffineStoreOp>([&](auto op) { return this->emitAffineStore(op); })

        // --- Fallback ---
        .Default([&](Operation* op) {
            return this->printOperation(op);
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

    // 拿到结果变量的名称
    std::string resName = nameFor(op->getResult(0));
    Type resType = op->getResult(0).getType();

    // 1. 根据 MLIR 的类型系统，自动在 C++ 中生成预分配语句
    if (isa<mlir::libra::sisd::SISDCipherType>(resType)) {
        os << "  Pointer<cuTLWE<lwe_enc_lvl>> " << resName << "(" << size << ");\n";
    } else if (isa<mlir::libra::simd::SIMDCipherType>(resType)) {
        os << "  PhantomCiphertext " << resName << ";\n";
    } else if (isa<MemRefType>(resType)) {
        os << "  std::vector<double> " << resName << "(" << size << ");\n";
    } else {
        // Fallback，如果是标量或其他未知类型
        os << "  auto " << resName << ";\n";
    }

    // 2. 调用修改后的 void 返回值的 Wrapper 函数
    // 格式：FlyHE_SISDAdd(hectx, v3, v1, v2, 10);
    os << "  " << funcName << "(hectx, " << resName;
    for (Value arg : op->getOperands()) {
        os << ", " << nameFor(arg);
    }
    os << ", " << size << ");\n";

    return success();
}

LogicalResult LibraBackendEmitter::emitFunc(func::FuncOp funcOp) {
    // 忽略辅助测试函数
    if (funcOp.isPrivate() || funcOp.getName() == "random_real" || funcOp.getName() == "rand")
        return success();

    if (funcOp.getName() == "main") {
        os << "int main() {\n";

        // --- 从 Module 中提取并生成 FlyHEConfig 初始化代码 ---
        std::string configStr = "FlyHEConfig::CreateSISD()";
        auto module = funcOp->getParentOfType<ModuleOp>();

        if (auto dictAttr = module->getAttrOfType<DictionaryAttr>("flyhe.config")) {
            // 注意这里：改成了全局的 dyn_cast_or_null<T>(...)
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
                // if (isa<func::ReturnOp>(op))
                //     continue;
                if (failed(translate(op)))
                    return failure();
            }
        }

        // os << "  return 0;\n";
        os << "}\n";
        return success();
    }

    // 翻译其他函数...
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
    // 处理其他调用...
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

LogicalResult LibraBackendEmitter::emitAffineLoad(affine::AffineLoadOp op) {
    os << "  auto " << nameFor(op.getResult()) << " = " << nameFor(op.getMemRef()) << "[" << nameFor(op.getIndices()[0]) << "];\n";
    return success();
}

LogicalResult LibraBackendEmitter::emitAffineStore(affine::AffineStoreOp op) {
    os << "  " << nameFor(op.getMemRef()) << "[" << nameFor(op.getIndices()[0]) << "] = " << nameFor(op.getValueToStore()) << ";\n";
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