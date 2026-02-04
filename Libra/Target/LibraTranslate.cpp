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
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

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

// ============================================================================
// 2. Dispatcher
// ============================================================================

LogicalResult LibraBackendEmitter::emitModule(ModuleOp module) {
    os << "// === Libra FlyHE Backend Interface Output ===\n\n";
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

    return llvm::TypeSwitch<Operation*, LogicalResult>(&op)
        // --- Functions ---
        .Case<func::FuncOp>([&](auto op) { return this->emitFunc(op); })
        .Case<func::ReturnOp>([&](auto op) { return this->emitReturn(op); })
        .Case<func::CallOp>([&](auto op) { return this->emitFuncCall(op); })

        // --- SIMD Ops ---
        .Case<mlir::libra::simd::SIMDAddOp>([&](auto op) {
            return this->emitBackendCall(op, backend->getSIMDAddFunc());
        })
        .Case<mlir::libra::simd::SIMDSubOp>([&](auto op) {
            return this->emitBackendCall(op, backend->getSIMDSubFunc());
        })
        .Case<mlir::libra::simd::SIMDMultOp>([&](auto op) {
            return this->emitBackendCall(op, backend->getSIMDMultFunc());
        })
        .Case<mlir::libra::simd::SIMDEncryptOp>([&](auto op) {
            return this->emitBackendCall(op, backend->getSIMDEncrypt());
        })
        .Case<mlir::libra::simd::SIMDDecryptOp>([&](auto op) {
            return this->emitBackendCall(op, backend->getSIMDDecrypt());
        })
        .Case<mlir::libra::simd::SIMDDivOp>([&](auto op) {
            return this->emitBackendCall(op, "FlyHE.SIMDDiv");
        })
        .Case<mlir::libra::simd::SIMDExpOp>([&](auto op) {
            return this->emitBackendCall(op, "FlyHE.SIMDExp");
        })
        .Case<mlir::libra::simd::SIMDStoreOp>([&](auto op) {
            // 修复: 使用通用接口获取操作数 (0: value, 1: memref)
            os << "  FlyHE.Store(" << nameFor(op->getOperand(0)) << ", " << nameFor(op->getOperand(1)) << ");\n";
            return success();
        })
        .Case<mlir::libra::simd::SIMDReduceAddOp>([&](auto op) {
            return this->emitBackendCall(op, "FlyHE.ReduceAdd");
        })

        // --- SISD Ops ---
        .Case<mlir::libra::sisd::SISDAddOp>([&](auto op) {
            return this->emitBackendCall(op, backend->getSISDAddFunc());
        })
        .Case<mlir::libra::sisd::SISDSubOp>([&](auto op) {
            return this->emitBackendCall(op, backend->getSISDSubFunc());
        })
        .Case<mlir::libra::sisd::SISDMinOp>([&](auto op) {
            return this->emitBackendCall(op, backend->getSISDMinFunc());
        })

        // --- SCFHE Ops ---
        .Case<mlir::libra::scfhe::SCFHEThresholdCountOp>([&](auto op) {
            return this->emitBackendCall(op, "FlyHE.ThresholdCount");
        })

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
        .Case<arith::AddIOp>([&](auto op) { return emitBinary(op, "+"); })
        .Case<arith::MulIOp>([&](auto op) { return emitBinary(op, "*"); })
        .Case<arith::CmpFOp>([&](auto op) { return emitBinary(op, "cmp"); })
        .Case<arith::XOrIOp>([&](auto op) { return emitBinary(op, "^"); })
        .Case<arith::SelectOp>([&](auto op) {
            os << "  auto " << nameFor(op.getResult()) << " = " << nameFor(op.getCondition())
               << " ? " << nameFor(op.getTrueValue()) << " : " << nameFor(op.getFalseValue()) << ";\n";
            return success();
        })
        // 修复: 使用 op->getResult(0) 和 op->getOperand(0)
        .Case<arith::IndexCastOp>([&](auto op) {
            os << "  auto " << nameFor(op->getResult(0)) << " = (size_t)" << nameFor(op->getOperand(0)) << ";\n";
            return success();
        })
        .Case<arith::SIToFPOp>([&](auto op) {
            os << "  auto " << nameFor(op->getResult(0)) << " = (double)" << nameFor(op->getOperand(0)) << ";\n";
            return success();
        })

        // --- Memory / Vector ---
        .Case<memref::AllocOp>([&](auto op) { return this->emitMemRefAlloc(op); })
        .Case<affine::AffineLoadOp>([&](auto op) { return this->emitAffineLoad(op); })
        .Case<affine::AffineStoreOp>([&](auto op) { return this->emitAffineStore(op); })
        .Case<vector::TransferReadOp>([&](auto op) {
            // 修复: 使用通用接口 op->getOperand(0) 获取 source
            os << "  auto " << nameFor(op->getResult(0)) << " = vec_load("
               << nameFor(op->getOperand(0)) << ");\n";
            return success();
        })
        .Case<vector::TransferWriteOp>([&](auto op) {
            // 修复: 0 is vector, 1 is source
            os << "  vec_store(" << nameFor(op->getOperand(0)) << ", " << nameFor(op->getOperand(1)) << ");\n";
            return success();
        })
        .Case<vector::BroadcastOp>([&](auto op) {
            return this->emitBackendCall(op, "vec_broadcast");
        })
        .Case<vector::CreateMaskOp>([&](auto op) {
            return this->emitBackendCall(op, "vec_mask");
        })

        // --- Casts ---
        .Case<mlir::UnrealizedConversionCastOp>([&](auto op) {
            nameTable[op->getResult(0)] = nameFor(op->getOperand(0));
            return success();
        })

        // --- Fallback ---
        .Default([&](Operation* op) {
            return this->printOperation(op);
        });
}

// ============================================================================
// Emit Logic Implementation
// ============================================================================

LogicalResult LibraBackendEmitter::emitBackendCall(Operation* op, StringRef funcName) {
    // 修复: 使用 op->getResult(0)
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

LogicalResult LibraBackendEmitter::emitFunc(func::FuncOp funcOp) {
    if (funcOp.isPrivate())
        return success();
    os << "\nfunc " << funcOp.getName() << "(";
    auto args = funcOp.getArguments();
    for (unsigned i = 0; i < args.size(); ++i) {
        if (i > 0)
            os << ", ";
        os << "arg" << i;
        nameTable[args[i]] = "arg" + std::to_string(i);
    }
    os << ") {\n";
    for (Block& block : funcOp.getBlocks()) {
        for (Operation& op : block) {
            if (failed(translate(op)))
                return failure();
        }
    }
    os << "}\n";
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
    os << "  ";
    if (op.getNumResults() > 0)
        os << "auto " << nameFor(op.getResult(0)) << " = ";
    os << op.getCallee() << "(";
    bool first = true;
    for (Value arg : op.getOperands()) {
        if (!first)
            os << ", ";
        first = false;
        os << nameFor(arg);
    }
    os << ");\n";
    return success();
}

LogicalResult LibraBackendEmitter::emitAffineFor(affine::AffineForOp op) {
    std::string iv = nameFor(op.getInductionVar());
    // 支持动态边界
    std::string lb = op.hasConstantLowerBound() ? std::to_string(op.getConstantLowerBound()) : nameFor(op.getLowerBoundOperands()[0]);
    std::string ub = op.hasConstantUpperBound() ? std::to_string(op.getConstantUpperBound()) : nameFor(op.getUpperBoundOperands()[0]);

    os << "  for (" << iv << " in range(" << lb << ", " << ub << ")) {\n";
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
    os << "  auto " << resName << " = ";
    if (auto floatAttr = dyn_cast<FloatAttr>(op.getValue())) {
        os << floatAttr.getValueAsDouble();
    } else if (auto intAttr = dyn_cast<IntegerAttr>(op.getValue())) {
        os << intAttr.getInt();
    } else {
        os << "const";
    }
    os << ";\n";
    return success();
}

LogicalResult LibraBackendEmitter::emitMemRefAlloc(memref::AllocOp op) {
    os << "  auto " << nameFor(op.getResult()) << " = alloc_mem();\n";
    return success();
}

LogicalResult LibraBackendEmitter::emitAffineLoad(affine::AffineLoadOp op) {
    os << "  auto " << nameFor(op.getResult()) << " = " << nameFor(op.getMemRef()) << "[...];\n";
    return success();
}

LogicalResult LibraBackendEmitter::emitAffineStore(affine::AffineStoreOp op) {
    os << "  " << nameFor(op.getMemRef()) << "[...] = " << nameFor(op.getValueToStore()) << ";\n";
    return success();
}

LogicalResult LibraBackendEmitter::printOperation(Operation* op) {
    StringRef name = op->getName().getStringRef();
    // 修复: 使用 starts_with
    if (name.starts_with("llvm.") || name.starts_with("memref.dealloc") || name.starts_with("polygeist")) {
        if (op->getNumResults() > 0)
            nameTable[op->getResult(0)] = "ignored";
        return success();
    }
    os << "  // " << name << "\n";
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