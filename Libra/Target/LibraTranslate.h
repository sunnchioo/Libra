#pragma once
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

// 引入你的自定义 Dialect 头文件
#include "SCFHEDialect.h"
#include "SIMDDialect.h"
#include "SISDDialect.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

// 确保包含 BackendDescriptor 的前向声明或头文件
namespace mlir {
    namespace libra {
        namespace backend {

            class BackendDescriptor; // 前向声明

            class LibraBackendEmitter {
            public:
                explicit LibraBackendEmitter(llvm::raw_ostream& os, const BackendDescriptor* backend = nullptr)
                    : os(os), backend(backend) {}

                LogicalResult emitModule(ModuleOp module);

            private:
                llvm::raw_ostream& os;
                const BackendDescriptor* backend;
                llvm::DenseMap<Value, std::string> nameTable;

                std::string nameFor(Value v);
                LogicalResult translate(Operation& op);

                // Emitters
                LogicalResult emitFunc(func::FuncOp op);
                LogicalResult emitReturn(func::ReturnOp op);
                LogicalResult emitFuncCall(func::CallOp op);

                LogicalResult emitArithConstant(arith::ConstantOp op);

                // 注意：这里删除了 emitBinaryOp, emitCast 等不再需要的声明，
                // 或者如果你想保留它们，必须在 .cpp 中提供空实现。
                // 为了通过编译，这里只列出 .cpp 中实际实现的函数：

                LogicalResult emitAffineFor(affine::AffineForOp op);
                LogicalResult emitAffineStore(affine::AffineStoreOp op);
                LogicalResult emitAffineLoad(affine::AffineLoadOp op);

                LogicalResult emitScfIf(scf::IfOp op);

                LogicalResult emitMemRefAlloc(memref::AllocOp op);

                // 核心后端调用函数 [必须匹配 .cpp]
                LogicalResult emitBackendCall(Operation* op, StringRef funcName);

                LogicalResult emitContextAwareCall(Operation* op, llvm::StringRef funcName);

                LogicalResult printOperation(Operation* op);
            };

            void registerLibraBackendTranslation();

        } // namespace backend
    } // namespace libra
} // namespace mlir