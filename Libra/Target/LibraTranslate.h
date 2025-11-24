#pragma once

// 标准 MLIR / LLVM 头
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-translate/Translation.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace mlir {
    namespace libra {
        namespace backend {

            /// 前向声明（你的工程里应有这个类）
            class BackendDescriptor;

            /// 注册翻译器（供 mlir-translate 使用）
            /// 例如：mlir-translate --emit-libra-backend input.mlir
            void registerLibraBackendTranslation();

            /// LibraBackendEmitter
            ///
            /// 负责将 MLIR (ModuleOp) 翻译/打印成后端目标代码（可通过 BackendDescriptor 定制）。
            /// 设计原则：
            // —- 1) 非侵入式：构造时可传入后端描述符（或留空）
            // —- 2) 可扩展：提供 translate/emitModule/printOperation 等钩子，方便在 .cpp 中按 op 类型做分发
            /// 典型用法（伪）：
            ///   LibraBackendEmitter emitter(output, backendDesc);
            ///   emitter.emitModule(module);
            class LibraBackendEmitter {
            public:
                /// 构造：必须传入输出流；后端描述符可选（nullptr 表示使用默认/延迟注入）
                explicit LibraBackendEmitter(llvm::raw_ostream& os,
                                             const BackendDescriptor* backend = nullptr)
                    : os(os), backend(backend) {}

                /// 主接口：翻译并输出整个 module
                LogicalResult emitModule(ModuleOp module);

                /// 可选：在运行前设置或替换后端描述符
                void setBackend(const BackendDescriptor* b) { backend = b; }

            private:
                llvm::raw_ostream& os;
                const BackendDescriptor* backend;

                /// 值 -> 名称映射表（为每个 SSA Value 生成/缓存一个名字）
                llvm::DenseMap<Value, std::string> nameTable;

                /// 给定 Value，返回一个唯一的名称（实现可在 .cpp 中完善）
                std::string nameFor(Value v);

                /// 翻译单个顶层操作（如 func.func / global / …），由 emitModule 调用
                LogicalResult translate(Operation& op);

                /// 为 func.func 生成代码（或分发函数体）
                LogicalResult emitFunc(func::FuncOp funcOp);

                /// 通用打印器：按操作类型重载
                LogicalResult printOperation(func::ReturnOp op);
                LogicalResult printOperation(func::FuncOp funcOp);

                /// 通用 fallback：指针/引用两种接口，供不同场景调用
                LogicalResult printOperation(Operation* op);  // pointer-based fallback
                LogicalResult printOperation(Operation& op);  // reference-based dispatcher

                /// 针对特定自定义 op 的打印（如果需要，可在 .cpp 中实现更多重载）
                LogicalResult printOperation(Operation* op, StringRef name);

                /// 你可以在这里添加更多辅助方法（例如：emitBackendCall, emitTypeDecl 等）
            };

        }  // namespace backend
    }  // namespace libra
}  // namespace mlir
