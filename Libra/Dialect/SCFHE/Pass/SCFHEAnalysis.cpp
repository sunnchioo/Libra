// SCFHEAnalysis.cpp
#include "SCFHEAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "scfhe-pass"

using namespace mlir;
using namespace mlir::libra::scfhe;

void ArgAnalysis::run(ModuleOp module) {
    LLVM_DEBUG(llvm::dbgs() << "\n=== [Analysis] Starting Parameter Analysis ===\n");

    module.walk([&](func::FuncOp func) {
        if (!func->hasAttr("scfhe.crypto"))
            return;

        LLVM_DEBUG(llvm::dbgs() << "Analyzing Function: @" << func.getName() << "\n");

        FunctionArgInfo info;
        unsigned numArgs = func.getNumArguments();
        info.inputArgs.resize(numArgs);
        info.outputArgs.resize(numArgs);

        for (unsigned i = 0; i < numArgs; ++i) {
            BlockArgument arg = func.getArgument(i);

            LLVM_DEBUG(llvm::dbgs() << "  Arg[" << i << "] Type: " << arg.getType());

            if (!isa<MemRefType>(arg.getType())) {
                LLVM_DEBUG(llvm::dbgs() << " -> [Ignored] (Not MemRef)\n");
                continue;
            }

            bool isWritten = false;
            for (Operation* user : arg.getUsers()) {
                if (auto storeOp = dyn_cast<affine::AffineStoreOp>(user)) {
                    if (storeOp.getMemRef() == arg) {
                        isWritten = true;
                        break;
                    }
                }
            }

            if (isWritten) {
                info.outputArgs.set(i);
                LLVM_DEBUG(llvm::dbgs() << " -> [Output] (Will be Decrypted)\n");
            } else {
                info.inputArgs.set(i);
                LLVM_DEBUG(llvm::dbgs() << " -> [Input] (Will be Encrypted)\n");
            }
        }
        infoMap[func] = info;
    });
    LLVM_DEBUG(llvm::dbgs() << "=== [Analysis] Finished ===\n\n");
}