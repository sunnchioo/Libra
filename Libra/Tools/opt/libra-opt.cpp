
#include "SCFHEDialect.h"
#include "SCFHEPass.h"

#include "SIMDDialect.h"
#include "SIMDPass.h"

#include "SISDDialect.h"
#include "SISDPass.h"

#include "ModeSelectPass.h"

#include "ParallelPass.h"

#include "mlir-c/Debug.h"
#include "mlir/Config/mlir-config.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

int main(int argc, char** argv) {
    mlir::registerAllPasses();
    mlir::DialectRegistry registry;
    registerAllDialects(registry);
    registry.insert<mlir::libra::scfhe::SCFHEDialect>();
    registry.insert<mlir::libra::simd::SIMDDialect>();
    registry.insert<mlir::libra::sisd::SISDDialect>();
    registerAllExtensions(registry);
    mlir::libra::scfhe::registerSCFHEOptPasses();
    mlir::libra::simd::registerSIMDOptPasses();
    mlir::libra::sisd::registerSISDOptPasses();
    mlir::libra::mdsel::registerMDSELOptPasses();
    mlir::libra::parallel::registerPARALLELOptPasses();
    // mlirEnableGlobalDebug(true);
    return mlir::asMainReturnCode(
        mlir::MlirOptMain(argc, argv, "libra optimizer driver\n", registry));
}
