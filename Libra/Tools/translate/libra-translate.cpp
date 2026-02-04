#include "LibraTranslate.h"

#include "llvm/Support/LogicalResult.h"                   // from @llvm-project
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"  // from @llvm-project

using namespace mlir;
using namespace mlir::libra::backend;

int main(int argc, char** argv) {
    registerLibraBackendTranslation();
    return failed(mlirTranslateMain(argc, argv, "Libra Translation Tool."));
}
