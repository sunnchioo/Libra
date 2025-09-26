# Libra
FHE CUDA Compiler

## Build
1. Using unified LLVM, MLIR, Clang, and Polygeist build

```bash
cd Tool/Polygeist
mkdir build
cd build
cmake -G Ninja ../llvm-project/llvm \
  -DLLVM_ENABLE_PROJECTS="clang;mlir" \
  -DLLVM_EXTERNAL_PROJECTS="polygeist" \
  -DLLVM_EXTERNAL_POLYGEIST_SOURCE_DIR=.. \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_BUILD_TYPE=DEBUG
ninja
ninja check-polygeist-opt && ninja check-cgeist
```

2. Build New LLVM, MLIR, and Clang

```bash
cd Tool/llvm-project
mkdir build
cd build
cmake -G Ninja ../llvm \
  -DLLVM_ENABLE_PROJECTS="mlir;clang" \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_BUILD_TYPE=DEBUG
ninja
ninja check-mlir
```


3. Build Libra compiler

```bash
cd Libra
mkdir build
cd build
cmake -G Ninja ..
ninja
```

## Tools
After the build process completes, two tools will be generated: `libra-opt` and `libra-translate`.
1. libra-opt

libra-opt will convert individual passes.

```bash
libra-opt  --convert-to-simd \
 --mlir-print-ir-after-all $fileName$.mlir \
 -o $fileName$-opt.mlir
```


2. libra-translate

libra-translate will translate the entire program.

```bash
libra-translate $fileName$.mlir --mlir-to-cuda
```


## Program
The completed program and its results can be found in the `Libra/program`.


