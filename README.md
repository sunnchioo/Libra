# About Libra
This artifact provides the source code and evaluation scripts for the 
paper Libra: Pattern-Scheduling Co-Optimization for Cross-Scheme FHE Code 
Generation over GPGPU. Libra is an end-to-end fully homomorphic encryption (FHE) 
compiler for GPGPUs that transforms high-level C programs into efficient GPU 
FHE implementations. Libra automates efficient code generation by coupling 
cross-scheme computational patterns with hardware-aware scheduling strategies.

# Repository's Structure

The repository is organized as follows:

```text
Libra                       – Root directory of the Libra compiler project
 |- build                   – Directory for compiled binaries and build artifacts
 |- doc                     – Project documentation and architecture designs
 |- HElib                   – Foundational FHE libraries
 |  └- FlyHE                – High-performance CUDA FHE library
 |- Libra                   – Core compiler source code
 |  |- Dialect              – Custom FHE dialects for Libra
 |  |- Target               – Backend code generation logic for CUDA
 |  |- Tools                – Libra Compiler utility passes and transformation source
 |  └- CMakeLists.txt       – Build configuration for core compiler components
 |- Script                  – Automation scripts for the compilation pipeline
 |  |- libra-mlir.sh        – Script for C to MLIR frontend conversion
 |  |- libra-opt.sh         – Script for middle-end optimization passes
 |  |- libra-translate.sh   – Translation of IR to CUDA code
 |  └- libra-trans.py       – Script for final CUDA code generation 
 |- Tool                    – External compiler infrastructure
 |  |- llvm-project         – Base LLVM/MLIR framework
 |  └- Polygeist            – C-to-MLIR frontend transformation tool
 |- CMakeLists.txt          – Global project build configuration
 |- LICENSE                 – Project license file
 └- README.md               – Main setup and usage instructions
```

For detailed functions and descriptions of each directory, please 
refer to [Libra/doc/1.Project_Structure.md](doc/1.Project_Structure.md).


## Prerequisites

> **Notice:** To simplify the setup process and ensure environment consistency, we 
provide a pre-configured [**Docker image**](https://hub.docker.com/r/suen0/libra_ae) with all toolchains 
and dependencies pre-installed. The details for starting Docker are provided in 
[Libra/doc/3.Docker_Setup.md](doc/3.Docker_Setup.md). You can pull the image 
and start a container to build the Libra compiler immediately. 


### System Requirements

Ensure your Ubuntu 22.04 system has the following toolchains installed:
* **CPU Architecture**: amd64
* **CMake** >= 3.31.1
* **Host Compilers**:
    * **GCC & G++** >= 13.1.0
    * **Clang & Clang++** >= 22.0.0
    * **Ninja** >= 1.10.1
* **CUDA Toolkit** >= 12.4
* **Hardware**: NVIDIA A100 GPGPU (>= 40 GB)


### Software Dependencies & Libraries

The following libraries must be installed or pre-compiled:
* **LLVM & MLIR** >= 22.0
* **NTL (Number Theory Library)** >= 11.5.1
* **GMP (GNU Multi-Precision Library)** >= 6.2.1
* **Python** >= 3.10
* **matplotlib** >= 3.10.6
* **pandas** >= 2.3.3


### Project Language Standards
* **C Standard**: 17 and 20
* **C++ Standard**: 17 and 20
* **CUDA Standard**: 20


## Build Libra compiler

After completing the prerequisites, you need to build the Libra compiler by following 
the steps below. 

> **Notice:** If you are using Docker, you still need to start from step `1. Build Polygeist`.

### 1. Build Polygeist

This project uses `Security_Artifact/Libra` as the root directory.

```bash
cd Security_Artifact/Libra/Tool/Polygeist
mkdir build
cd build
cmake -G Ninja ../llvm-project/llvm \
  -DLLVM_ENABLE_PROJECTS="clang;mlir" \
  -DLLVM_EXTERNAL_PROJECTS="polygeist" \
  -DLLVM_EXTERNAL_POLYGEIST_SOURCE_DIR=.. \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DLLVM_ENABLE_ASSERTIONS=OFF \
  -DCMAKE_BUILD_TYPE=Release
ninja -j32
ninja check-polygeist-opt && ninja check-cgeist
```

### 2. Build LLVM, MLIR, and Clang

```bash
cd Security_Artifact/Libra/Tool/llvm-project
mkdir build
cd build
cmake -G Ninja ../llvm \
  -DLLVM_ENABLE_PROJECTS="mlir;clang" \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DLLVM_ENABLE_ASSERTIONS=OFF \
  -DCMAKE_BUILD_TYPE=Release
ninja -j32
ninja check-mlir
```

### 3. Build the Libra Compiler

```bash
cd Security_Artifact/Libra
mkdir build
cd build
cmake -G Ninja ..
ninja
```

## Experiment Evaluation

The benchmarks for this artifact are in the `Libra_full_bench` directory. It has 
two parts: `Microbenchmarks` and `Applications`, covering all experiments in the 
paper. Benchmark source programs are provided in plaintext C, Libra compiles them 
into FHE CUDA C++ implementations. The scripts in `Security_Artifact/Libra/Script` 
can automatically convert them into the target FHE CUDA C++ code. 

You can reproduce the final experimental results using the steps below.

### 1. Generate code

Here are the steps to run the scripts and generate the FHE CUDA C++ code.

(1) Run the Libra frontend script.
```bash
cd Security_Artifact/Libra
./Script/libra-mlir.sh
```

(2) Run the Libra middle-end script.
```bash
cd Security_Artifact/Libra
./Script/libra-opt.sh
```

(3) Run the Libra backend script.
```bash
cd Security_Artifact/Libra
./Script/libra-translate.sh
```

After that, the generated target code will be written to `Security_Artifact/Libra/HElib/FlyHE/compiled`. 
Each subdirectory contains the corresponding FHE CUDA C++ code.

### 2. Building Libra Compiled Code

The compiled code can be found in `Security_Artifact/Libra/HElib/FlyHE/compiled`. 
The next step is to compile them with the foundational FHE libraries `FlyHE` to generate 
the executable using the following command.

```bash
cd Security_Artifact/Libra/HElib/FlyHE
mkdir build
cd build
cmake ..
make -j
```

## Appendix

The above steps describe how to reproduce the Libra compiler workflow. For more details on using Libra, please refer to [Libra/doc/2.Toolchain_of_Libra.md](doc/2.Toolchain_of_Libra.md).


## License
This project is released under the license in the [LICENSE](./LICENSE) file.
Third-party components (e.g., LLVM/MLIR, Polygeist, NTL, GMP, and any included 
dependencies) are subject to their respective licenses.