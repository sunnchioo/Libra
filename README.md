# Libra: Pattern-Scheduling Co-Optimization for Cross-Scheme FHE Code Generation over GPGPU
Libra is an end-to-end fully homomorphic encryption (FHE) compiler for GPGPUs. It takes high-level 
C programs as input and generates efficient GPU-based FHE implementations by co-optimizing cross-scheme 
computation patterns and hardware-aware GPU scheduling. 
This repository is under active development.

# Repository's Structure

The repository is organized as follows:

```text
Libra                       – Root directory of the Libra compiler project
 |- build                   – Directory for compiled binaries and build artifacts
 |- Doc                     – Project documentation and architecture designs
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
 |  └- libra-translate.sh   – Translation of IR to CUDA code
 |- Tool                    – External compiler infrastructure
 |  |- llvm-project         – Base LLVM/MLIR framework
 |  └- Polygeist            – C-to-MLIR frontend transformation tool
 |- CMakeLists.txt          – Global project build configuration
 |- LICENSE                 – Project license file
 └- README.md               – Main setup and usage instructions
```

For detailed functions and descriptions of each directory, please 
refer to [Libra/Doc/1.Project_Structure.md](Doc/1.Project_Structure.md).


## Prerequisites

> **Notice:** To simplify the setup process and ensure environment consistency, we 
provide a pre-configured [**Docker image**](https://hub.docker.com/r/suen0/libra_ae) with all toolchains 
and dependencies pre-installed. The details for starting Docker are provided in 
[Libra/Doc/3.Docker_Setup.md](Doc/3.Docker_Setup.md). You can pull the image 
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


### Project Language Standards
* **C Standard**: 17 and 20
* **C++ Standard**: 17 and 20
* **CUDA Standard**: 20


## Build Libra compiler

After completing the prerequisites, you need to build the Libra compiler by following 
the steps below. 

> **Notice:** If you are using Docker, you still need to start from step `1. Build Polygeist`.

### 1. Build Polygeist

This project uses `Libra` as the root directory.

```bash
cd Libra/Tool/Polygeist
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
cd Libra/Tool/llvm-project
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
cd Libra
mkdir build
cd build
cmake -G Ninja ..
ninja
```

## Usage

The scripts in `Libra/Script` can convert a C program into the target
FHE CUDA C++ code through three stages: **frontend (C → MLIR)**, **middle-end (MLIR opt)**, and
**backend (MLIR → CUDA C++)**.

### 1. Generate code

Here are the steps to run the scripts and generate the FHE CUDA C++ code.

(1) Run the Libra frontend script.
```bash
cd Libra
./Script/libra-mlir.sh <input.c> <output.mlir>
```

(2) Run the Libra middle-end script.
```bash
cd Libra
./Script/libra-opt.sh <input.mlir> <output.opt.mlir>
```

(3) Run the Libra backend script.
```bash
cd Libra
./Script/libra-translate.sh <input.opt.mlir> <output.cu>
```

After these three steps, the final generated code is the `.cu` file.

### 2. Building the Generated Code

The backend CUDA FHE library relies on our open-source project [`FlyHE`](https://github.com/sunnchioo/FlyHE.git).

The generated CUDA C++ code can be compiled together with `FlyHE` to produce the final executable.

```bash
cd Libra/HElib/FlyHE
mkdir build
cd build
cmake ..
make -j
```

## Future Work

- Provide additional examples and tutorials to further improve usability and accessibility.
- Expand compatibility with other FHE backend libraries.

## Appendix

The above steps describe how to reproduce the Libra compiler workflow. For more details on using Libra, please refer to [Libra/Doc/2.Toolchain_of_Libra.md](Doc/2.Toolchain_of_Libra.md).


## License
This project is released under the license in the [LICENSE](./LICENSE) file.
Third-party components (e.g., LLVM/MLIR, Polygeist, NTL, GMP, and any included 
dependencies) are subject to their respective licenses.

## Citation

```text
@inproceedings{Libra,
 author = {Song Bian and Yintai Sun and Zian Zhao and Haowen Pan and Mingzhe Zhang and Jiafeng Hua and Zhenyu Guan},
 title = {Libra: Pattern-Scheduling Co-Optimization for Cross-Scheme FHE Code Generation over GPGPU},
 booktitle = {{USENIX} Security Symposium ({USENIX} Security)},
 year = {2026}
}
```

## 📊 Project Statistics 

### 🌟 Star History 
<a href="https://star-history.com/#sunnchioo/Libra&Date">
  <img src="https://api.star-history.com/svg?repos=sunnchioo/Libra&type=Date" alt="Star History Chart">
</a> 

### 📉 Traffic Analytics 
GitHub's native traffic insights only retain data for the past 14 days. This project utilizes an automated GitHub Action to persistently archive traffic data and generate long-term historical trends. 

[![View Traffic Report](https://img.shields.io/badge/View-Traffic_Report-blue?style=for-the-badge&logo=google-analytics)](https://sunnchioo.github.io/Libra/sunnchioo/Libra/latest-report/report.html) 

> **Note:** Click the button above to view the detailed interactive report, including historical Views, Clones, and Top Referrers.