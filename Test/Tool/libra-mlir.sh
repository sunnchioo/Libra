#!/bin/bash

# --- 1. Get and validate input ---

# Check if at least one argument (the working directory name) was provided
# [修改]: 允许输入 1 个或 2 个参数，所以这里改为判断是否小于 1
if [ "$#" -lt 1 ]; then
    echo "Error: No working directory name specified."
    echo "Usage: $0 <WorkDirName> [VectorSize]"
    echo "Example: $0 MinED"
    echo "Example: $0 MinED 16"
    exit 1
fi

WORK_DIR_NAME=$1

VECTOR_SIZE=${2:-8}

# --- 2. Build paths ---

# Get the directory where this script is located
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Build the absolute paths for the working directory and files
BASE_WORK_DIR="${SCRIPT_DIR}/../${WORK_DIR_NAME}"
FILENAME_STEM=$WORK_DIR_NAME

INPUT_FILE="${BASE_WORK_DIR}/${FILENAME_STEM}.c"
OUTPUT_FILE="${BASE_WORK_DIR}/${FILENAME_STEM}.O0.mlir"
TMP_MLIR_FILE="${OUTPUT_FILE}.tmp"

# Check if the input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file not found!"
    echo "Checked path: $INPUT_FILE"
    exit 1
fi

echo "Working Directory: $BASE_WORK_DIR"
echo "Input File:      $INPUT_FILE"
echo "Output File:     $OUTPUT_FILE"
echo "Vector Size:     $VECTOR_SIZE"  # 打印一下当前的向量大小

# --- 3. Set tool paths ---

POLYGEIST_BIN_DIR=~/Libra/Tool/Polygeist/build/bin
# LLVM_BIN_DIR=~/Libra/Tool/llvm-project/build/bin

CGEIST_BIN=$POLYGEIST_BIN_DIR/cgeist
POLYGEIST_OPT_BIN=$POLYGEIST_BIN_DIR/polygeist-opt
# MLIR_OPT_BIN=$LLVM_BIN_DIR/mlir-opt

# --- 4. Run cgeist ---

echo "Step 1: Running cgeist..."
$CGEIST_BIN "$INPUT_FILE" \
  -function=* \
  -S \
  -O0 \
  -memref-fullrank \
  -raise-scf-to-affine \
  -o "$TMP_MLIR_FILE"

if [ $? -ne 0 ]; then
    echo "cgeist failed! Keeping $TMP_MLIR_FILE for debugging."
    exit 1
fi

# --- 5. Run polygeist-opt ---

echo "Step 2: Running polygeist-opt with vector size ${VECTOR_SIZE}..."

# [修改]: 将 virtual-vector-size=8 改为 virtual-vector-size=${VECTOR_SIZE}
$POLYGEIST_OPT_BIN "$TMP_MLIR_FILE" \
 -affine-super-vectorize="virtual-vector-size=${VECTOR_SIZE}" \
 -canonicalize \
 -cse \
 -vectorize-slp \
 -affine-loop-normalize="promote-single-iter" \
 -affine-simplify-structures \
 -o "$OUTPUT_FILE"

if [ $? -ne 0 ]; then
    echo "polygeist-opt failed! Keeping $TMP_MLIR_FILE for debugging."
    exit 1
fi

# Success! Remove the temporary file.
rm -f "$TMP_MLIR_FILE"

# --- 6. Clean up MLIR ---

echo "Step 3: Optimizing MLIR output..."
sed -i -E 's/^module attributes.*/module {/' "$OUTPUT_FILE"
sed -i -E 's/(func\.func[^{]*?)\s+attributes\s*\{[^}]+\}/\1/' "$OUTPUT_FILE"

echo "MLIR generated at: $OUTPUT_FILE"