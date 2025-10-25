#!/bin/bash
INPUT_FILE=~/Libra/Test/Tool/toy/sum.c
OUTPUT_FILE=sum-main.O3.mlir

POLYGEIST_BIN_DIR=~/Libra/Tool/Polygeist/build/bin
# LLVM_BIN_DIR=~/Libra/Tool/llvm-project/build/bin

CGEIST_BIN=$POLYGEIST_BIN_DIR/cgeist
POLYGEIST_OPT_BIN=$POLYGEIST_BIN_DIR/polygeist-opt
# MLIR_OPT_BIN=$LLVM_BIN_DIR/mlir-opt

TMP_MLIR_FILE="${OUTPUT_FILE}.tmp"

echo "Step 1: Running cgeist..."
$CGEIST_BIN $INPUT_FILE \
  -function=* \
  -S \
  -memref-fullrank \
  -raise-scf-to-affine \
  -O3 \
  -o $TMP_MLIR_FILE

if [ $? -ne 0 ]; then
    echo "cgeist failed!"
    rm -f $TMP_MLIR_FILE
    exit 1
fi

echo "Step 2: Running polygeist-opt..."
$POLYGEIST_OPT_BIN $TMP_MLIR_FILE \
  -canonicalize \
  -cse \
  -affine-super-vectorize="virtual-vector-size=8" \
  -vectorize-slp \
  -affine-loop-normalize="promote-single-iter" \
  -affine-simplify-structures \
  -o $OUTPUT_FILE

if [ $? -ne 0 ]; then
    echo "mlir-opt failed!"
    rm -f $TMP_MLIR_FILE
    exit 1
fi

# rm -f $TMP_MLIR_FILE

echo "Step 3: Cleaning up MLIR output..."
sed -i -E 's/^module attributes.*/module {/' $OUTPUT_FILE
sed -i -E 's/(func\.func[^{]*?)\s+attributes\s*\{[^}]+\}/\1/' $OUTPUT_FILE

echo "MLIR generated at: $OUTPUT_FILE"
