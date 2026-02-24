#!/bin/bash
# ---------------------------------------------------------
# Polygeist Compiler Script
# Purpose: C -> MLIR -> polygeist-opt (No Vectorization)
# Output: Placed in the same directory as the input file
# ---------------------------------------------------------

set -euo pipefail

# Check if at least input file is provided
if [ $# -lt 1 ]; then
  echo "Usage: $0 <input.c> [output.mlir]" >&2
  exit 1
fi

INPUT_FILE="$1"
IN_DIR=$(cd "$(dirname "$INPUT_FILE")" && pwd)
IN_BASE=$(basename "$INPUT_FILE")

# Handle default output filename: if $2 is missing, use input_name.mlir
if [ -z "${2:-}" ]; then
    # Strip the extension from input filename and append .mlir
    OUTPUT_FILENAME="${IN_BASE%.*}.mlir"
else
    OUTPUT_FILENAME="$2"
fi

OUTPUT_FILE="${IN_DIR}/${OUTPUT_FILENAME}"
TMP_MLIR_FILE="${OUTPUT_FILE}.tmp"

# --- Tool paths ---
REPO_ROOT="/mnt/data0/home/syt/Libra"
POLYGEIST_BIN_DIR="${REPO_ROOT}/Tool/Polygeist/build/bin"
CGEIST_BIN="${POLYGEIST_BIN_DIR}/cgeist"
POLYGEIST_OPT_BIN="${POLYGEIST_BIN_DIR}/polygeist-opt"

# --- Validate tools ---
for tool in "$CGEIST_BIN" "$POLYGEIST_OPT_BIN"; do
  if [ ! -x "$tool" ]; then
    echo "Error: Tool not found or not executable: $tool" >&2
    exit 3
  fi
done

echo "--------------------------------------------------"
echo "Input:  $INPUT_FILE"
echo "Output: $OUTPUT_FILE"

# 1. Convert C to initial MLIR using cgeist
"$CGEIST_BIN" "$INPUT_FILE" \
  -function=* \
  -S \
  -O0 \
  -memref-fullrank \
  -raise-scf-to-affine \
  -o "$TMP_MLIR_FILE"

# 2. Optimize MLIR using polygeist-opt (Vectorization removed)
"$POLYGEIST_OPT_BIN" "$TMP_MLIR_FILE" \
  --canonicalize --cse \
  --sccp \
  --affine-loop-normalize \
  --canonicalize --cse \
  -o "$OUTPUT_FILE"


# Remove the temporary intermediate file
rm -f "$TMP_MLIR_FILE"

# 3. Post-processing: Clean module and function attributes
if [ -f "$OUTPUT_FILE" ]; then
    sed -i -E 's/^module attributes.*/module {/' "$OUTPUT_FILE"
    sed -i -E 's/(func\.func[^{]*?)\s+attributes\s*\{[^}]+\}/\1/' "$OUTPUT_FILE"
    echo "Done: Successfully generated $OUTPUT_FILE"
else
    echo "Error: Failed to generate $OUTPUT_FILE" >&2
    exit 4
fi