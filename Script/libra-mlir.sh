#!/bin/bash
# ---------------------------------------------------------
# Compile ONE input C file to ONE output MLIR file.
#
# Usage:
#   ./libra-mlir.sh <input.c> <output.mlir> [vector_size]
# ---------------------------------------------------------

set -euo pipefail

if [ $# -lt 2 ]; then
  echo "Usage: $0 <input.c> <output.mlir> [vector_size]" >&2
  exit 1
fi

INPUT_FILE="$1"
OUTPUT_FILE="$2"
VECTOR_SIZE="${3:-32768}"

# --- Resolve paths based on script location ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# --- Tool paths ---
POLYGEIST_BIN_DIR="${REPO_ROOT}/Tool/Polygeist/build/bin"
CGEIST_BIN="${POLYGEIST_BIN_DIR}/cgeist"
POLYGEIST_OPT_BIN="${POLYGEIST_BIN_DIR}/polygeist-opt"

# --- Validate ---
if [ ! -f "$INPUT_FILE" ]; then
  echo "Error: input file not found: $INPUT_FILE" >&2
  exit 2
fi

if [ ! -x "$CGEIST_BIN" ]; then
  echo "Error: cgeist not found or not executable: $CGEIST_BIN" >&2
  exit 3
fi

if [ ! -x "$POLYGEIST_OPT_BIN" ]; then
  echo "Error: polygeist-opt not found or not executable: $POLYGEIST_OPT_BIN" >&2
  exit 4
fi

# Ensure output dir exists
OUT_DIR="$(dirname "$OUTPUT_FILE")"
mkdir -p "$OUT_DIR"

TMP_MLIR_FILE="${OUTPUT_FILE}.tmp"

echo "--------------------------------------------------"
echo "Input:      $INPUT_FILE"
echo "Output:     $OUTPUT_FILE"

"$CGEIST_BIN" "$INPUT_FILE" \
  -function=* \
  -S \
  -O0 \
  -memref-fullrank \
  -raise-scf-to-affine \
  -o "$TMP_MLIR_FILE"

"$POLYGEIST_OPT_BIN" "$TMP_MLIR_FILE" \
  -canonicalize -cse \
  -affine-loop-invariant-code-motion \
  -affine-loop-normalize \
  -affine-scalrep \
  -canonicalize -cse \
  -affine-super-vectorize="virtual-vector-size=${VECTOR_SIZE} test-fastest-varying=0 vectorize-reductions=true" \
  -canonicalize -cse \
  -affine-simplify-structures \
  -o "$OUTPUT_FILE"

rm -f "$TMP_MLIR_FILE"

# Post-process output (keep same behavior as original)
sed -i -E 's/^module attributes.*/module {/' "$OUTPUT_FILE"
sed -i -E 's/(func\.func[^{]*?)\s+attributes\s*\{[^}]+\}/\1/' "$OUTPUT_FILE"

echo "Done: $OUTPUT_FILE"
