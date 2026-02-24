#!/bin/bash
# ---------------------------------------------------------
# libra-opt.sh (single input -> single output)
#
# Usage:
#   ./libra-opt.sh <input.mlir> <output.mlir> [log_file]
# ---------------------------------------------------------

set -euo pipefail

if [ $# -lt 2 ]; then
  echo "Usage: $0 <input.mlir> <output.mlir> [log_file]" >&2
  exit 1
fi

INPUT_FILE="$1"
OUTPUT_FILE="$2"
LOG_FILE="${3:-${OUTPUT_FILE}.log}"

# --- Resolve paths relative to this script ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../Libra/Script (or wherever)
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"                  # .../Libra

# --- Tool path ---
LIBRA_OPT_BIN="${REPO_ROOT}/build/Libra/Tools/opt/libra-opt"

# --- Cost table path (as in original) ---
COST_TABLE_PATH="${REPO_ROOT}/Libra/Dialect/Analysis/cost_table.json"

# --- Checks ---
if [ ! -x "$LIBRA_OPT_BIN" ]; then
  echo "Error: libra-opt not found or not executable:" >&2
  echo "  $LIBRA_OPT_BIN" >&2
  echo "Hint: build it first under: ${REPO_ROOT}/build" >&2
  exit 2
fi

if [ ! -f "$INPUT_FILE" ]; then
  echo "Error: input not found: $INPUT_FILE" >&2
  exit 3
fi

# Ensure output/log directories exist
mkdir -p "$(dirname "$OUTPUT_FILE")"
mkdir -p "$(dirname "$LOG_FILE")"

echo "--------------------------------------------------"
echo "Input :  $INPUT_FILE"
echo "Output:  $OUTPUT_FILE"
echo "Log  :   $LOG_FILE"

: > "$LOG_FILE"

## opt for all
"$LIBRA_OPT_BIN" \
  --verify-each \
  --mlir-print-ir-after-all \
  --debug-only=scfhe-pass \
  --auto-annotate-scfhe \
  --convert-to-scfhe \
  --canonicalize \
  --inline \
  --canonicalize \
  --convert-to-simd \
  --debug-only=convert-to-simd \
  --mode-select \
  --mode-select-cost-table="$COST_TABLE_PATH" \
  --debug-only=mode-select \
  "$INPUT_FILE" \
  -o "$OUTPUT_FILE" &> "$LOG_FILE"

## opt for scfhe
# "$LIBRA_OPT_BIN" \
#   --verify-each \
#   --mlir-print-ir-after-all \
#   --debug-only=scfhe-pass \
#   --auto-annotate-scfhe \
#   --convert-to-scfhe \
#   --canonicalize \
#   --inline \
#   --canonicalize \
#   "$INPUT_FILE" \
#   -o "$OUTPUT_FILE" &> "$LOG_FILE"

## opt for simd
# "$LIBRA_OPT_BIN" \
#   --verify-each \
#   --mlir-print-ir-after-all \
#   --convert-to-simd \
#   --debug-only=convert-to-simd \
#   "$INPUT_FILE" \
#   -o "$OUTPUT_FILE" &> "$LOG_FILE"

## opt for mode-select
# "$LIBRA_OPT_BIN" \
#   --verify-each \
#   --mlir-print-ir-after-all \
#   --mode-select \
#   --mode-select-cost-table="$COST_TABLE_PATH" \
#   --debug-only=mode-select \
#   "$INPUT_FILE" \
#   -o "$OUTPUT_FILE" &> "$LOG_FILE"

echo "Done: $OUTPUT_FILE"
