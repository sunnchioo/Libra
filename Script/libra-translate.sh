#!/bin/bash
# ---------------------------------------------------------
# libra-translate.sh (single input -> single output)
#
# Usage:
#   ./libra-translate.sh <input.mlir> <output.cu> [log_file]
# ---------------------------------------------------------

set -euo pipefail

if [ $# -lt 2 ]; then
  echo "Usage: $0 <input.mlir> <output.cu> [log_file]" >&2
  exit 1
fi

INPUT_FILE="$1"
OUTPUT_FILE="$2"
LOG_FILE="${3:-${OUTPUT_FILE}.log}"

# --- Resolve paths relative to this script ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# --- Tool path (relative to repo root) ---
LIBRA_TRANSLATE_BIN="${REPO_ROOT}/build/Libra/Tools/translate/libra-translate"

# --- Checks ---
if [ ! -x "$LIBRA_TRANSLATE_BIN" ]; then
  echo "Error: libra-translate not found or not executable:" >&2
  echo "  $LIBRA_TRANSLATE_BIN" >&2
  echo "Hint: check your build output path under: ${REPO_ROOT}/build" >&2
  exit 2
fi

if [ ! -f "$INPUT_FILE" ]; then
  echo "Error: input not found: $INPUT_FILE" >&2
  exit 3
fi

mkdir -p "$(dirname "$OUTPUT_FILE")"
mkdir -p "$(dirname "$LOG_FILE")"

echo "--------------------------------------------------"
echo "Input :  $INPUT_FILE"
echo "Output:  $OUTPUT_FILE"
echo "Log  :   $LOG_FILE"
echo "--------------------------------------------------"

: > "$LOG_FILE"

"$LIBRA_TRANSLATE_BIN" \
  -emit-libra-backend \
  --allow-unregistered-dialect \
  "$INPUT_FILE" \
  -o "$OUTPUT_FILE" &> "$LOG_FILE"

echo "Done: $OUTPUT_FILE"
