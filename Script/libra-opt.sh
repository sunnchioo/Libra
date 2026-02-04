#!/bin/bash

# ---------------------------------------------------------
# libra-opt.sh  (for repo root: .../Libra)
# Usage:
#   ./Script/libra-opt.sh
#   ./Script/libra-opt.sh all
#   ./Script/libra-opt.sh Applications/KMeans
#   ./Script/libra-opt.sh Applications/KMeans/KMeans_p16c2.O0.mlir
# ---------------------------------------------------------

set -e

TARGET=${1:-all}

# --- 1) Resolve paths relative to this script (robust no matter where you run it) ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../Libra/Script
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"                  # .../Libra

IN_ROOT="${REPO_ROOT}/Libra_full_bench/MLIR"
OUT_ROOT="${REPO_ROOT}/Libra_full_bench/OPT"

# --- 2) Tool path (relative to repo root) ---
LIBRA_OPT_BIN="${REPO_ROOT}/build/Libra/Tools/opt/libra-opt"

# --- 3) Checks ---
if [ ! -x "$LIBRA_OPT_BIN" ]; then
  echo "Error: libra-opt not found or not executable:"
  echo "  $LIBRA_OPT_BIN"
  echo "Hint: build it first under: ${REPO_ROOT}/build"
  exit 1
fi

if [ ! -d "$IN_ROOT" ]; then
  echo "Error: Input MLIR directory not found:"
  echo "  $IN_ROOT"
  exit 1
fi

mkdir -p "$OUT_ROOT"

# --- helper: run libra-opt on one input file (REL_FILE is relative to IN_ROOT) ---

COST_TABLE_PATH="${REPO_ROOT}/Libra/Dialect/Analysis/cost_table.json"

run_one_file() {
  local REL_FILE="$1"   # e.g. Applications/KMeans/KMeans_p16c2.O0.mlir

  local INPUT_FILE="${IN_ROOT}/${REL_FILE}"
  if [ ! -f "$INPUT_FILE" ]; then
    echo "Skip: input not found: $INPUT_FILE"
    return
  fi

  local REL_DIR
  REL_DIR="$(dirname "$REL_FILE")"

  local BASENAME
  BASENAME="$(basename "$REL_FILE")"          # xxx.mlir
  local STEM="${BASENAME%.mlir}"              # xxx

  local OUT_DIR="${OUT_ROOT}/${REL_DIR}"
  mkdir -p "$OUT_DIR"

  local OUTPUT_FILE="${OUT_DIR}/opt-${STEM}.mlir"
  local LOG_FILE="${OUT_DIR}/opt-${STEM}.log"

  echo "--------------------------------------------------"
  echo "Input :  $INPUT_FILE"
  echo "Output:  $OUTPUT_FILE"
  echo "Log  :   $LOG_FILE"
#   echo "--------------------------------------------------"

  : > "$LOG_FILE"

  "$LIBRA_OPT_BIN" \
    --verify-each \
    --convert-to-scfhe \
    --convert-to-simd \
    --mode-select \
    --mode-select-cost-table="$COST_TABLE_PATH" \
    --add-bounded-stream-id \
    --allow-unregistered-dialect \
    --mlir-print-ir-after-all \
    "$INPUT_FILE" \
    -o "$OUTPUT_FILE" &> "$LOG_FILE"

    # "$LIBRA_OPT_BIN" \
    # --verify-each \
    # --convert-to-scfhe \
    # -allow-unregistered-dialect \
    # --mlir-print-ir-after-all \
    # "$INPUT_FILE" \
    # -o "$OUTPUT_FILE" &> "$LOG_FILE"

  echo "Done: $OUTPUT_FILE"
}

# --- helper: run on all *.mlir in one relative directory (non-recursive) ---
run_dir_all_mlir() {
  local REL_DIR="$1"   # e.g. Applications/KMeans

  shopt -s nullglob
  local files=("${IN_ROOT}/${REL_DIR}"/*.mlir)
  shopt -u nullglob

  if [ ${#files[@]} -eq 0 ]; then
    echo "Skip: no .mlir files in ${IN_ROOT}/${REL_DIR}"
    return
  fi

  for f in "${files[@]}"; do
    run_one_file "${REL_DIR}/$(basename "$f")"
  done
}

# --- 4) Dispatch ---
if [ "$TARGET" = "all" ]; then
  # Traverse MLIR/Applications/* and MLIR/Microbenchmarks/*
  for d in "${IN_ROOT}/Applications"/* "${IN_ROOT}/Microbenchmarks"/*; do
    [ -d "$d" ] || continue
    rel="${d#${IN_ROOT}/}"
    run_dir_all_mlir "$rel"
  done
else
  if [[ "$TARGET" == *.mlir ]]; then
    # Single file relative to MLIR root
    run_one_file "$TARGET"
  else
    # Directory relative to MLIR root
    run_dir_all_mlir "$TARGET"
  fi
fi

echo "All done."
