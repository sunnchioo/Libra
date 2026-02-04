#!/bin/bash

# ---------------------------------------------------------
# libra-translate.sh  (for repo root: .../Libra)
# Usage:
#   ./Script/libra-translate.sh
#   ./Script/libra-translate.sh all
#   ./Script/libra-translate.sh Applications/DataAnalysis
#   ./Script/libra-translate.sh Applications/DataAnalysis/opt-Database32.O0.mlir
# ---------------------------------------------------------

set -e

TARGET=${1:-all}

# --- 1) Resolve paths relative to this script ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../Libra/Script
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"                  # .../Libra

IN_ROOT="${REPO_ROOT}/Libra_full_bench/OPT"
OUT_ROOT="${REPO_ROOT}/Libra_full_bench/TRANS"

# --- 2) Tool path (relative to repo root) ---
LIBRA_TRANSLATE_BIN="${REPO_ROOT}/build/Libra/Tools/translate/libra-translate"

# --- 3) Checks ---
if [ ! -x "$LIBRA_TRANSLATE_BIN" ]; then
  echo "Error: libra-translate not found or not executable:"
  echo "  $LIBRA_TRANSLATE_BIN"
  echo "Hint: check your build output path under: ${REPO_ROOT}/build"
  exit 1
fi

if [ ! -d "$IN_ROOT" ]; then
  echo "Error: Input OPT directory not found:"
  echo "  $IN_ROOT"
  exit 1
fi

mkdir -p "$OUT_ROOT"

# --- helper: translate one OPT mlir file (REL_FILE is relative to IN_ROOT) ---
run_one_file() {
  local REL_FILE="$1"   # e.g. Applications/DataAnalysis/opt-Database32.O0.mlir

  local INPUT_FILE="${IN_ROOT}/${REL_FILE}"
  if [ ! -f "$INPUT_FILE" ]; then
    echo "Skip: input not found: $INPUT_FILE"
    return
  fi

  local REL_DIR
  REL_DIR="$(dirname "$REL_FILE")"

  local BASENAME
  BASENAME="$(basename "$REL_FILE")"        # opt-XXX.O0.mlir

  # Output name: remove .mlir, then (optionally) remove leading "opt-"
  local STEM="${BASENAME%.mlir}"            # opt-XXX.O0
  local OUT_STEM="${STEM#opt-}"             # XXX.O0 (if it had opt- prefix)

  local OUT_DIR="${OUT_ROOT}/${REL_DIR}"
  mkdir -p "$OUT_DIR"

  local OUTPUT_FILE="${OUT_DIR}/${OUT_STEM}.cu"
  local LOG_FILE="${OUT_DIR}/translate-${OUT_STEM}.log"

  echo "--------------------------------------------------"
  echo "Input :  $INPUT_FILE"
  echo "Output:  $OUTPUT_FILE"
  echo "Log  :   $LOG_FILE"
  # echo "--------------------------------------------------"

  : > "$LOG_FILE"

  "$LIBRA_TRANSLATE_BIN" \
    -emit-libra-backend \
    --allow-unregistered-dialect \
    "$INPUT_FILE" \
    -o "$OUTPUT_FILE" &> "$LOG_FILE"

  echo "Done: $OUTPUT_FILE"
}

# --- helper: run on all *.mlir in one relative directory (non-recursive) ---
run_dir_all_mlir() {
  local REL_DIR="$1"   # e.g. Applications/DataAnalysis

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
  # Traverse OPT/Applications/* and OPT/Microbenchmarks/*
  for d in "${IN_ROOT}/Applications"/* "${IN_ROOT}/Microbenchmarks"/*; do
    [ -d "$d" ] || continue
    rel="${d#${IN_ROOT}/}"
    run_dir_all_mlir "$rel"
  done
else
  if [[ "$TARGET" == *.mlir ]]; then
    # Single file relative to OPT root
    run_one_file "$TARGET"
  else
    # Directory relative to OPT root
    run_dir_all_mlir "$TARGET"
  fi
fi

echo "All done."
