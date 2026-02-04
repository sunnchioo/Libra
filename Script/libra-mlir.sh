#!/bin/bash

# ---------------------------------------------------------
# Usage:
#   ./libra-mlir.sh                              # default: all 32768
#   ./libra-mlir.sh all                          # all 32768
#   ./libra-mlir.sh all 1024                     # all 1024
# ---------------------------------------------------------

set -e

# --- 1. Defaults (no args -> all 32768) ---
TARGET=${1:-all}
VECTOR_SIZE=${2:-32768}

# --- 2. Resolve relative paths based on script location ---

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

IN_ROOT="${REPO_ROOT}/Libra_full_bench"

OUT_ROOT="${IN_ROOT}/MLIR"

# --- 3. Tool paths ---

POLYGEIST_BIN_DIR="${REPO_ROOT}/Tool/Polygeist/build/bin"
# POLYGEIST_BIN_DIR="/mnt/data0/home/syt/Libra/Tool/Polygeist/build/bin"


CGEIST_BIN="${POLYGEIST_BIN_DIR}/cgeist"
POLYGEIST_OPT_BIN="${POLYGEIST_BIN_DIR}/polygeist-opt"

# --- helper: compile one .c file under a relative program directory ---
#   run_one "Applications/KMeans" "KMeans_p16c2.c"
run_one() {
  local REL_DIR="$1"        # e.g. Applications/KMeans
  local SRC_FILE="$2"       # e.g. KMeans_p16c2.c

  local BASE_IN_DIR="${IN_ROOT}/${REL_DIR}"
  local BASE_OUT_DIR="${OUT_ROOT}/${REL_DIR}"

  local STEM="${SRC_FILE%.c}"

  local INPUT_FILE="${BASE_IN_DIR}/${SRC_FILE}"
  local OUTPUT_FILE="${BASE_OUT_DIR}/${STEM}.O0.mlir"
  local TMP_MLIR_FILE="${OUTPUT_FILE}.tmp"

  if [ ! -f "$INPUT_FILE" ]; then
    echo "Skip: $INPUT_FILE not found"
    return
  fi

  mkdir -p "$BASE_OUT_DIR"

  echo "--------------------------------------------------"
  echo "ProgramDir:   $REL_DIR"
  echo "Source:       $SRC_FILE"
  echo "Input:        $INPUT_FILE"
  echo "Output:       $OUTPUT_FILE"
  # echo "VectorSize:   $VECTOR_SIZE"
  # echo "--------------------------------------------------"

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


  # "$POLYGEIST_OPT_BIN" "$TMP_MLIR_FILE" \
  # -canonicalize -cse \
  # -affine-loop-normalize \
  # -affine-scalrep \
  # -canonicalize -cse \
  # -affine-super-vectorize="virtual-vector-size=${VECTOR_SIZE} test-fastest-varying=0 vectorize-reductions=true" \
  # -debug-only=early-vect \
  # -canonicalize -cse \
  # -affine-simplify-structures \
  # -o "$OUTPUT_FILE"


  rm -f "$TMP_MLIR_FILE"

  sed -i -E 's/^module attributes.*/module {/' "$OUTPUT_FILE"
  sed -i -E 's/(func\.func[^{]*?)\s+attributes\s*\{[^}]+\}/\1/' "$OUTPUT_FILE"

  echo "Done: $OUTPUT_FILE"
}

# --- helper: compile all .c files in one relative directory (non-recursive) ---
run_dir_all_c() {
  local REL_DIR="$1"  # e.g. Applications/KMeans

  shopt -s nullglob
  local files=("${IN_ROOT}/${REL_DIR}"/*.c)
  shopt -u nullglob

  if [ ${#files[@]} -eq 0 ]; then
    echo "Skip: no .c files in ${REL_DIR}"
    return
  fi

  for f in "${files[@]}"; do
    run_one "$REL_DIR" "$(basename "$f")"
  done
}

# --- 4. Run ---

if [ "$TARGET" = "all" ]; then
  # Traverse each subfolder under Applications and Microbenchmarks,
  # compile all .c files inside each subfolder.
  for d in "${IN_ROOT}/Applications"/* "${IN_ROOT}/Microbenchmarks"/*; do
    [ -d "$d" ] || continue
    rel="${d#${IN_ROOT}/}"
    run_dir_all_c "$rel"
  done

else
  # Two modes:
  # 1) TARGET ends with .c -> compile that single file
  #    e.g. Applications/KMeans/KMeans_p16c2.c
  # 2) TARGET is a directory -> compile all .c in that directory
  #    e.g. Applications/KMeans

  if [[ "$TARGET" == *.c ]]; then
    rel_dir="$(dirname "$TARGET")"
    src_file="$(basename "$TARGET")"
    run_one "$rel_dir" "$src_file"
  else
    run_dir_all_c "$TARGET"
  fi
fi

echo "All done."
