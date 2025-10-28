#!/bin/bash

# --- 1. Get and validate input ---

# Check if one argument (the working directory name) was provided
if [ "$#" -ne 1 ]; then
    echo "Error: No working directory name specified."
    echo "Usage: $0 <WorkDirName>"
    echo "Example: $0 MinED"
    exit 1
fi

WORK_DIR_NAME=$1

# --- 2. Build paths ---

# Get the directory where this script is located
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Build the absolute paths for the working directory and files
# (e.g., ../MinED)
BASE_WORK_DIR="${SCRIPT_DIR}/../${WORK_DIR_NAME}"
FILENAME_STEM=$WORK_DIR_NAME

# Set the input file name based on the output of the previous script
# (e.g., ../MinED/MinED.O0.mlir)
INPUT_FILE="${BASE_WORK_DIR}/${FILENAME_STEM}.O0.mlir"

# Set the output file name to be inside the working directory
# (e.g., ../MinED/opt-MinED.O0.mlir)
OUTPUT_FILE="${BASE_WORK_DIR}/opt-${FILENAME_STEM}.O0.mlir"

# Set the log file to be inside the working directory
# (e.g., ../MinED/out.log)
LOG_FILE="${BASE_WORK_DIR}/out.log"

# --- 3. Set variables ---

LIBRA_OPT_BIN=~/Libra/build/Libra/tools/opt/libra-opt

# --- 4. Check if tools/inputs exist ---

if [ ! -f "$LIBRA_OPT_BIN" ]; then
    echo "Error: $LIBRA_OPT_BIN not found."
    echo "Please check your build path."
    exit 1
fi

if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file not found!"
    echo "Expected file at: $INPUT_FILE"
    echo "(Did you run libra-mlir.sh first?)"
    exit 1
fi

# --- 5. Run the command ---

echo "Running libra-opt..."
echo "  Input: $INPUT_FILE"
echo "  Output: $OUTPUT_FILE"
echo "  Log: $LOG_FILE"

# Clear old log (if you want to preserve history, remove this line)
> "$LOG_FILE"

$LIBRA_OPT_BIN \
  --convert-to-scfhe \
  -allow-unregistered-dialect \
  --mlir-print-ir-after-all \
  "$INPUT_FILE" \
  -o "$OUTPUT_FILE" &> "$LOG_FILE"

# --- 6. Check the result ---

if [ $? -ne 0 ]; then
    echo "libra-opt failed! See $LOG_FILE for details."
    exit 1
fi

echo "libra-opt succeeded."
echo "Final output is at $OUTPUT_FILE"

