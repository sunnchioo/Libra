~/Libra/build/Libra/tools/opt/libra-opt \
 --convert-to-simd \
 --mlir-print-ir-after-all sum-main.O0.mlir \
 -o opt-sum-main.O0.mlir  &> ./out.log