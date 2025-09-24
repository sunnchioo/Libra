~/Libra/build/Libra/tools/opt/libra-opt \
 --convert-to-simd \
 --mlir-print-ir-after-all sum.mlir \
 -o opt-sum.mlir  &> ./out.log