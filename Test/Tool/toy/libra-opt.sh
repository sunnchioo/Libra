~/Libra/build/Libra/tools/opt/libra-opt \
 --convert-to-scfhe \
 -allow-unregistered-dialect \
 --mlir-print-ir-after-all \
 sum-main.O3.mlir \
 -o opt-sum-main.O3.mlir  &> ./out.log