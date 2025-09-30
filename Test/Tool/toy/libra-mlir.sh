~/Libra/Tool/Polygeist/build/bin/cgeist sum.cpp \
  -function=* \
  --S \
  -raise-scf-to-affine \
  --memref-fullrank -O0 \
  -o sum-main.O0.mlir

# ~/Libra/Tool/Polygeist/build/bin/cgeist sum.cpp \
#   -function=* -S \
#   -raise-scf-to-affine \
#   --memref-fullrank -O1 \
#   -o sum-main.O1.mlir

# ~/Libra/Tool/Polygeist/build/bin/cgeist sum.cpp \
#   -function=* -S \
#   -raise-scf-to-affine \
#   --memref-fullrank -O2 \
#   -o sum-main.O2.mlir

# ~/Libra/Tool/Polygeist/build/bin/cgeist sum.cpp \
#   -function=* -S \
#   -raise-scf-to-affine \
#   --memref-fullrank -O3 \
#   -o sum-main.O3.mlir