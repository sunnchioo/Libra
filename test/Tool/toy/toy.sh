../../Tool/Polygeist/build/bin/cgeist toy.cpp \
  -function=* -S \
  -raise-scf-to-affine \
  --memref-fullrank -O3 \
  -o toy.O3.mlir
