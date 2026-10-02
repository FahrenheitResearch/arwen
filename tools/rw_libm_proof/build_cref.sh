#!/bin/bash
# Build the CORE-MATH C references twice (no contraction / native FMA with
# GCC's default contraction) plus the MPFR shim into static libraries.
set -euo pipefail
# B: a scratch directory holding a core-math clone (git clone
# https://gitlab.inria.fr/core-math/core-math, revision 284b3b0e1980) and an
# MPFR 4.2.2 mpfr.h in $B/inc; the libraries land in $B/cref.
B="${B:?set B to the scratch directory}"
CM=$B/core-math/src
mkdir -p $B/cref
cd $B/cref
rm -f *.o *.a
for f in binary32/sin/sinf binary32/cos/cosf binary32/tan/tanf binary32/atan/atanf binary32/atan2/atan2f \
         binary32/exp/expf binary32/log/logf binary32/log10/log10f binary32/pow/powf \
         binary64/asin/asin binary64/acos/acos; do
  n=$(basename $f)
  gcc -O2 -ffp-contract=off -fno-builtin -fPIC -c $CM/$f.c -o nofma_$n.o
  gcc -O3 -march=native -ffp-contract=fast -fPIC -Dcr_$n=crf_$n -c $CM/$f.c -o fma_$n.o
done
gcc -O2 -fPIC -I$B/inc -c "$(dirname "$0")/mpref.c" -o mpref.o
ar rcs libcmref.a nofma_*.o fma_*.o
ar rcs libmpref.a mpref.o
ls -la *.a
nm libcmref.a | grep -E " T (cr|crf)_" | sort | tr '\n' ' '; echo
