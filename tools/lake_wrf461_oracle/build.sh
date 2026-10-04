#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
build=${1:?build directory inside owned scratch required}
mkdir -p "$build/tmp"
export TMPDIR="$build/tmp"
cd "$build"
gfortran -c -O0 -fPIC -cpp -DEM_CORE=1 -ffree-form -ffree-line-length-none \
  "$here/upstream/module_model_constants.F" "$here/stub_wrf.F90" \
  "$here/upstream/module_sf_lake.F" "$here/column_wrapper.F90"
gfortran -shared -o lake_fortran.so module_model_constants.o stub_wrf.o module_sf_lake.o column_wrapper.o
g++ -std=c++17 -O0 -fPIC -fno-fast-math -ffp-contract=off -shared \
  -x c++ "$here/column_cpp.cpp" -I "$here/../../gpuwm/core/kernels" -o lake_cpp.so
gfortran -O0 "$here/abs_control.F90" -o abs_fortran
g++ -std=c++17 -O0 "$here/abs_control.cpp" \
  -I "$here/../../gpuwm/core/kernels" -o abs_cpp
./abs_fortran > abs-fortran.txt
./abs_cpp > abs-cpp.txt
diff -u abs-fortran.txt abs-cpp.txt
