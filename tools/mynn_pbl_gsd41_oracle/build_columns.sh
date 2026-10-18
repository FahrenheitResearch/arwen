#!/usr/bin/env bash
# Builds the 84-column gsd_41 MYNN driver oracle from the unmodified
# NOAA-EMC WRF 3.9 branch source (tag v4.1.21,
# sorc/hrrr_wrfarw.fd/WRFV3.9/phys/module_bl_mynn.F, sha256
# 42178a22693ad14b1a3adcf3e44be3047f051abf80f3a64c0ce9ed9ac2ec6d8a):
#   driver-columns-gsd41.bin.gz       the one :995 line squared (the
#                                     bl_mynn_gsd41_unsquared_qtke = false
#                                     default form), as build.sh patches it
#   driver-columns-gsd41-asis.bin.gz  the file as written
# Same compiler flags as build.sh's driver families oracle (gfortran -O0,
# the harness also -ffp-contract=off).  The columns come from
# columns_gsd41.py (python3 with numpy).
# usage: build_columns.sh FORK_module_bl_mynn.F BUILD_DIR [NSTEP]
set -euo pipefail
[[ $# -ge 2 ]] || { echo "usage: build_columns.sh module_bl_mynn.F BUILD_DIR [NSTEP]" >&2; exit 2; }
src=$(realpath "$1"); build=$(realpath -m "$2"); nstep=${3:-12}
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
test "$(sha256sum < "$src" | cut -d' ' -f1)" = \
    42178a22693ad14b1a3adcf3e44be3047f051abf80f3a64c0ce9ed9ac2ec6d8a
mkdir -p "$build/asis" "$build/sq"
cp "$src" "$build/asis/module_bl_mynn.F"
line='           qtke(k) = 0.5\*qkw(k)  ! qkw -> TKE'
test "$(grep -c "$line" "$src")" = 1
sed "s/$line/           qtke(k) = 0.5*(qkw(k)**2)  ! qkw -> TKE/" \
    "$src" > "$build/sq/module_bl_mynn.F"
test "$(grep -c 'qtke(k) = 0.5\*(qkw(k)\*\*2)  ! qkw -> TKE' "$build/sq/module_bl_mynn.F")" = 1
python3 "$here/columns_gsd41.py" "$build/columns-in.bin" "$nstep"
for v in asis sq; do
  cd "$build/$v"
  gfortran -c -O0 -cpp -ffree-form -ffree-line-length-none "$here/stub_wrf39.F90"
  gfortran -c -O0 -cpp -ffree-form -ffree-line-length-none -I. module_bl_mynn.F
  gfortran -c -O0 -ffp-contract=off -ffree-form -ffree-line-length-none -I. \
      "$here/run_driver_columns_gsd41.F90"
  gfortran -o run_driver_columns_gsd41 stub_wrf39.o module_bl_mynn.o \
      run_driver_columns_gsd41.o
done
rm -f "$build/driver-columns-gsd41.bin" "$build/driver-columns-gsd41-asis.bin"
"$build/sq/run_driver_columns_gsd41" "$build/columns-in.bin" "$build/driver-columns-gsd41.bin"
"$build/asis/run_driver_columns_gsd41" "$build/columns-in.bin" "$build/driver-columns-gsd41-asis.bin"
gzip -n -9 -f -k "$build/driver-columns-gsd41.bin" "$build/driver-columns-gsd41-asis.bin"
cd "$build"
sha256sum "$src" "$here/stub_wrf39.F90" "$here/columns_gsd41.py" \
    "$here/run_driver_columns_gsd41.F90" columns-in.bin \
    driver-columns-gsd41.bin driver-columns-gsd41-asis.bin \
    driver-columns-gsd41.bin.gz driver-columns-gsd41-asis.bin.gz \
    > columns-sha256sums.txt
{ gfortran --version | sed -n 1p; ldd --version | sed -n 1p; } > columns-compiler.txt
