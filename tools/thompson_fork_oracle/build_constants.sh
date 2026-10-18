#!/usr/bin/env bash
# Dump the HRRR v4.1.21 fork's derived Thompson constants (thompson_init and
# radar_init) as exact bit patterns, beside a pristine oracle build
# (build.sh's BUILD_DIR, whose tables thompson_init reads).
#
# usage: ./build_constants.sh FORK_PHYS_DIR BUILD_DIR OUT_DIR
# writes OUT_DIR/fork_constants.txt; make_constants_header.py turns it into
# gpuwm/core/kernels/thompson_fork_constants.cuh.
set -euo pipefail
fc=${FC:-gfortran}
opt_flags=${OPT_FLAGS:--O2 -fno-tree-vectorize}
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
phys=$(realpath "$1"); build=$(realpath "$2"); out=$(realpath -m "$3")
echo "4d60011188443eb432294f7693beb64bdbc8f812541a15c7800013060c877283  $phys/module_mp_thompson.F" | sha256sum -c -
mkdir -p "$out"; cd "$out"
python3 "$here/make_dump_module.py" "$phys/module_mp_thompson.F" \
  module_mp_thompson_dump.F
$fc -c $opt_flags -ffree-form -ffree-line-length-none "$here/stub_wrf.F90"
$fc -c $opt_flags -ffree-form -ffree-line-length-none "$phys/module_mp_radar.F"
$fc -c $opt_flags -cpp -DWRF_CHEM=0 -ffree-form -ffree-line-length-none \
  module_mp_thompson_dump.F -o module_mp_thompson.o
$fc -c $opt_flags -ffree-form -ffree-line-length-none "$here/dump_constants.F90"
$fc $opt_flags -o dump_constants stub_wrf.o module_mp_radar.o \
  module_mp_thompson.o dump_constants.o
for t in qr_acr_qg.dat qr_acr_qs.dat freezeH2O.dat CCN_ACTIVATE.BIN; do
  ln -sf "$build/$t" "$t"
done
GFORTRAN_CONVERT_UNIT='big_endian:20' ./dump_constants
wc -l fork_constants.txt
