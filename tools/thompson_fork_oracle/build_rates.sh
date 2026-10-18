#!/usr/bin/env bash
# Build the rate-instrumented fork column driver beside a pristine oracle
# build (tools/thompson_fork_oracle/build.sh's BUILD_DIR), sharing its
# tables.  The instrumented module only adds WRITE statements;
# fork_stage_parity.py refuses unless both drivers return identical bytes.
#
# usage: ./build_rates.sh FORK_PHYS_DIR BUILD_DIR RATES_DIR
set -euo pipefail
fc=${FC:-gfortran}
opt_flags=${OPT_FLAGS:--O2 -fno-tree-vectorize}
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
phys=$(realpath "$1"); build=$(realpath "$2"); rates=$(realpath -m "$3")
echo "4d60011188443eb432294f7693beb64bdbc8f812541a15c7800013060c877283  $phys/module_mp_thompson.F" | sha256sum -c -
mkdir -p "$rates"; cd "$rates"
python3 "$here/instrument_fork_rates.py" "$phys/module_mp_thompson.F" \
  module_mp_thompson_rates.F schema.json
$fc -c $opt_flags -ffree-form -ffree-line-length-none "$here/stub_wrf.F90"
$fc -c $opt_flags -ffree-form -ffree-line-length-none "$phys/module_mp_radar.F"
$fc -c $opt_flags -cpp -DWRF_CHEM=0 -ffree-form -ffree-line-length-none \
  module_mp_thompson_rates.F -o module_mp_thompson.o
$fc -c $opt_flags -ffree-form -ffree-line-length-none "$here/run_columns_fork.F90"
$fc $opt_flags -o run_columns_fork stub_wrf.o module_mp_radar.o \
  module_mp_thompson.o run_columns_fork.o
if nm -D run_columns_fork 2>/dev/null | grep -q '_ZGV'; then
  echo "libmvec SIMD math linked; refused" >&2; exit 3
fi
for t in qr_acr_qg.dat qr_acr_qs.dat freezeH2O.dat CCN_ACTIVATE.BIN; do
  ln -sf "$build/$t" "$t"
done
echo "rates build ready: $rates"
