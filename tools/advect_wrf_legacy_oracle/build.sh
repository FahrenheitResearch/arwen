#!/usr/bin/env bash
# Build the HRRR fork (NOAA-EMC/HRRR 40ee6058c, WRFV3.9) module_advect_em
# behind the same binary driver as the WRF 4.7.1 oracle.  The only source
# transformation is the hybrid-mass macro expansion of expand_hybrid.py,
# which the receipt records.  Run CPU compilation under nice.
set -euo pipefail
if [[ $# -ne 2 ]]; then
  echo 'usage: build.sh FORK_SOURCE_ROOT BUILD_DIR   (root holds dyn_em/ share/ frame/)' >&2
  exit 2
fi
src=$(realpath "$1")
out=$(realpath -m "$2")
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
driver="$here/../advect_wrf471_oracle/run_advect.F90"
( cd "$src" && sha256sum -c "$here/SOURCES.sha256" )
mkdir -p "$out"
cd "$out"
python3 "$here/expand_hybrid.py" "$src/dyn_em/module_advect_em.F" "$out/module_advect_em.hybrid.F" \
  --receipt "$out/hybrid-expansion.json"
defs=(-Dwrfmodel -DEM_CORE=1 -DNMM_CORE=0 -DRWORDSIZE=4 -DIWORDSIZE=4 -DDWORDSIZE=8 -DLWORDSIZE=4)
base=(-cpp -ffree-form -ffree-line-length-none -fallow-argument-mismatch)
ref=(-O0 -ffp-contract=off -fcheck=all -fbacktrace -g)
nice -n 10 gfortran -c "${base[@]}" "${ref[@]}" "$here/stub_wrf.F90"
nice -n 10 gfortran -c "${base[@]}" "${ref[@]}" "${defs[@]}" "$src/share/module_model_constants.F"
nice -n 10 gfortran -c "${base[@]}" "${ref[@]}" "${defs[@]}" "$out/module_advect_em.hybrid.F" -o module_advect_em.o
nice -n 10 gfortran -c "${base[@]}" "${ref[@]}" "$driver"
nice -n 10 gfortran -o run_advect stub_wrf.o module_model_constants.o module_advect_em.o run_advect.o
nm -u module_advect_em.o | sort > undefined.txt
if grep -q '_ZGV' undefined.txt; then
  echo 'reference uses vector libm; compiled arithmetic would not be scalar WRF' >&2
  exit 4
fi
for routine in advect_u advect_v advect_w advect_scalar advect_scalar_pd advect_scalar_mono; do
  nm --defined-only module_advect_em.o | grep "__module_advect_em_MOD_${routine}$" > "symbol-${routine}.txt"
done
{
  gfortran --version | head -1
  ldd --version | head -1
  printf 'reference flags:'
  printf ' %s' "${base[@]}" "${ref[@]}" "${defs[@]}"
  printf '\n'
} > compiler.txt
( cd "$src" && sha256sum dyn_em/module_advect_em.F share/module_model_constants.F frame/module_wrf_error.F ) > source-sha256sums.txt
{
  ( cd "$here" && sha256sum stub_wrf.F90 expand_hybrid.py build.sh )
  ( cd "$here/../advect_wrf471_oracle" && sha256sum run_advect.F90 )
  sha256sum module_advect_em.hybrid.F run_advect module_advect_em.o
} > oracle-sha256sums.txt
echo 'HRRR fork advection reference built with complete routines and bounds checking'
