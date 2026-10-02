#!/usr/bin/env bash
# Build actual WRF v4.7.1 module_advect_em and model constants, without edits.
# Run CPU compilation under nice. The caller accepts full WRF bounds and halos.
set -euo pipefail
if [[ $# -ne 2 ]]; then
  echo 'usage: build.sh WRF_SOURCE_ROOT BUILD_DIR' >&2
  exit 2
fi
src=$(realpath "$1")
out=$(realpath -m "$2")
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
( cd "$src" && sha256sum -c "$here/SOURCES.sha256" )
mkdir -p "$out"
cd "$out"
defs=(-Dwrfmodel -DEM_CORE=1 -DNMM_CORE=0 -DRWORDSIZE=4 -DIWORDSIZE=4 -DDWORDSIZE=8 -DLWORDSIZE=4)
base=(-cpp -ffree-form -ffree-line-length-none -fallow-argument-mismatch)
ref=(-O0 -ffp-contract=off -fcheck=all -fbacktrace -g)
nice -n 10 gfortran -c "${base[@]}" "${ref[@]}" "$here/stub_wrf.F90"
nice -n 10 gfortran -c "${base[@]}" "${ref[@]}" "${defs[@]}" "$src/share/module_model_constants.F"
nice -n 10 gfortran -c "${base[@]}" "${ref[@]}" "${defs[@]}" "$src/frame/module_wrf_error.F"
nice -n 10 gfortran -c "${base[@]}" "${ref[@]}" "${defs[@]}" "$src/dyn_em/module_advect_em.F"
nice -n 10 gfortran -c "${base[@]}" "${ref[@]}" "$here/run_advect.F90"
nice -n 10 gfortran -o run_advect stub_wrf.o module_model_constants.o module_wrf_error.o module_advect_em.o run_advect.o
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
  ( cd "$here" && sha256sum stub_wrf.F90 run_advect.F90 build.sh )
  sha256sum run_advect module_advect_em.o
} > oracle-sha256sums.txt
# Independent compiler variants are evidence only. Their outputs are checked
# byte for byte against the reference; neither replaces the reference fixture.
for variant in o2 snan; do
  mkdir -p "$out/$variant"
  (
    cd "$out/$variant"
    if [[ "$variant" == o2 ]]; then
      opt=(-O2 -ftree-vectorize -funroll-loops -ffp-contract=off -fcheck=all)
    else
      opt=("${ref[@]}" -finit-real=snan -finit-integer=-999999 -finit-derived)
    fi
    nice -n 10 gfortran -c "${base[@]}" "${opt[@]}" "$here/stub_wrf.F90"
    nice -n 10 gfortran -c "${base[@]}" "${opt[@]}" "${defs[@]}" "$src/share/module_model_constants.F"
    nice -n 10 gfortran -c "${base[@]}" "${opt[@]}" "${defs[@]}" "$src/frame/module_wrf_error.F"
    nice -n 10 gfortran -c "${base[@]}" "${opt[@]}" "${defs[@]}" "$src/dyn_em/module_advect_em.F"
    nice -n 10 gfortran -c "${base[@]}" "${opt[@]}" "$here/run_advect.F90"
    nice -n 10 gfortran -o run_advect stub_wrf.o module_model_constants.o module_wrf_error.o module_advect_em.o run_advect.o
    nm -u module_advect_em.o | sort > undefined.txt
    sha256sum run_advect module_advect_em.o > oracle-sha256sums.txt
  )
done
echo 'reference and compiler controls built with complete routines and bounds checking'
