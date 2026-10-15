#!/usr/bin/env bash
# Evidence arm: the unmodified WRF file at WRF's own GNU physics flags
# (-O2 -ftree-vectorize -funroll-loops, which pull in glibc's libmvec vector
# math in vectorised loops), to show how far stock WRF moves from the -O0
# scalar reference by compile flags alone.  Never graded against WOOF.
# Run after build.sh; reuses build.sh's stock objects for everything but WDM6.
# usage: measure_o2.sh WRF_SOURCE_ROOT BUILD_DIR FIXTURE_DIR OUT_ROOT
set -euo pipefail
source_root=$(realpath "$1"); build_dir=$(realpath "$2"); fixture=$3; out=$4
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
wdm6_source="${source_root}/phys/module_mp_wdm6.F"
echo "78b7c148f7c29b55670f137636d632aba040bf338d8214e1a5dd191dacf67d70  ${wdm6_source}" \
    | sha256sum -c --quiet
mkdir -p "${build_dir}/stockO2"
cd "${build_dir}/stockO2"
cp ../stock/*.mod ../stock/stub_wrf_error.o ../stock/module_model_constants.o \
   ../stock/module_mp_radar.o ../stock/libmassv.o .
gfortran -c -O2 -ftree-vectorize -funroll-loops -cpp -ffree-form \
    -ffree-line-length-none -DRWORDSIZE=4 -I . -o module_mp_wdm6.o "${wdm6_source}"
gfortran -c -O0 -cpp -ffree-form -ffree-line-length-none -DRWORDSIZE=4 -I . \
    "${script_dir}/run_wdm6.F90"
gfortran -o ../run_wdm6_stockO2 stub_wrf_error.o module_model_constants.o \
    module_mp_radar.o libmassv.o module_mp_wdm6.o run_wdm6.o
nm -u module_mp_wdm6.o | grep _ZGV || true
cd - > /dev/null
mkdir -p "${out}/stockO2"
for f in "${fixture}"/*.in; do
    n=$(basename "$f" .in)
    "${build_dir}/run_wdm6_stockO2" "$f" "${out}/stockO2/${n}.out" \
        "${out}/stockO2/constants-${n}.txt"
done
echo "stockO2 arm written to ${out}/stockO2"
