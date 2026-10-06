#!/usr/bin/env bash
# Compile generated full-ARW columns against the unchanged tagged RUC.
set -euo pipefail
if [[ $# -ne 3 ]]; then
    echo 'usage: build_real_columns.sh HRRR_WRF_ROOT COLUMN_DIR {ifort|gfortran}' >&2
    exit 2
fi
source_root=$(realpath "$1")
column_dir=$(realpath "$2")
compiler=$3
tool_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR="${TMPDIR:-$column_dir/tmp}"
mkdir -p "$TMPDIR"
# A source check prevents recording another lineage as this fork's oracle.
printf '%s  %s\n' \
    cb2b54114afc247e16ea03d547a510b19842e2ee012da1202eba19105bef6aab "$source_root/phys/module_sf_ruclsm.F" \
    0d0ee7d243fe673288efafcdcdc470ee028ad8d083648f12225ce2d168f8e87c "$source_root/share/module_model_constants.F" \
    1a13be6f04332dd31dcd1d90a54d00f1ed7c7537722d30fdf2d1081ae66dcd76 "$source_root/run/VEGPARM.TBL" \
    b143d644c1ce118634270a25d08fea7e32490268142c22a151631fd76f69f920 "$source_root/run/SOILPARM.TBL" \
    9c02832a0e4a2ecaf47fcee485539aad95cd732c379c5c258161a88eb3d25ea2 "$source_root/run/GENPARM.TBL" \
    | sha256sum --check
cd "$column_dir"
if [[ "$compiler" == ifort ]]; then
    flags=(-O0 -fp-model precise -free -fpp -extend-source)
    output=oracle-real-intel.csv
elif [[ "$compiler" == gfortran ]]; then
    flags=(-O0 -cpp -ffree-form -ffree-line-length-none -fallow-argument-mismatch)
    output=oracle-real-gnu.csv
else
    echo 'compiler must be ifort or gfortran' >&2
    exit 2
fi
"$compiler" --version > "$compiler-compiler.txt"
"$compiler" -c "${flags[@]}" -DNMM_CORE=0 "$source_root/share/module_model_constants.F"
"$compiler" -c "${flags[@]}" "$tool_dir/column_stubs.F90"
"$compiler" -c "${flags[@]}" -DEM_CORE=1 -Dwrf_chem=0 "$source_root/phys/module_sf_ruclsm.F"
"$compiler" -c "${flags[@]}" run_real.F90
"$compiler" -o "run_real_$compiler" module_model_constants.o column_stubs.o module_sf_ruclsm.o run_real.o
cp "$source_root/run/VEGPARM.TBL" "$source_root/run/SOILPARM.TBL" "$source_root/run/GENPARM.TBL" .
"./run_real_$compiler" "$output"
sha256sum "$source_root/phys/module_sf_ruclsm.F" "$source_root/share/module_model_constants.F" \
    "$tool_dir/column_stubs.F90" run_real.F90 inputs.bin \
    VEGPARM.TBL SOILPARM.TBL GENPARM.TBL "$output" > "$compiler-sha256.txt"
touch "$compiler.done"
