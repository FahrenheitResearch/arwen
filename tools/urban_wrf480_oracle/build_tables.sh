#!/usr/bin/env bash
# Build unchanged release schemes, with fresh module state for each table.
set -euo pipefail
src=$(realpath "$1")
build=$(realpath -m "$2")
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$build"
flags=(-O0 -fno-fast-math -ffp-contract=off -ffree-form -ffree-line-length-none
       -cpp -Dwrfmodel -DEM_CORE=1 -DNMM_CORE=0 -DRWORDSIZE=4 -DIWORDSIZE=4
       -DDWORDSIZE=8 -DLWORDSIZE=4 -fcheck=all -fbacktrace)
gfortran --version > "$build/compiler.txt"
printf '%q ' "${flags[@]}" > "$build/flags.txt"
for tag in v4.6.1 v4.7.0 v4.7.1 v4.8.0; do
  mkdir -p "$build/$tag"
  (
    cd "$build/$tag"
    sha256sum "$src/$tag/"{module_sf_urban.F,module_model_constants.F,module_wrf_error.F,URBPARM.TBL,URBPARM_LCZ.TBL} > source-sha256.txt
    gfortran "${flags[@]}" -c "$here/../urban_wrf471_oracle/stub_ucm.F90"
    gfortran "${flags[@]}" -c "$src/$tag/module_model_constants.F"
    gfortran "${flags[@]}" -fallow-argument-mismatch -c "$src/$tag/module_wrf_error.F"
    gfortran "${flags[@]}" -fallow-argument-mismatch -c "$src/$tag/module_sf_urban.F"
    extra=()
    if [[ $tag == v4.8.0 ]]; then extra=(-DWRF480); fi
    gfortran "${flags[@]}" "${extra[@]}" -c "$here/table_driver.F90"
    gfortran -o table_driver ./*.o
    cp "$src/$tag/"{URBPARM.TBL,URBPARM_LCZ.TBL} .
    for opt in 1 2 3; do
      for lcz in 0 1; do
        ./table_driver "$opt" "$lcz" > "opt${opt}_lcz${lcz}.csv"
      done
    done
  )
done
