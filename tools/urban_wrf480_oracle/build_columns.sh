#!/usr/bin/env bash
# Reference-only tree corpus. No engine canopy parity is claimed.
set -euo pipefail
src=$(realpath "$1")
tables=$(realpath "$2")
out=$(realpath -m "$3")
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$out"
python "$here/make_tree_driver.py" "$out/tree_driver.F90"
flags=(-O0 -g -fcheck=all -fno-fast-math -ffp-contract=off
       -ffree-form -ffree-line-length-none -cpp -Dwrfmodel -DEM_CORE=1 -DRWORDSIZE=4)
sha256sum "$src/"{module_sf_urban.F,URBPARM.TBL,URBPARM_LCZ.TBL} > "$out/source-sha256.txt"
printf '%q ' "${flags[@]}" > "$out/flags.txt"
for mode in ref checked snan; do
  mkdir -p "$out/$mode"
  (
    cd "$out/$mode"
    extra=()
    if [[ $mode == checked ]]; then extra=(-ffpe-trap=invalid,zero,overflow); fi
    if [[ $mode == snan ]]; then extra=(-finit-real=snan -ffpe-trap=invalid,zero,overflow); fi
    gfortran "${flags[@]}" "${extra[@]}" -fallow-argument-mismatch -I"$tables" -c "$src/module_sf_urban.F"
    gfortran "${flags[@]}" "${extra[@]}" -I"$tables" -c "$out/tree_driver.F90"
    gfortran "${extra[@]}" -o tree_driver tree_driver.o module_sf_urban.o "$tables/"{module_model_constants.o,module_wrf_error.o,stub_ucm.o}
    cp "$src/"{URBPARM.TBL,URBPARM_LCZ.TBL} .
    ulimit -c 0
    for table in nlcd lcz; do
      for variant in baseline diagnostic prognostic; do
        tree=1; scheme=0; fraction=.2
        if [[ $variant == baseline ]]; then tree=0; fraction=0; fi
        if [[ $variant == prognostic ]]; then scheme=1; fi
        count=3; if [[ $table == lcz ]]; then count=11; fi
        fraction_text=$(printf "${fraction} %.0s" $(seq 1 "$count"))
        set +e
        ./tree_driver "$table" "$variant" "$out/$mode/$table-$variant" -9 -9 -9 -9 -9 -9 -9 -9 -9 "$tree" "$scheme" "$fraction_text" > "$out/$mode/$table-$variant.log" 2>&1
        status=$?
        set -e
        printf '%s %s %s %s\n' "$mode" "$table" "$variant" "$status" >> "$out/exits.txt"
      done
    done
  )
done
