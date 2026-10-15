#!/usr/bin/env bash
# Adversarial column sets for the WSM6 WRF v4.6.1 oracle: build the oracle
# (../build.sh, which also reruns the committed column set), build a second
# WRF driver that runs with the CPU's flush-to-zero and denormals-are-zero
# modes on, write the refute column sets, and run WRF over them.  CPU only.
#
#   bash run_refute.sh WRF_SOURCE_ROOT WORK_DIR
#
# Then, on a GPU (through the box's GPU mutex), grade.sh WORK_DIR.
#
# Sets (make_refute_columns.py): fuzz45 (24 targeted + 360 random columns),
# sub45 (64 random columns with subnormal inputs), deep80, nz64, nz65, nz2,
# nz3 (random columns at those level counts; the launcher picks the tier).
#
# The FTZ+DAZ driver is evidence for the default build only: CuPy compiles
# every module with -ftz=true, and WRF's own words under FTZ+DAZ are what
# that build should reproduce if flushing is the only difference.  It is
# not the reference; the -O0 build is.
set -euo pipefail
if [[ $# -ne 2 ]]; then
    echo "usage: run_refute.sh WRF_SOURCE_ROOT WORK_DIR" >&2
    exit 2
fi
src=$(realpath "$1")
work=$(realpath -m "$2")
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$work"
bash "$here/../build.sh" "$src" "$work/oracle" > "$work/oracle.log" 2>&1

# --- WRF under FTZ+DAZ (same sources, same -O0 flags, one call added) -------
ftz=$work/ftz
mkdir -p "$ftz"
gcc -c -O0 "$here/ftz_daz.c" -o "$ftz/ftz_daz.o"
sed 's/^  if (kind(1.0_kind_phys) \/= 4)/  call set_ftz_daz()\n  if (kind(1.0_kind_phys) \/= 4)/' \
    "$here/../run_wsm6_columns.F90" > "$ftz/run_wsm6_columns_ftz.F90"
grep -q 'call set_ftz_daz()' "$ftz/run_wsm6_columns_ftz.F90"
fl=(-O0 -cpp -DRWORDSIZE=4 -ffree-form -ffree-line-length-none -J "$ftz" -I "$ftz")
objs=()
for f in phys/ccpp_kind_types.F share/module_model_constants.F \
         phys/physics_mmm/module_libmassv.F90 phys/physics_mmm/mp_radar.F90 \
         phys/physics_mmm/mp_wsm6.F90 phys/physics_mmm/mp_wsm6_effectRad.F90 \
         phys/module_mp_wsm6.F; do
    o=$ftz/$(basename "${f%.*}").o
    gfortran -c "${fl[@]}" -o "$o" "$src/$f"
    objs+=("$o")
done
gfortran -c "${fl[@]}" -o "$ftz/run_ftz.o" "$ftz/run_wsm6_columns_ftz.F90"
gfortran -o "$ftz/run_wsm6_columns_ftz" "${objs[@]}" "$ftz/run_ftz.o" "$ftz/ftz_daz.o"

# --- the sets, and WRF over them ------------------------------------------------
python3 "$here/make_refute_columns.py" "$work/sets"
FULL=("0 5 24" "1 5 24" "0 20 10" "1 20 10" "0 37.5 8" "1 37.5 8" "0 60 6" "1 60 6"
      "0 90 5" "1 90 5" "0 120 4" "1 120 4" "0 121 4" "1 121 4" "0 180 4" "1 180 4"
      "0 600 3" "1 600 3" "0 1800 2" "1 1800 2")
SHORT=("0 20 8" "1 20 8" "0 60 5" "1 60 5" "0 180 4" "1 180 4" "0 600 3" "1 600 3")
run_set() {   # set name, then "hail dt nsteps" triples
    local s=$1; shift
    for r in "$@"; do
        set -- $r
        tag=$(python3 -c "print(f'h$1-dt{float($2):g}-n$3')")
        "$work/oracle/O0/run_wsm6_columns" "$work/sets/$s/columns.bin" \
            "$work/sets/$s/wrf-O0-$tag.bin" "$1" "$2" "$3"
        "$ftz/run_wsm6_columns_ftz" "$work/sets/$s/columns.bin" \
            "$work/sets/$s/wrf-FTZ-$tag.bin" "$1" "$2" "$3"
    done
}
run_set fuzz45 "${FULL[@]}"
run_set sub45 "${FULL[@]}"
for s in deep80 nz64 nz65 nz2 nz3; do run_set "$s" "${SHORT[@]}"; done
echo "refute sets written and run under $work/sets"
