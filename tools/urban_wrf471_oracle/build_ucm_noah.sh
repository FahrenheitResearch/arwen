#!/usr/bin/env bash
# Build and run the UCM Noah-coupling oracle: lsm (module_sf_noahdrv.F,
# WRF v4.7.1, byte-unmodified) with sf_urban_physics = 1.
#
#   bash tools/urban_wrf471_oracle/build_ucm_noah.sh WRF_SRC_DIR GFDLETA_F BUILD_DIR [OUT_DIR]
#
# WRF_SRC_DIR holds the pinned files flat (the design stage's wrf-src/, with
# SOURCES.sha256).  GFDLETA_F is phys/module_ra_gfdleta.F of WRF v4.7.1,
# from which CAL_MON_DAY -- the calendar lsm calls for the UCM's month -- is
# extracted VERBATIM (sed, by its SUBROUTINE/END SUBROUTINE lines) into a
# one-routine module_ra_gfdleta rather than compiling the whole radiation
# scheme; its file hash is pinned below.
#
# The tap.  One statement, CALL ucm_oracle_tap(...), is inserted into a COPY
# of module_sf_noahdrv.F immediately before its first
# `UTYPE_URB = UTYPE_URB2D(I,J)` (:1332, inside lsm's UCM block), so the
# fixture can carry the rural values WRF held at the UCM's entry -- the
# hand-off gpuwm's LSM makes.  Both the pristine and the tapped objects are
# built and run on the same inputs, and the build FAILS unless every output
# column they share is byte-identical.
set -euo pipefail
if [[ $# -lt 3 ]]; then
    echo "usage: build_ucm_noah.sh WRF_SRC_DIR GFDLETA_F BUILD_DIR [OUT_DIR]" >&2
    exit 2
fi
src=$(realpath "$1")
gfdl=$(realpath "$2")
build=$(realpath -m "$3")
out=$(realpath -m "${4:-$3/fixture}")
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
gfdl_sha=b6b2e9e006eff0b61f637a30e31a70f21b3c2bd8275182acb3c49aaf3fac0b4a

files=(module_model_constants.F module_wrf_error.F module_sf_noahlsm.F
       module_sf_noahlsm_glacial_only.F module_sf_urban.F module_bep_bem_helper.F
       module_sf_bem.F module_sf_bep.F module_sf_bep_bem.F module_sf_noahdrv.F
       URBPARM.TBL VEGPARM.TBL SOILPARM.TBL GENPARM.TBL)
for f in "${files[@]}"; do
    want=$(awk -v n="$f" '$2 == n {print $1}' "$src/SOURCES.sha256")
    got=$(sha256sum "$src/$f" | cut -d' ' -f1)
    if [[ -z "$want" || "$want" != "$got" ]]; then
        echo "$f does not match SOURCES.sha256" >&2
        exit 3
    fi
done
if [[ "$(sha256sum "$gfdl" | cut -d' ' -f1)" != "$gfdl_sha" ]]; then
    echo "module_ra_gfdleta.F is not the pinned v4.7.1 file" >&2
    exit 3
fi

mkdir -p "$build" "$out"
cd "$build"
{
    echo "module module_ra_gfdleta"
    echo "contains"
    sed -n '/^ *SUBROUTINE CAL_MON_DAY/,/^ *END SUBROUTINE CAL_MON_DAY/p' "$gfdl"
    echo "end module module_ra_gfdleta"
} > gfdleta_cal_mon_day.F90
[[ $(grep -c 'CAL_MON_DAY' gfdleta_cal_mon_day.F90) -eq 2 ]] || { echo "CAL_MON_DAY extraction failed" >&2; exit 4; }

anchor='UTYPE_URB = UTYPE_URB2D(I,J)'
first=$(grep -n -F "$anchor" "$src/module_sf_noahdrv.F" | head -1 | cut -d: -f1)
[[ "$first" == 1332 ]] || { echo "tap anchor moved (found line $first)" >&2; exit 4; }
tapline='            CALL ucm_oracle_tap(I, J, T1, SHEAT, ETA_KINEMATIC, ETA, SSOIL, ALBEDOK, Q1, SFCTMP, Q2K, SFCPRS, ZLVL, SOLDN, RAINBL(I,J), CHS(I,J), CHS2(I,J), CQS2(I,J), UST(I,J), ZNT(I,J), GLW(I,J))'
awk -v n="$first" -v t="$tapline" 'NR == n {print t} {print}' "$src/module_sf_noahdrv.F" > noahdrv_tapped.F
[[ $(diff "$src/module_sf_noahdrv.F" noahdrv_tapped.F | grep -c '^>') -eq 1 ]] || { echo "tap insertion is not exactly one line" >&2; exit 4; }

defines="-Dwrfmodel -DEM_CORE=1 -DNMM_CORE=0 -DRWORDSIZE=4 -DIWORDSIZE=4 -DDWORDSIZE=8 -DLWORDSIZE=4"
free="-ffree-form -ffree-line-length-none"
fc() { gfortran -c -O0 -cpp ${defines} ${free} -fallow-argument-mismatch -I. "$@"; }
build_one() {   # $1 = tag, $2 = noahdrv source
    local tag=$1 drv=$2
    mkdir -p "$tag"
    ( cd "$tag"
      gfortran -c -O0 ${free} "$here/stub_ucm_noah.F90"
      fc "$src/module_model_constants.F"
      fc "$src/module_wrf_error.F"
      fc "$src/module_sf_noahlsm.F"
      fc "$src/module_sf_noahlsm_glacial_only.F"
      fc "$src/module_sf_urban.F"
      fc "$src/module_bep_bem_helper.F"
      fc "$src/module_sf_bem.F"
      fc "$src/module_sf_bep.F"
      fc "$src/module_sf_bep_bem.F"
      gfortran -c -O0 ${free} ../gfdleta_cal_mon_day.F90
      cp "$drv" module_sf_noahdrv.F
      fc module_sf_noahdrv.F
      gfortran -c -O0 ${free} -I. "$here/ucm_noah_oracle.F90"
      gfortran -o run_ucm_noah stub_ucm_noah.o module_model_constants.o \
          module_wrf_error.o module_sf_noahlsm.o module_sf_noahlsm_glacial_only.o \
          module_sf_urban.o module_bep_bem_helper.o module_sf_bem.o module_sf_bep.o \
          module_sf_bep_bem.o gfdleta_cal_mon_day.o module_sf_noahdrv.o ucm_noah_oracle.o
      cp "$src/URBPARM.TBL" "$src/VEGPARM.TBL" "$src/SOILPARM.TBL" "$src/GENPARM.TBL" .
      ./run_ucm_noah "ucm-noah.csv" )
}
build_one pristine "$src/module_sf_noahdrv.F"
build_one tapped "$build/noahdrv_tapped.F"

# Every column except the tap's must agree byte for byte.
python3 - "$build/pristine/ucm-noah.csv" "$build/tapped/ucm-noah.csv" <<'PY'
import csv, sys
a = list(csv.DictReader(open(sys.argv[1])))
b = list(csv.DictReader(open(sys.argv[2])))
assert len(a) == len(b) and a, "row counts differ"
keys = [k for k in a[0] if not k.startswith("tap")]
bad = [(i, k) for i, (ra, rb) in enumerate(zip(a, b)) for k in keys if ra[k] != rb[k]]
if bad:
    sys.exit(f"the tap perturbed {len(bad)} words, first {bad[:3]}")
urban = sum(1 for r in b if r["tapped"] == "1")
print(f"pristine == tapped on {len(keys)} shared columns x {len(a)} rows; {urban} rows tapped")
PY
cp tapped/ucm-noah.csv "$out/ucm-noah.csv"
gzip -n -9 -f "$out/ucm-noah.csv"
( cd "$out" && sha256sum ucm-noah.csv.gz >> oracle-sha256sums.txt )
echo "ucm noah oracle written to $out"
