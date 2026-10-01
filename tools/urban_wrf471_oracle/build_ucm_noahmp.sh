#!/usr/bin/env bash
# Build and run the UCM Noah-MP coupling oracle: noahmp_urban
# (phys/noahmp/drivers/wrf/module_sf_noahmpdrv.F at noahmp e5c0859, WRF
# v4.7.1) with sf_urban_physics = 1, plus WRF's own option-1 surface-driver
# override block (module_surface_driver.F:3383-3405) extracted verbatim.
#
#   bash tools/urban_wrf471_oracle/build_ucm_noahmp.sh WRF_SRC_DIR WRF_TREE BUILD_DIR [OUT_DIR]
#
# WRF_SRC_DIR: the design stage's pinned flat copy (SOURCES.sha256).
# WRF_TREE:    a v4.7.1 checkout with phys/noahmp at e5c0859, for the three
#              Noah-MP physics modules noahmp_urban's module USEs, the GECROS
#              module they USE, and CAL_MON_DAY; each is pinned by sha256
#              below.
set -euo pipefail
if [[ $# -lt 3 ]]; then
    echo "usage: build_ucm_noahmp.sh WRF_SRC_DIR WRF_TREE BUILD_DIR [OUT_DIR]" >&2
    exit 2
fi
src=$(realpath "$1")
tree=$(realpath "$2")
build=$(realpath -m "$3")
out=$(realpath -m "${4:-$3/fixture}")
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

files=(module_model_constants.F module_wrf_error.F module_sf_urban.F
       module_bep_bem_helper.F module_sf_bem.F module_sf_bep.F module_sf_bep_bem.F
       module_sf_noahmpdrv.F module_surface_driver.F URBPARM.TBL)
for f in "${files[@]}"; do
    want=$(awk -v n="$f" '$2 == n {print $1}' "$src/SOURCES.sha256")
    got=$(sha256sum "$src/$f" | cut -d' ' -f1)
    if [[ -z "$want" || "$want" != "$got" ]]; then
        echo "$f does not match SOURCES.sha256" >&2
        exit 3
    fi
done
declare -A pinned=(
    [phys/noahmp/src/module_sf_noahmplsm.F]=74b475600ad1c999fe43299bcaf4d3e5eaf59b9f861c47fdbff4d69739e3a45b
    [phys/noahmp/src/module_sf_noahmp_glacier.F]=a35d61ea29f660f4f4c6384bc019d4a57ddc8b0c2ca0f3ef7c6e621b79b4dece
    [phys/noahmp/src/module_sf_noahmp_groundwater.F]=ff4b2d7bcb4e948d43af2797e29cbec1e3bb2875b3e9bbf2421c39a6f840cf50
    [phys/module_sf_gecros.F]=ad2864562e95678a25276df82ef96395cca61c3a1bd0ab48ddfb8402902cf2f6
    [phys/module_ra_gfdleta.F]=b6b2e9e006eff0b61f637a30e31a70f21b3c2bd8275182acb3c49aaf3fac0b4a
)
for f in "${!pinned[@]}"; do
    [[ "$(sha256sum "$tree/$f" | cut -d' ' -f1)" == "${pinned[$f]}" ]] || { echo "$f is not the pinned file" >&2; exit 3; }
done

mkdir -p "$build" "$out"
cd "$build"
{
    echo "module module_ra_gfdleta"
    echo "contains"
    sed -n '/^ *SUBROUTINE CAL_MON_DAY/,/^ *END SUBROUTINE CAL_MON_DAY/p' "$tree/phys/module_ra_gfdleta.F"
    echo "end module module_ra_gfdleta"
} > gfdleta_cal_mon_day.F90
[[ $(grep -c 'CAL_MON_DAY' gfdleta_cal_mon_day.F90) -eq 2 ]] || { echo "CAL_MON_DAY extraction failed" >&2; exit 4; }
sed -n '3383,3405p' "$src/module_surface_driver.F" > ucm_sd_overrides.inc
head -1 ucm_sd_overrides.inc | grep -q 'IF(SF_URBAN_PHYSICS.eq.1) THEN' || { echo "override block moved" >&2; exit 4; }
tail -1 ucm_sd_overrides.inc | grep -q 'ENDIF' || { echo "override block end moved" >&2; exit 4; }
fixture=ucm-noahmp.csv
if [[ "${UCM_T2_TEMPERATURE_FIX:-0}" == 1 ]]; then
    # The one named divergence gpuwm carries (gpuwm/core/kernels/urban_ucm.cu,
    # ucm_overrides): WRF's line 3393 converts the UCM's 2 m value as a
    # potential temperature, but module_sf_urban.F:1686 builds it from TS and
    # TA, both absolute temperatures.  This build blends it as the temperature
    # it is and writes a second fixture; nothing else in the block changes.
    before=$(md5sum < ucm_sd_overrides.inc)
    sed -i 's#(TH2_URB2D(i,j)/((1.E5/PSFC(i,j))\*\*RCP))\*FRC_URB2D(I,J)#TH2_URB2D(i,j)*FRC_URB2D(I,J)#' ucm_sd_overrides.inc
    [[ "$(md5sum < ucm_sd_overrides.inc)" != "$before" ]] || { echo "the T2 blend moved; the fix did not apply" >&2; exit 4; }
    [[ $(grep -c 'TH2_URB2D(i,j)\*FRC_URB2D(I,J)' ucm_sd_overrides.inc) -eq 1 ]] || { echo "T2 fix applied more than once" >&2; exit 4; }
    fixture=ucm-noahmp-t2fix.csv
fi

defines="-Dwrfmodel -DEM_CORE=1 -DNMM_CORE=0 -DRWORDSIZE=4 -DIWORDSIZE=4 -DDWORDSIZE=8 -DLWORDSIZE=4"
free="-ffree-form -ffree-line-length-none"
fc() { gfortran -c -O0 -cpp ${defines} ${free} -fallow-argument-mismatch -I. "$@"; }
gfortran -c -O0 ${free} "$here/stub_ucm_noahmp.F90"
fc "$src/module_model_constants.F"
fc "$src/module_wrf_error.F"
fc "$src/module_sf_urban.F"
fc "$src/module_bep_bem_helper.F"
fc "$src/module_sf_bem.F"
fc "$src/module_sf_bep.F"
fc "$src/module_sf_bep_bem.F"
fc "$tree/phys/module_sf_gecros.F"
fc "$tree/phys/noahmp/src/module_sf_noahmplsm.F"
fc "$tree/phys/noahmp/src/module_sf_noahmp_glacier.F"
fc "$tree/phys/noahmp/src/module_sf_noahmp_groundwater.F"
gfortran -c -O0 ${free} gfdleta_cal_mon_day.F90
fc "$src/module_sf_noahmpdrv.F"
gfortran -c -O0 -cpp ${free} -I. "$here/ucm_noahmp_oracle.F90"
gfortran -o run_ucm_noahmp stub_ucm_noahmp.o module_model_constants.o \
    module_wrf_error.o module_sf_urban.o module_bep_bem_helper.o module_sf_bem.o \
    module_sf_bep.o module_sf_bep_bem.o module_sf_gecros.o module_sf_noahmplsm.o \
    module_sf_noahmp_glacier.o module_sf_noahmp_groundwater.o gfdleta_cal_mon_day.o \
    module_sf_noahmpdrv.o ucm_noahmp_oracle.o
cp "$src/URBPARM.TBL" .
./run_ucm_noahmp "$fixture"
cp "$fixture" "$out/"
gzip -n -9 -f "$out/$fixture"
( cd "$out" && sha256sum "$fixture.gz" >> oracle-sha256sums.txt )
echo "ucm noahmp oracle written to $out"
