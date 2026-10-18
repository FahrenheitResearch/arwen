#!/usr/bin/env bash
# Control: is the WRF 4.6.1 / 4.7.1 difference in the mosaic UCM families
# exactly the three IMP_SCHEME=2 water-retention lines of module_sf_urban.F?
#
# Starts from a finished 4.6.1 oracle build (build.sh output, built with the
# same NOAH_MOSAIC_CR_LIBM setting as the 4.7.1 build it is compared with),
# copies it, recompiles a copy of v4.6.1 module_sf_urban.F in which only
# lines 1035, 1287 and 1289 take v4.7.1's RHOO/1000. vapour-flux-to-depth
# conversion, relinks run_mosaic_ucm and reruns it. compare_raw.py of the
# result against the 4.7.1 build must show no differing word.
#
# The pristine WRF tree is only read; the patched copy lives in OUT_DIR.
# Usage: unit_fix_control.sh WRF461_SOURCE_ROOT BUILD461_DIR OUT_DIR
set -euo pipefail
src=$(realpath "$1"); b461=$(realpath "$2"); out=$(realpath -m "$3")
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
urban="${src}/phys/module_sf_urban.F"
want=$(awk '$2=="phys/module_sf_urban.F"{print $1}' "${here}/SOURCES.sha256")
have=$(sha256sum "${urban}" | cut -d' ' -f1)
[[ "${have}" == "${want}" ]] || { echo "module_sf_urban.F is not the pinned v4.6.1 file" >&2; exit 3; }
rm -rf "${out}"; mkdir -p "${out}"
cp -a "${b461}/." "${out}/"
rm -rf "${out}/fixtures"
cd "${out}"
cp "${urban}" module_sf_urban_unitfix.F
sed -i \
  -e '1035s|(RAIN1-FLXHUMRP)\*DELT|(RAIN1-FLXHUMRP*RHOO/1000.)*DELT|' \
  -e '1287s|(RAIN1-FLXHUMBP)\*DELT|(RAIN1-FLXHUMBP*RHOO/1000.)*DELT|' \
  -e '1289s|(RAIN1-FLXHUMGP)\*DELT|(RAIN1-FLXHUMGP*RHOO/1000.)*DELT|' \
  module_sf_urban_unitfix.F
diff "${urban}" module_sf_urban_unitfix.F > unitfix.diff || true
[[ $(grep -c '^>' unitfix.diff) == 3 ]] || { cat unitfix.diff; echo "expected exactly 3 changed lines" >&2; exit 4; }
defines="-Dwrfmodel -DEM_CORE=1 -DNMM_CORE=0 -DRWORDSIZE=4 -DIWORDSIZE=4 -DDWORDSIZE=8 -DLWORDSIZE=4"
gfortran -c -O0 -cpp ${defines} -ffree-form -ffree-line-length-none -fallow-argument-mismatch \
    -I "${out}" -o module_sf_urban.o module_sf_urban_unitfix.F
objs=(stub_wrf.o module_model_constants.o module_wrf_error.o module_sf_urban.o
      module_bep_bem_helper.o module_sf_bep.o module_sf_bem.o module_sf_bep_bem.o
      module_sf_noahlsm.o module_sf_noahlsm_glacial_only.o module_ra_gfdleta_cal_mon_day.o
      module_sf_noahdrv.o oracle_io.o)
[[ -f cr_libm_control.o ]] && objs+=(cr_libm_control.o)
gfortran -o run_mosaic_ucm "${objs[@]}" run_mosaic_ucm.o
mkdir -p fixtures/mosaic_ucm
./run_mosaic_ucm fixtures/mosaic_ucm > run_mosaic_ucm-stdout.txt
echo "unit-fix control written to ${out}/fixtures/mosaic_ucm"
