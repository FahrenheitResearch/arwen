#!/usr/bin/env bash
# Build the legacy RRTMG column oracle against UNMODIFIED WRF v4.6.1 text.
#
# Arms (each in BUILD_DIR/<arm>/, each its own executable rad_column_oracle):
#   stock     WRF v4.6.1, -O0, no FP contraction (the 0 ULP referee)
#   fork      the same, but aer_opt = 3 optics from the operational HRRR fork
#             (NOAA-EMC/HRRR v4.1.21, fetched from its public commit and
#             sha256-pinned by tools/hrrr_radiation_driver_oracle/build.py)
#   stock-o2  WRF v4.6.1 with WRF's own GNU flags (-O2 -ftree-vectorize
#             -funroll-loops), for reference only: WRF against itself
#   fork-cloud  the fork arm with the operational branch's two wrapper cloud
#             rules in place of v4.6.1's (rrtmg_cloud_optics_form =
#             noaa_wrf39; tools/fork_forms_oracle/cloud_form_transplant.py)
#
# cpp state mirrors WRF's group build for this bundle: EM_CORE=1,
# WRF_CHEM=0, HWRF=0, RWORDSIZE=4 (kind_rb = FP32), -fconvert=big-endian
# (BYTESWAPIO; the RRTMG_*_DATA lookup reads); the oracle's own input and
# output files pin little-endian per unit.
#
# usage: build.sh WRF461_SOURCE_ROOT BUILD_DIR [ARMS...]
set -euo pipefail
src=$(realpath "$1"); out=$(realpath -m "$2"); shift 2
arms=("$@"); [[ ${#arms[@]} -gt 0 ]] || arms=(stock fork stock-o2)
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PY=${PYTHON:-python3}
COMMON="-cpp -ffree-form -ffree-line-length-none -fconvert=big-endian \
  -fallow-argument-mismatch -DEM_CORE=1 -DWRF_CHEM=0 -DHWRF=0 \
  -DRWORDSIZE=4 -DIWORDSIZE=4 -DDWORDSIZE=8 -DLWORDSIZE=4"
for arm in "${arms[@]}"; do
  d="$out/$arm"; mkdir -p "$d"; cd "$d"
  case "$arm" in
    stock|fork|fork-cloud) OPT="-O0 -ffp-contract=off" ;;
    stock-o2)   OPT="-O2 -ftree-vectorize -funroll-loops" ;;
    *) echo "unknown arm $arm" >&2; exit 2 ;;
  esac
  FFLAGS="$OPT $COMMON"
  "$PY" "$here/build_units.py" "$arm" "$src" "$out/fork-src" "$d"
  gfortran -c $FFLAGS "$here/stub_wrf_col.F90"
  rrtmg="$src/phys"
  if [[ $arm == fork-cloud ]]; then
    "$PY" "$here/../fork_forms_oracle/fetch_fork.py" "$out/fork-forms-src" > /dev/null
    "$PY" "$here/../fork_forms_oracle/cloud_form_transplant.py" "$src" "$out/fork-forms-src" "$d"
    rrtmg="$d"
  fi
  gfortran -c $FFLAGS -I "$d" "$rrtmg/module_ra_rrtmg_lw.F"
  gfortran -c $FFLAGS -I "$d" "$rrtmg/module_ra_rrtmg_sw.F"
  gfortran -c $FFLAGS -I "$d" raddrv_extract.F90
  objs="stub_wrf_col.o module_ra_rrtmg_lw.o module_ra_rrtmg_sw.o raddrv_extract.o aer_unit.o"
  if [[ $arm == fork || $arm == fork-cloud ]]; then
    gfortran -c $FFLAGS -I "$d" aer_unit.F90
    gfortran -c $FFLAGS -DFORK_AER -I "$d" "$here/rad_column_oracle.F90"
  else
    gfortran -c $FFLAGS -I "$d" "$src/phys/module_ra_aerosol.F"
    gfortran -c $FFLAGS -I "$d" aer_unit.F90
    objs="$objs module_ra_aerosol.o"
    gfortran -c $FFLAGS -I "$d" "$here/rad_column_oracle.F90"
  fi
  gcc -O0 -c "$here/pad_malloc.c" -o pad_malloc.o
  gfortran -o rad_column_oracle $objs rad_column_oracle.o pad_malloc.o
  cp -f "$src/run/RRTMG_LW_DATA" "$src/run/RRTMG_SW_DATA" \
        "$src/run/ozone.formatted" "$src/run/ozone_lat.formatted" \
        "$src/run/ozone_plev.formatted" .
  { gfortran --version | head -1; echo "flags: $FFLAGS";
    sha256sum "$here"/*.F90 "$here"/*.py "$here"/*.c "$here"/build.sh RRTMG_LW_DATA RRTMG_SW_DATA \
      ozone.formatted ozone_lat.formatted ozone_plev.formatted raddrv_extract.F90 aer_unit.F90;
    echo "vfmadd/vfmsub in the executable: $(objdump -d rad_column_oracle | grep -c -E 'vfn?m(add|sub)' || true)";
  } > build-receipt.txt
  echo "built $d/rad_column_oracle"
done
