#!/usr/bin/env bash
# Make HRRR's cold-start guess the way the tag's
# scripts/conus/exhrrr_makeguess.sh does: ungrib the RAP native-level file,
# metgrid it onto the HRRR grid, run real. Runs inside the "full" image.
#
#   run_makeguess.sh <valid YYYYMMDDHH> <RAP grib2 file> <fix dir> <work dir> [ranks]
#
# The executables, Vtable, METGRID.TBL, namelists and physics tables are the
# tag's own, unedited; the namelist date edits below are the ones the tag's
# script makes with sed. Differences from the tag's script, all of them
# outside the programs:
#   * it runs one valid time, the one named on the command line, from the RAP
#     analysis of that hour; the tag's script takes the RAP file one hour
#     before its cycle, a name NCEP does not publish
#     (rap.tHHz.awp130bgrbf00df), and falls back to older RAP forecasts;
#   * the rank count is this host's, not 128.
# Like the tag's script it repacks the GRIB file to simple packing with
# wgrib2 first: WPS 3.9.1's ungrib misreads the published complex packing
# (MEASURED 2026-10-03: metgrid pressures of 1.3e11 Pa without the repack).
set -euo pipefail
valid="$1"; rap_grib="$2"; FIXhrrr="$3"; DATA="$4"; ranks="${5:-8}"
PARMhrrr=/opt/hrrr/parm/conus
EXEChrrr=/opt/hrrr/exec
stamp="${valid:0:4}-${valid:4:2}-${valid:6:2}_${valid:8:2}:00:00"
mkdir -p "$DATA" && cd "$DATA"
ulimit -s unlimited || true

# --- ungrib -----------------------------------------------------------------
ln -sf "$PARMhrrr/hrrr_vtable" Vtable
sed -e "s/\(start_date\)[[:blank:]]*=[[:blank:]]*'[^']*'/\1 = '$stamp'/" \
    -e "s/\(end_date\)[[:blank:]]*=[[:blank:]]*'[^']*'/\1 = '$stamp'/" \
    -e "s/\(interval_seconds\)[[:blank:]]*=[[:blank:]]*[[:digit:]]\{1,\}/\1 = 10800/" \
    -e "s/\(prefix\)[[:blank:]]*=[[:blank:]]*'[[:alnum:]]\{1,\}'/\1 = 'RAP'/" \
    "$PARMhrrr/hrrr_namelist.wps" > namelist.wps
WGRIB2="${WGRIB2:-/opt/wgrib2/bin/wgrib2}"
cp "$rap_grib" GRIBFILE_COMPLEX.AAA
# script: $WGRIB2 GRIBFILE_COMPLEX.AAA -set_grib_type s -grib_out GRIBFILE_SIMPLE.AAA
"$WGRIB2" GRIBFILE_COMPLEX.AAA -set_grib_type s -grib_out GRIBFILE_SIMPLE.AAA > wgrib2.out 2>&1 \
  || { tail -20 wgrib2.out; exit 1; }
ln -sf GRIBFILE_SIMPLE.AAA GRIBFILE.AAA
"$EXEChrrr/hrrr_wps_ungrib" > ungrib.out 2>&1 || { tail -30 ungrib.out; exit 1; }
test -s "RAP:${stamp:0:13}" || { echo "ungrib wrote no RAP:${stamp:0:13}"; tail -30 ungrib.out; exit 1; }

# --- metgrid ----------------------------------------------------------------
ln -sf "$FIXhrrr/hrrr_geo_em.d01.nc" geo_em.d01.nc
ln -sf "$PARMhrrr/hrrr_METGRID.TBL" METGRID.TBL
sed -i -e "s/\(fg_name\)[[:blank:]]*=.*/\1 = 'RAP',/" namelist.wps
mpiexec -n "$ranks" "$EXEChrrr/hrrr_wps_metgrid" > metgrid.out 2>&1 || { tail -30 metgrid.out; exit 1; }
test -s "met_em.d01.$stamp.nc" || { echo "metgrid wrote no met_em.d01.$stamp.nc"; tail -30 metgrid.log metgrid.out; exit 1; }

# --- real -------------------------------------------------------------------
# script: the WRF_DAT_FILES list, linked without their hrrr_run_ prefix.
for name in LANDUSE.TBL RRTM_DATA RRTM_DATA_DBL RRTMG_LW_DATA RRTMG_LW_DATA_DBL \
            RRTMG_SW_DATA RRTMG_SW_DATA_DBL VEGPARM.TBL GENPARM.TBL SOILPARM.TBL \
            MPTABLE.TBL URBPARM.TBL URBPARM_UZE.TBL ETAMPNEW_DATA \
            ETAMPNEW_DATA.expanded_rain ETAMPNEW_DATA.expanded_rain_DBL \
            ETAMPNEW_DATA_DBL co2_trans ozone.formatted ozone_lat.formatted \
            ozone_plev.formatted tr49t85 tr49t67 tr67t85 grib2map.tbl gribmap.txt \
            aerosol.formatted aerosol_lat.formatted aerosol_lon.formatted \
            aerosol_plev.formatted bulkdens.asc_s_0_03_0_9 bulkradii.asc_s_0_03_0_9 \
            capacity.asc CCN_ACTIVATE.BIN coeff_p.asc coeff_q.asc constants.asc \
            kernels.asc_s_0_03_0_9 kernels_z.asc masses.asc termvels.asc \
            wind-turbine-1.tbl freezeH2O.dat qr_acr_qg.dat qr_acr_qs.dat \
            eclipse_besselian_elements.dat; do
  test -s "$PARMhrrr/hrrr_run_$name" || { echo "missing $PARMhrrr/hrrr_run_$name"; exit 1; }
  ln -sf "$PARMhrrr/hrrr_run_$name" "$name"
done
rm -f namelist.input
Y="${valid:0:4}"; M="${valid:4:2}"; D="${valid:6:2}"; H="${valid:8:2}"
sed -e "s/\(run_days\)[[:blank:]]*=[[:blank:]]*[[:digit:]]\{1,\}/\1 = 0/" \
    -e "s/\(run_hours\)[[:blank:]]*=[[:blank:]]*[[:digit:]]\{1,\}/\1 = 0/" \
    -e "s/\(start_year\)[[:blank:]]*=[[:blank:]]*[[:digit:]]\{4\}/\1 = $Y/" \
    -e "s/\(start_month\)[[:blank:]]*=[[:blank:]]*[[:digit:]]\{2\}/\1 = $M/" \
    -e "s/\(start_day\)[[:blank:]]*=[[:blank:]]*[[:digit:]]\{2\}/\1 = $D/" \
    -e "s/\(start_hour\)[[:blank:]]*=[[:blank:]]*[[:digit:]]\{2\}/\1 = $H/" \
    -e "s/\(end_year\)[[:blank:]]*=[[:blank:]]*[[:digit:]]\{4\}/\1 = $Y/" \
    -e "s/\(end_month\)[[:blank:]]*=[[:blank:]]*[[:digit:]]\{2\}/\1 = $M/" \
    -e "s/\(end_day\)[[:blank:]]*=[[:blank:]]*[[:digit:]]\{2\}/\1 = $D/" \
    -e "s/\(end_hour\)[[:blank:]]*=[[:blank:]]*[[:digit:]]\{2\}/\1 = $H/" \
    -e "s/\(interval_seconds\)[[:blank:]]*=[[:blank:]]*[[:digit:]]\{1,\}/\1 = 10800/" \
    "$PARMhrrr/hrrr_real.nl" > namelist.input
mpiexec -n "$ranks" "$EXEChrrr/hrrr_wrfarw_real" > real.out 2>&1 || { tail -30 rsl.error.0000 real.out; exit 1; }
grep -q "SUCCESS COMPLETE REAL_EM INIT" rsl.out.0000 || { tail -40 rsl.error.0000; exit 1; }
ls -l wrfinput_d01 wrfbdy_d01 2>/dev/null || ls -l wrfinput_d01
