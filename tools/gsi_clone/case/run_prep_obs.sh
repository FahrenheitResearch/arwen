#!/usr/bin/env bash
# Prepare the two cloud-analysis observation files HRRR's analysis reads
# beside the BUFR dumps, the way the tag's scripts do: the MRMS reflectivity
# mosaic on the model grid (scripts/conus/exhrrr_prep_radar.sh) and the NASA
# LaRC cloud product (scripts/conus/exhrrr_prep_cloud.sh). Runs inside the
# "gsi" or "full" image.
#
#   run_prep_obs.sh <valid YYYYMMDDHH> <case dir> <fix dir> <work dir> [ranks]
#
# Reads <case dir>/mrms and <case dir>/obs (fetch_case.sh); writes
# hrrr.tHHz.NSSLRefInGSI.bufr and hrrr.tHHz.NASALaRCCloudInGSI.bufr into
# <case dir>/obs, the names the analysis script links. The executables, the
# BUFR table and the namelist values are the tag's own. Lightning is skipped
# when no lightning dump was fetched, as the tag's script skips it.
set -euo pipefail
valid="$1"; case_dir="$2"; FIXhrrr="$3"; DATA="$4"; ranks="${5:-8}"
PARMhrrr=/opt/hrrr/parm/conus
EXEChrrr=/opt/hrrr/exec
PDY="${valid:0:8}"; cyc="${valid:8:2}"
if [[ "$cyc" == 00 || "$cyc" == 12 ]]; then prefix=rap_e; else prefix=rap; fi
# The programs keep whole grids on the stack: under the default 8 MB limit
# every rank of process_mosaic is killed before it reads a level (MEASURED
# 2026-10-03).
ulimit -s unlimited

# --- radar mosaic ------------------------------------------------------------
mkdir -p "$DATA/radar" && cd "$DATA/radar"
cp "$PARMhrrr/hrrr_prepobs_prep.bufrtable" ./prepobs_prep.bufrtable
ln -sf "$FIXhrrr/hrrr_geo_em.d01.nc" geo_em.d01.nc
count="$( (ls "$case_dir"/mrms/MergedReflectivityQC_*_"${PDY}-${cyc}"0???.grib2.gz 2>/dev/null || true) | wc -l)"
if [[ "$count" -eq 33 ]]; then
  cp "$case_dir"/mrms/MergedReflectivityQC_*_"${PDY}-${cyc}"0???.grib2.gz .
  gzip -d -f ./*.gz
  ls MergedReflectivityQC_*_"${PDY}-${cyc}"????.grib2 > filelist_mrms
  echo "$valid" > ./mosaic_cycle_date
  # script: tversion=1 selects the single-tile GRIB2 mosaic.
  printf ' &setup\n  tversion=1,\n  analysis_time = %s,\n  dataPath = %s,\n /\n' "$valid" "'./'" > mosaic.namelist
  # script: mpiexec -n 36. The program gives each of the 33 levels its own
  # rank and stops with fewer than 33, so the tag's count is kept on any
  # host; on fewer cores the ranks share them.
  mpiexec -n 36 "$EXEChrrr/hrrr_process_mosaic" > process_mosaic.out 2>&1 \
    || { tail -30 process_mosaic.out; exit 1; }
  test -s NSSLRefInGSI.bufr || { echo "process_mosaic wrote no NSSLRefInGSI.bufr"; tail -30 process_mosaic.out; exit 1; }
  cp NSSLRefInGSI.bufr "$case_dir/obs/hrrr.t${cyc}z.NSSLRefInGSI.bufr"
  rm -f ./*.grib2
else
  echo "Warning: $count of 33 MRMS levels present; no reflectivity for the analysis"
fi

# --- NASA LaRC cloud ---------------------------------------------------------
mkdir -p "$DATA/cloud" && cd "$DATA/cloud"
export F_UFMTENDIAN="big;little:10,15,66"      # script: F_UFMTENDIAN
cp "$PARMhrrr/hrrr_prepobs_prep.bufrtable" ./prepobs_prep.bufrtable
ln -sf "$FIXhrrr/hrrr_geo_em.d01.nc" geo_em.d01.nc
if [[ -r "$case_dir/obs/$prefix.t${cyc}z.lgycld.tm00.bufr_d" ]]; then
  ln -sf "$case_dir/obs/$prefix.t${cyc}z.lgycld.tm00.bufr_d" NASA_LaRC_cloud.bufr
  echo "$valid" > nasaLaRC_cycle_date
  printf '&SETUP\n  analysis_time = %s,\n  bufrfile=%s,\n  npts_rad=3,\n  ioption = 2,\n/\n' "$valid" "'NASALaRCCloudInGSI.bufr'" > namelist_nasalarc
  mpiexec -n 4 "$EXEChrrr/hrrr_process_cloud" > process_cloud.out 2>&1 \
    || { tail -30 process_cloud.out; exit 1; }
  test -s NASALaRCCloudInGSI.bufr || { echo "process_cloud wrote no NASALaRCCloudInGSI.bufr"; tail -30 process_cloud.out; exit 1; }
  cp NASALaRCCloudInGSI.bufr "$case_dir/obs/hrrr.t${cyc}z.NASALaRCCloudInGSI.bufr"
else
  echo "Warning: no $prefix.t${cyc}z.lgycld.tm00.bufr_d; no LaRC cloud for the analysis"
fi
ls -l "$case_dir/obs"
