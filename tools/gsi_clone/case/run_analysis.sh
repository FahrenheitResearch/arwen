#!/usr/bin/env bash
# Run HRRR's GSI analysis on one background the way the tag's
# scripts/conus/exhrrr_analysis.sh does, with the ensemble off. Runs inside
# the "gsi" or "full" image.
#
#   run_analysis.sh <valid YYYYMMDDHH> <background file> <obs dir> <fix dir> \
#                   <crtm dir> <work dir> [ranks] [threads]
#
# Environment:
#   GRID_RATIO           grid_ratio_wrfmass for the variational step
#   CLOUD_ANALYSIS_TYPE  i_gsdcldanal_type for the variational step
#   CLOUD_STEP=1         follow it with the tag's second step: the cloud
#                        analysis alone (type 6, grid ratio 1) on the
#                        variational analysis
#
# The executable, the namelist template (parm/conus/hrrr_gsiparm.anl.sh), the
# fix files and the BUFR table are the tag's own, unedited. What the tag's
# script sets and this one sets the same way is marked "script:" with the
# script's own values. Differences, all outside the programs:
#   * no ensemble: the script's hybrid branches are not taken, so
#     ifhyb=.false. and beta1_inv=1.0 as the script initialises them. The
#     script leaves grid_ratio and cloudanalysistype unset on that path;
#     here they come from GRID_RATIO and CLOUD_ANALYSIS_TYPE, and are left
#     unset (GSI's own defaults) when those are not given;
#   * the public ".nr" prepbufr stands in for the restricted one under the
#     name GSI opens;
#   * no land-surface cycling, sea-surface temperature update or snow trim
#     (they need the previous cycle or other products);
#   * rank and thread counts are this host's, not 360 x 4.
set -euo pipefail
valid="$1"; background="$2"; OBS="$3"; FIXhrrr="$4"; FIXcrtm="$5"; DATA="$6"
ranks="${7:-8}"; threads="${8:-1}"
PARMhrrr=/opt/hrrr/parm/conus
EXEChrrr=/opt/hrrr/exec
PDY="${valid:0:8}"; cyc="${valid:8:2}"
mkdir -p "$DATA" && cd "$DATA"
ulimit -s unlimited || true
export OMP_STACKSIZE=500M              # script: OMP_STACKSIZE=500M
export OMP_NUM_THREADS="$threads"      # script: OMP_NUM_THREADS=4

if [[ ! -s wrf_inout ]]; then cp "$background" wrf_inout; fi

# Observations, under the names the script links them to.
link_obs() {  # link_obs <name GSI opens> <file in the observation folder>...
  local name="$1"; shift
  local file
  for file in "$@"; do
    if [[ -s "$OBS/$file" ]]; then ln -sf "$OBS/$file" "$name"; echo "obs: $name <- $file"; return 0; fi
  done
  echo "Warning: no file for $name"
}
if [[ "$cyc" == 00 || "$cyc" == 12 ]]; then prefix=rap_e; else prefix=rap; fi
link_obs prepbufr   "$prefix.t${cyc}z.prepbufr.tm00" "$prefix.t${cyc}z.prepbufr.tm00.nr"
link_obs refInGSI   "hrrr.t${cyc}z.NSSLRefInGSI.bufr"
link_obs lghtInGSI  "hrrr.t${cyc}z.LightningInGSI.bufr"
link_obs larcInGSI  "hrrr.t${cyc}z.NASALaRCCloudInGSI.bufr"
link_obs l2rwbufr   "$prefix.t${cyc}z.nexrad.tm00.bufr_d"
link_obs satwndbufr "$prefix.t${cyc}z.satwnd.tm00.bufr_d"

# Fix files, copied under the names the script copies them to.
cp "$FIXhrrr/hrrr_anavinfo_arw_netcdf"         anavinfo
cp "$FIXhrrr/hrrr_berror_stats_global"         berror_stats
cp "$FIXhrrr/hrrr_global_satinfo.txt"          satinfo
cp "$FIXhrrr/hrrr_nam_regional_convinfo"       convinfo
cp "$FIXhrrr/hrrr_global_ozinfo.txt"           ozinfo
cp "$FIXhrrr/hrrr_global_pcpinfo.txt"          pcpinfo
cp "$FIXhrrr/hrrr_nam_errtable.r3dv"           errtable
cp "$FIXhrrr/hrrr_current_bad_aircraft.txt"    current_bad_aircraft
cp "$FIXhrrr/hrrr_current_mesonet_uselist.txt" gsd_sfcobs_uselist.txt
cp "$FIXhrrr/hrrr_gsd_sfcobs_provider.txt"     gsd_sfcobs_provider.txt
for name in AerosolCoeff.bin CloudCoeff.bin Nalli.IRwater.EmisCoeff.bin \
            NPOESS.IRice.EmisCoeff.bin NPOESS.IRsnow.EmisCoeff.bin NPOESS.IRland.EmisCoeff.bin \
            NPOESS.VISice.EmisCoeff.bin NPOESS.VISland.EmisCoeff.bin NPOESS.VISsnow.EmisCoeff.bin \
            NPOESS.VISwater.EmisCoeff.bin FASTEM5.MWwater.EmisCoeff.bin; do
  ln -sf "$FIXcrtm/$name" "./$name"
done
for sensor in $(awk '{if($1!~"!"){print $1}}' ./satinfo | sort | uniq); do
  ln -sf "$FIXcrtm/$sensor.SpcCoeff.bin" ./
  ln -sf "$FIXcrtm/$sensor.TauCoeff.bin" ./
done
cp "$PARMhrrr/hrrr_prepobs_prep.bufrtable" ./prepobs_prep.bufrtable

run_gsi() {  # run_gsi <label>: write gsiparm.anl from the tag's template, run
  local label="$1"
  export JCAP=62 LEVS=60                         # script: JCAP=62, LEVS=60
  export DELTIM=$((3600/(JCAP/20)))              # script: DELTIM
  export YYYYMMDDHH="$valid"
  # The template names JCAP_B, NLAT and LONA, which the script never sets;
  # they expand empty, a Fortran namelist null value that leaves GSI's
  # default. Sourced without the unset-variable check for that reason.
  set +u
  # shellcheck disable=SC1091
  . "$PARMhrrr/hrrr_gsiparm.anl.sh"
  set -u
  printf '%s\n' "$gsi_namelist" > gsiparm.anl
  cp gsiparm.anl "gsiparm.anl_$label"
  local started=$SECONDS
  mpiexec -n "$ranks" "$EXEChrrr/hrrr_gsi" < gsiparm.anl > "stdout_$label" 2> "errfile_$label" \
    || { echo "hrrr_gsi ($label) failed"; tail -40 "stdout_$label" "errfile_$label"; exit 1; }
  echo "hrrr_gsi ($label): $((SECONDS - started)) s on $ranks rank(s) x $threads thread(s)"
  grep -q "PROGRAM GSI_ANL HAS ENDED" "stdout_$label" \
    || { echo "hrrr_gsi ($label) did not end normally"; tail -40 "stdout_$label"; exit 1; }
}

# script: beta1_inv=1.0, ifhyb=.false. before any ensemble is found. The
# script counts members as "lines of `more filelist` minus 3" (the pager's
# three header lines on its host), which is 0 with no member listed.
beta1_inv=1.0; ifhyb=.false.; nummem=0; ifsoilnudge=.true.
regional_ensemble_option=; grid_ratio_ens=; i_en_perts_io=; ens_fast_read=
grid_ratio="${GRID_RATIO:-}"; cloudanalysistype="${CLOUD_ANALYSIS_TYPE:-}"
run_gsi var

# script: the diagnostic files of the first and last outer loop, and the fit
# files under the script's names.
for loop in 01 03; do
  case $loop in 01) string=ges;; 03) string=anl;; esac
  if ls pe*.conv_${loop}* > /dev/null 2>&1; then
    cat pe*.conv_${loop}* > "diag_conv_${string}.${PDY}${cyc}"
  fi
done
for pair in 201:fit_p1 202:fit_w1 203:fit_t1 204:fit_q1 205:fit_pw1 206:fit_oz1 \
            207:fit_rad1 208:fit_pcp1 209:fit_rw1 213:fit_sst1; do
  if [[ -e "fort.${pair%%:*}" ]]; then mv "fort.${pair%%:*}" "${pair##*:}"; fi
done
cat fit_p1 fit_w1 fit_t1 fit_q1 fit_pw1 fit_rad1 fit_rw1 > "hrrr.t${cyc}z.fits" 2>/dev/null || true
cat fort.210 fort.211 fort.212 fort.214 fort.215 fort.217 fort.220 > "hrrr.t${cyc}z.fits2" 2>/dev/null || true

if [[ "${CLOUD_STEP:-0}" == 1 ]]; then
  # script: the second step, taken when the variational step skipped the
  # cloud analysis: grid_ratio=1, cloudanalysistype=6, ifhyb=.false.
  grid_ratio=1; cloudanalysistype=6; ifhyb=.false.; ifsoilnudge=.true.
  mv sigf03 sigf03_step1
  mv siganl sigf03
  run_gsi cloud
fi
rm -f pe0*.* obs_input.* 2>/dev/null || true
ls -l wrf_inout "hrrr.t${cyc}z.fits" 2>/dev/null
