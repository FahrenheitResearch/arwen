#!/usr/bin/env bash
# GSI's single-observation mode on one WRF-format background with a hybrid
# ensemble, under one namelist profile (profiles/*.profile). Runs inside a
# pin image (Dockerfile.pin), whose gsi.x reads WRF files in Regional mode.
#
#   run_singleob.sh <profile> <background> <member dir> <fix root> <work dir> [ranks]
#
# <member dir>  WRF-format history files on the background's grid, used as
#               the ensemble (every wrfout_d01_* in it, sorted)
# <fix root>    holds the folders the profile's FIX_MAP and ANAVINFO_TABLE
#               name (hrrr/, gsi-fix/)
# Environment (optional): OBLAT, OBLON (degrees east, 0-360), OBPRES (hPa).
# The default observation sits at the domain centre (CEN_LAT, CEN_LON) at
# 500 hPa, with the profile's innovation and error.
#
# Everything about the system being imitated comes from the profile; what
# comes from the background file (level count, hybrid coordinate, valid
# time, centre) is read from the file. The run writes receipt.txt.
#
# Start the container with --shm-size=16g (as case/in_container.sh does):
# Intel MPI's shared-memory transport overruns podman's 64 MB default and
# ranks die with SIGBUS (MEASURED 2026-10-03).
set -euo pipefail
profile="$1"; background="$2"; members="$3"; fixroot="$4"; work="$5"; ranks="${6:-8}"
here="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1090
. "$profile"
GSI_EXE="${GSI_EXE:-/opt/gsi-pin/exec/gsi.x.enkf-wrf-build}"
image_pin=$( . /opt/gsi-pin.pin; printf '%s' "$PIN_NAME" )
[[ "$image_pin" == "$GSI_PIN" ]] || { echo "profile requires $GSI_PIN but image carries $image_pin; refusing a different GSI source" >&2; exit 1; }
[[ "$ranks" =~ ^[1-9][0-9]*$ ]] || { echo "ranks must be a positive integer" >&2; exit 1; }
mkdir -p "$work"; cd "$work"
ulimit -s unlimited || true
export OMP_NUM_THREADS=1 OMP_STACKSIZE=500M
export MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TMPDIR="${TMPDIR:-$work/tmp}"
mkdir -p "$TMPDIR"

# --- what the background file says --------------------------------------
hdr="$(ncdump -h "$background")"
for name in ${STATIC_FIELDS//,/ }; do
  grep -Eq "^[[:space:]]+(float|double|int|char) ${name}\\(" <<< "$hdr" || {
    echo "background lacks $name, which GSI reads without checking the read status; prepare_history.sh copies it from the matching input template" >&2
    exit 1
  }
done
nz=$(awk '$1=="bottom_top" && $2=="=" {print $3+0}' <<< "$hdr")
hyb=$(awk -F'= ' '/:HYBRID_OPT =/ {gsub(/[ ;]/,"",$2); print $2}' <<< "$hdr")
cen_lat=$(awk -F'= ' '/:CEN_LAT =/ {gsub(/[ f;]/,"",$2); print $2}' <<< "$hdr")
cen_lon=$(awk -F'= ' '/:CEN_LON =/ {gsub(/[ f;]/,"",$2); print $2}' <<< "$hdr")
times=$(ncdump -v Times "$background" | awk -F'"' '/^ *Times =|^  "/ {for(i=2;i<=NF;i+=2) if($i ~ /^[0-9]{4}-/) {print $i; exit}}')
adate="${times:0:4}${times:5:2}${times:8:2}${times:11:2}"
[[ -n "$nz" && -n "$adate" ]] || { echo "cannot read level count or time from $background" >&2; exit 1; }
if [[ "${hyb:-0}" == 2 ]]; then hybridcord=.true.; else hybridcord=.false.; fi
OBLAT="${OBLAT:-$cen_lat}"
OBLON="${OBLON:-$(awk -v x="$cen_lon" 'BEGIN{x+=0; if (x<0) x+=360; print x}')}"
OBPRES="${OBPRES:-500.}"

# --- background and members ----------------------------------------------
cp "$background" wrf_inout
cp wrf_inout wrf_inout.background
# Members are linked under short names: GSI's hydrometeor-aware ensemble
# reader (general_read_wrf_mass2) keeps only the first characters of a long
# path (MEASURED 2026-10-03: it opened "/w/woof-case/members/wrf").
filelist="filelist$(printf %02d "$NHR_ASSIMILATION")"
: > "$filelist"; n_ens=0
for m in "$members"/wrfout_d01_*; do
  [[ -f "$m" ]] || { echo "no WRF members found in $members" >&2; exit 1; }
  n_ens=$((n_ens + 1)); name=$(printf 'wrf_en%03d' "$n_ens")
  ln -sf "$m" "$name"; echo "$name" >> "$filelist"
done

# --- variable table ---------------------------------------------------------
table="$here/$ANAVINFO_TABLE"; [[ -f "$table" ]] || table="$fixroot/$ANAVINFO_TABLE"
if grep -q '@NZ@' "$table"; then
  sed -e "s/@NZ1@/$((nz + 1))/g" -e "s/@NZ@/$nz/g" "$table" > anavinfo
else
  # A table written for a fixed level count: its most common level count
  # above one is the 3-D count; that value and one more are rewritten.
  old=$(awk '/::/{t=$1} t!~/control_vector_enkf/ && $2 ~ /^[0-9]+$/ && $2>9 {c[$2]++} END{m=0; for(k in c) if(c[k]>m){m=c[k]; v=k}; print v}' "$table")
  awk -v old="$old" -v nz="$nz" -v drop=" ${ANAVINFO_DROP_MET_GUESS:-} " -v ren="${ANAVINFO_RENAME_MET_GUESS:-}" '
    BEGIN { n=split(ren, r, " "); for (x=1; x<=n; x++) { split(r[x], p, ":"); to[p[1]]=p[2] } }
    /::$/ {sect=$1}
    sect=="met_guess::" && index(drop, " " $1 " ") {next}
    sect=="met_guess::" && ($1 in to) { sub($1, to[$1]) }
    $2 == old   {$2 = nz}
    $2 == old+1 {$2 = nz+1}
    {print}' "$table" > anavinfo
fi

# --- fix files ------------------------------------------------------------
for pair in $FIX_MAP; do cp "$fixroot/${pair#*=}" "./${pair%%=*}"; done
usenew="${SUBSTITUTE_USENEWGFSBERROR:-}"; [[ -n "$usenew" ]] || usenew="$USENEWGFSBERROR"

# --- namelist -------------------------------------------------------------
cat > gsiparm.anl <<EOF
 &SETUP
   miter=${MITER},niter(1)=${NITER1},niter(2)=${NITER2},
   write_diag(1)=.true.,write_diag(2)=.false.,write_diag(3)=.true.,
   qoption=${QOPTION},print_obs_para=.true.,
   gencode=${GENCODE},factqmin=0.0,factqmax=0.0,
   iguess=-1,
   oneobtest=.true.,retrieval=.false.,
   nhr_assimilation=${NHR_ASSIMILATION},l_foto=.false.,
   use_pbl=.false.,use_prepb_satwnd=.false.,
   newpc4pred=.true.,adp_anglebc=.true.,angord=4,
   passive_bc=.true.,use_edges=.false.,emiss_bc=.true.,
   diag_precon=.true.,step_start=1.e-3,
   l4densvar=.false.,nhr_obsbin=3,
   use_gfs_nemsio=.false.,use_gfs_ncio=.true.,reset_bad_radbc=.true.,
   netcdf_diag=.false.,binary_diag=.true.,
   ${SETUP_EXTRA:-}
 /
 &GRIDOPTS
   JCAP=62,JCAP_B=62,nsig=${nz},
   wrf_nmm_regional=.false.,wrf_mass_regional=.true.,
   diagnostic_reg=.false.,
   filled_grid=.false.,half_grid=.true.,netcdf=.true.,
   grid_ratio_wrfmass=${GRID_RATIO},
   wrf_mass_hybridcord=${hybridcord},
 /
 &BKGERR
   vs=${BKGERR_VS},
   hzscl=${BKGERR_HZSCL},
   bw=0.,fstat=.true.,
   usenewgfsberror=${usenew},
 /
 &ANBKGERR
   anisotropic=.false.,
 /
 &JCOPTS
 /
 &STRONGOPTS
 /
 &OBSQC
   ${OBSQC}
 /
 &OBS_INPUT
   ${OBS_INPUT}
 /
OBS_INPUT::
!  dfile          dtype       dplat     dsis                 dval    dthin dsfcalc
   prepbufr       t           null      t                    1.0     0     0
::
 &SUPEROB_RADAR
 /
 &LAG_DATA
 /
 &HYBRID_ENSEMBLE
   l_hyb_ens=${L_HYB_ENS},
   uv_hyb_ens=${UV_HYB_ENS},
   q_hyb_ens=${Q_HYB_ENS},
   aniso_a_en=.false.,generate_ens=.false.,
   n_ens=${n_ens},
   beta_s0=${BETA_S0},s_ens_h=${S_ENS_H},s_ens_v=${S_ENS_V},
   regional_ensemble_option=3,
   pseudo_hybens=.false.,
   grid_ratio_ens=${GRID_RATIO_ENS},
   l_ens_in_diff_time=${L_ENS_IN_DIFF_TIME},
   ensemble_path='',
   i_en_perts_io=0,
   jcap_ens=574,
   readin_localization=${READIN_LOCALIZATION},
   nsclgrp=${NSCLGRP},l_timloc_opt=.false.,ngvarloc=${NGVARLOC},naensloc=${NAENSLOC},
   r_ensloccov4tim=${R_ENSLOCCOV4TIM},r_ensloccov4var=${R_ENSLOCCOV4VAR},r_ensloccov4scl=${R_ENSLOCCOV4SCL},
   assign_vdl_nml=${ASSIGN_VDL_NML},
 /
 &RAPIDREFRESH_CLDSURF
   ${CLDSURF}
 /
 &CHEM
 /
 &NST
 /
 &SINGLEOB_TEST
   maginnov=${SINGLEOB_MAGINNOV},magoberr=${SINGLEOB_MAGOBERR},oneob_type='${SINGLEOB_TYPE}',
   oblat=${OBLAT},oblon=${OBLON},obpres=${OBPRES},obdattim=${adate},
   obhourset=0.,
 /
EOF

# --- run ------------------------------------------------------------------
started=$SECONDS
set +e
mpiexec -n "$ranks" "$GSI_EXE" < gsiparm.anl > stdout 2> stderr
rc=$?
set -e
wall=$((SECONDS - started))
ended=no; grep -q "PROGRAM GSI_ANL HAS ENDED" stdout && ended=yes
nonfinite=no
grep -Eq '(Initial cost function|cost,grad,step|^analysis ).*(NaN|Infinity)' stdout && nonfinite=yes

{
  echo "profile        $PROFILE_NAME ($(sha256sum "$profile" | cut -c1-16))"
  echo "gsi executable $GSI_EXE $(sha256sum "$GSI_EXE" | cut -c1-16)"
  echo "background     $background $(sha256sum wrf_inout.background | cut -c1-16)"
  echo "               nz=$nz hybrid_opt=${hyb:-0} valid=$adate centre=$cen_lat,$cen_lon"
  echo "members        $n_ens from $members"
  echo "observation    $SINGLEOB_TYPE innovation=$SINGLEOB_MAGINNOV error=$SINGLEOB_MAGOBERR at $OBLAT,$OBLON,$OBPRES hPa"
  echo "localization   s_ens_h=$S_ENS_H s_ens_v=$S_ENS_V nsclgrp=$NSCLGRP ngvarloc=$NGVARLOC naensloc=$NAENSLOC r_ensloccov4var=$R_ENSLOCCOV4VAR beta_s0=$BETA_S0"
  echo "static B       usenewgfsberror=$usenew (profile value $USENEWGFSBERROR) hzscl=$BKGERR_HZSCL"
  echo "fix map        $FIX_MAP"
  echo "ranks          $ranks"
  echo "exit           rc=$rc ended_normally=$ended nonfinite_analysis=$nonfinite wall_s=$wall"
} > receipt.txt
cat receipt.txt
[[ $rc == 0 && $ended == yes && $nonfinite == no ]] || { tail -40 stdout; tail -20 stderr; echo "GSI did not produce a finite completed analysis" >&2; exit 1; }
