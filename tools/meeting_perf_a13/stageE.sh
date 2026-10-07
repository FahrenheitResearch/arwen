#!/usr/bin/env bash
# Stage case c3 on box E (CPU only): raw inputs, geography overlay, authored experiment, preparation.
set -u
W=/work/meeting-perf-a13; c=$W/case/c3; RAW=$W/raw/2026100321
mkdir -p $RAW $c $W/geog
for d in /work/da-e/WPS_GEOG/*; do ln -sfn $d $W/geog/$(basename $d); done
get() { [ -s "$2" ] || { curl -fsSL --retry 5 -o "$2.part" "$1" && mv "$2.part" "$2"; }; }
HB=https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.20261003/conus
RB=https://noaa-rap-pds.s3.amazonaws.com/rap.20261003
for f in wrfnatf00 wrfprsf00 wrfsfcf00; do get $HB/hrrr.t21z.$f.grib2 $RAW/hrrr.t21z.$f.grib2 & done
for L in 03 06; do get $RB/rap.t18z.awp130bgrbf$L.grib2 $RAW/rap.t18z.awp130bgrbf$L.grib2 & done
[ -d $W/geog/orogwd3_10m ] || { get https://www2.mmm.ucar.edu/wrf/src/wps_files/orogwd3_10m.tar.bz2 $W/o.tbz && tar -xjf $W/o.tbz -C $W/geog && rm $W/o.tbz; }
[ -d $W/geog/lake_depth ] || { get https://www2.mmm.ucar.edu/wrf/src/wps_files/lake_depth.tar.bz2 $W/l.tbz && tar -xjf $W/l.tbz -C $W/geog && rm $W/l.tbz; }
wait
ls -la $RAW
sed 's#/work/earth2-1km/kit/hrrr-door-c24s.toml#/work/meeting-perf-a13/hrrr-door-c24s.toml#' $W/mkcase.py > $W/mkcase-e.py
python3 $W/mkcase-e.py conus3 $c/experiment.toml 2026100321 2026100318 3 || exit 1
cat > $c/initial-inputs.json <<J
{"schema": "gpuwm-initial-source-v1", "source": "hrrr-native",
 "input_files": ["$RAW/hrrr.t21z.wrfnatf00.grib2"],
 "supplements": {"soil_surface_data": ["$RAW/hrrr.t21z.wrfprsf00.grib2"],
                 "vegetation_surface_data": ["$RAW/hrrr.t21z.wrfsfcf00.grib2"]}}
J
printf '%s\n' $RAW/rap.t18z.awp130bgrbf03.grib2 $RAW/rap.t18z.awp130bgrbf06.grib2 > $c/rap-files.txt
source $W/envE.sh
python - $c <<'PY'
import sys
from pathlib import Path
import gpuwm
assert gpuwm.__file__.startswith("/work/meeting-perf-a13/trees/tip"), gpuwm.__file__
from gpuwm.experiment import load_experiment
from gpuwm.hrrr_prepared_bundle import render_wps_namelist
c = Path(sys.argv[1]); e = load_experiment(str(c / 'experiment.toml')); r = e.root.run
(c / 'namelist.wps').write_text(render_wps_namelist(e, interval_seconds=10800))
print('config ok', (r.nx, r.ny, r.nz), r.dx, e.start_time, e.run_seconds, 'lake', r.sf_lake_physics)
PY
T=96
export CUDA_VISIBLE_DEVICES= GPUWM_NO_LOCAL_GPU=1
export OMP_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T RAYON_NUM_THREADS=$T GPUWM_MAPPED_ENGINE_THREADS=$T GPUWM_PREPROCESS_THREADS=$T
A=(--source rap-native)
while read -r f; do A+=(--input $f); done < $c/rap-files.txt
while read -r f; do A+=(--supplement $f); done < $c/rap-files.txt
A+=(--author-input-manifest $c/proof-inputs.json --initial-inputs $c/initial-inputs.json)
rm -rf $c/prepared $c/prepared.partial $c/proof-inputs.json
t0=$(date +%s)
nice -n 10 python -m gpuwm.cli prep "${A[@]}" --experiment-config $c/experiment.toml --wps-namelist $c/namelist.wps \
    --geog-root $W/geog --preprocess-backend cpu --preprocess-workers $T --no-stock-wrf-export \
    --output-root $c/prepared.partial > $W/prep.log 2>&1
rc=$?
echo "prep rc=$rc wall=$(( $(date +%s) - t0 ))" >> $W/prep.log
[ $rc = 0 ] && mv $c/prepared.partial $c/prepared
echo "PREP DONE rc=$rc"
