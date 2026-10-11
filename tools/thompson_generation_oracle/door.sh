#!/usr/bin/env bash
# Real installed WRF door, on a whole card for timing. Compare before cleanup.
set -euo pipefail
L=$1; LEG=$2; GENERATION=$3
T=$L/src
source /work/env.sh
export PYTHONPATH=$T:$T/tests PYTHON_GIL=0
export GPUWM_RW_NETCDF=/work/engine-base/tools/rustwx/target/release/rw_netcdf
export GPUWM_STATIC_BRIDGE=/work/engine-base/tools/rustwx/target/release/libstatic_fields.so
export GPUWM_NCWRITE_BRIDGE=/work/engine-base/tools/rustwx/target/release/libnetcdf_writer.so
export CUPY_CACHE_DIR=$L/cupy-cache TMPDIR=$L/tmp
unset GPUWM_WRF_EXACT GPUWM_NO_LOCAL_GPU
DATA=/work/c289-nv/data/v4.4_bench_conus12km
RUN=$L/runs/$LEG-$GENERATION
mkdir -p "$RUN"
printf '%s\n' "$L/runs" "$RUN" >> "$L/MANIFEST.txt"
cd "$RUN"
git -C "$T" rev-parse HEAD > source-tip.txt
cmd=("$L/venv/bin/gpuwm" run)
if [ "$LEG" = restart ]; then cmd+=(--restart-from "$DATA/wrfrst_d01_2019-11-26_23:00:00.gnu"); REF=$DATA/wrfout_d01_2019-11-27_00:00:00.gnu.restart
else cmd+=("$DATA"); REF=$DATA/wrfout_d01_2019-11-27_00:00:00.gnu; fi
cmd+=(--outdir "$RUN/out")
if [ "$GENERATION" != auto ]; then cmd+=(--thompson-version "$GENERATION"); fi
printf '%q ' "${cmd[@]}" > command.txt; printf '\n' >> command.txt
start=$(date +%s.%N)
set +e
"${cmd[@]}" > run.log 2>&1
rc=$?
set -e
end=$(date +%s.%N)
"$L/venv/bin/python" -c 'import json,sys; from pathlib import Path; Path("rc.json").write_text(json.dumps({"rc":int(sys.argv[1]),"wall_s":float(sys.argv[3])-float(sys.argv[2])},indent=2)+"\n")' "$rc" "$start" "$end"
if [ "$rc" -ne 0 ]; then tail -20 run.log; exit "$rc"; fi
find out -type f -name 'wrfout*' -print > history-files.txt
FILE=out/wrfout/wrfout_d01_2019-11-27_00_00_00
if [ ! -f "$FILE" ]; then echo 'missing final 00Z history frame'; exit 2; fi
sha256sum "$FILE" > final-history.sha256
"$L/venv/bin/python" "$DATA/diffwrf.py" "$FILE" "$REF" > diffwrf.txt 2>&1
printf '%s\n' "door $LEG $GENERATION rc=$rc"
