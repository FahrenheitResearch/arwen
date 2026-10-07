#!/usr/bin/env bash
# go.sh ARM TAG [DAY]: one CONUS 9 km DA run of lane/da-iau-7 on box K, hydrometeor-boundary case
# (/work/da-tune/build-hydro/<DAY>, CASES.md), through the mutex:
#   /opt/gpu-mutex/run.sh da-iau-7 --cards 1 --wait S /work/da-iau-7/run/go.sh ARM TAG
# ARM: m8-<base|hyd|3d|3dh|4d|4dh> (8 members, serial, one card) or smoke-<arm>.
set -u
ARM=$1 TAG=$2 DAY=${3:-2026-10-01}
R=/work/da-iau-7/run; OUT=$R/out/$TAG; D=/work/da-tune/build-hydro/${DAY//-/}
_CVD=${CUDA_VISIBLE_DEVICES:-}
# /work/da-e tables (verified: CCN_ACTIVATE.BIN 35288 B, sha256 f2b8d391...); /work/da-286/tables/thompson is empty
# on K (00:13Z: mp=28 refused at the first microphysics call). Natives and venv are this lane's own below.
source /work/da-e/env.sh
export DA_ENGINE=/work/da-iau-7/src
export CUDA_VISIBLE_DEVICES=$_CVD NUMPY_MADVISE_HUGEPAGE=0 TMPDIR=/work/da-iau-7/tmp
export CUPY_CACHE_DIR=/work/da-iau-7/cache/cupy XDG_CACHE_HOME=/work/da-iau-7/cache
[ -n "$CUDA_VISIBLE_DEVICES" ] || { echo "go: no cards; run through the mutex" >&2; exit 2; }
mkdir -p "$OUT" $TMPDIR; [ -e "$OUT/da" ] && { echo "go: $OUT/da exists" >&2; exit 2; }
# Disk guard (breakage it prevents: 00:06Z on K, three arms started at 37 GB free and failed ENOSPC within seconds;
# an arm that runs the disk out mid-run takes other lanes' writes with it). One arm needs about 15 GB; start only
# with 60 GB free (the queue waits for space before it takes a card; this is the fail-fast check under the card).
[ -s "$GPUWM_THOMPSON_TABLE_ROOT/CCN_ACTIVATE.BIN" ] || { echo "go: no CCN_ACTIVATE.BIN under $GPUWM_THOMPSON_TABLE_ROOT" >&2; exit 3; }
[ "$(df -BG --output=avail /work | tail -1 | tr -dc 0-9)" -ge 60 ] || { echo "go: under 60 GB free on /work" >&2; exit 3; }
IFS=, read -ra U <<< "$CUDA_VISIBLE_DEVICES"
source $R/bridges.snip
# The case binds its planner to the engine it was prepared with (91b69817b = /work/da-286/engine); plan there, run
# this lane's tree.
( cd /work/da-286/engine && PYTHONPATH=/work/da-286/engine CUDA_VISIBLE_DEVICES= GPUWM_NO_LOCAL_GPU=1 \
  /work/da-286/venv/bin/python -m tools.da_recent_run --manifest $D/shared/case/run-manifest.json --seed 20261001 \
  --arm da --device-uuids "${U[@]}" > $OUT/run-plan.jsonl 2> $OUT/run-plan.err ) || { echo "plan failed" >&2; exit 2; }
cd "$DA_ENGINE"; export PYTHONPATH=$DA_ENGINE PATH=/work/da-iau-7/venv/bin:$PATH
python3 $R/mkargv.py $OUT/run-plan.jsonl "$ARM" "$OUT" "${U[@]}" > "$OUT/command.txt" || exit 2
python3 -c "import gpuwm, tools.da_cycle_prepared as d; print(gpuwm.__file__, d.__file__)" > "$OUT/import-check.txt" 2>&1
echo "{\"engine\": \"$DA_ENGINE\", \"lane_sync\": \"$(cat $DA_ENGINE/.lane-sync 2>/dev/null)\", \"arm\": \"$ARM\", \"case\": \"$D\"}" > "$OUT/engine.json"
t0=$(date +%s)
bash -c "exec $(cat "$OUT/command.txt")" > "$OUT/controller.log" 2>&1
rc=$?
echo "{\"rc\": $rc, \"wall_seconds\": $(( $(date +%s) - t0 )), \"cards\": \"${GPU_MUTEX_CARDS:-?}\", \"arm\": \"$ARM\"}" > "$OUT/done.json"
exit $rc
