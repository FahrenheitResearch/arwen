#!/usr/bin/env bash
# go_e.sh ARM TAG: one CONUS 9 km DA run of lane/da-iau-7 on box E's 10-01 case (/work/da-e/conus run plan).
#   /opt/gpu-mutex/run.sh da-iau-7 --cards N --wait S /work/da-iau-7/run/go_e.sh ARM TAG
# ARM: m8-base m8-hyd m8-3d m8-3dh m8-4d m8-4dh (8 members, serial, one card), or smoke-<arm>.
set -u
ARM=$1 TAG=$2
R=/work/da-iau-7/run; OUT=$R/out/$TAG
_CVD=${CUDA_VISIBLE_DEVICES:-}
export DA_ENGINE=/work/da-iau-7/src
source /work/da-e/env.sh
export CUDA_VISIBLE_DEVICES=$_CVD NUMPY_MADVISE_HUGEPAGE=0 TMPDIR=/work/da-iau-7/tmp
export CUPY_CACHE_DIR=/work/da-iau-7/cache/cupy XDG_CACHE_HOME=/work/da-iau-7/cache
[ -n "$CUDA_VISIBLE_DEVICES" ] || { echo "go: no cards; run through the mutex" >&2; exit 2; }
mkdir -p "$OUT" $TMPDIR; [ -e "$OUT/da" ] && { echo "go: $OUT/da exists" >&2; exit 2; }
IFS=, read -ra U <<< "$CUDA_VISIBLE_DEVICES"
# The natives env.sh names are /work/da-e/engine's (55e9206b8); this tree's Rust half moved with the lane 0 merge,
# so each bridge this tree has built replaces the shared one (the shared one is kept where this tree has none).
for v in GPUWM_CPU_PREPROCESS_BRIDGE GPUWM_DEALIAS_REGION_BRIDGE GPUWM_GDT101_REMAP GPUWM_GFS_GRIB2_BRIDGE \
         GPUWM_GRIB1_BRIDGE GPUWM_GRIB2_DUMP GPUWM_GRIB2_INVENTORY GPUWM_HRRR_DECODER GPUWM_MAPPED_ENGINE_BIN \
         GPUWM_NCWRITE_BRIDGE GPUWM_OBSSCORE_BRIDGE GPUWM_RW_ASOS GPUWM_RW_MRMS GPUWM_RW_NETCDF GPUWM_RW_NEXRAD \
         GPUWM_STATIC_BRIDGE; do
  cur=${!v:-}
  case "$cur" in /work/da-e/engine/*) cand="$DA_ENGINE/${cur#/work/da-e/engine/}"; [ -e "$cand" ] && export "$v=$cand";; esac
done
export GPUWM_BRIDGE_SOURCE_REV=$(cat $DA_ENGINE/.engine-export-sha)
env | grep "^GPUWM_.*=/work/" | sort > "$OUT/bridges.txt"
# the 10-01 case on the hydrometeor-boundary cache (needs lane/gfs-hydro-lbc: lane/da-line 91b69817b has it)
cp /work/da-e/fx4s/run-plan.jsonl $OUT/run-plan.jsonl
cd "$DA_ENGINE"; export PYTHONPATH=/work/da-e/conus/sfchook:$DA_ENGINE PATH=/work/da-iau-7/venv/bin:$PATH
python3 $R/mkargv.py $OUT/run-plan.jsonl "$ARM" "$OUT" "${U[@]}" > "$OUT/command.txt" || exit 2
python3 -c "import sitecustomize, gpuwm, tools.da_cycle_prepared as d; print(sitecustomize.__file__, gpuwm.__file__, d.__file__)" > "$OUT/import-check.txt" 2>&1
echo "{\"engine\": \"$DA_ENGINE\", \"lane_sync\": \"$(cat $DA_ENGINE/.lane-sync 2>/dev/null)\", \"arm\": \"$ARM\"}" > "$OUT/engine.json"
t0=$(date +%s)
bash -c "exec $(cat "$OUT/command.txt")" > "$OUT/controller.log" 2>&1
rc=$?
echo "{\"rc\": $rc, \"wall_seconds\": $(( $(date +%s) - t0 )), \"cards\": \"${GPU_MUTEX_CARDS:-?}\", \"arm\": \"$ARM\"}" > "$OUT/done.json"
exit $rc
