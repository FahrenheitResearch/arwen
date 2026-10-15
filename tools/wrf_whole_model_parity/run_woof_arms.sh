#!/usr/bin/env bash
# run_woof_arms.sh ROOT CASE_DIR TREE TAG MODE ARM...
#   WOOF runs of whole-model arms from the same per-arm real.exe copy WRF read,
#   one card, arms one after another.  MODE = default | strict.  TREE is the
#   engine checkout the active Python environment runs (an editable install);
#   the native bridges come from the environment (GPUWM_RW_NETCDF,
#   GPUWM_NCWRITE_BRIDGE, GPUWM_STATIC_BRIDGE, GPUWM_CPU_PREPROCESS_BRIDGE).
set -u
ROOT=$1; CASE=$2; TREE=$3; TAG=$4; MODE=$5; shift 5
if [ "$MODE" = strict ]; then export GPUWM_WRF_EXACT=1; else unset GPUWM_WRF_EXACT; fi
for ARM in "$@"; do
  IN=$ROOT/woof-in/$TAG-$MODE/$ARM
  mkdir -p "$IN"
  ln -sf "$ROOT/inputs/$ARM/wrfinput_d01" "$IN/wrfinput_d01"; ln -sf "$CASE/wrfbdy_d01" "$IN/wrfbdy_d01"
  # The one edit to the namelist WRF ran: restart_interval 100000 min is not a
  # whole number of steps and WOOF refuses it; no restart falls inside the
  # run on either side, so 1440 min changes nothing integrated.
  sed -E 's/^( restart_interval +)= *100000,/\1= 1440,/' "$ROOT/cases/$ARM/namelist.input" > "$IN/namelist.input"
  grep -q 'restart_interval *= 1440,' "$IN/namelist.input" || { echo "restart_interval edit failed for $ARM"; exit 3; }
  steps_seconds=$(awk -F= '/run_minutes/{m=$2+0} /run_seconds/{s=$2+0} END{print m*60+s}' "$IN/namelist.input")
  OUT=$ROOT/woof/$TAG/$MODE/$ARM
  rm -rf "$OUT"; mkdir -p "$OUT"
  EXTRA=()
  if grep -qE 'ra_lw_physics *= *4' "$IN/namelist.input"; then EXTRA+=(--rrtmg-variant rrtmg_legacy); fi
  start=$(date +%s)
  ( cd "$TREE" && git rev-parse HEAD > "$OUT/revision.txt"; git status --short > "$OUT/status.txt" )
  python -m gpuwm.cli run --wrfinput "$IN" --run-seconds "$steps_seconds" --outdir "$OUT/run" "${EXTRA[@]}" > "$OUT/stdout.txt" 2> "$OUT/stderr.txt"
  rc=$?
  echo "{\"arm\":\"$ARM\",\"tag\":\"$TAG\",\"mode\":\"$MODE\",\"rc\":$rc,\"seconds\":$(( $(date +%s) - start )),\"extra\":\"${EXTRA[*]}\"}" > "$OUT/result.json"
  echo "$TAG $MODE $ARM rc=$rc $(tail -1 "$OUT/stderr.txt")"
done
