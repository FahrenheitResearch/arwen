#!/usr/bin/env bash
# cycle-l6.sh ARM [smoke]: DA lane 6 cycled confirmation on box N, the 2026-10-01 9 km CONUS hydro-boundary case
# (lane 0's plan and case, /work/da-line/case on N, read only), engine lane/da-l6-radar-heating 9f9e163da on lane/da-line 7280e8abc.
# Three observed legs (analyses 19, 20, 21Z) and a 3 h free forecast in 1 h legs; leg-end and free-forecast composites saved.
#   /opt/gpu-mutex/run.sh da-l6 --cards 8 --wait 14400 /work/da-l6/cycle-l6.sh heat-clear
# ARM: off (lane 0 recipe, the reference) | clear (no heating, Z update limited to clearing: heat-clear's control) | heat (radar heating, campaign Z update) | heat-clear (heating, Z update
#      limited to clearing) | heat-strict (heat-clear + strict suppression + PBL extension) | lhn-clear | heat-weak (lane 3 tree: heating + the weak Z update)
# smoke: 1 card, 2 members, one observed leg, a 120 s free leg: proves the windowed heating path end to end.
set -u
ARM=$1; MODE=${2:-full}
L=/work/da-l6; OUT=$L/cycled/$ARM; [ "$MODE" = smoke ] && OUT=$L/cycled/smoke-$ARM
WIN=/work/da-l6-grid/windows-20261001
_CVD=${CUDA_VISIBLE_DEVICES:-}
TREE=engine3; [ "$ARM" = heat-weak ] && TREE=engine3-l3
export DA_ENGINE=$L/$TREE
source /work/da-e/env.sh
export PATH=$L/$TREE.venv/bin:$PATH NUMPY_MADVISE_HUGEPAGE=0 CUDA_VISIBLE_DEVICES=$_CVD
export PYTHONPATH=$DA_ENGINE TMPDIR=$L/tmp CUPY_CACHE_DIR=$L/cache/cupy XDG_CACHE_HOME=$L/cache
mkdir -p $TMPDIR $CUPY_CACHE_DIR
[ -n "$CUDA_VISIBLE_DEVICES" ] || { echo "no cards; run through the mutex" >&2; exit 2; }
mkdir -p $OUT; [ -e $OUT/da ] && { echo "$OUT/da exists" >&2; exit 2; }
IFS=, read -ra U <<< "$CUDA_VISIBLE_DEVICES"
python3 /work/da-l6/mkargv-hydro.py da $OUT "${U[@]}" > $OUT/command.base.txt || exit 2
python3 - $OUT/command.base.txt $ARM $MODE $WIN $TREE > $OUT/command.txt <<'PY' || exit 2
import shlex, sys
src, arm, mode, win, tree = sys.argv[1:]
a = [x.replace("/work/da-e/shared-conus-hydro", "/work/da-line/case/shared-conus-hydro") for x in shlex.split(open(src).read())]
def span(flag):
    i = a.index(flag); j = i + 1
    while j < len(a) and not a[j].startswith("--"): j += 1
    return i, j
def setv(flag, vals):
    if flag in a:
        i, j = span(flag); a[i + 1:j] = vals
    else:
        a.extend([flag, *vals])
def drop(flag):
    if flag in a:
        i, j = span(flag); del a[i:j]
def keep_first(flag, n):
    idx = [k for k, x in enumerate(a) if x == flag]
    for k in reversed(idx[n:]): del a[k:k + 2]
a[0] = f"/work/da-l6/{tree}.venv/bin/python"
drop("--save-ensemble")
if mode == "smoke":
    keep_first("--obs", 1); keep_first("--grid-wrfout", 1)
    setv("--leg-durations-seconds", ["3600.0", "120.0"])
    setv("--members", ["2"])
    setv("--forecast-members-per-card", ["serial"])
else:
    # three observed legs, then the free forecast as three 1 h legs so every
    # member writes a composite at f01, f02 and f03
    setv("--leg-durations-seconds", ["3600.0"] * 6)
setv("--free-legs", ["1" if mode == "smoke" else "3"])
i, j = span("--leg-durations-seconds")
setv("--run-seconds", [f"{sum(float(x) for x in a[i + 1:j]):.0f}"])
# heat-weak runs the lane 3 test tree: its default field rules (design, theta/qv weight 0.1 inside echo) are the
# weak update
heat = {"heat": [], "heat-weak": [], "heat-clear": [], "heat-strict": ["--radar-tten-strict-suppression", "--radar-tten-pbl-extension"],
        "lhn-clear": ["--radar-tten-mode", "lhn"]}
if arm in heat:
    a += ["--radar-tten", "--radar-tten-windows", win, "--radar-tten-perturb-members", "--radar-tten-control",
          *heat[arm]]
elif arm not in ("off", "clear"):
    raise SystemExit(f"unknown arm {arm}")
if arm.endswith("-clear") or arm in ("clear", "heat-strict"):
    # Z update limited to clearing: no echo analysis; clear-air zeroes and the GSD radar clearing still run
    if "--reflectivity-analysis" in a:
        a.remove("--reflectivity-analysis")
    setv("--precip-analysis", ["clear"])
print(shlex.join(a))
PY
cd $DA_ENGINE
( while :; do echo "$(date +%s),$(/usr/bin/nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | tr "\n" ";")"; sleep 5; done ) > $OUT/gpu-samples.csv 2>/dev/null &
SP=$!
trap "kill $SP 2>/dev/null" EXIT
t0=$(date +%s)
bash -c "exec $(cat $OUT/command.txt)" > $OUT/controller.log 2>&1 &
CPID=$!; echo $CPID > $OUT/controller.pid
wait $CPID; rc=$?
echo "{\"arm\": \"$ARM\", \"mode\": \"$MODE\", \"rc\": $rc, \"wall_seconds\": $(( $(date +%s) - t0 ))}" > $OUT/done.json
echo "$(date -u +%FT%TZ) $ARM $MODE rc=$rc" >> $L/cycled/runs.log
if [ $rc -eq 0 ]; then
  # spent once the arm ended cleanly: the recovery restart sets (~186 GB) and the stage (~22 GB);
  # the composites, the report and the increments stay
  du -sh $OUT/da/packed-recovery $OUT/da/stage 2>/dev/null | tr "
" " " >> $L/cycled/cleanup.log
  rm -rf $OUT/da/packed-recovery $OUT/da/stage && echo "$(date -u +%FT%TZ) $ARM cleared" >> $L/cycled/cleanup.log
fi
exit $rc
