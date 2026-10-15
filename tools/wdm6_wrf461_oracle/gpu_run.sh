#!/usr/bin/env bash
# Run WOOF's WDM6 on every fixture input, strict and default builds.
# usage: gpu_run.sh TREE FIXTURE_DIR OUT_ROOT   (run it through the GPU mutex)
set -euo pipefail
T=$1; FIX=$2; OUT=$3
export T
source /work/pverify/base/env.sh
mkdir -p "$OUT/strict" "$OUT/default"
args_s=(); args_d=()
for f in "$FIX"/*.in; do
    n=$(basename "$f" .in)
    args_s+=("$f" "$OUT/strict/$n.out"); args_d+=("$f" "$OUT/default/$n.out")
done
ORACLE=$T/tools/wdm6_wrf461_oracle
GPUWM_WRF_EXACT=1 python3 "$ORACLE/run_woof.py" "${args_s[@]}" > "$OUT/strict/receipt.json"
env -u GPUWM_WRF_EXACT python3 "$ORACLE/run_woof.py" "${args_d[@]}" > "$OUT/default/receipt.json"
echo "done $OUT"
