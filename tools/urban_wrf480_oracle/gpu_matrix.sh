#!/usr/bin/env bash
# Run only inside the lane's authorized card hold.
set -uo pipefail
lane=$(realpath "$1")
repo=$(cd "$(dirname "$0")/../.." && pwd)
failed=0
for mode in default strict;do
  if [[ $mode == strict ]];then export GPUWM_WRF_EXACT=1;else unset GPUWM_WRF_EXACT;fi
  for table in nlcd lcz;do
    lc=0;if [[ $table == lcz ]];then lc=1;fi
    for arm in tt0 tt1 tree0;do
      tree=1;tt=0;fraction=.2
      if [[ $arm == tt1 ]];then tt=1;fi
      if [[ $arm == tree0 ]];then tree=0;fraction=0;fi
      name="$mode-$table-$arm"
      python "$repo/tools/urban_wrf480_oracle/grade_full.py" --gpu \
        --listing "$repo/gpuwm/core/kernels/urban_ucm480.cuh" \
        --csv "$repo/tests/data/oracles/urban/ucm480/$table-$arm.csv.gz" \
        --tables "$repo/gpuwm/data/urban/wrf480" --out "$lane/$name.json" \
        --lcz "$lc" --tree "$tree" --tt "$tt" --fvg "$fraction" --switch groption=0 \
        > "$lane/$name.log" 2>&1
      status=$?;printf '%s %s\n' "$name" "$status";tail -n 3 "$lane/$name.log"
      if [[ $status != 0 ]];then printf '%s\n' 1 > "$lane/gpu-matrix.exit";exit 1;fi
    done
  done
done
printf '%s\n' "$failed" > "$lane/gpu-matrix.exit"
exit "$failed"
