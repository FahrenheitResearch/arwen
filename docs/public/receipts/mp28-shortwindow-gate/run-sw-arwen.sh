#!/bin/bash
# ArWen short-window runs, each configuration twice (dual-run byte screen;
# the card has no ECC).  Initial condition: build A's t=1800 s history frame.
set -x
cd mp28sw/arwen || exit 1
export GPUWM_THOMPSON_TABLE_ROOT=mp28sw/arwen/gpuwm/data/thompson/tables
T=mp28sw
PY=/venv/mp28/bin/python
for MP in 8 28; do
  M=$(printf %02d $MP)
  for REP in a b; do
    OUT=$T/runs/sw-arwen-mp$M; [ $REP = b ] && OUT=$T/runs/sw-arwen-mp$M-b
    rm -rf $OUT
    $PY tools/mp28_matched/run_arwen.py \
      --wrfinput $T/runs/ic-mp$M/wrfinput_d01 \
      --restart-from $(ls $T/runs/sw-wrf-mp$M/wrfout_d01_* | head -1) \
      --restart-frame 0 --out $OUT --mp $MP --steps 50 --frame-steps 5 --dt 12
    echo "SW-ARWEN-RC-$M-$REP=$?"
  done
done
echo DONE > $T/swarwen.done
