#!/bin/bash
# ArWen side of the matched case.  Each configuration is run TWICE into
# separate directories: the 5090 has no ECC, so a byte comparison of two
# independent runs is the corruption detector.
set -x
cd mp28traj/arwen || exit 1
export GPUWM_THOMPSON_TABLE_ROOT=mp28traj/arwen/gpuwm/data/thompson/tables
PY=/venv/main/bin/python
for MP in 8 28; do
  M=$(printf %02d $MP)
  for REP in a b; do
    $PY tools/mp28_matched/run_arwen.py \
        --wrfinput mp28traj/runs/ic-mp$M/wrfinput_d01 \
        --out mp28traj/runs/arwen-mp$M-$REP \
        --mp $MP --steps 600 --frame-steps 50 --dt 12
    echo "RC-mp$M-$REP=$?"
  done
done
echo DONE > mp28traj/arwen.done
