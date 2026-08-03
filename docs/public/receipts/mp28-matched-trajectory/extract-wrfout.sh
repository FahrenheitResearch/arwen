#!/bin/bash
# Wait for all four WRF runs, then extract each to the comparison layout.
T=mp28traj
cd $T/runs
for i in $(seq 1 200); do
  n=$(ls wrf-*/wrf.rc 2>/dev/null | wc -l)
  [ "$n" = "4" ] && break
  sleep 20
done
echo "rc files: $(cat wrf-*/wrf.rc 2>/dev/null | tr '\n' ' ')"
for TAG in vec-mp08 vec-mp28 novec-mp08 novec-mp28; do
  D=$T/runs/wrf-$TAG
  MP=$(echo $TAG | sed 's/.*mp0*//')
  OUT=$T/runs/wrf-$TAG-x
  W=$(ls $D/wrfout_d01_* 2>/dev/null | head -1)
  echo "extract $TAG mp=$MP from $W"
  /venv/main/bin/python $T/arwen/tools/mp28_matched/extract_wrfout.py \
      --wrfout "$W" --out "$OUT" --mp $MP
done
echo DONE > $T/extract.done
