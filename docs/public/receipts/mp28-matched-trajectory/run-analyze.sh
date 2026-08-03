#!/bin/bash
T=mp28traj
PY=/venv/main/bin/python
for i in $(seq 1 300); do [ -f $T/extract.done ] && break; sleep 15; done
ls $T/runs/wrf-*-x/series.json
$PY $T/arwen/tools/mp28_matched/compare.py --runs $T/runs --out $T/out --wrf-build vec --as-model arwen
echo "--- CONTROL: WRF-novec in the tested slot ---"
$PY $T/arwen/tools/mp28_matched/compare.py --runs $T/runs --out $T/out-control --wrf-build vec --as-model novec
$PY $T/arwen/tools/mp28_matched/plots.py --runs $T/runs --wrfinput $T/runs/ic-mp28/wrfinput_d01 --out $T/png --comparison $T/out/comparison.json
echo DONE > $T/analyze.done
