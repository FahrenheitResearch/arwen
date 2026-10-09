#!/bin/bash
# step7b.sh : the 12 km local-face door again with the Rust static-fields bridge installed (step7 ran on the pure-Python
# projection fallback, logged as WORKAROUND); the fixture arrays and the receipt must match step7s exactly. CPU only.
set -u; source /work/tclock/env6.sh; export CUDA_VISIBLE_DEVICES= GPUWM_NO_LOCAL_GPU=1 PYTHONPATH=$W/tree7
O=$W/results/step2-tree7/bridge-check; mkdir -p $O
D=$W/ncar/door-12-local_face-bridge; rm -rf $D
$PY $W/ncar_door_dir.py $W/ncar/c12/v4.4_bench_conus12km $D local_face > $O/dir.log 2>&1
$PY $W/tree7/tools/terrain_clock_local_check.py door $D --run-seconds 43200 --out $O/door-12-local_face > $O/door.log 2>&1
rm -rf $O/door-12-local_face/prep
$PY - <<EOP > $O/compare.txt 2>&1
import json, numpy as np
a = np.load("$W/results/step2-tree7/door-12-local_face/faces-d01.npz"); b = np.load("$O/door-12-local_face/faces-d01.npz")
same = {k: bool(np.array_equal(a[k], b[k])) for k in a.files}
ra = json.load(open("$W/results/step2-tree7/door-12-local_face/terrain-clock.json")); rb = json.load(open("$O/door-12-local_face/terrain-clock.json"))
print("arrays", same); print("receipt identical", ra == rb)
EOP
grep -c WORKAROUND $O/door.log > $O/workaround-lines.txt
echo done > $O/done
