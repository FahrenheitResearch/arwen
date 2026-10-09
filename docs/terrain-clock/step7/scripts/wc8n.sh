#!/bin/bash
# wc8n.sh : step 7, the two NCAR grids of wc8.sh again (the fixture route imports tools/terrain_clock_local_check.py
# beside the script; linked from the tree).
set -u; source /work/tclock/s7/env7.sh
export CUDA_VISIBLE_DEVICES= GPUWM_NO_LOCAL_GPU=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 RAYON_NUM_THREADS=2 TMPDIR=/work/tclock/s7/tmp
O=/work/tclock/s7/wc; cd /work/tclock/s7
R=/opt/dlami/nvme/tclock/runs
run() { L=$1; shift; s=$(date +%s); nice -n 10 $PY /work/tclock/s7/wind7.py "$@" > $O/$L.log 2>&1; echo "rc=$? wall=$(( $(date +%s)-s ))" > $O/$L.done; }
run ncar12 fixture /work/tclock/results/step2-tree7/door-12-local_face/faces-d01.npz /work/tclock/s3/results/ncar12-local_face/run-receipt.json $R/ncar12-local_face/out/wrfout --out $O/ncar12.json --label ncar12 &
run ncar25 fixture /work/tclock/results/step2-tree7/door-25-local_face/faces-d01.npz /work/tclock/s3/results/ncar25-local_face/run-receipt.json $R/ncar25-local_face/out/wrfout --out $O/ncar25.json --label ncar25 &
wait; echo done > $O/NCAR.done
