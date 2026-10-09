#!/bin/bash
# wc8.sh : step 7. The wind check with the never-worse rule (wind7.py) on all six cases, CPU only, on the step-7 tree.
# The hrrr cases run the tree runner's own preflight (the clock decision as a run takes it); the NCAR grids re-derive
# from the door's saved fields and are checked against the step-3 run receipts.
set -u; source /work/tclock/s7/env7.sh
export CUDA_VISIBLE_DEVICES= GPUWM_NO_LOCAL_GPU=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 RAYON_NUM_THREADS=2 TMPDIR=/work/tclock/s7/tmp
O=/work/tclock/s7/wc; mkdir -p $O $TMPDIR; cd /work/tclock/s7
R=/opt/dlami/nvme/tclock/runs
run() { L=$1; shift; s=$(date +%s); nice -n 10 $PY /work/tclock/s7/wind7.py "$@" > $O/$L.log 2>&1; echo "rc=$? wall=$(( $(date +%s)-s ))" > $O/$L.done; }
run ncar12 fixture /work/tclock/results/step2-tree7/door-12-local_face/faces-d01.npz /work/tclock/s3/results/ncar12-local_face/run-receipt.json $R/ncar12-local_face/out/wrfout --out $O/ncar12.json --label ncar12 &
run ncar25 fixture /work/tclock/results/step2-tree7/door-25-local_face/faces-d01.npz /work/tclock/s3/results/ncar25-local_face/run-receipt.json $R/ncar25-local_face/out/wrfout --out $O/ncar25.json --label ncar25 &
for c in la-santaana-250107-48h sf-diablo-210118-48h tor18z-48h bos0400-48h; do
  P=$(ls -d $R/$c-prep/run-*/chain/prep | head -1); W=$(ls -d $R/$c-cand/run-*/wrfout | head -1)
  run $c hrrr $P /work/tclock/s3/cfg/$c.cand.toml $W --out $O/$c.json --label $c &
done
wait; echo done > $O/ALL.done
