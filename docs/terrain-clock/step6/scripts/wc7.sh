#!/bin/bash
# wc7.sh : step 6. The step-4 wind check (wind5.py) on all six cases, CPU only, on the step-6 tree (lane 0a52e0a7f's
# engine with the step-6 files: the 1.5 input-wind floor, the 100 m/s peak rule, the re-derived map and adaptive rows),
# then the test-suite fixtures (face groups) regenerated on the same tree.
set -u; source /work/tclock/s6/env6.sh
export CUDA_VISIBLE_DEVICES= GPUWM_NO_LOCAL_GPU=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 RAYON_NUM_THREADS=2 TMPDIR=/work/tclock/s6/tmp
O=/work/tclock/s6/wc; mkdir -p $O $O/fixtures $TMPDIR; cd /work/tclock/s6
R=/opt/dlami/nvme/tclock/runs
run() { L=$1; shift; s=$(date +%s); nice -n 10 $PY /work/tclock/s4d/wind5.py "$@" > $O/$L.log 2>&1; echo "rc=$? wall=$(( $(date +%s)-s ))" > $O/$L.done; }
run ncar12 fixture /work/tclock/results/step2-tree7/door-12-local_face/faces-d01.npz /work/tclock/s3/results/ncar12-local_face/run-receipt.json $R/ncar12-local_face/out/wrfout --out $O/ncar12.json --label ncar12 &
run ncar25 fixture /work/tclock/results/step2-tree7/door-25-local_face/faces-d01.npz /work/tclock/s3/results/ncar25-local_face/run-receipt.json $R/ncar25-local_face/out/wrfout --out $O/ncar25.json --label ncar25 &
for c in la-santaana-250107-48h sf-diablo-210118-48h tor18z-48h bos0400-48h; do
  P=$(ls -d $R/$c-prep/run-*/chain/prep | head -1); W=$(ls -d $R/$c-cand/run-*/wrfout | head -1)
  run $c hrrr $P /work/tclock/s3/cfg/$c.cand.toml $W --out $O/$c.json --label $c &
done
wait
for c in la-santaana-250107-48h sf-diablo-210118-48h; do
  P=$(ls -d $R/$c-prep/run-*/chain/prep | head -1)
  nice -n 10 $PY /work/tclock/s4d/slim5.py $P /work/tclock/s3/cfg/$c.cand.toml $O/fixtures/$c-d01-groups.json > $O/fixtures/$c.log 2>&1 &
done
( cd $T4 && nice -n 10 $PY tools/terrain_clock_local_check.py slim /work/tclock/results/step2-tree7/door-25-local_face/faces-d01.npz --out $O/fixtures/ncar-conus2p5km-groups.json > $O/fixtures/ncar25.log 2>&1 ) &
wait; echo done > $O/ALL.done
