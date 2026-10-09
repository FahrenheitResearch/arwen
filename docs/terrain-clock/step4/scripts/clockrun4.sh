#!/bin/bash
# clockrun4.sh : CPU only, no card. The candidate clock's reading on the four HRRR long-run cases with the step-4 adaptive
# rows (clock4.py, variants as-is and none-held-cap), plus clock3.py unchanged as the control (must equal step 3's).
set -u; source /work/tclock/s4/env-s4.sh; export CUDA_VISIBLE_DEVICES= GPUWM_NO_LOCAL_GPU=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 RAYON_NUM_THREADS=2
R=/opt/dlami/nvme/tclock/runs; O=$W4/results/clock4; mkdir -p $O; cd $W4
for c in sf-diablo-210118-48h la-santaana-250107-48h tor18z-48h bos0400-48h; do
  P=$(sed -n 's/^prepared: //p' $S/results/$c-cand/meta.txt); CF=$S/cfg/$c.cand.toml
  nice -n 10 $PY $S/clock3.py $P $CF $O/$c.control.json > $O/$c.control.log 2>&1
  for v in as-is none-held-cap; do
    nice -n 10 $PY $W4/clock4.py $W4/delta4.json $v $P $CF $O/$c.$v.json > $O/$c.$v.log 2>&1
  done
done
echo done > $O/done
