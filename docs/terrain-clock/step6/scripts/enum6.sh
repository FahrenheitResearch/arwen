#!/bin/bash
source /work/tclock/s6/env6.sh
export CUDA_VISIBLE_DEVICES= GPUWM_NO_LOCAL_GPU=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 RAYON_NUM_THREADS=2 TMPDIR=/work/tclock/s6/tmp
cd /work/tclock/s6; mkdir -p enum
for c in ncar12 ncar25 la-santaana-250107-48h sf-diablo-210118-48h tor18z-48h bos0400-48h; do
  nice -n 10 $PY enum6.py enum/$c.json $c > enum/$c.log 2>&1 &
done
wait; echo done > enum/ALL.done
