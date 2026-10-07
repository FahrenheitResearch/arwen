#!/usr/bin/env bash
# sheets.sh: S1 (MRMS | member 0 | PMM | no-DA control | HRRR) for lane 7 E arms, 21Z analysis rows. CPU only.
set -u
R=/work/da-iau-7/run; cd /work/da-iau-7/src
export PYTHONPATH=/work/da-iau-7/run:/work/da-iau-7/src CUDA_VISIBLE_DEVICES= GPUWM_NO_LOCAL_GPU=1
for arm in m8-base m8-3dh m8-4dh; do
  nice -n 10 /work/da-iau-7/venv/bin/python /work/da-iau-7/run/da_cycle_sheets.py --run $R/out/e-$arm/da --noda control \
    --start 2026-10-01T18 --hrrr-cycle 2026-10-01T21 \
    --rw-compare /work/da-iau-7/src/tools/rustwx/target/release/rw_compare --reference-dir $R/refs \
    --name e1001-$arm --out $R/sheets/$arm --source-label "WOOF DA lane 7 $arm" \
    --step "21Z analysis + 2 min=history:10920@3.0333333" --step "f00:30=history:12600@3.5" \
    --step "f01=history:14400@4" --step "f02=history:18000@5" --step "f03=history:21600@6" \
    > $R/sheets-$arm.log 2>&1
  echo "$arm rc $?" >> $R/sheets.events
done
echo done >> $R/sheets.events
