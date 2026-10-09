#!/bin/bash
# validate4.sh : inside the mutex, 1 card, 1 process. The step-4 probe (adaptive cell, map criterion, generated dynamics,
# 3 h, first step 5 s/km) re-run on two shipped adaptive cells, to tie the instrument to the shipped entries:
# 2 km 4500 m ridge 0.15 at 50 m/s (shipped: none held, tried from 15 s/km) and 2 km 3000 m ridge 0.25 at 50 m/s
# (shipped: 9 s/km held, 10 s/km and longer stopped), each at 15 s/km on both target pairs, and the 3000 m cell at 9 s/km.
source /work/tclock/s4/env-s4.sh; cd $W4; O=$W4/results/validate4.jsonl
for pair in "1.2 0.84" "1.4 0.98"; do set -- $pair
  $PY tools/terrain_clock_probe.py adaptive --dx 2000 --crest 4500 --ridge-slopes 0.15 --winds 50 --max-steps 30 --start-step 10 --seconds 10800 --target-cfl $1 --target-hcfl $2 --out $O
  $PY tools/terrain_clock_probe.py adaptive --dx 2000 --crest 3000 --ridge-slopes 0.25 --winds 50 --max-steps 30,18 --start-step 10 --seconds 10800 --target-cfl $1 --target-hcfl $2 --out $O
done > $W4/logs/validate4.log 2>&1
echo done >> $W4/logs/validate4.log
