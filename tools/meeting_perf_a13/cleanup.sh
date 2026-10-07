#!/usr/bin/env bash
# After r2 ends: keep small receipts under keep/, delete the bulk (prepared case, run outputs, engine copies, geog overlay).
W=/work/meeting-perf-a13; cd $W
until [ -f GPU-DONE ]; do sleep 30; done
mkdir -p keep
cp gpuE.out GPU-DONE price22.json hrrrplot.py hexplot.py case/c3/experiment.toml case/c3/proof-inputs.json keep/ 2>/dev/null
cp case/c3/prepared/proof.json keep/prepared-proof.json 2>/dev/null
for r in runs/*; do n=$(basename $r); mkdir -p keep/$n; cp $r/run.log $r/nvml.csv $r/frames.sha256 keep/$n/ 2>/dev/null
  cp $r/out/report.json $r/out/progress.jsonl keep/$n/ 2>/dev/null; done
b=$(du -sb . | cut -f1)
rm -rf case runs engine engine-meas geog cache tmp wif
echo "freed $(( (b - $(du -sb . | cut -f1)) / 1048576 )) MiB; left $(du -sh . | cut -f1)" > CLEANED
