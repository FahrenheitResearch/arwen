#!/usr/bin/env bash
# reap.sh: once a lane 7 arm writes done.json, delete its spent stage and increment dumps (its composites,
# cycle report and logs stay). Lane 7 paths only.
R=/work/da-iau-7/run
end=$(( $(date +%s) + 6*3600 ))
while [ $(date +%s) -lt $end ]; do
  for d in $R/out/k-m8-*; do
    [ -f $d/done.json ] && [ ! -f $d/reaped ] || continue
    before=$(du -sm $d | cut -f1)
    rm -rf $d/da/stage; rm -f $d/da/increments_*.npz
    echo "$(date -u +%FT%TZ) freed $(( before - $(du -sm $d | cut -f1) )) MB" > $d/reaped
  done
  sleep 60
done
