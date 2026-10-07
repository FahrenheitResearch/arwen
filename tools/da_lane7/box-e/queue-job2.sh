#!/usr/bin/env bash
# /work/da-iau-7/queue-job.sh: DA lane 7's slot in box E's queue. Three one-card arms in parallel on the 10-01
# hydro-boundary case (8 members, serial, 3 analyses + 6 h free forecast), each through the E mutex:
#   m8-base (one-shot insertion, today's default), m8-3dh (3D-IAU 30 min + hydrostatic rebalance),
#   m8-4dh (4D-IAU 60 min centred + rebalance). The no-DA reference is each run's own unanalysed control.
set -u
R=/work/da-iau-7/run
log() { echo "$(date -u +%FT%TZ) $*" >> $R/queue.log; }
pids=()
for arm in m8-basep m8-3dhp m8-4dhp; do
  ( log "queued $arm"
    /opt/gpu-mutex/run.sh da-iau-7 --cards 1 --wait 14400 $R/go_e.sh $arm e-$arm > $R/e-$arm.mutex.log 2>&1
    log "done $arm rc $?" ) &
  pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait $p || rc=1; done
log "queue-job end rc $rc"
exit $rc
