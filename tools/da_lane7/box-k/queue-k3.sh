#!/usr/bin/env bash
# queue-k.sh: DA lane 7 on box K after the gate resume. At most 3 cards at once (three chains):
#   chain 1: GPU tests (1 card, ~1 min), then m8-base, m8-hyd   chain 2: m8-3d, m8-3dh   chain 3: m8-4d, m8-4dh
# Case: da-tune's 10-01 build (/work/da-tune/build/20261001), 8 members serial; no-DA = each run's own control.
set -u
R=/work/da-iau-7/run
log() { echo "$(date -u +%FT%TZ) $*" >> $R/queue.log; }
gputest() {
  log "queued gputest"
  /opt/gpu-mutex/run.sh da-iau-7 --cards 1 --wait 14400 bash -c 'cd /work/da-iau-7/src && NUMPY_MADVISE_HUGEPAGE=0 CUPY_CACHE_DIR=/work/da-iau-7/cache/cupy TMPDIR=/work/da-iau-7/tmp nice -n 10 ../venv/bin/python -m pytest -q -s -p no:cacheprovider tests/test_da_iau_forcing_gpu.py' > $R/gputest.log 2>&1
  log "done gputest rc $?"
}
chain() {
  for arm in "$@"; do
    log "queued $arm"
    # wait for disk BEFORE taking a card (go.sh refuses under 60 GB free; a card held while waiting is an idle card)
    until [ "$(df -BG --output=avail /work | tail -1 | tr -dc 0-9)" -ge 75 ]; do sleep 60; done
    /opt/gpu-mutex/run.sh da-iau-7 --cards 1 --wait 14400 $R/go.sh $arm k-$arm > $R/k-$arm.mutex.log 2>&1
    log "done $arm rc $?"
  done
}
chain m8-base m8-hyd &
chain m8-3d m8-3dh &
chain m8-4d m8-4dh &
wait
log "queue-k end"
