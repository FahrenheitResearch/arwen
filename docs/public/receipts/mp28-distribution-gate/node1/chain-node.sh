#!/bin/bash
# Orchestration only: runs the four banked scripts in order with done
# flags and wall times.  The banked scripts themselves are byte-identical
# to the committed receipts except the work-root path, per the
# declaration's one permitted adaptation.
T=mp28dist
log(){ echo "$(date -u +%F,%H:%M:%S) $*" >> $T/chain.log; }
log chain-start
{ echo "-- coexistence snapshot (before) --"; uptime; free -g | head -2; ps -eo pid,comm,pcpu,etime --sort=-pcpu | head -12; } >> $T/chain.log 2>&1
S0=$SECONDS; bash $T/build-vec.sh   > $T/build-vec.out 2>&1;  log "build-vec rc=$? wall_s=$((SECONDS-S0))"
S0=$SECONDS; bash $T/build-novec.sh > $T/build-novec.out 2>&1; log "build-novec rc=$? wall_s=$((SECONDS-S0))"
for E in WRF-4.6.1/main/ideal.exe WRF-4.6.1/main/wrf.exe WRF-4.6.1-novec/main/ideal.exe WRF-4.6.1-novec/main/wrf.exe; do
  [ -f $T/$E ] || { log "MISSING $E"; echo FAIL > $T/chain.fail; exit 1; }
done
S0=$SECONDS; bash $T/stage-sw.sh    > $T/stage.out 2>&1;  log "stage rc=$? wall_s=$((SECONDS-S0))"
S0=$SECONDS; bash $T/run-sw-wrf.sh  > $T/swwrf.out 2>&1;  log "swwrf rc=$? wall_s=$((SECONDS-S0))"
bash $T/provenance-node.sh >> $T/chain.log 2>&1
{ echo "-- coexistence snapshot (after) --"; uptime; } >> $T/chain.log 2>&1
echo DONE > $T/chain.done
log chain-complete
