#!/bin/bash
# Self-driving remainder of the mp28 short-window gate lane.  Every stage
# fires on its predecessor's done-flag; no controlling session is needed.
T=mp28sw
log(){ echo "$(date -u +%F,%H:%M:%S) $*" >> $T/chain.log; }
log chain-start
for i in $(seq 1 480); do [ -f $T/swwrf.done ] && break; sleep 15; done
[ -f $T/swwrf.done ] || { log TIMEOUT-swwrf; echo TIMEOUT-SWWRF > $T/chain.fail; exit 1; }
log swwrf-done
bash $T/run-sw-arwen.sh >> $T/chain.log 2>&1
log arwen-done
bash $T/run-gate.sh >> $T/chain.log 2>&1
log gate-done
bash $T/provenance.sh >> $T/chain.log 2>&1
log prov-done
tar czf $T/out-bundle.tar.gz -C $T runs out swwrf.out chain.log build-vec.out build-novec.out configure-vec.log configure-novec.log stage.log venv.log 2>> $T/chain.log
sha256sum $T/out-bundle.tar.gz > $T/out-bundle.sha256
log bundle-done
for i in $(seq 1 240); do [ -f $T/arwen-les.ready ] && break; sleep 15; done
if [ -f $T/arwen-les.ready ]; then
  bash $T/run-crosscard.sh >> $T/chain.log 2>&1
  log crosscard-done
  tar czf $T/crosscard-bundle.tar.gz -C $T crosscard 2>> $T/chain.log
  sha256sum $T/crosscard-bundle.tar.gz > $T/crosscard-bundle.sha256
  log crosscard-bundled
else
  log crosscard-SKIPPED-no-les-tree
fi
echo DONE > $T/chain.done
log chain-complete
