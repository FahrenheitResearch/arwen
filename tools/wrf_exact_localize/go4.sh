#!/bin/bash
# go4.sh TAG CASE:SECONDS... : WOOF arm of the localize pair on the base4 tree (strict, controls on by default),
# dumps for steps 1..WOOF_LOCDUMP_STEPS (default 2), history every step, hashes.  GPU: call under the mutex.
set -u
L=/work/pverify/combo/localize; B=/work/pverify/base4; TAG=$1; shift
echo "cards=${GPU_MUTEX_CARDS:-?} start $(date -u +%FT%TZ)"
for case in "$@"; do
  S=${case#*:}; case=${case%%:*}
  O=$B/runs/woof-$case-$TAG; rm -rf $O; mkdir -p $O/stage; CD=$L/cases/$case; [ -d $B/cases/$case ] && CD=$B/cases/$case
  for f in wrfinput_d01 wrfbdy_d01; do ln -s $(readlink -f $CD/$f) $O/stage/$f; done
  cp $CD/namelist.input $O/stage/
  ( cd $B && env GPUWM_WRF_EXACT=1 CUPY_CACHE_DIR=$B/cupy-cache PYTHONPATH=$B/src:$B PYTHONDONTWRITEBYTECODE=1 \
      WOOF_LOCDUMP=$O/dump WOOF_LOCDUMP_STEPS=${WOOF_LOCDUMP_STEPS:-2} WOOF_HOOK3=1 \
      /work/pverify/base2/venv/bin/python $B/locrun4.py $L/woof_run.py $O/stage $O/run $S > $O/run.log 2>&1 )
  echo "$case rc=$? $(tail -1 $O/run.log)"
  /work/pverify/base2/venv/bin/python $L/woof_hashes.py $O/run/wrfout $O/rec > $O/hash.log 2>&1; echo "hash rc $?"
done
echo "end $(date -u +%FT%TZ)"
