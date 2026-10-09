#!/bin/bash
# step7.sh TREE : terrain-clock local-face lane, STEP 2 (code), CPU only, no card. On NCAR's v4.4 CONUS benchmark inputs
# (fetched from NCAR straight onto this box by ncar/fetch.sh; sha256 of each tarball stream in ncar/*.tgz.sha256): the
# WRF-input door's own clock derivation under terrain_clock = "local_face" and "measured"
# (tools/terrain_clock_local_check.py door), the fields the local reading read saved as fixtures, the wind margin checked
# against NCAR's own reference forecast at the end hour (margin), the reading reproduced from the fixture (reproduce),
# the face groups for the test suite (slim) and the shipped rows the candidate moved (moved).
set -u; source /work/tclock/env6.sh; export CUDA_VISIBLE_DEVICES= GPUWM_NO_LOCAL_GPU=1
TREE=${1:-tree7}; export T=$W/$TREE PYTHONPATH=$W/$TREE
O=$W/results/step2-$TREE; mkdir -p $O; cd $W
echo "tree $(cat $T/TREE-COMMIT 2>/dev/null | tr '\n' ' ') $TREE" > $O/tree.txt
$PY $T/tools/terrain_clock_local_check.py moved --out $O/moved-rows.json > $O/moved.log 2>&1
for c in 12:43200:c12/v4.4_bench_conus12km 25:21600:c25/v4.4_bench_conus2.5km; do
  IFS=: read tag sec src <<< "$c"
  for clock in local_face measured; do
    D=$W/ncar/door-$tag-$clock; rm -rf $D
    $PY $W/ncar_door_dir.py $W/ncar/$src $D $clock > $O/dir-$tag-$clock.log 2>&1
    $PY $T/tools/terrain_clock_local_check.py door $D --run-seconds $sec --out $O/door-$tag-$clock > $O/door-$tag-$clock.log 2>&1
    rm -rf $O/door-$tag-$clock/prep
  done
  F=$O/door-$tag-local_face/faces-d01.npz
  $PY $T/tools/terrain_clock_local_check.py margin $F $W/ncar/$src/wrfout_d01_2019-11-27_00:00:00.gnu --out $O/margin-$tag.json > $O/margin-$tag.log 2>&1
  $PY $T/tools/terrain_clock_local_check.py reproduce $F $O/door-$tag-local_face/terrain-clock.json > $O/reproduce-$tag.json 2>&1
  $PY $T/tools/terrain_clock_local_check.py slim $F --out $O/groups-$tag.json --receipt $O/door-$tag-local_face/terrain-clock.json > $O/slim-$tag.log 2>&1
  sha256sum $W/ncar/$src/wrfinput_d01 $W/ncar/$src/wrfbdy_d01 $W/ncar/$src/namelist.input $W/ncar/$src/wrfout_d01_2019-11-27_00:00:00.gnu > $O/inputs-$tag.sha256
done
cp $W/ncar/*.tgz.sha256 $W/ncar/*.members $O/ 2>/dev/null
echo STEP2_DONE > $O/done
