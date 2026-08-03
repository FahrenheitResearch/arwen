#!/bin/bash
# The four WRF short-window runs: continuous 0->2400 s, history every 60 s
# from 1800 s (the banked take-3 sed recipe), both builds, both schemes.
# mp08 of each build runs first and generates that build's Thompson tables;
# mp28 of the same build consumes copies of them.
set -x
T=mp28sw
for BUILD in vec novec; do
  case $BUILD in
    vec)   SRC=$T/WRF-4.6.1;       TAGP=sw-wrf ;;
    novec) SRC=$T/WRF-4.6.1-novec; TAGP=sw-wrf-novec ;;
  esac
  for MP in 8 28; do
    M=$(printf %02d $MP)
    D=$T/runs/$TAGP-mp$M
    rm -rf $D; mkdir -p $D; cd $D
    ln -sf $SRC/main/wrf.exe .
    cp $T/runs/ic-mp$M/wrfinput_d01 .
    cp $SRC/run/CCN_ACTIVATE.BIN .
    sed -e 's/^ run_minutes .*/ run_minutes                         = 40,/' \
        -e 's/^ end_minute .*/ end_minute                          = 40,/' \
        -e 's/^ history_interval .*/ history_interval                    = 1,\n history_begin_m                     = 30,/' \
        $T/runs/ic-mp$M/namelist.input > namelist.input
  done
done
run_one () {
  cd $T/runs/$1
  local S0=$SECONDS
  ./wrf.exe > wrf.log 2>&1
  local RC=$?
  echo "wall_seconds=$((SECONDS-S0)) rc=$RC" > time.log
  echo "RC-$1=$?"
  ls wrfout_d01_* 2>/dev/null
}
# mp08 pair concurrently (each generates its build's tables)...
run_one sw-wrf-mp08 & P1=$!
run_one sw-wrf-novec-mp08 & P2=$!
wait $P1 $P2
# ...then hand each build's tables to its mp28 twin and run those.
cp $T/runs/sw-wrf-mp08/freezeH2O.dat $T/runs/sw-wrf-mp08/qr_acr_qg_V4.dat \
   $T/runs/sw-wrf-mp08/qr_acr_qsV2.dat $T/runs/sw-wrf-mp28/
cp $T/runs/sw-wrf-novec-mp08/freezeH2O.dat $T/runs/sw-wrf-novec-mp08/qr_acr_qg_V4.dat \
   $T/runs/sw-wrf-novec-mp08/qr_acr_qsV2.dat $T/runs/sw-wrf-novec-mp28/
sha256sum $T/runs/sw-wrf-mp08/*.dat $T/runs/sw-wrf-novec-mp08/*.dat
run_one sw-wrf-mp28 & P3=$!
run_one sw-wrf-novec-mp28 & P4=$!
wait $P3 $P4
echo DONE > $T/swwrf.done
