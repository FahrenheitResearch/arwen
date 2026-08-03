#!/bin/bash
# Short-window paired test, take 3.  No WRF restart: run continuously from
# t=0 to t=2400 s and begin writing history at t=1800 s, every 60 s.  Both
# models are then compared from the SAME mature WRF state at t=1800 s.
set -x
T=mp28traj
for MP in 8 28; do
  M=$(printf %02d $MP)
  D=$T/runs/sw-wrf-mp$M
  rm -rf $D; mkdir -p $D; cd $D
  ln -sf $T/WRF-4.6.1/main/wrf.exe .
  cp $T/runs/ic-mp$M/wrfinput_d01 .
  cp $T/WRF-4.6.1/run/CCN_ACTIVATE.BIN .
  cp $T/probe28/freezeH2O.dat $T/probe28/qr_acr_qg_V4.dat $T/probe28/qr_acr_qsV2.dat .
  sed -e 's/^ run_minutes .*/ run_minutes                         = 40,/' \
      -e 's/^ end_minute .*/ end_minute                          = 40,/' \
      -e 's/^ history_interval .*/ history_interval                    = 1,\n history_begin_m                     = 30,/' \
      $T/runs/ic-mp$M/namelist.input > namelist.input
  ./wrf.exe > wrf.log 2>&1; echo "SW3-RC-$M=$?"
  ls wrfout_d01_* 2>/dev/null
  ncdump -v Times wrfout_d01_* 2>/dev/null | grep -c "0001-01-01"
done
cd $T/arwen
export GPUWM_THOMPSON_TABLE_ROOT=$T/arwen/gpuwm/data/thompson/tables
for MP in 8 28; do
  M=$(printf %02d $MP)
  for REP in a b; do
    OUT=$T/runs/sw-arwen-mp$M; [ $REP = b ] && OUT=$T/runs/sw-arwen-mp$M-b
    rm -rf $OUT
    /venv/main/bin/python tools/mp28_matched/run_arwen.py \
      --wrfinput $T/runs/ic-mp$M/wrfinput_d01 \
      --restart-from $(ls $T/runs/sw-wrf-mp$M/wrfout_d01_* | head -1) \
      --restart-frame 0 --out $OUT --mp $MP --steps 50 --frame-steps 5 --dt 12
    echo "SW3-ARWEN-RC-$M-$REP=$?"
  done
done
/venv/main/bin/python $T/arwen/tools/mp28_matched/shortwindow.py --runs $T/runs --out $T/out/shortwindow.json
echo DONE > $T/shortwin3.done
