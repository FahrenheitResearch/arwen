#!/bin/bash
# wrf_run.sh EXE RUNDIR CASEDIR [dumpdir] : run a scratch wrf.exe on CASEDIR's wrfinput/wrfbdy/namelist; hashes into RUNDIR/rec
set -u
L=/work/pverify/combo/localize; EXE=$1; R=$2; C=$3; DUMP=${4:-}
W=$L/wrf/WRF-4.6.1; P=/work/pverify/combo/recordings/provenance/sweep
rm -rf $R; mkdir -p $R/rec; cd $R
for t in $W/run/*; do n=$(basename $t); case $n in *.exe|namelist.input) continue;; esac; ln -s $t $n; done
for t in /work/pverify/wrf-build/thompson-tables/*.dat; do ln -sf $t .; done
ln -s $(readlink -f $C/wrfinput_d01) wrfinput_d01; ln -s $(readlink -f $C/wrfbdy_d01) wrfbdy_d01
cp $C/namelist.input .; cp $P/iofields.txt .
if [ -n "$DUMP" ]; then mkdir -p $DUMP; export WRF_LOCDUMP=$DUMP; fi
timeout 1800 nice -n 5 $EXE > wrf.log 2>&1; echo "wrf rc $?"; tail -1 wrf.log
/work/pverify/base/venv/bin/python $P/extract.py $(ls wrfout_d01_* | head -1) $R/rec 1,5,20,last 1 > $R/rec/extract.log 2>&1; echo "extract rc $?"
