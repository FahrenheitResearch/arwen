#!/usr/bin/env bash
set -u; D=/work/da-iau-7; E=$D/src; LOG=$D/logs
source $HOME/.cargo/env
export CUDA_VISIBLE_DEVICES= GPUWM_NO_LOCAL_GPU=1 GPUWM_BRIDGE_SOURCE_REV=$(cat $E/.engine-export-sha) TMPDIR=$D/tmp
(cd $E/tools/grib1_bridge && nice -n 10 cargo build --release --locked --offline --bins --lib -j 24) > $LOG/build-grib.log 2>&1 & P1=$!
(cd $E/tools/rustwx && nice -n 10 cargo build --release --locked --offline -p netcdf-writer -p static-fields -p rw-netcdf -p rw-fetch -p rw-wrfbatch -p obs-score -p rw-obs -p rw-nexrad -j 32) > $LOG/build-rustwx.log 2>&1 & P2=$!
(cd $E/tools/rw_wps && nice -n 10 cargo build --release --locked --offline -j 24) > $LOG/build-rwwps.log 2>&1 & P3=$!
(cd $E/tools/region_global_dealias && nice -n 10 cargo build --release --locked --offline -j 12) > $LOG/build-dealias.log 2>&1 & P4=$!
for p in $P1 $P2 $P3 $P4; do wait $p; echo "cargo pid $p rc $?" >> $LOG/build.events; done; echo built
