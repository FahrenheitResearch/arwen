#!/bin/bash
# fetch.sh: NCAR v4.4 CONUS benchmark tarballs straight from NCAR onto this box (no box-to-box copy). Keeps the inputs,
# the namelist and NCAR reference wrfouts; sha256 of each tarball taken from the same stream; members listed. CPU only.
cd /work/tclock/ncar
B=https://www2.mmm.ucar.edu/wrf/users/benchmark/v44
get() {  # NAME DIR
  mkdir -p $2
  curl -sS --retry 5 $B/$1 | tee >(sha256sum > $2.tgz.sha256) | tar -xvz -C $2 --wildcards "*wrfinput_d01" "*wrfbdy_d01" "*namelist.input" "*wrfout*" > $2.members 2>&1
  echo "$2 done rc=${PIPESTATUS[*]} $(date -u +%FT%TZ)"
}
echo "start $(date -u +%FT%TZ)"
get v4.4_bench_conus12km.tar.gz c12 > c12.log 2>&1 &
get v4.4_bench_conus2.5km.tar.gz c25 > c25.log 2>&1 &
wait
echo "end $(date -u +%FT%TZ)"
