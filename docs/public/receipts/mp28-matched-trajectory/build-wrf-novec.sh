#!/bin/bash
# Second pristine WRF v4.6.1 tree, identical SOURCE, one optimization flag
# changed: -fno-tree-vectorize, the flag the mp=28 column oracle's build
# scripts pin.  Gives WRF-vs-WRF flag sensitivity as the comparison yardstick.
set -x
cd mp28traj || exit 1
export NETCDF=/usr
export J="-j 16"
rm -rf WRF-4.6.1-novec
mkdir WRF-4.6.1-novec
tar xzf WRFv4.6.1.tar.gz -C WRF-4.6.1-novec --strip-components=1
cd WRF-4.6.1-novec || exit 2
printf '32\n0\n' | ./configure > mp28traj/configure-novec.log 2>&1
sed -i 's/^FCOPTIM *=.*/FCOPTIM         =       -O2 -fno-tree-vectorize/' configure.wrf
grep -E "^FCOPTIM" configure.wrf
./compile em_quarter_ss > mp28traj/compile-novec.log 2>&1
echo "COMPILE-RC=$?"
ls -la main/*.exe
echo DONE > mp28traj/build-novec.done
