#!/bin/bash
# Banked recipe: build-wrf-novec.sh, adapted only in its root path.
set -x
cd mp28sw || exit 1
export NETCDF=/usr
export J="-j 16"
rm -rf WRF-4.6.1-novec
mkdir WRF-4.6.1-novec
tar xzf WRFv4.6.1.tar.gz -C WRF-4.6.1-novec --strip-components=1
cd WRF-4.6.1-novec || exit 2
sha256sum phys/module_mp_thompson.F phys/module_mp_radar.F
printf '32\n0\n' | ./configure > mp28sw/configure-novec.log 2>&1
sed -i 's/^FCOPTIM *=.*/FCOPTIM         =       -O2 -fno-tree-vectorize/' configure.wrf
grep -E "^FCOPTIM" configure.wrf
./compile em_quarter_ss > mp28sw/compile-novec.log 2>&1
echo "COMPILE-RC=$?"
ls -la main/*.exe
sha256sum main/*.exe 2>/dev/null
echo DONE > mp28sw/build-novec.done
