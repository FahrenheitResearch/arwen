#!/bin/bash
# Banked recipe: build-wrf-novec.sh, adapted only in its root path.
set -x
cd mp28dist || exit 1
export NETCDF=mp28dist/nc
export J="-j 16"
rm -rf WRF-4.6.1-novec
mkdir WRF-4.6.1-novec
tar xzf WRFv4.6.1.tar.gz -C WRF-4.6.1-novec --strip-components=1
cd WRF-4.6.1-novec || exit 2
sha256sum phys/module_mp_thompson.F phys/module_mp_radar.F
printf '32\n0\n' | ./configure > mp28dist/configure-novec.log 2>&1
sed -i '/^SCC[ 	]*=[ 	]*gcc$/c\SCC = gcc -std=gnu89 -fpermissive -Wno-implicit-function-declaration -Wno-implicit-int -Wno-incompatible-pointer-types' configure.wrf
grep -E "^SCC" configure.wrf
sed -i 's/^FCOPTIM *=.*/FCOPTIM         =       -O2 -fno-tree-vectorize/' configure.wrf
grep -E "^FCOPTIM" configure.wrf
./compile em_quarter_ss > mp28dist/compile-novec.log 2>&1
echo "COMPILE-RC=$?"
ls -la main/*.exe
sha256sum main/*.exe 2>/dev/null
echo DONE > mp28dist/build-novec.done
