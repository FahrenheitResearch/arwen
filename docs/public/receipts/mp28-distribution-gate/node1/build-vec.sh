#!/bin/bash
# Banked recipe: docs/public/receipts/mp28-matched-trajectory/build-wrf-vec.sh
# adapted only in its root path (mp28dist) and log names.
set -x
cd mp28dist || exit 1
export NETCDF=mp28dist/nc
export J="-j 16"
rm -rf WRF-4.6.1
tar xzf WRFv4.6.1.tar.gz
mv WRFV4.6.1 WRF-4.6.1 2>/dev/null || mv WRFV4.61 WRF-4.6.1 2>/dev/null || true
ls -d WRF*
cd WRF-4.6.1 || { ls mp28dist; exit 2; }
sha256sum phys/module_mp_thompson.F phys/module_mp_radar.F run/CCN_ACTIVATE.BIN
printf '32\n0\n' | ./configure > mp28dist/configure-vec.log 2>&1
sed -i '/^SCC[ 	]*=[ 	]*gcc$/c\SCC = gcc -std=gnu89 -fpermissive -Wno-implicit-function-declaration -Wno-implicit-int -Wno-incompatible-pointer-types' configure.wrf
grep -E "^SCC" configure.wrf
echo "CONFIGURE-RC=$?"
grep -E "^(SFC|SCC|FCOPTIM|FCBASEOPTS_NO_G)" configure.wrf
./compile em_quarter_ss > mp28dist/compile-vec.log 2>&1
echo "COMPILE-RC=$?"
ls -la main/*.exe
sha256sum main/*.exe 2>/dev/null
echo DONE > mp28dist/build-vec.done
