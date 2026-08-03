#!/bin/bash
# Pristine WRF v4.6.1 build for the mp=28 matched-trajectory lane.
# Isolated in mp28traj -- touches nothing else on the node.
set -x
cd mp28traj || exit 1
export NETCDF=/usr
export J="-j 16"
rm -rf WRF-4.6.1
tar xzf WRFv4.6.1.tar.gz
mv WRFV4.6.1 WRF-4.6.1 2>/dev/null || mv WRF-4.6.1-* WRF-4.6.1 2>/dev/null || true
ls -d WRF*
cd WRF-4.6.1 || { ls mp28traj; exit 2; }
# 32 = GNU gfortran/gcc serial ; nesting 0 (single domain, idealized)
printf '32\n0\n' | ./configure > mp28traj/configure.log 2>&1
echo "CONFIGURE-RC=$?"
grep -E "^(SFC|SCC|FCOPTIM|FCBASEOPTS|FCDEBUG)" configure.wrf | head -20
./compile em_quarter_ss > mp28traj/compile.log 2>&1
echo "COMPILE-RC=$?"
ls -la main/*.exe
echo DONE > mp28traj/build.done
