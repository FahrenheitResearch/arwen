#!/bin/bash
# CPU-node variant of the banked provenance.sh: same pins, no GPU/ArWen
# lines (those are collected on the machine that runs ArWen).
T=mp28dist
mkdir -p $T/out
cd $T
{
  echo "== node =="
  hostname; nproc; uname -r
  gfortran --version | head -1; gcc --version | head -1
  ldd --version | head -1; python3 --version 2>&1
  nf-config --version 2>/dev/null; nc-config --version 2>/dev/null
  echo "== tarball =="
  sha256sum WRFv4.6.1.tar.gz
  echo "== executables =="
  sha256sum WRF-4.6.1/main/ideal.exe WRF-4.6.1/main/wrf.exe WRF-4.6.1-novec/main/ideal.exe WRF-4.6.1-novec/main/wrf.exe
  echo "== source pins =="
  sha256sum WRF-4.6.1/phys/module_mp_thompson.F WRF-4.6.1/phys/module_mp_radar.F \
            WRF-4.6.1-novec/phys/module_mp_thompson.F WRF-4.6.1-novec/phys/module_mp_radar.F \
            WRF-4.6.1/run/CCN_ACTIVATE.BIN
  echo "== tables =="
  sha256sum runs/sw-wrf-mp08/*.dat runs/sw-wrf-novec-mp08/*.dat 2>/dev/null
  echo "== initial conditions =="
  sha256sum runs/ic-mp08/wrfinput_d01 runs/ic-mp28/wrfinput_d01
  echo "== case inputs =="
  sha256sum input_sounding mknml.sh runs/ic-mp08/namelist.input runs/ic-mp28/namelist.input \
            runs/sw-wrf-mp08/namelist.input runs/sw-wrf-mp28/namelist.input \
            runs/sw-wrf-novec-mp08/namelist.input runs/sw-wrf-novec-mp28/namelist.input
} > $T/out/provenance.txt 2>&1
( cd $T && find runs -type f \( -name 'wrfout_d01_*' -o -name 'wrfinput_d01' -o -name 'namelist.input' -o -name '*.log' -o -name '*.dat' \) -print0 | sort -z | xargs -0 sha256sum ) > $T/out/SHA256SUMS-node.txt
echo DONE > $T/prov.done
