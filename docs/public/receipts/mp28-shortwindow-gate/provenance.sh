#!/bin/bash
# Hash every input and output that determines or carries the result.
T=mp28sw
cd $T
{
  echo "== node =="
  hostname; nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader
  gfortran --version | head -1; ldd --version | head -1; python3 --version
  /venv/mp28/bin/python -c "import numpy, netCDF4, cupy; print('numpy', numpy.__version__, 'netCDF4', netCDF4.__version__, 'cupy', cupy.__version__)"
  echo "== tarball =="
  sha256sum WRFv4.6.1.tar.gz arwen-src.tar.gz
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
( cd $T && find runs out -type f \( -name 'wrfout_d01_*' -o -name '*.npz' -o -name '*.json' -o -name 'namelist.input' -o -name '*.log' \) -print0 | sort -z | xargs -0 sha256sum ) > $T/out/SHA256SUMS-node.txt
echo DONE > $T/prov.done
