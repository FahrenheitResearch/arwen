#!/usr/bin/env bash
# Build both pristine generation oracles and read v4.4's init coefficients.
# Sources are public WRF tag snapshots; tables are reused from the benchmark.
# Usage: build.sh OWNED_LANE_ROOT WRF44_PHYS WRF461_PHYS BENCHMARK_DIRECTORY
set -euo pipefail
L=$(realpath "$1"); P44=$(realpath "$2"); P461=$(realpath "$3"); DATA=$(realpath "$4")
T=$(CDPATH= cd -- "$(dirname "$0")/../.." && pwd)
echo 'c89bd2ec24a2e6bc95a6ec6d47ad7239a1656c1eca3cc1e70f5089345c6211cb' " $P44/module_mp_thompson.F" | sha256sum -c -
echo 'fabf19e2a9073cff886e882b187080bfdf089d3fd40c0fce1d19bc93b1e5e802' " $P461/module_mp_thompson.F" | sha256sum -c -
for P in "$P44" "$P461"; do
  echo 'aa99da858be41efa579966680708d230123a7417560af0eb2e24f4c94e253688' " $P/module_mp_radar.F" | sha256sum -c -
done
flags=(-O2 -fno-tree-vectorize -ffree-form -ffree-line-length-none)
for version in 44 461; do
  B=$L/oracle$version; mkdir -p "$B/run"
  printf '%s\n' "$B" >> "$L/MANIFEST.txt"
  P=$P461; if [ "$version" = 44 ]; then P=$P44; fi
  cd "$B"
  gfortran -c "${flags[@]}" "$T/tools/thompson_wrf461_oracle/stub_wrf.F90" "$P/module_mp_radar.F"
  gfortran -c "${flags[@]}" -cpp -DWRF_CHEM=0 "$P/module_mp_thompson.F"
  gfortran -c "${flags[@]}" "$T/tools/thompson_real_column_parity/run_columns_classic.F90"
  gfortran "${flags[@]}" -o run_columns_classic stub_wrf.o module_mp_radar.o module_mp_thompson.o run_columns_classic.o
  if nm -D run_columns_classic | grep _ZGV > /dev/null; then echo 'libmvec changes the scalar arithmetic'; exit 2; fi
  if [ "$version" = 44 ]; then
    for name in qr_acr_qgV3.dat qr_acr_qsV2.dat freezeH2O.dat; do
      if [ ! -e "$B/run/$name" ]; then ln -s "$DATA/$name" "$B/run/$name"; fi
    done
  else
    for name in qr_acr_qg_V4.dat qr_acr_qsV2.dat freezeH2O.dat; do
      if [ ! -e "$B/run/$name" ]; then ln -s /work/engine-base/gpuwm-data/gpuwm_data/data/thompson/tables/$name "$B/run/$name"; fi
    done
  fi
  { gfortran --version | sed -n '1p'; ldd --version | sed -n '1p'; printf '%s\n' "${flags[*]}"; sha256sum "$P/module_mp_thompson.F" "$P/module_mp_radar.F" run_columns_classic; } > "$B/BUILD.txt"
done
B=$L/coeff44; mkdir -p "$B"; printf '%s\n' "$B" >> "$L/MANIFEST.txt"
python "$T/tools/thompson_generation_oracle/translate.py" dump-init "$P44/module_mp_thompson.F" "$B"
cd "$B"
gfortran -c "${flags[@]}" "$T/tools/thompson_wrf461_oracle/stub_wrf.F90" "$P44/module_mp_radar.F"
gfortran -c "${flags[@]}" -cpp -DWRF_CHEM=0 module_mp_thompson-dump.F dump_coefficients.F90
gfortran "${flags[@]}" -o dump_coefficients stub_wrf.o module_mp_radar.o module_mp_thompson-dump.o dump_coefficients.o
cd "$L/oracle44/run"
GFORTRAN_CONVERT_UNIT=big_endian:63,64,65 "$B/dump_coefficients" > "$B/init.log"
cp coefficients.bin "$B/coefficients.bin"
python "$T/tools/thompson_generation_oracle/translate.py" generate "$P44/module_mp_thompson.F" "$B" --output "$B/thompson_column_generated.cuh"
sha256sum "$B/coefficients.bin" "$B/thompson_column_generated.cuh" > "$B/GENERATED.sha256"
