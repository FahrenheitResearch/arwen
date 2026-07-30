#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: $0 /path/to/WRF-v4.6.1 /empty/build-directory" >&2
  exit 2
fi

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
wrf_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
radar_source="$wrf_root/phys/module_mp_radar.F"
thompson_source="$wrf_root/phys/module_mp_thompson.F"

for source in "$radar_source" "$thompson_source"; do
  if [ ! -f "$source" ]; then
    echo "missing official WRF source: $source" >&2
    exit 2
  fi
done
mkdir -p "$build_dir"
for output in qr_acr_qg_V4.dat qr_acr_qsV2.dat freezeH2O.dat \
              thompson_aux_tables.dat; do
  if [ -e "$build_dir/$output" ]; then
    echo "refusing to reuse existing oracle output: $build_dir/$output" >&2
    exit 2
  fi
done

cd "$build_dir"
gfortran -c -O2 -ffree-form -ffree-line-length-none \
  "$script_dir/stub_wrf.F90"
gfortran -c -O2 -ffree-form -ffree-line-length-none "$radar_source"
gfortran -c -O2 -cpp -DWRF_CHEM=0 -ffree-form \
  -ffree-line-length-none "$thompson_source"
gfortran -c -O2 -ffree-form -ffree-line-length-none \
  "$script_dir/generate_tables.F90"
gfortran -O2 -o generate_tables stub_wrf.o module_mp_radar.o \
  module_mp_thompson.o generate_tables.o
./generate_tables | tee generate.log

# A second call must take WRF's read path successfully before the cache is
# accepted.  This catches record-layout or short-write failures immediately.
./generate_tables | tee readback.log

gfortran -c -O2 -ffree-form -ffree-line-length-none \
  "$script_dir/dump_aux_tables.F90"
gfortran -O2 -o dump_aux_tables stub_wrf.o module_mp_radar.o \
  module_mp_thompson.o dump_aux_tables.o
./dump_aux_tables | tee auxiliary.log

gfortran -c -O2 -ffree-form -ffree-line-length-none \
  "$script_dir/run_column.F90"
gfortran -O2 -o run_column stub_wrf.o module_mp_radar.o \
  module_mp_thompson.o run_column.o
mkdir -p column-oracle
for scenario in warm mixed ice condense rain-sed ice-sed cloud-sed snow-sed \
                graupel-sed warm-auto rain-self warm-accrete ice-auto \
                rain-evap snow-subl graupel-subl ice-dep ice-nuc \
                snow-dep graupel-dep snow-melt graupel-melt rain-freeze \
                cloud-freeze graupel-rime snow-rime snow-ice rain-ice \
                rain-snow rain-graupel snow-rime-convert warm-overlap \
                rain-snow-graupel-overlap rain-ice-graupel-overlap \
                frozen-vapor-overlap cold-cloud-overlap \
                frozen-vapor-nucleation-overlap cold-ice-rain-overlap \
                cold-full-overlap cold-cloud-rain-overlap \
                cloud-condense-sed cloud-condense-nofall \
                cloud-rain-condense-sed cloud-rain-condense-nofall \
                condense-fall-attempt warm-frozen-subsat; do
  ./run_column "$scenario" column-oracle | tee "column-$scenario.log"
done

stat -c '%n %s' qr_acr_qg_V4.dat qr_acr_qsV2.dat freezeH2O.dat \
  thompson_aux_tables.dat
sha256sum qr_acr_qg_V4.dat qr_acr_qsV2.dat freezeH2O.dat \
  thompson_aux_tables.dat | tee SHA256SUMS
sha256sum column-oracle/*.csv | tee COLUMN_SHA256SUMS
