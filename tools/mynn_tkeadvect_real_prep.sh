#!/usr/bin/env bash
# Public input acquisition and CPU preparation for the one-hour suite gate.
set -euo pipefail
: "${L:?}" "${T:?}"
export GPUWM_FETCH_LOCK_ROOT="$L/fetch-locks"
export GPUWM_RW_NETCDF=/work/engine-base/tools/rustwx/target/release/rw_netcdf
export GPUWM_NCWRITE_BRIDGE=/work/engine-base/tools/rustwx/target/release/libnetcdf_writer.so
export GPUWM_STATIC_BRIDGE=/work/engine-base/tools/rustwx/target/release/libstatic_fields.so
export GPUWM_RW_FETCH=/work/engine-base/tools/rustwx/target/release/rw_fetch
export GPUWM_GRIB2_INVENTORY=/work/engine-base/tools/grib1_bridge/target/release/grib2_inventory
export GPUWM_GRIB2_DUMP=/work/engine-base/tools/grib1_bridge/target/release/grib2_dump
export GPUWM_CPU_PREPROCESS_BRIDGE=/work/engine-base/tools/grib1_bridge/target/release/libgpuwm_preprocess_cpu.so
export GPUWM_HRRR_DECODER=/work/engine-base/tools/grib1_bridge/target/release/hrrr_grib2_bridge
export PATH=/work/engine-base/tools/grib1_bridge/target/release:/work/engine-base/tools/rw_wps/target/release:$PATH
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES= GPUWM_NO_LOCAL_GPU=1
printf '%s\n' "$L/real-inputs" "$L/geog" "$L/real-prepared" "$L/real-config" "$L/fetch-locks" "$L/fetch-cache" >> "$L/MANIFEST.txt"
mkdir -p "$L/real-config"
cd "$T"
python - <<'PY'
import os
from pathlib import Path
lane = Path(os.environ['L'])
root = Path(os.environ['T'])
config = (root/'configs/hrrr_native_quick_demo.toml').read_text()
config = config.replace('[shared]', '[shared]\nbl_mynn_tkeadvect = true\nbl_mynn_mixlength = 2\nbl_mynn_version = "gsd_41"')
config = config.replace('sf_surface_physics = 2', 'sf_surface_physics = 3')
config = config.replace('num_soil_layers = 4', 'num_soil_layers = 9')
config = config.replace('history_interval_s = 900.0', 'history_interval_s = 3600.0')
config = config.replace('restart_interval_s = 0.0', 'restart_interval_s = 3600.0')
(lane/'real-config/experiment.toml').write_text(config)
namelist = (root/'configs/hrrr_native_quick_demo.namelist.input').read_text()
namelist = namelist.replace('&physics', '&physics\n bl_mynn_tkeadvect = .true.,\n bl_mynn_mixlength = 2,')
import re
namelist = re.sub(r'(sf_surface_physics\s*=\s*)2', r'\g<1>3', namelist)
namelist = re.sub(r'(num_soil_layers\s*=\s*)4', r'\g<1>9', namelist)
(lane/'real-config/namelist.input').write_text(namelist)
PY
python -m gpuwm.cli fetch-geog --static-source hrrr-conus-v4 --root "$L/geog"
# The native static receipt also reads the published land-use index metadata.
python -m gpuwm.cli fetch-geog --datasets modis_landuse_20class_30s_with_lakes --root "$L/geog"
python -m gpuwm.cli fetch --source hrrr --cycle 2026-08-20T00 --hours 1 \
  --out "$L/real-inputs" --transport s3 --mode full-file --fetch-workers 4 \
  --cache-dir "$L/fetch-cache"
python -m gpuwm.cli prep --source hrrr --source-root "$L/real-inputs" \
  --source-sha256s "$L/real-inputs/SHA256SUMS" \
  --source-sha256s-sha256 "$(sha256sum "$L/real-inputs/SHA256SUMS" | cut -d' ' -f1)" \
  --domain-spec "$T/configs/hrrr_native_quick_demo.d01-target.json" \
  --namelist-input "$L/real-config/namelist.input" \
  --wps-namelist "$T/configs/hrrr_native_quick_demo.namelist.wps" \
  --experiment-config "$L/real-config/experiment.toml" \
  --geog-root "$L/geog" --valid-time 2026-08-20_00:00:00 \
  --run-seconds 3600 --history-interval-seconds 3600 \
  --preprocess-backend cpu --preprocess-workers 4 --output-root "$L/real-prepared"
