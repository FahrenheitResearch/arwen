# Source only on the rented box after build_box.sh completed.
# Prepared data exists once in /workspace/shared-prepared.
unset GPUWM_NO_LOCAL_GPU CUDA_VISIBLE_DEVICES
export DA_ROOT=${DA_ROOT:-/workspace/da-rerun-286}
export DA_ENGINE=${DA_ENGINE:-$DA_ROOT/engine}
export DA_VENV=${DA_VENV:-$DA_ROOT/venv}
export PATH="$DA_VENV/bin:$DA_ENGINE/tools/rw_wps/target/release:$DA_ENGINE/tools/rustwx/target/release:$DA_ENGINE/tools/grib1_bridge/target/release:$PATH"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
export GPUWM_MAPPED_ENGINE_THREADS=12
# No GPUWM_MAPPED_ENGINE_MEMORY_BUDGET_BYTES here. The packed member wave
# sets each worker's decoder budget itself (gpuwm.da.member_wave.run_wave,
# at most 5 GiB per worker under the 160 GiB total); a 5 GiB export here
# also bound case preparation, which refused one 241x241 source time that
# needs 27.5 GB (2026-10-05).
export GPUWM_PREPROCESS_THREADS=12 RAYON_NUM_THREADS=12
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export GPUWM_BRIDGE_SOURCE_REV
GPUWM_BRIDGE_SOURCE_REV=$(cat "$DA_ENGINE/.engine-export-sha")
export GPUWM_STATIC_BRIDGE="$DA_ENGINE/tools/rustwx/target/release/libstatic_fields.so"
export GPUWM_NCWRITE_BRIDGE="$DA_ENGINE/tools/rustwx/target/release/libnetcdf_writer.so"
export GPUWM_RW_NETCDF="$DA_ENGINE/tools/rustwx/target/release/rw_netcdf"
export GPUWM_CPU_PREPROCESS_BRIDGE="$DA_ENGINE/tools/grib1_bridge/target/release/libgpuwm_preprocess_cpu.so"
export GPUWM_MAPPED_ENGINE_BIN="$DA_ENGINE/tools/rw_wps/target/release/gpuwm_mapped_engine"
export GPUWM_GRIB1_BRIDGE="$DA_ENGINE/tools/grib1_bridge/target/release/grib1_bridge"
export GPUWM_GFS_GRIB2_BRIDGE="$DA_ENGINE/tools/grib1_bridge/target/release/gfs_grib2_bridge"
export GPUWM_HRRR_DECODER="$DA_ENGINE/tools/grib1_bridge/target/release/hrrr_grib2_bridge"
export GPUWM_GRIB2_INVENTORY="$DA_ENGINE/tools/grib1_bridge/target/release/grib2_inventory"
export GPUWM_GRIB2_DUMP="$DA_ENGINE/tools/grib1_bridge/target/release/grib2_dump"
export GPUWM_GDT101_REMAP="$DA_ENGINE/tools/grib1_bridge/target/release/gdt101_remap"
export GPUWM_THOMPSON_TABLE_ROOT="$DA_ROOT/tables/thompson"
export GPUWM_WIF_DATA_ROOT="$DA_ROOT/tables/wif"
export GPUWM_WIF_CLIMATOLOGY="$DA_ROOT/tables/wif/QNWFA_QNIFA_SIGMA_MONTHLY.dat"
export CUPY_CACHE_DIR="$DA_ROOT/cache/cupy" XDG_CACHE_HOME="$DA_ROOT/cache"
export TMPDIR="$DA_ROOT/tmp"

export GPUWM_OBSSCORE_BRIDGE="$DA_ENGINE/tools/rustwx/target/release/libobs_score.so"
export GPUWM_RW_MRMS="$DA_ENGINE/tools/rustwx/target/release/rw_mrms"
