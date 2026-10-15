#!/usr/bin/env bash
set -euo pipefail
T=$(cd "$(dirname "$0")/../.." && pwd)
L=$(dirname "$T")
source /work/sprint/env.sh
export GPUWM_RW_NETCDF=/work/sprint/engine-base/tools/rustwx/target/release/rw_netcdf
export GPUWM_NCWRITE_BRIDGE=/work/sprint/engine-base/tools/rustwx/target/release/libnetcdf_writer.so
export GPUWM_WRF_EXACT=${GPUWM_SOLAR_REAL_MODE:-0}
cd "$T"
python -m tools.solar_geometry_oracle.real_identity
