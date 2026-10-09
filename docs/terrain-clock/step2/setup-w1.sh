#!/bin/bash
# setup-w1.sh : how box W1 (AWS us-west-2, 8x RTX PRO 6000) was set up for the terrain-clock local-face lane, step 2,
# 2026-10-07, as run.  Everything was fetched or built on the box itself; nothing came from another box.
#  - tree6.tgz / tree7.tgz: the lane worktree's gpuwm/ + tools/terrain_clock_*.py (git ls-files plus the uncommitted
#    lane changes), uploaded from the workstation; TREE-COMMIT records HEAD and that it carried uncommitted changes.
#  - NCAR's v4.4 CONUS benchmark tarballs: ncar/fetch.sh, straight from www2.mmm.ucar.edu (sha256 of each stream kept).
#  - rw_netcdf and libstatic_fields.so (the WRF-input door decodes NetCDF only through the Rust bridge): built from the
#    lane's tools/rustwx (+ tools/grib1_bridge, tools/region_global_dealias path deps) with cargo --offline on the box.
set -eu; W=/work/tclock
python3 -m venv $W/venv6
$W/venv6/bin/pip install -q "numpy==2.5.3" "scipy>=1.11" "netCDF4>=1.6" "jsonschema>=4.0" "threadpoolctl>=3.1" "cupy-cuda12x[ctk]>=14.0"
export CARGO_HOME=$W/cargo RUSTUP_HOME=$W/rustup
curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
. $W/cargo/env
cd $W/rust/tools/rustwx
cargo build --release --locked --offline -p rw-netcdf
cargo build --release --locked --offline -p static-fields
mkdir -p "$W/bridges" && cp target/release/libstatic_fields.so "$W/bridges/"
echo "export GPUWM_STATIC_BRIDGE=$W/bridges/libstatic_fields.so" >> "$W/env6.sh"
echo "export GPUWM_RW_NETCDF=$W/rust/tools/rustwx/target/release/rw_netcdf" >> $W/env6.sh
