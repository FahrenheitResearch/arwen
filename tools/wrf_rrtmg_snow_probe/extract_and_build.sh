#!/usr/bin/env bash
# Extract the WRF v4.6.1 option-4 snow-radiation coupling statements
# byte-for-byte from the read-only authority, verify provenance, and build
# the FP32 probe.  Run under WSL:
#   wsl bash /mnt/c/.../tools/wrf_rrtmg_snow_probe/extract_and_build.sh
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
auth="${WRF_SOURCE_ROOT:?export WRF_SOURCE_ROOT=<path>/WRF_source_v4.6.1_group (the read-only v4.6.1 authority)}"
sw="$auth/phys/module_ra_rrtmg_sw.F"

expect_sha=447345d2658cd370e6bc97ff2ab582a5d12b84adffc58f72a938b353e017987e
got_sha=$(sha256sum "$sw" | cut -d' ' -f1)
if [ "$got_sha" != "$expect_sha" ]; then
  echo "FATAL: authority module_ra_rrtmg_sw.F SHA-256 drifted: $got_sha" >&2
  exit 1
fi

cd "$here"
# resnow floor + m->um conversion (has_reqs branch, iceflgsw=5 selection).
sed -n '10823,10825p' "$sw" > wrf_resnow_floor.inc
# snow mass discount + 130 um cap.
sed -n '11055,11067p' "$sw" > wrf_snow_discount.inc
sha256sum wrf_resnow_floor.inc wrf_snow_discount.inc

# Same effective language configuration as the group build / Phase-A oracle:
# default REAL = FP32 (RWORDSIZE=4), free form, -O0.
gfortran -O0 -ffree-form -ffree-line-length-none -o wrf_snow_probe \
  probe_main.F90
echo "probe built OK"

if [ -f probe_inputs.txt ]; then
  ./wrf_snow_probe
fi
