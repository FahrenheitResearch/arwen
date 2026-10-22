#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: $0 /path/to/WRF-v4.6.1 /new/build-directory" >&2
  exit 2
fi

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
wrf_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
official_source="$wrf_root/phys/module_mp_nssl_2mom.F"
# The pinned d66e442f Git blob is LF. The original oracle recorded the
# byte-identical statements in a Windows CRLF checkout; admit exactly those
# two byte forms and preserve the actual input hash in SOURCE_SHA256.
expected_source_sha_lf=1eb1b138b75ff3b0cfe33c23779f4ec9b72e57a5455a53ef11c9e55ae0f42722
expected_source_sha_crlf=5aaae368289694c929d38365d77d445e4f22291a30a48555df7a21d470b72ae3

if [ ! -f "$official_source" ]; then
  echo "missing official WRF source: $official_source" >&2
  exit 2
fi
actual_source_sha=$(sha256sum "$official_source" | awk '{print $1}')
if [ "$actual_source_sha" != "$expected_source_sha_lf" ] \
   && [ "$actual_source_sha" != "$expected_source_sha_crlf" ]; then
  echo "unexpected module_mp_nssl_2mom.F SHA-256: $actual_source_sha" >&2
  exit 2
fi
if [ -e "$build_dir" ]; then
  echo "refusing to reuse oracle build directory: $build_dir" >&2
  exit 2
fi

mkdir -p "$build_dir"
cp "$official_source" "$build_dir/module_mp_nssl_2mom.F.orig"
cp "$official_source" "$build_dir/module_mp_nssl_2mom.F"
cd "$build_dir"
# Git's Windows checkout may contain CRLF while the semantic WRF source and
# visibility patch use LF.  Normalize only the disposable compilation copy;
# the byte-pinned original remains untouched beside it.
sed -i 's/\r$//' module_mp_nssl_2mom.F
patch --fuzz=0 < "$script_dir/visibility.patch"

# nssl_2mom_init always attempts an internal namelist read.  An empty file
# gives the intended nonzero iostat; the false monitor stub prevents output.
touch namelist.input
gfortran -c -O2 -cpp -DWRF_CHEM=0 -ffree-form \
  -ffree-line-length-none module_mp_nssl_2mom.F
drivers=(
  effective_radius
  initial_state
  self_collection
  warm_autoconversion
  rain_sedimentation
  snow_sedimentation
  ice_sedimentation
  graupel_sedimentation
  hail_sedimentation
  primary_ice_nucleation
  ice_cloud_riming
  snow_cloud_riming
  graupel_cloud_riming
  hail_cloud_riming
  rain_ice_collection_freezing
  frozen_cross_collection
  melting_liquid_shedding
  secondary_ice_conversions
  rain_cloud_accretion
  rain_evaporation
  clear_air_activation
  cloudy_water_adjustment
  cloud_interior_renucleation
  snow_aggregation
  ice_deposition_conversion
  frozen_vapor_exchange
  graupel_hail_vapor_exchange
  bigg_rain_freezing
)
sources=("$script_dir/stub_wrf.F90")
for driver in "${drivers[@]}"; do
  sources+=("$script_dir/$driver.F90")
done
gfortran -c -O2 -ffree-form -ffree-line-length-none "${sources[@]}"
for driver in "${drivers[@]}"; do
  gfortran -O2 -o "nssl2_${driver}_oracle" \
    module_mp_nssl_2mom.o stub_wrf.o "$driver.o"
done
tee_args=()
for driver in "${drivers[@]:0:17}"; do
  "./nssl2_${driver}_oracle" "${driver//_/-}.csv" | tee "${tee_args[@]}" oracle.log
  tee_args=(-a)
done
for mode in baseline contact homogeneous hm ice_to_g snow_to_g g_to_h all; do
  ./nssl2_secondary_ice_conversions_oracle \
    "secondary-ice-conversions-${mode}-raw.csv" "$mode" \
    | tee -a oracle.log
  cp namelist.input "secondary-ice-conversions-${mode}.namelist.input"
  python3 "$script_dir/audit_nssl_namelist.py" \
    module_mp_nssl_2mom.F namelist.input \
    | tee "secondary-ice-conversions-${mode}.namelist-audit.txt"
done
secondary_inputs=()
for mode in contact homogeneous hm ice_to_g snow_to_g g_to_h all; do
  secondary_inputs+=("secondary-ice-conversions-${mode}-raw.csv")
done
for mode in mapped all; do
  output=secondary-ice-conversions.csv
  if [ "$mode" = mapped ]; then
    output=secondary-ice-conversions-family-isolated.csv
  fi
  python3 "$script_dir/combine_secondary_oracle.py" \
    "$mode" secondary-ice-conversions-baseline-raw.csv "$output" \
    "${secondary_inputs[@]}"
done
if cmp -s secondary-ice-conversions-all-raw.csv \
    secondary-ice-conversions-baseline-raw.csv; then
  echo 'secondary-ice all-target and baseline outputs are identical' >&2
  exit 1
fi
for driver in "${drivers[@]:18}"; do
  "./nssl2_${driver}_oracle" "${driver//_/-}.csv" | tee -a oracle.log
done

printf '%s  %s\n' "$actual_source_sha" module_mp_nssl_2mom.F.orig \
  > SOURCE_SHA256
outputs=()
for driver in "${drivers[@]}"; do
  outputs+=("${driver//_/-}.csv")
  if [ "$driver" = secondary_ice_conversions ]; then
    outputs+=(secondary-ice-conversions-family-isolated.csv)
  fi
done
sha256sum "${outputs[@]}" | tee ORACLE_SHA256
