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
expected_source_sha=5aaae368289694c929d38365d77d445e4f22291a30a48555df7a21d470b72ae3

if [ ! -f "$official_source" ]; then
  echo "missing official WRF source: $official_source" >&2
  exit 2
fi
actual_source_sha=$(sha256sum "$official_source" | awk '{print $1}')
if [ "$actual_source_sha" != "$expected_source_sha" ]; then
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
gfortran -c -O2 -ffree-form -ffree-line-length-none \
  "$script_dir/stub_wrf.F90" "$script_dir/effective_radius.F90" \
  "$script_dir/initial_state.F90" "$script_dir/self_collection.F90" \
  "$script_dir/warm_autoconversion.F90" \
  "$script_dir/rain_sedimentation.F90" \
  "$script_dir/snow_sedimentation.F90" \
  "$script_dir/ice_sedimentation.F90" \
  "$script_dir/graupel_sedimentation.F90" \
  "$script_dir/hail_sedimentation.F90" \
  "$script_dir/primary_ice_nucleation.F90" \
  "$script_dir/ice_cloud_riming.F90" \
  "$script_dir/snow_cloud_riming.F90" \
  "$script_dir/graupel_cloud_riming.F90" \
  "$script_dir/hail_cloud_riming.F90" \
  "$script_dir/rain_ice_collection_freezing.F90" \
  "$script_dir/frozen_cross_collection.F90" \
  "$script_dir/melting_liquid_shedding.F90" \
  "$script_dir/secondary_ice_conversions.F90" \
  "$script_dir/rain_cloud_accretion.F90" \
  "$script_dir/rain_evaporation.F90" \
  "$script_dir/clear_air_activation.F90" \
  "$script_dir/cloudy_water_adjustment.F90" \
  "$script_dir/cloud_interior_renucleation.F90" \
  "$script_dir/snow_aggregation.F90" \
  "$script_dir/ice_deposition_conversion.F90" \
  "$script_dir/frozen_vapor_exchange.F90" \
  "$script_dir/graupel_hail_vapor_exchange.F90" \
  "$script_dir/bigg_rain_freezing.F90"
gfortran -O2 -o nssl2_effective_radius_oracle \
  module_mp_nssl_2mom.o stub_wrf.o effective_radius.o
gfortran -O2 -o nssl2_initial_state_oracle \
  module_mp_nssl_2mom.o stub_wrf.o initial_state.o
gfortran -O2 -o nssl2_self_collection_oracle \
  module_mp_nssl_2mom.o stub_wrf.o self_collection.o
gfortran -O2 -o nssl2_warm_autoconversion_oracle \
  module_mp_nssl_2mom.o stub_wrf.o warm_autoconversion.o
gfortran -O2 -o nssl2_rain_sedimentation_oracle \
  module_mp_nssl_2mom.o stub_wrf.o rain_sedimentation.o
gfortran -O2 -o nssl2_snow_sedimentation_oracle \
  module_mp_nssl_2mom.o stub_wrf.o snow_sedimentation.o
gfortran -O2 -o nssl2_ice_sedimentation_oracle \
  module_mp_nssl_2mom.o stub_wrf.o ice_sedimentation.o
gfortran -O2 -o nssl2_graupel_sedimentation_oracle \
  module_mp_nssl_2mom.o stub_wrf.o graupel_sedimentation.o
gfortran -O2 -o nssl2_hail_sedimentation_oracle \
  module_mp_nssl_2mom.o stub_wrf.o hail_sedimentation.o
gfortran -O2 -o nssl2_primary_ice_nucleation_oracle \
  module_mp_nssl_2mom.o stub_wrf.o primary_ice_nucleation.o
gfortran -O2 -o nssl2_ice_cloud_riming_oracle \
  module_mp_nssl_2mom.o stub_wrf.o ice_cloud_riming.o
gfortran -O2 -o nssl2_snow_cloud_riming_oracle \
  module_mp_nssl_2mom.o stub_wrf.o snow_cloud_riming.o
gfortran -O2 -o nssl2_graupel_cloud_riming_oracle \
  module_mp_nssl_2mom.o stub_wrf.o graupel_cloud_riming.o
gfortran -O2 -o nssl2_hail_cloud_riming_oracle \
  module_mp_nssl_2mom.o stub_wrf.o hail_cloud_riming.o
gfortran -O2 -o nssl2_rain_ice_collection_freezing_oracle \
  module_mp_nssl_2mom.o stub_wrf.o rain_ice_collection_freezing.o
gfortran -O2 -o nssl2_frozen_cross_collection_oracle \
  module_mp_nssl_2mom.o stub_wrf.o frozen_cross_collection.o
gfortran -O2 -o nssl2_melting_liquid_shedding_oracle \
  module_mp_nssl_2mom.o stub_wrf.o melting_liquid_shedding.o
gfortran -O2 -o nssl2_secondary_ice_conversions_oracle \
  module_mp_nssl_2mom.o stub_wrf.o secondary_ice_conversions.o
gfortran -O2 -o nssl2_rain_cloud_accretion_oracle \
  module_mp_nssl_2mom.o stub_wrf.o rain_cloud_accretion.o
gfortran -O2 -o nssl2_rain_evaporation_oracle \
  module_mp_nssl_2mom.o stub_wrf.o rain_evaporation.o
gfortran -O2 -o nssl2_clear_air_activation_oracle \
  module_mp_nssl_2mom.o stub_wrf.o clear_air_activation.o
gfortran -O2 -o nssl2_cloudy_water_adjustment_oracle \
  module_mp_nssl_2mom.o stub_wrf.o cloudy_water_adjustment.o
gfortran -O2 -o nssl2_cloud_interior_renucleation_oracle \
  module_mp_nssl_2mom.o stub_wrf.o cloud_interior_renucleation.o
gfortran -O2 -o nssl2_snow_aggregation_oracle \
  module_mp_nssl_2mom.o stub_wrf.o snow_aggregation.o
gfortran -O2 -o nssl2_ice_deposition_conversion_oracle \
  module_mp_nssl_2mom.o stub_wrf.o ice_deposition_conversion.o
gfortran -O2 -o nssl2_frozen_vapor_exchange_oracle \
  module_mp_nssl_2mom.o stub_wrf.o frozen_vapor_exchange.o
gfortran -O2 -o nssl2_graupel_hail_vapor_exchange_oracle \
  module_mp_nssl_2mom.o stub_wrf.o graupel_hail_vapor_exchange.o
gfortran -O2 -o nssl2_bigg_rain_freezing_oracle \
  module_mp_nssl_2mom.o stub_wrf.o bigg_rain_freezing.o
./nssl2_effective_radius_oracle effective-radius.csv | tee oracle.log
./nssl2_initial_state_oracle initial-state.csv | tee -a oracle.log
./nssl2_self_collection_oracle self-collection.csv | tee -a oracle.log
./nssl2_warm_autoconversion_oracle warm-autoconversion.csv | tee -a oracle.log
./nssl2_rain_sedimentation_oracle rain-sedimentation.csv | tee -a oracle.log
./nssl2_snow_sedimentation_oracle snow-sedimentation.csv | tee -a oracle.log
./nssl2_ice_sedimentation_oracle ice-sedimentation.csv | tee -a oracle.log
./nssl2_graupel_sedimentation_oracle \
  graupel-sedimentation.csv | tee -a oracle.log
./nssl2_hail_sedimentation_oracle \
  hail-sedimentation.csv | tee -a oracle.log
./nssl2_primary_ice_nucleation_oracle \
  primary-ice-nucleation.csv | tee -a oracle.log
./nssl2_ice_cloud_riming_oracle \
  ice-cloud-riming.csv | tee -a oracle.log
./nssl2_snow_cloud_riming_oracle \
  snow-cloud-riming.csv | tee -a oracle.log
./nssl2_graupel_cloud_riming_oracle \
  graupel-cloud-riming.csv | tee -a oracle.log
./nssl2_hail_cloud_riming_oracle \
  hail-cloud-riming.csv | tee -a oracle.log
./nssl2_rain_ice_collection_freezing_oracle \
  rain-ice-collection-freezing.csv | tee -a oracle.log
./nssl2_frozen_cross_collection_oracle \
  frozen-cross-collection.csv | tee -a oracle.log
./nssl2_melting_liquid_shedding_oracle \
  melting-liquid-shedding.csv | tee -a oracle.log
for mode in baseline contact homogeneous hm ice_to_g snow_to_g g_to_h all; do
  ./nssl2_secondary_ice_conversions_oracle \
    "secondary-ice-conversions-${mode}-raw.csv" "$mode" \
    | tee -a oracle.log
  cp namelist.input "secondary-ice-conversions-${mode}.namelist.input"
  python3 "$script_dir/audit_nssl_namelist.py" \
    module_mp_nssl_2mom.F namelist.input \
    | tee "secondary-ice-conversions-${mode}.namelist-audit.txt"
done
python3 "$script_dir/combine_secondary_oracle.py" \
  mapped \
  secondary-ice-conversions-baseline-raw.csv \
  secondary-ice-conversions-family-isolated.csv \
  secondary-ice-conversions-contact-raw.csv \
  secondary-ice-conversions-homogeneous-raw.csv \
  secondary-ice-conversions-hm-raw.csv \
  secondary-ice-conversions-ice_to_g-raw.csv \
  secondary-ice-conversions-snow_to_g-raw.csv \
  secondary-ice-conversions-g_to_h-raw.csv \
  secondary-ice-conversions-all-raw.csv
python3 "$script_dir/combine_secondary_oracle.py" \
  all \
  secondary-ice-conversions-baseline-raw.csv \
  secondary-ice-conversions.csv \
  secondary-ice-conversions-contact-raw.csv \
  secondary-ice-conversions-homogeneous-raw.csv \
  secondary-ice-conversions-hm-raw.csv \
  secondary-ice-conversions-ice_to_g-raw.csv \
  secondary-ice-conversions-snow_to_g-raw.csv \
  secondary-ice-conversions-g_to_h-raw.csv \
  secondary-ice-conversions-all-raw.csv
if cmp -s secondary-ice-conversions-all-raw.csv \
    secondary-ice-conversions-baseline-raw.csv; then
  echo 'secondary-ice all-target and baseline outputs are identical' >&2
  exit 1
fi
./nssl2_rain_cloud_accretion_oracle rain-cloud-accretion.csv | tee -a oracle.log
./nssl2_rain_evaporation_oracle rain-evaporation.csv | tee -a oracle.log
./nssl2_clear_air_activation_oracle clear-air-activation.csv | tee -a oracle.log
./nssl2_cloudy_water_adjustment_oracle cloudy-water-adjustment.csv | tee -a oracle.log
./nssl2_cloud_interior_renucleation_oracle \
  cloud-interior-renucleation.csv | tee -a oracle.log
./nssl2_snow_aggregation_oracle snow-aggregation.csv | tee -a oracle.log
./nssl2_ice_deposition_conversion_oracle \
  ice-deposition-conversion.csv | tee -a oracle.log
./nssl2_frozen_vapor_exchange_oracle \
  frozen-vapor-exchange.csv | tee -a oracle.log
./nssl2_graupel_hail_vapor_exchange_oracle \
  graupel-hail-vapor-exchange.csv | tee -a oracle.log
./nssl2_bigg_rain_freezing_oracle \
  bigg-rain-freezing.csv | tee -a oracle.log

printf '%s  %s\n' "$actual_source_sha" module_mp_nssl_2mom.F.orig \
  > SOURCE_SHA256
sha256sum effective-radius.csv initial-state.csv self-collection.csv \
  warm-autoconversion.csv rain-sedimentation.csv snow-sedimentation.csv \
  ice-sedimentation.csv \
  graupel-sedimentation.csv \
  hail-sedimentation.csv \
  primary-ice-nucleation.csv \
  ice-cloud-riming.csv \
  snow-cloud-riming.csv \
  graupel-cloud-riming.csv \
  hail-cloud-riming.csv \
  rain-ice-collection-freezing.csv \
  frozen-cross-collection.csv \
  melting-liquid-shedding.csv \
  secondary-ice-conversions.csv \
  secondary-ice-conversions-family-isolated.csv \
  rain-cloud-accretion.csv \
  rain-evaporation.csv clear-air-activation.csv \
  cloudy-water-adjustment.csv \
  cloud-interior-renucleation.csv snow-aggregation.csv \
  ice-deposition-conversion.csv frozen-vapor-exchange.csv \
  graupel-hail-vapor-exchange.csv bigg-rain-freezing.csv \
  | tee ORACLE_SHA256
