#!/bin/sh
# End-to-end 20CRv3 demo: fetch a real window, prepare it, run it, plot it.
#
# 20CRv3 is the reanalysis that reaches back to 1836, so it is the only
# source in this product that can initialise a case from before the
# satellite era.  This runs the whole path on the smallest honest window:
# two three-hourly analyses over a 25x20 degree box, about 0.7 MB of real
# NOAA data (MEASURED 2026-08-16: 737,620 bytes for the 1974-04-03 18Z
# window below, fourteen files plus the recovered invariant supplement).
#
# Nothing here is 20CRv3-specific except the source name and the window.
# The decode is the declarative mapped route reading a PACKAGED profile:
# `gpuwm prep --source 20crv3-cf` fills in the mapping, composition and
# provenance from the wheel and byte-checks them, so this script never
# names a mapping file and neither does a user.
#
# Two things to know before you read the output as truth:
#   * NOAA PSL's 20CRv3 NetCDF distribution is the ENSEMBLE MEAN
#     analysis, not a member.  For a member state use `--source 20crv3`
#     over the every-member GRIB2 archive.
#   * PSL publishes no orography and no land mask for 20CRv3, so both are
#     recovered from 20CRv3's own published fields; the supplement's
#     provenance receipt states the method and the divergence.
#
# Usage:
#   tools/demo_20crv3_netcdf.sh WORKDIR GEOG_ROOT
set -eu

WORK="${1:?usage: demo_20crv3_netcdf.sh WORKDIR GEOG_ROOT}"
GEOG="${2:?usage: demo_20crv3_netcdf.sh WORKDIR GEOG_ROOT}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"

START="1974-04-03T18:00:00"
CONFIG="$REPO/configs/twentycrv3_netcdf_demo.toml"
NAMELIST="$REPO/configs/twentycrv3_netcdf_demo.namelist.wps"

SUBSET="$WORK/subset"
PREPARED="$WORK/prepared"
RUN="$WORK/run"
PLOTS="$WORK/plots"

echo "== 1/4 fetch a real 20CRv3 window (NOAA PSL THREDDS subset) =="
python "$REPO/tools/download_20crv3_native_subset.py" \
    --start "$START" --frames 2 \
    --north 50 --south 30 --west -105 --east -80 \
    --output "$SUBSET" > "$WORK/subset-receipt.json"

echo "== 2/4 prepare, through the packaged 20CRv3 NetCDF profile =="
set -- gpuwm prep --source 20crv3-cf
for FILE in air hgt shum uwnd vwnd pres.sfc skt air.2m shum.2m \
            uwnd.10m vwnd.10m tsoil soilw invariant; do
    set -- "$@" --input "$SUBSET/$FILE.nc"
done
"$@" \
    --supplement "$SUBSET/invariant.nc" \
    --author-input-manifest "$SUBSET/inputs.json" \
    --wps-namelist "$NAMELIST" \
    --geog-root "$GEOG" \
    --experiment-config "$CONFIG" \
    --output-root "$PREPARED" > "$WORK/preparation-proof.json"

echo "== 3/4 run the forecast on the prepared tree =="
gpuwm sim "$PREPARED" \
    --experiment-config "$CONFIG" \
    --wps-namelist "$NAMELIST" \
    --outdir "$RUN"

echo "== 4/4 render the product PNGs with the Rust renderer =="
gpuwm render "$RUN"/wrfout/wrfout_d01_* \
    --engine rust --products t2,wind10,refl --out "$PLOTS" \
    --source-label "ArWen -- NOAA 20CRv3 NetCDF (ensemble mean)"

echo
echo "prepared tree : $PREPARED"
echo "forecast      : $RUN"
echo "plots         : $PLOTS"
