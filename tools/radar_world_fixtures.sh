#!/usr/bin/env bash
# Fetch the real files rw_odim's non-ODIM readers are tested on, into $1
# (default: ./radar-world-fixtures), and check each against its pinned
# SHA-256. They are fetched on demand rather than committed: the JMA tars'
# redistribution terms are an open question, and the others are megabytes.
#
#   tools/radar_world_fixtures.sh DIR
#   RW_ODIM_WORLD_DATA=DIR cargo test --release -p rw-odim --test world_real_files -- --ignored
#
# Sources (all public, read-only):
#   JMA via NICT mirror  pawr.nict.go.jp  2019-10-12 09:00Z (Typhoon Hagibis landfall)
#   IDEAM (Colombia)     s3-radaresideam  Tablazo IRIS RAW, Bogota CfRadial 1 (CC BY 4.0)
#   openradar            open-radar-data  RVP8 Universal Format sample (MIT)
#   BoM via NCI rq0      thredds.nci.org.au Melbourne (site 2) ODIM_H5 (CC BY 4.0),
#                        one member of the day zip, fetched by HTTP range by
#                        gpuwm.obs.radar_archives (needs python3 and this checkout)
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dir="${1:-radar-world-fixtures}"
mkdir -p "$dir"
cd "$dir"

fetch() {  # name sha256 url
  local name="$1" sum="$2" url="$3"
  if [ ! -f "$name" ] || ! echo "$sum  $name" | sha256sum -c --status; then
    curl -fsSL --max-time 300 -o "$name.part" "$url"
    mv "$name.part" "$name"
  fi
  echo "$sum  $name" | sha256sum -c
}

jma=https://pawr.nict.go.jp/jmadata/JMA-PolarCoordsRadar/2019/10/12
fetch Z__C_RJTD_20191012090000_RDR_JMAGPV_N5_grib2.tar \
  59628ec14724c07e2e86c6e2eacb10c250893534d99a40020e4843f94a89140f \
  "$jma/Z__C_RJTD_20191012090000_RDR_JMAGPV_N5_grib2.tar"
fetch Z__C_RJTD_20191012090000_RDR_JMAGPV_N6_grib2.tar \
  fc304da3cffd85b5f7d91ddac1e4d2a461ccf003dd368746c711b4b1af2ffd07 \
  "$jma/Z__C_RJTD_20191012090000_RDR_JMAGPV_N6_grib2.tar"

ideam=https://s3-radaresideam.s3.amazonaws.com/l2_data/2025/06/01
fetch TAB250601000003.RAWYSBR \
  dc73d3e87cbcf3ac12ca4a6d3ab29af9b14526f745e04417e6d789c18315102f \
  "$ideam/Tablazo/TAB250601000003.RAWYSBR"
fetch 1399BOG-20250601-000042-PPIVol-7550.nc.gz \
  d5225b3e25a951cf2bad03e2211d0824cea52d33dc4431e80300aee044baf8d4 \
  "$ideam/Bogota/1399BOG-20250601-000042-PPIVol-7550.nc.gz"

fetch 20110427_164233_rvp8-rel_v001_SUR.uf.gz \
  17ea67041bb4f97218f70b2206eb6a42c9eadfc2acd533827b78bcc74d36b837 \
  https://github.com/openradar/open-radar-data/raw/main/data/20110427_164233_rvp8-rel_v001_SUR.uf.gz

nci=2_20261013_120000.pvol.h5
nci_sum=991a0d70a83b0f1a8180396316286b020070e63c8d13fc5c8b5dcdc66d96551c
if [ ! -f "$nci" ] || ! echo "$nci_sum  $nci" | sha256sum -c --status; then
  PYTHONPATH="$root" python3 -c "
from datetime import datetime
from gpuwm.obs.radar_archives import fetch
fetch('bom-nci', site='2', valid_time=datetime(2026, 10, 13, 12, 0), out_dir='.')
"
fi
echo "$nci_sum  $nci" | sha256sum -c
