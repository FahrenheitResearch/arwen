#!/usr/bin/env bash
# Fetch the public inputs of one analysis case into <case dir>: the
# observation files HRRR's analysis script links, the RAP native-level file
# its makeguess script starts from, and the fix files and CRTM coefficients
# the analysis reads. Every file is logged with its URL, UTC fetch time and
# SHA-256 in <case dir>/SOURCES.txt.
#
#   fetch_case.sh <YYYYMMDD> <HH> <case dir> <fix dir> <crtm dir>
#
# NOMADS keeps the observation dumps for about two days; a case older than
# that cannot be fetched from there again, which is why the hashes are kept.
set -euo pipefail
PDY="$1"; cyc="$2"; case_dir="$3"; fix_dir="$4"; crtm_dir="$5"
obsproc=https://nomads.ncep.noaa.gov/pub/data/nccf/com/obsproc/prod
mkdir -p "$case_dir/obs" "$case_dir/rap" "$fix_dir" "$crtm_dir"

get() {  # get <directory> <url> [<name>]
  local dir="$1" url="$2" name="${3:-$(basename "$2")}"
  if [[ -s "$dir/$name" ]]; then echo "have $dir/$name"; return 0; fi
  # NOMADS answers HTTP/2 with a header curl refuses; HTTP/1.1 is clean.
  if curl --http1.1 -fsS --retry 6 --retry-delay 5 --retry-all-errors -o "$dir/$name.part" "$url"; then
    mv "$dir/$name.part" "$dir/$name"
    echo "$(date -u +%FT%TZ) $(sha256sum "$dir/$name" | cut -d' ' -f1) $name $url" >> "$case_dir/SOURCES.txt"
  else
    rm -f "$dir/$name.part"
    echo "$(date -u +%FT%TZ) MISSING $name $url" >> "$case_dir/SOURCES.txt"
    echo "missing: $url"
  fi
}

# Observations. The analysis script takes the early dump (rap_e) at 00 and
# 12 UTC and the regular one otherwise (scripts/conus/exhrrr_analysis.sh).
# The public prepbufr is the ".nr" file: the restricted reports are removed.
if [[ "$cyc" == 00 || "$cyc" == 12 ]]; then prefix=rap_e; else prefix=rap; fi
for kind in prepbufr.tm00.nr satwnd.tm00.bufr_d nexrad.tm00.bufr_d lgycld.tm00.bufr_d lghtng.tm00.bufr_d; do
  get "$case_dir/obs" "$obsproc/$prefix.$PDY/$prefix.t${cyc}z.$kind"
done

# MRMS three-dimensional reflectivity mosaic: the 33 height levels at the
# analysis minute, named as the radar preparation script finds them
# (MergedReflectivityQC_<level>_<date>-<time>.grib2.gz). The scan nearest
# after the top of the hour is taken, within two minutes.
mrms=https://noaa-mrms-pds.s3.amazonaws.com
mkdir -p "$case_dir/mrms"
for level in 00.50 00.75 01.00 01.25 01.50 01.75 02.00 02.25 02.50 02.75 03.00 \
             03.50 04.00 04.50 05.00 05.50 06.00 06.50 07.00 07.50 08.00 08.50 \
             09.00 10.00 11.00 12.00 13.00 14.00 15.00 16.00 17.00 18.00 19.00; do
  product="MergedReflectivityQC_$level"
  key="$(curl -fsS --retry 6 --retry-delay 3 --retry-all-errors \
              "$mrms/?list-type=2&prefix=CONUS/$product/$PDY/MRMS_${product}_$PDY-${cyc}0" \
          | grep -o "CONUS/$product/$PDY/MRMS_${product}_$PDY-${cyc}0[0-2][0-9]\{2\}\.grib2\.gz" | sort | head -1 || true)"
  if [[ -z "$key" ]]; then
    echo "$(date -u +%FT%TZ) MISSING $product $mrms/CONUS/$product/$PDY/" >> "$case_dir/SOURCES.txt"
    echo "missing: MRMS level $level"
    continue
  fi
  name="$(basename "$key")"
  get "$case_dir/mrms" "$mrms/$key" "${name#MRMS_}"
done

# The RAP native-level analysis the cold-start guess is made from.
get "$case_dir/rap" "https://noaa-rap-pds.s3.amazonaws.com/rap.$PDY/rap.t${cyc}z.awp130bgrbf00.grib2"

# Fix files of the tag, from NCEP's published copy of the production tree
# (the git tag carries no fix directory).
fix=https://www.nco.ncep.noaa.gov/pmb/codes/nwprod/hrrr.v4.1.21/fix/conus
for name in hrrr_anavinfo_arw_netcdf hrrr_berror_stats_global hrrr_current_bad_aircraft.txt \
            hrrr_current_mesonet_uselist.txt hrrr_global_ozinfo.txt hrrr_global_pcpinfo.txt \
            hrrr_global_satinfo.txt hrrr_gsd_sfcobs_provider.txt hrrr_nam_errtable.r3dv \
            hrrr_nam_regional_convinfo hrrr_geo_em.d01.nc; do
  get "$fix_dir" "$fix/$name"
done

# CRTM 2.3.0 coefficients (2.6 GB). The analysis script links them whether or
# not a radiance file is present.
get "$crtm_dir" https://ftp.emc.ncep.noaa.gov/jcsda/CRTM/REL-2.3.0/crtm_v2.3.0.tar.gz

cat "$case_dir/SOURCES.txt"
