#!/usr/bin/env bash
# Add missing static records to a history background through the Rust
# NetCDF rewriter. Existing history fields, attributes and time are kept.
#
#   prepare_history.sh <profile> <history> <matching input template> <output>
#
# NC_REWRITE names the nc_rewrite binary built with the rewrite feature.
# The template must be the same grid and vertical coordinate as the history.
# The rewriter refuses dimensions that differ and existing-field replacement.
set -euo pipefail
profile="$1"; history="$2"; template="$3"; output="$4"
. "$profile"
: "${NC_REWRITE:?set NC_REWRITE to the Rust nc_rewrite binary}"
[[ ! -e "$output" ]] || { echo "$output already exists; refusing to overwrite a background" >&2; exit 1; }
header=$(ncdump -h "$history")
template_header=$(ncdump -h "$template")
geometry=':(MAP_PROJ|DX|DY|CEN_LAT|CEN_LON|TRUELAT1|TRUELAT2|STAND_LON|HYBRID_OPT) ='
if [[ "$(grep -E "$geometry" <<< "$header" | sort)" != "$(grep -E "$geometry" <<< "$template_header" | sort)" ]]; then
  echo "history and template grid attributes differ; copying static records would give GSI a different grid or coordinate" >&2
  exit 1
fi
missing=""
for name in ${STATIC_FIELDS//,/ }; do
  if ! grep -Eq "^[[:space:]]+(float|double|int|char) ${name}\\(" <<< "$header"; then
    missing="${missing:+$missing,}$name"
  fi
done
if [[ -n "$missing" ]]; then
  "$NC_REWRITE" "$history" "$output" --fields-from "$template" "$missing"
else
  cp "$history" "$output"
fi
{
  echo "static fields copied: ${missing:-none}"
  sha256sum "$history" "$template" "$output"
} > "$output.receipt.txt"
cat "$output.receipt.txt"
