#!/usr/bin/env bash
# The short version completed on the 16 GB proof card; see README.md.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
"$earth2_cli" earth2 --model "$earth2_model" --export earth2 --diagnose derived-ws10m \
    --cycle 2024-09-24T12 --hours "$earth2_hours" \
    --area 24,-90,30,-84 --out "$earth2_out"

"$earth2_cli" render \
    --cf "$earth2_out/diagnostics/derived-ws10m/diagnostic.nc" \
    --variable ws10m --bounds=-90,-84,24,30 \
    --out "$earth2_out/diagnostic-maps" \
    --receipt "$earth2_out/diagnostic-map-receipt.json"
