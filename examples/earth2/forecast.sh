#!/usr/bin/env bash
# The 12 hour UCast short version completed on the 16 GB proof card.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
"$earth2_cli" earth2 --model "$earth2_model" \
    --cycle 2024-09-24T12 --hours "$earth2_hours" \
    --area 24,-90,30,-84 --out "$earth2_out"
