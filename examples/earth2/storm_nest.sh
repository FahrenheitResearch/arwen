#!/usr/bin/env bash
# The short version completed on the 16 GB proof card; see README.md.
earth2_short_model=fcn3
earth2_short_hours=6
earth2_short_requires_fetch=true
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
earth2_card=()
if "$earth2_short"; then earth2_card=(--card 16gb); fi
"$earth2_cli" earth2 --model "$earth2_model" --storm auto --nest 2.5km "${earth2_card[@]}" \
    --cycle 2024-09-24T12 --hours "$earth2_hours" --out "$earth2_out"
