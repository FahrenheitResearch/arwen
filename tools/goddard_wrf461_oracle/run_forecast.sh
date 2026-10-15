#!/usr/bin/env bash
set -euo pipefail
# Run inside the named card queue, with T and L bound to this lane.
: "${T:?}" "${L:?}" "${RECORDINGS:?}"
I=$L/forecast-input; O=${GODDARD_FORECAST_OUT:-$L/forecast-output}
mkdir -p "$I"
printf '%s\n' "$I" "$O" "$L/forecast-import.log" >> "$L/MANIFEST.txt"
cp "$T/tools/goddard_wrf461_oracle/forecast/namelist."{input,wps} "$I/"
test -L "$I/wrfinput_d01" || ln -s "$RECORDINGS/inputs/iowa-convective/A154/wrfinput_d01" "$I/wrfinput_d01"
test -L "$I/wrfbdy_d01" || ln -s "$RECORDINGS/inputs/iowa-convective/A154/wrfbdy_d01" "$I/wrfbdy_d01"
python -m gpuwm.cli import-namelist "$I/namelist.wps" "$I/namelist.input" --output "$I/imported.toml" --explain > "$L/forecast-import.log" 2>&1
python -m gpuwm.cli run --wrfinput "$I" --run-seconds 30 --outdir "$O" --no-supervise
