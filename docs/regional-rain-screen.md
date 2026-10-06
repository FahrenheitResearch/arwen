# Regional rain scorer

`tools/regional_rain_score.py` measures the last hour at forecast leads 1, 3 and 6 hours. It writes 36 rows: footprint rain ratio, whole common-domain rain ratio, time-integrated 35-dBZ area multiple, and FSS at every combination of 1/5/10 mm per hour and 10/25/50 km square widths.

The computation follows `regional-rain/v1`. Projection, geographic cell bounds, polygon overlap, time integration and FSS run through Rust. The common grid has exact 3000-m equal-area square cells on the WRF sphere, radius 6370000 m. Native observed footprint fractions survive remapping. Observed rain is masked on its native footprint before remap; model rain on the common grid is weighted by that footprint fraction. Observed objects crossing a supplied truth-region edge, missing footprint support, missing hourly endpoints and missing time intervals stay pending. Full observed coverage is required inside each admitted FSS neighborhood. Fractional edge-cell weights preserve the exact physical square width. Dry observed events do not provide wet-event FSS evidence.

Model rain uses cumulative `RAINC + RAINNC` in native mm over `[L-1,L]`, plus explicit before-reset carry if a declared reset occurs. An unexplained counter decrease is missing support. There is no negative-rain clipping. Model fields require a maximum 120-second interval. MRMS product stamps can carry completion-second jitter; the fixed archive maximum is 150 seconds, which admits that jitter and rejects a missing two-minute frame. Rate and echo clocks are separate. A bracketing trapezoid is clipped to the exact hourly endpoints without extrapolation. Echo area integrates thresholded native cell occupancy in square metres times seconds, rather than using one endpoint image.

Build in the pinned engine export:

```sh
cd tools/rustwx
cargo build --release --offline --locked -p obs-score -p static-fields -j 2
cargo build --release --offline --locked -p rw-obs --bin rw_mrms -p rw-netcdf --bin rw_netcdf -j 2
cd ../..
export GPUWM_OBSSCORE_BRIDGE="$PWD/tools/rustwx/target/release/libobs_score.so"
export GPUWM_RW_NETCDF="$PWD/tools/rustwx/target/release/rw_netcdf"
export GPUWM_RW_MRMS="$PWD/tools/rustwx/target/release/rw_mrms"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
```

Use the shared prepared copy and one serial anonymous fetch process. This example uses a case region and an analysis time supplied by the frozen case manifest. Set `START` to at least two minutes before final analysis and `END` to at least two minutes after its six-hour forecast end. The bbox must contain the complete observed storm footprint and the full 50-km neighborhoods. Do not shrink it to the model footprint.

```sh
export START=2026-06-14T19:08:00Z END=2026-06-15T01:14:00Z
export BBOX=-104,30,-96,38
mkdir -p /workspace/shared-prepared/mrms/raw-rate /workspace/shared-prepared/mrms/raw-echo
mkdir -p /workspace/shared-prepared/mrms/rate /workspace/shared-prepared/mrms/echo
"$GPUWM_RW_MRMS" fetch --product PrecipRate_00.00 --start "$START" --end "$END" --cache /workspace/shared-prepared/mrms/raw-rate > rate-fetch.json
"$GPUWM_RW_MRMS" fetch --product MergedReflectivityQCComposite_00.50 --start "$START" --end "$END" --cache /workspace/shared-prepared/mrms/raw-echo > echo-fetch.json
while IFS= read -r f; do
  "$GPUWM_RW_MRMS" decode --product PrecipRate_00.00 --file "$f" --bbox "$BBOX" --out "/workspace/shared-prepared/mrms/rate/$(basename "$f").obspack" > "/workspace/shared-prepared/mrms/rate/$(basename "$f").json"
done < <(find /workspace/shared-prepared/mrms/raw-rate -type f -name '*.grib2*' -print)
while IFS= read -r f; do
  "$GPUWM_RW_MRMS" decode --product MergedReflectivityQCComposite_00.50 --file "$f" --bbox "$BBOX" --out "/workspace/shared-prepared/mrms/echo/$(basename "$f").obspack" > "/workspace/shared-prepared/mrms/echo/$(basename "$f").json"
done < <(find /workspace/shared-prepared/mrms/raw-echo -type f -name '*.grib2*' -print)
first_rate=$(find /workspace/shared-prepared/mrms/raw-rate -type f -name '*.grib2*' -print -quit)
"$GPUWM_RW_MRMS" grid --product PrecipRate_00.00 --file "$first_rate" --bbox "$BBOX" --out /workspace/shared-prepared/mrms/grid.geopack > grid.json
```

The shared-kit operator should use its existing file inventory rather than refetch a prepared product. The truth manifest preserves product-specific missing-data masks. RQI can be fetched and decoded with product `RadarQualityIndex_00.00`, then included as an explicitly frozen mask threshold. No additional RQI cutoff is invented by the scorer. `RadarOnly_QPE_01H_00.00` and MultiSensor Pass2 remain sensitivity checks; they are not substitutes for the required PrecipRate primary clock.

The model run must save the issuance frame and every two-minute free-forecast frame, including exact 1, 3 and 6 hour endpoints. The updated public cycle driver publishes native rain in its composite NPZ and wrfout sidecars, with a rain reset identity. A reflectivity-only old export cannot recover rain scores. Select one member and one domain per invocation.

```sh
python tools/regional_rain_manifest.py model --frames /workspace/runs/da-seed1/composites --pattern 'wrfout_*_0_d02.nc' --analysis-end 2026-06-14T19:10:30Z --center-lon -100 --center-lat 34 --out /workspace/scores/da-seed1-member0.json
python tools/regional_rain_manifest.py model --frames /workspace/runs/control-seed1/composites --pattern 'wrfout_*_control_d02.nc' --analysis-end 2026-06-14T19:10:30Z --center-lon -100 --center-lat 34 --out /workspace/scores/control-seed1.json
python tools/regional_rain_manifest.py truth --rate-packs /workspace/shared-prepared/mrms/rate --echo-packs /workspace/shared-prepared/mrms/echo --geo-pack /workspace/shared-prepared/mrms/grid.geopack --analysis-end 2026-06-14T19:10:30Z --center-lon -100 --center-lat 34 --out /workspace/scores/truth.json
python tools/regional_rain_score.py --forecast /workspace/scores/da-seed1-member0.json --truth /workspace/scores/truth.json --baseline /workspace/scores/control-seed1.json --analysis-end 2026-06-14T19:10:30Z --event june14-development --seed 20260614 --product member0 --out /workspace/scores/da-seed1-member0.jsonl
```

Freeze the actual case analysis time, bbox and equal-area centre before executing; the values above illustrate the interface. For root-domain output use `wrfout_*_0.nc` or `wrfout_*_control.nc`. `--assert-no-resets` is available only for an independently documented non-resetting archive that lacks the native reset attribute. A reset needs a manual manifest with the exact `reset_carry_mm` amount for every cell, not that assertion.

`regional-rain/input.v1` can also read saved NPZ arrays and explicit projected cell bounds. Fields use descriptors `{path, variable}`, `{paths, variable}`, `{sum: [descriptor, ...]}`, and explicit `{constant: 1, shape: [time, ny, nx]}` masks for complete model output. NetCDF uses `gpuwm.netcdf_bridge`, with no Python decoder fallback. Native MRMS `.obspack` and `.geopack` containers preserve the Rust decoder's metadata and masks. Explicit `footprint_masks` can pin the observed tracked-object union for each lead. Their source hashes enter the receipt.

The JSONL rows include unrounded numerator, denominator, value, common-support hashes, completeness, paired baseline and paired FSS gain. A companion `.receipt.json` records exact argv, resolved configuration, source hashes, clock rules and conservative integral closure. `measurement_status=complete` means the arithmetic has complete support. `screen_status` compares an individual ratio with 0.8-1.2 for footprint/domain rain or 0.7-1.3 for echo area. The scientific `status` stays `pending`: one event and two seeds cannot supply simultaneous event-blocked 95-percent bounds, unseen-event power or the spec's ancillary loss tests.

Tests:

```sh
python -m pytest tests/test_regional_rain_gate.py -q --basetemp /workspace/test-tmp/rain-gate
```

The independent synthetic cases cover analytic conservative volume, partial-cell footprint area, reset accounting, shifted hourly windows, bracketing MRMS clocks, missing cadence/coverage, dry cases, exact neighborhood weights, native reader routing and the complete saved-field CLI. They do not establish forecast skill. Surviving legacy archive reproduction within 1 percent, real storm regrading, gauges, event-blocked confidence bounds and the remaining gate losses still require measured receipts.
