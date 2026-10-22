# Native derived heat, fire and lightning diagnostics

`gpuwm render --engine rust --products
heat_index_2m,apparent_temperature_2m,wetbulb_2m,wbgt_2m,hot_dry_windy,haines_low,haines_mid,haines_high,lightning_threat_1,lightning_threat_2,lightning_threat_3,pr92_flash_rate`
imports native WRF fields into the standard Rust store and renders them with
`rw_wrfbatch`. These rows also belong to the standard temperature, fire and
convective product sets. Supply the CLI's normal wrfout and output arguments.

| Product | Method and units | Required native fields beyond grid metadata |
| --- | --- | --- |
| heat_index_2m | NWS two-step Steadman/Rothfusz, °C | T2, Q2, PSFC, U10, V10 |
| apparent_temperature_2m | NWS heat index / wind chill / air temperature convention, °C | T2, Q2, PSFC, U10, V10 |
| wetbulb_2m | Thermodynamic wet bulb, °C | T2, Q2, PSFC, U10, V10 |
| wbgt_2m | Liljegren 2008 natural wet bulb and globe sensor balances, °C | Above, SWDOWN, native temperature and height profiles, valid UTC time |
| hot_dry_windy | Independent maxima of wind and surface-adjusted VPD in lowest 500 m, hPa m/s | Above and native pressure, temperature, vapor, horizontal wind, geopotential, terrain |
| haines_low/mid/high | Published pressure-layer stability and dewpoint-depression categories, 2–6 | Native pressure, temperature, vapor, geopotential, terrain |
| lightning_threat_1/2/3 | McCaul calibrated updraft–graupel, ice-column and 0.95/0.05 blend, flashes/km²/5 min | Native pressure, temperature, vertical velocity, graupel, snow, cloud ice, vapor, interfaces, terrain |
| pr92_flash_rate | WRF PR92 domain maximum updraft and widest >20 dBZ echo area, flashes/s/cell | Native vertical velocity and reflectivity inputs |

HDW is instantaneous; daily maximum and climatological percentile are separate
statistics. Haines levels below terrain remain missing. Missing microphysics
never becomes zero lightning. LTG fields are experimental calibrated proxies;
they are sensitive to microphysics and resolution. PR92 depends on the entire
imported domain. WBGT uses the original rural SRDT wind reduction to sensor height,
with temperature at 10 m interpolated from native levels. It requires radiation;
the thermodynamic wet bulb product is a different quantity.

## Observation verification

The existing `rw_verify --request REQUEST.json --json RECEIPT.json --image CARD.png`
door accepts ASOS `heat_index_2m` and `apparent_temperature_2m` station quantity
rows, in `degC`, with explicit QC bounds. Map those quantities to the identically
named native store fields in the request arm. Decoded ASOS screen temperature,
dewpoint and wind supply the observed values in Rust. Original observation times
are retained, and missing wind remains missing. METAR does not measure the solar
radiation needed to verify WBGT.

For lightning, `rw_verify --request REQUEST.json --json COUNTS.json` also accepts
schema `gpuwm.lightning-verification.request.v1`:

```json
{
  "schema": "gpuwm.lightning-verification.request.v1",
  "valid_time": "2026-08-19T21:00:00Z",
  "window_start": "2026-08-19T21:00:00Z",
  "window_end": "2026-08-19T21:05:00Z",
  "arm": {"label": "forecast", "kind": "store", "path": "store/model/run/f000.rws"},
  "area_source_path": "native-grid-invariant.nc",
  "glm_paths": ["contiguous-native-GLM-L2-LCFA-files.nc"],
  "bbox": [-110, 25, -70, 48]
}
```

The five-minute interval must start at the model valid time. The GLM list must
cover it exactly without gaps or overlaps; use the provider's observed footprint
for `bbox`. Only quality-zero flashes are counted. Native DX, DY and MAPFAC_M
provide physical cell areas, with coordinate agreement checked against the
forecast store. The receipt includes four count-score rows, source hashes,
missing support and the legacy file licence attributes. An instantaneous model
threat is held across the interval; this comparison is not a time-integrated
lightning forecast. GLM measures total lightning, whereas NLDN cloud-to-ground
counts measure a different population.

Python orchestration uses `gpuwm.rustwx.verify_observations(...)` for ASOS and
`gpuwm.rustwx.verify_lightning_counts(request, receipt_path=...)` for GLM. All
weather reads, coordinate matching, transforms and scoring occur in Rust.

ASOS archive terms: [IEM disclaimer](https://mesonet.agron.iastate.edu/disclaimer.php),
public domain. GLM distribution terms: [NOAA NODD GOES terms](https://registry.opendata.aws/noaa-goes/),
public use without restriction. No proprietary NLDN archive is downloaded by
this implementation. Source acknowledgments and the full Liljegren licence are
in [derived-diagnostics notices](../licenses/derived-diagnostics.txt).
