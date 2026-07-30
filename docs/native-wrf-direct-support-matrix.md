# Native source to CPU-WRF support matrix

This matrix is fail closed.  `accepted` means the exporter validates and emits
the combination; `stock-WRF verified` additionally means an unchanged WRF
v4.6.1 executable has opened the files and advanced.  Anything not listed as
accepted must be rejected rather than substituted.

The runtime invariant is permanent: native gpuwm reads the source GRIB and
emits `wrfinput_dNN`/`wrfbdy_d01` directly.  WPS and `real.exe` are not runtime
dependencies.

| Capability | Accepted now | Stock-WRF verified | Required next work |
|---|---:|---:|---|
| HRRR f00/f01 hourly forcing | yes | yes, 10 seconds | none |
| Consecutive hourly f00..fNN forcing | yes | yes, 12-record f00..f12 file advanced 10 seconds | extend horizon matrix |
| Forecast start time | exact hourly cycle derived from `VALID_TIME` | 2026-07-18 00Z | broaden proof matrix |
| Projection: Lambert `map_proj=1`, polar stereographic `map_proj=2`, Mercator `map_proj=3` (`MAP_PROJ`/`MAP_PROJ_CHAR` derived per projection) | yes | Lambert yes; Mercator and polar are oracle-verified plus smoke-run verified, not matched-run/stock-WRF verified | non-Lambert stock-WRF proofs; derive all attributes dynamically |
| 500 x 500 x 49, approximately 1-km proof grid | yes | yes | none |
| Oklahoma 1000 x 1000 x 49 at 1 km | yes | yes, full 12-record boundary file | broaden geography matrix |
| Oklahoma/Ohio 192 x 160 x 49 at 3 km | yes | yes | broaden size/resolution matrix |
| Other positive equal `dx`/`dy` and horizontal extents | strict dynamic Lambert preflight; not broadly certified | no | add representative stock-WRF proofs |
| Other eta/vertical levels | no | no | dynamic coordinate/declaration work |
| Single specified outer domain | yes | yes | none |
| Multiple domains / nests | exact Oklahoma d01+d02 one-way slice; child initialization is worker-scheduled and emits `wrfinput_d01`, `wrfinput_d02`, and `wrfbdy_d01` | yes, unchanged WRF advanced both domains 15 seconds | generalize beyond the sealed two-domain layout |
| WSM6 + YSU + classic-MM5 + Noah | yes | yes | none |
| Other microphysics/PBL/surface/LSM state | no | no | scheme-specific state and metadata exporters |
| Radiation namelist choice | native hierarchy proof has LW off plus Dudhia SW | stock gate used RRTM LW plus Dudhia SW | exact stock-only runtime deltas are receipt-bound; broaden scheme matrix |
| Native-to-stock namelist deltas | fail-closed to `ra_lw_physics 0 -> 1`, `use_theta_m 0 -> 1`, and stock-only `ghg_input=0` | yes | remove a delta only when stock WRF supports the native representation directly |
| Non-hourly or gapped forcing | no | no | explicit time-axis generalization |
| Downloaded f00..f12 one-command input | yes, exact filenames and SHA manifest required | yes | generalize series discovery without weakening the manifest gate |
| GFS `pgrb2.0p25` pressure-level series | yes, uniform manifest-bound series beginning at f000 | yes, f000/f003 advanced 5 seconds | extend horizon/geography matrix |
| GFS 21 pressure levels, 1000..100 hPa | yes, complete inventory required at every time | yes | add alternate vertical slices explicitly |
| GFS four exact Noah soil slabs | yes, both GRIB2 fixed surfaces validated; direct layer copy | yes | none for Noah |
| GFS packing/process identity | DRT 5.0 only; process 81 at f000 and 96 for forecasts | yes for the certified NOAA sample | certify additional operational product variants explicitly |
| GFS masked surface/soil interpolation | native land-aware nearest/global lake-donor policy | structurally accepted; not WPS numerical parity | add field-by-field METGRID comparison if parity is required |

Current safe operational inputs are the output path and a valid time that
exactly matches the sealed preparation cache.  The boundary interval is fixed
within a run and must equal the validated source cadence: 3600 seconds for the
certified HRRR series and 10800 seconds for the certified GFS series.  Physics,
grid, projection, vertical coordinate, specified boundary width, and nest
topology are state-shaping options and therefore are not safe namelist-only
substitutions.

Current implementation order is:

1. broaden arbitrary-Lambert stock-WRF geometry proofs;
2. generalize the one-command f00..f12 series to manifest-bound f00..fNN;
3. support additional vertical coordinates and scheme-specific state;
4. generalize the sealed d01+d02 hierarchy to additional nests and layouts;
5. certify additional Rusty Weather source adapters one at a time.
