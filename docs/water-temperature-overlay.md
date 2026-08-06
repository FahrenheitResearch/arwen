# High-resolution water-temperature overlay (ERA5 route)

Optional, off by default. A user-supplied high-resolution gridded
water-temperature analysis replaces ERA5 SST and SKINTEMP over WATER
source cells before horizontal interpolation. Configured on nothing,
the ERA5 route is byte-identical to a tree without the feature.

## The problem it fixes

ERA5's water-surface temperature is coarse structure at 0.25 degrees,
and a convection-permitting nest inherits it as hard rectangular
quantization of near-surface fields over lakes and coasts (2 m
dewpoint is where users see it first). This is a documented stock
WRF + ERA5 limitation, not something specific to this model's ingest;
the WRF community threads below describe the same artifact and its two
remedies:

- <https://forum.mmm.ucar.edu/threads/lake-sst.10558/>
- <https://forum.mmm.ucar.edu/threads/issue-with-era5-sst-fields.12525/>
- <https://forum.mmm.ucar.edu/threads/sst-input-resolution-and-landmask.15850/>
- <https://forum.mmm.ucar.edu/threads/resolved-weird-sst-skintemp-artifacts-along-coast.9083/>

The community remedies are (1) drop ERA5 SST and let SKINTEMP carry the
water temperature, or (2) substitute a high-resolution SST/lake product.
This overlay is remedy (2) implemented natively: what WRF users
accomplish by feeding metgrid an extra hi-res SST source (or running
`avg_tsfc.exe`), declared as one input file here. Behavior where WRF
defines one is matched in spirit; the mechanical difference (we replace
values on the ERA5 source grid before interpolation, metgrid
interpolates the extra source directly) is deliberate and documented
here.

### Why not the SKINTEMP-preference remedy as a default?

Measured on the reproducing case (Lake Erie, 1985-05-31 12Z ERA5):

- ERA5 SKINTEMP over the lake is the FLake model state: 287.4 to
  295.1 K across the lake with adjacent-0.25-degree-cell jumps up to
  about 6 K. It IS the blocky field.
- ERA5 SST exists over the lake there (HadISST2-era analysis) and is
  smooth: 284.6 to 289.0 K.
- The two disagree by up to about 8 K at the same cells, so any chain
  that switches between them mid-lake (ours matches `real.exe`: water
  skin takes SST where the interpolated SST is valid, SKINTEMP
  otherwise, `gpuwm/ingest/soil.py`) paints a seam of that amplitude.

Preferring SKINTEMP everywhere would standardize on the WORSE field
over lakes and keep the quantization. The overlay replaces BOTH fields
with one consistent analysis, which removes the quantization and the
seam at once. So remedy (1) ships as documentation, not as a default
change.

## Configuration

Runtime route (`gpuwm run`), one optional `[case_data]` key:

```toml
[case_data]
# ... forcing, vtable, wps_namelist, geog_root as usual ...
water_temperature_overlay = "/data/oisst-avhrr-v02r01.19850531.nc"
```

era5_direct route: the `--water-temperature-overlay ANALYSIS` flag (a
`water_temperature_overlay` keyword on `prepare_era5_wrf`). The file
must then also appear in the input manifest under the
`water_temperature_overlay` role, hash-bound like every other input.

Identity: the key never touches `ExperimentConfig`, so the experiment
fingerprint and restart identity of every existing experiment are
unchanged when it is absent (pinned by a golden-hash test in
`tests/test_water_overlay.py`). Declared, the file joins the
InputCatalog / input manifest and the supervisor's content-addressed
snapshot set, so the run identity moves exactly when the data does.

## Accepted files

Container is sniffed by magic, refusals are named:

- **netCDF** (HDF5 or classic CDF): the water-temperature variable is
  found by CF `standard_name` (`sea_surface_temperature` and kin,
  `lake_surface_water_temperature`, `sea_water_temperature`,
  `surface_temperature`) or by name (`analysed_sst`, `sst`,
  `water_temp`, `wtmp`, `lswt`); zero or several candidates is a
  refusal naming what was found (declare `variable=` explicitly in the
  loader call to disambiguate). The trailing two dimensions must be
  (latitude, longitude) with 1-D coordinate variables; leading
  dimensions must be size 1. `units` is required; kelvin and Celsius
  spellings are accepted and Celsius is converted.
- **GRIB2**: exactly one record with WMO (discipline, category,
  parameter) = (10, 3, 0) (water temperature, kelvin), regular
  latitude/longitude grid, decoded by the vendored Rust
  `grib2_inventory`/`grib2_dump` tools (env
  `GPUWM_GRIB2_INVENTORY`/`GPUWM_GRIB2_DUMP`, else the repo-local
  release build). GRIB edition 1 is refused by name.

Value guards, all named refusals: valid-cell median outside
[250, 340] K (with explicit "Celsius-shaped values under a kelvin
declaration" / "kelvin-shaped values under a Celsius declaration"
diagnoses in both directions) and any valid cell outside [230, 350] K
(the undeclared-fill-sentinel catcher). Ascending or descending axes
are handled; a 0..360 longitude ring is re-cut into a continuous
ascending axis.

## Semantics

Water source cells are `LANDSEA < 0.5` on the ERA5 grid. At each water
cell the overlay is sampled with a masked bilinear: the four
surrounding overlay cells contribute with weights renormalized over
VALID corners only. Covered cells replace SST and SKINTEMP (whichever
the snapshot carries); everything else -- land cells, uncovered water
cells, every other field -- is value-identical. A configured overlay
that covers none of the crop's water is refused by name (it is a
misconfiguration, not a fallback). The run prints a one-line receipt:

```
water-temperature overlay: replaced 41 of 44 water source cells per snapshot from ... (3 kept ERA5 fallback)
```

Documented v1 seams:

- Coverage ends at the overlay's bounding box / valid-data edge at
  ERA5-cell granularity; uncovered water keeps ERA5 values.
- A global overlay's longitude ring keeps one artificial cut (no
  periodic wrap in v1), the same stance the ERA5 ring itself gets;
  water cells in the one-cell gap fall back to ERA5.
- One analysis file serves all forcing times of the run (water
  temperature varies slowly at forecast range); a per-time overlay
  schedule is future work if a case ever demands it.

## Choosing a source

Verified during the 1985 acceptance work (fetch what your case's date
allows; always the full file, receipt-ed):

- **NOAA OISST v2.1** (daily, 0.25 degree, September 1981 onward):
  covers the Great Lakes; the pre-1995 choice for lake cases. The 1985
  verification below used it -- over Lake Erie its maximum
  adjacent-cell jump was 0.58 K against ERA5 SKINTEMP's ~6 K, and
  removing the FLake cell noise plus the SST/SKINTEMP seam is what
  kills the artifact even at equal grid spacing.
- **ESA CCI / C3S SST L4** (daily, 0.05 degree, September 1981 onward,
  CDS `satellite-sea-surface-temperature`): the hi-res OCEAN choice.
  Measured caveat: its analysed_sst is masked over the Great Lakes (0
  valid Lake Erie cells on 1985-05-31, CDR3.0), so it cannot fix a lake
  case alone.
- **GLSEA** (NOAA GLERL, Great Lakes, 1995 onward) and **OSTIA**
  (global 0.05 degree, 2006 onward): the modern-era choices for lakes
  and ocean respectively; any conforming netCDF/GRIB2 works.

## Acceptance measurement (the reproducing case)

Lake Erie, 1985-05-31 12Z ERA5, 24 km parent with a ratio-8 3 km nest,
f006, identical configs except `water_temperature_overlay` (NOAA OISST
v2.1 for the run date). 2 m dewpoint over the lake's 2827 water cells
(5381 adjacent-cell pairs) on the 3 km nest:

| adjacent-cell TD2 step   | ERA5 water temps | OISST overlay |
| ------------------------ | ---------------- | ------------- |
| median                   | 0.09 K           | 0.07 K        |
| P95                      | 0.96 K           | 0.35 K        |
| P99                      | 2.31 K           | 0.58 K        |
| cells with steps > 1 K   | 4.83 %           | 0.17 %        |

The rectangular quantization visible in the control render is absent in
the overlay render (before/after pair delivered with the reproduction
assets), and the lake-wide dewpoint dropped about 1.6 K as the FLake
warm patches (294+ K skin in late May) gave way to the analysis's
~286 K -- the climatologically right value for the date.
