# Composition initial and boundary inputs

`gpuwm sources merra2 --json` describes a composition supplement.
Meteorological forcing is still required. `chem_sources` selects enabled source
rows; each species' boundary alternatives select its first enabled provider.
Composition is imported before preparing a forecast; initialization never downloads
or reads account credentials.

Published files use one generic Rust decoder, selected entirely by source-table
coordinates, variable names, units, file groups and pressure metadata. A new source
with the same data contract needs table rows and species boundary alternatives.
The importer checks identical horizontal/native grids, CF valid times, exact units,
finite values, all requested fields at every time, and duplicate timestamps. It
refuses incomplete coverage rather than substituting clean air. Original inputs,
source metadata, decoder and decoded payloads have content identities; changing any
of them requires importing again.

For NASA MERRA-2, download matched `M2I3NVAER` and `M2I3NVASM` files through NASA's
Earthdata service using a **NASA Earthdata login**, then import the existing files:

```sh
python -m gpuwm.composition_source --source merra2 \
  --files aerosol=/data/aer.nc4 meteorology=/data/asm.nc4 --cache /data/chem-cache
```

Repeat `aerosol=...` and `meteorology=...` for every additional input file. Include
the bracketing 3-hourly frames for the whole forecast and its forcing times.
`GPUWM_DATA_STORE_CACHE` selects the same cache during forecast preparation.
Native pressure is reconstructed in Rust from the declared 1 Pa model top and all
72 DELP levels; PS is retained separately. QV is read
from the companion ASM collection and converts source moist-air mixing ratios to
WRF-Chem dry-air units. GEOS-FP uses the same import command with `--source geos-fp`
and matched `inst3_3d_aer_Nv`/`inst3_3d_asm_Nv` files. That row represents primary
aerosol; nitrate, ammonium and brown carbon in newer GEOS files are outside it.

CAMS EAC4 acquisition requires a **Copernicus ADS key**. It also requires all 60
native model levels, matched humidity and surface pressure.
Its L60 half-level A/B coefficients are table metadata; the existing Rust remapper
computes pressure. Below-ground missing values in a pressure-level export do not
meet this native-level contract. Import with:

```sh
python -m gpuwm.composition_source --source cams-reanalysis \
  --files composition=/data/model-levels.nc surface=/data/surface.nc \
  --cache /data/chem-cache
```

CAMS forecast GRIB uses the existing `gpuwm.data_store_fetch` plan/fetch/import
commands and `cams-global` row. CAMS data is CC-BY-4.0 (Licence to use Copernicus
Products), which permits commercial use with attribution, so neither CAMS route needs
an opt-in; the attribution is carried in the run receipts. A source row whose licence
limits use declares `research_only_licence` and then needs `--licence` naming it.
The source catalogue is the authority for current data terms. NASA data use follows
its unrestricted-data guidance; preserve collection citations in run receipts.

For aerosol-aware Thompson (`mp_physics=28`), the default
`aerosol_mp_coupling="auto"` diagnoses supplied aerosol mass into the number targets
present in the active chemistry table before cold-start droplet closure, then after
each chemistry step. An explicit `"none"` disables this coupling. Supplied zero
number fields remain data and do not trigger a synthetic fill. A dust-only set
supplies NIFA; its receipt does not claim supplied NWFA. NASA organic matter is
converted to transported organic carbon once, and the optical and Thompson
operators restore the documented organic-matter factor. Five source sea-salt bins
are mass-conservatively mapped to the four transported bins; sub-0.1 micrometre salt
is carried in the first bin, so that size distinction is unresolved.

The published-file source contract has a completed nine-hour, 3 km smoke forecast
from public GEOS-FP composition on 2020-09-09, with ten hourly AOD and near-surface
dust maps rendered by Rust. AERONET verification matched 262 pairs at six stations;
the forecast underpredicts smoke AOD (bias -0.5625, RMSE 1.2026). This proves the
source-to-product path; it does not establish accurate smoke-event skill. The run
uses Thompson mp8; source-backed aerosol-aware Thompson mp28 diagnosis is covered
separately by device checks. MERRA-2 and CAMS share the native decoder contract;
their authenticated upstream acquisitions were not exercised. Composition remains
a supplement to meteorological forcing in the inventory.

AERONET V3 direct-sun Level 2 files from the public web service can verify stored
forecast output with Rust data processing:

```sh
python -m gpuwm.aeronet_verify --observations /data/aeronet-level2.txt \
  --frames /data/wrfout_d01_* --out /data/aod-verification --workers 8
```

The operator interpolates positive observed AOD in log wavelength/log AOD between
measured band centres bracketing 550 nm. It matches the nearest model mass point
within 10 km, then linearly interpolates model AOD between bracketing history
frames at most one hour apart. Both limits have named CLI options. It never
extrapolates in wavelength or time. Summary JSON includes exclusion counts,
pair-weighted bias/MAE/RMSE, per-station scores and every matched pair. Without
`--frames`, it decodes the observations and explicitly reports `scored=false`.
This is preparation, not a forecast score. Input files must contain one history
time each; multiple-time files are refused with that requirement.

Native pressure uses each source row's declared reference. The NASA 72-layer rows
add `DELP` downward from the documented 1 Pa model top, taking the mean of each
layer's edges; `PS` remains a separate surface reference. Subtracting independently
packed/regridded `DELP` from `PS` can create negative upper-layer pressures. The
stack receipt records the reconstructed surface minus `PS` range. No `DELP`
rescaling or upper-layer pressure clipping is applied. See NASA [GEOS-FP file
specification, section 4.2](https://gmao.gsfc.nasa.gov/pubs/docs/Lucchesi1203.pdf).

Missing input or registration errors name the source acquisition requirement and
the missing chemistry IC/BC state. These messages do not inspect credentials or
claim that an account is absent; local import itself requires no login or key.
