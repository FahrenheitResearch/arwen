# High-resolution water-temperature overlay (ERA5 route)

Optional, off by default. A user-supplied high-resolution gridded
water-temperature analysis replaces ERA5 SST and SKINTEMP over WATER
source cells before horizontal interpolation. Configured on nothing,
the ERA5 route is byte-identical to a tree without the feature.

## The default: one provider per body of water

`water_temperature_policy` in `[case_data]` names which provider decides the
water-surface temperature. Silence selects `era5_class_coherent`, so a bare
run gets coherent lakes with no declared file and no network access.

| policy | what it does | when to declare it |
|---|---|---|
| `era5_class_coherent` (default) | Surface classes come from the target statics (land `LANDMASK >= 0.5`, lakes the MODIS inland-water category, ocean the rest). Each class is labelled into connected components and ONE provider is chosen per component: the ERA5 SST analysis, interpolated with a normalized masked bilinear over that component's own donors, where those donors cover at least half of it; otherwise the coherent SKINTEMP field for the whole body. | nothing to declare |
| `wrf_compat` | The historical per-cell selector, byte-for-byte: mapped SST where it lies in 170..400 K, mapped SKINTEMP elsewhere. | stock-WRF certification, parity batteries, reproducing an archived run |
| `external_overlay` | The overlay machinery below, unchanged. A declared `water_temperature_overlay` selects this on its own. | an observational analysis is available |

Two properties are structural rather than checked after the fact. Donors are
selected by component identity instead of by radius, so a lake cannot be
handed ocean water and no water cell can be handed a land donor. A donorless
body takes its component fallback rather than a global search, so a landlocked
lake with no analysis of its own cannot import another basin's water. Every
water cell carries a provider id in the diagnostic `WATER_TEMP_SOURCE`, and a
water cell that reaches the end without a provider is a refusal, not a fill.

### Which routes it covers, and how that stays true

The guarantee is enforced at the soil preprocessing routers, which are the one
seam every forcing route already crosses, and not route by route. A route that
hands the router a raw `SST` beside its `SKINTEMP` with no assembled field is
refused by name and told where the decision belongs; `water_temperature_policy
= "wrf_compat"` is the one declaration that reopens the historical selector. A
mapping that carries no `SST` passes, because the selector then reduces to
`SKINTEMP` on every water cell, which is exactly what the assembly returns for
a body with no donors: the two are the same array, and a test pins that.

| route | water temperature | lake class |
|---|---|---|
| ERA5 (`gpuwm run` and the direct adapter) | assembled from the source SST analysis | the land-use table's `ISLAKE` |
| nested children of either ERA5 route | assembled, under the parent's policy, on the child's own statics | the child's own `ISLAKE` |
| mapped / rw-wps, 20CRv3 included | assembled; these compositions declare a skin temperature and no SST, so the assembled field equals the mapped skin | the land-use table's `ISLAKE` |
| GFS | assembled after the lake skin is resolved, because GEOG lakes are smaller than a 0.25 degree GFS cell and this route classes them as land for the mapping | the land-use table's `ISLAKE` |
| native HRRR | not assembled: no SST in the inventory, so the assembly is the identity | not applicable |

One scope limit, stated because it is a real gap and not a rounding: the ERA5
direct adapter run with `--static-input` and no `--geog-root` cannot resolve a
land-use table at all, because the prebuilt static NPZ is numeric fields only.
That run prints a line saying so and falls back to the historical per-cell
selector, so its lakes stay blocky. Passing `--geog-root`, which is read only
for the land-use index, restores the default treatment.

A land-use table with no inland-water class at all (USGS 24-category, where
WPS writes `ISLAKE = -1`) is a declared state, not an empty mask: the advisory
says that inland water is classed with the ocean on that domain, so a lake
joined to the sea by a coarse coastline can share its provider there.

## What the ingest fixed first, and what is left for an overlay

Most of the blockiness this document was written about was this project's own
defect, not ERA5's content. Measured on the reproducing case (Lake Erie,
1985-05-31 12Z, 3 km nest, 4195 water cells, 2684 of them in Erie itself),
attributing every mechanism to the source or to the chain:

| mechanism | source or ours | what it did | disposition |
|---|---|---|---|
| The water temperature was selected PER CELL between mapped SST and mapped SKINTEMP (`gpuwm/ingest/soil.py`) | **ours** | The two providers disagree by a mean of -5.47 K and up to 12.37 K on the cells where both exist, so switching between them mid-lake painted a seam of that amplitude. 257 of the 269 intra-lake adjacent steps above 1 K sat exactly on the switch boundary. | fixed: one provider per connected body, `gpuwm/ingest/water_temperature.py` |
| WPS's SST operators quit at the coast (`sixteen_pt+four_pt`, both need a full stencil, SST is missing over land, `fill_missing=0.`) | shared with WPS | Only 46.6 % of the nest's water cells passed the validity test, which is what drove the switch above; the mapped SST field itself keeps this behaviour, because the reconciler and the parity batteries read it expecting METGRID.TBL. | no longer decides anything: the assembly reads the SOURCE analysis, not this field |
| ERA5 SKINTEMP over lakes is the FLake model state | source | Warmer than the analysis and quantized on the 0.25 degree cell. | still the fallback for a body with no analysis of its own, and coherent across that whole body |
| Mixed land/water source cells enter the water donor set (`LANDSEA` binarized at 0.5) | ours | SKINTEMP ramps toward the land mean near shore. | second order once the analysis reaches the lake. On its own it warms near-shore mapped SKINTEMP by a mean of 0.31 K and at most 2.59 K, and it does not build a shore-to-midlake gradient by itself: that gap is -0.18 K with the mixed donors and +0.14 K without. Isolated by remapping SKINTEMP twice on the same case, donors `LANDSEA <= 0.5` against donors `LANDSEA == 0.0`, and reading the near and far means defined below off the mapped field. |

Measured before and after at the same 18Z valid time on the same case, with
nothing declared:

| statistic | shipped per-cell selector | `era5_class_coherent` | stock WRF v4.6.1 oracle |
|---|---|---|---|
| intra-lake adjacent-step P99 | 7.32 K | 0.13 K | 0.60 K (1 km, 1974 case) |
| intra-lake steps above 1 K | 5.18 % | 0.00 % | 0.65 % |
| shore-to-midlake TSK gap | +5.21 K | +0.46 K | -2.33 K (12 km, 1974 case) |
| lake TSK range | 284.51..294.86 K | 284.49..289.41 K | 273..276 K coherent |

The instrument behind those four rows, stated so that a re-measurement lands
on the same numbers instead of on a neighbouring definition:

* **The field is TSK in the 18Z forecast frame** of the two runs, not the
  assembled water temperature at the 18Z forcing time. Those are two
  instruments and they disagree: read the assembled field instead and the
  shipped column becomes P99 5.36 K and 284.51..296.63 K, with the same
  5.18 % above 1 K. Six hours of surface integration is the whole of the
  difference, and the forecast frame is what a user sees.
* **The lake is the largest connected component of `LAKEMASK`**, 2684 cells,
  labelled 4-connected by `_label_components` in
  `gpuwm/ingest/water_temperature.py`. It carries the two step rows and the
  range row.
* **An adjacent-cell step** is `abs(TSK[i,j] - TSK[i,j+1])` and
  `abs(TSK[i,j] - TSK[i+1,j])` over pairs with both cells inside that
  component. The P99 row is that set's 99th percentile and the row under it
  is the fraction of the set above 1 K.
* **The shore-to-midlake gap is measured on a box, not on the component**:
  over water cells (`LANDMASK < 0.5`) inside 41.2 to 42.95 N and 83.6 to
  78.6 W, it is the mean TSK at a 4-connected distance of 2 cells or less
  from the nearest land cell, minus the mean at 9 cells or more. That is 673
  near cells and 699 far cells out of the box's 2826 on this 3 km nest.
* **The lake TSK range** is the minimum and the maximum over the component.

ERA5 carries an SST analysis over the Great Lakes in every era sampled (1985,
1995, 2000, 2010, 2020), so the default is not era-specific. A declared
overlay remains the higher-accuracy option and is what to declare when the
case turns on lake temperature.

### Limitation of the overlay path

The overlay is sampled onto the ERA5 SOURCE grid before horizontal
interpolation, so a high-resolution analysis loses detail finer than 0.25
degrees. Interpolating a declared analysis directly to the target is out of
scope here and would change the overlay's identity payload.

## The problem it fixes

ERA5's water-surface temperature is coarse structure at 0.25 degrees,
and a convection-permitting nest inherits it as hard rectangular
quantization of near-surface fields over lakes and coasts (2 m
dewpoint is where it shows first). Stock WRF + ERA5 has the same
artifact, for the reason in the table above: the SST operator chain is
METGRID.TBL's, so metgrid abandons the same coastal cells to
`fill_missing=0.` and real.exe falls back to the same FLake SKINTEMP.
That shared inheritance is why the community threads below describe
this artifact and its two remedies, and why this project stopped
matching WPS on the cells WPS leaves undefined:

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

What the default DOES move, stated plainly: water-surface temperature.
`era5_class_coherent` interpolates the source analysis itself rather than
reading METGRID.TBL's mapped SST field, so ocean and other SST-valid cells
are **not** bit-identical to the previous default. Measured on the
reproducing case over the 1955 water cells where the mapped SST was valid,
the assembled field differs from it by a mean of +0.002 K with a maximum of
0.169 K, and 0 of the 1955 are bit-identical. That is the two mapping paths
disagreeing at interpolation noise, not a physical change. It has NOT been
verified on a real coastal ocean domain, where the SST-valid fraction and
the component structure are both different; declare `wrf_compat` for a run
that must reproduce an archived result byte for byte.

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
