# Separate initial analysis and lateral forcing

`gpuwm prep --source rap-native` uses RAP's 13 km hybrid-level product for
the lateral sequence. The older `rap` source retains its 32 km pressure-level
product. Source names select packaged mappings and fetch-table rows.

`gpuwm go` can acquire the separate analysis automatically from recipe metadata:

```toml
[fetch]
source = "rap-native"
initial_source = "hrrr-native"
cycle = "2026-10-21T00"
boundary_cycle_offset_hours = -3
boundary_lead_offset_hours = 3
hours = 48
cadence = 1
```

The nominal cycle starts the forecast. This example fetches HRRR f00 at 00Z
and RAP f03 through f51 from the previous day's 21Z cycle. The offsets must
cancel so both inventories start at the same valid time; `hours` remains the
forecast duration. `--cycle latest` resolves the analysis source's cycle.
The packaged `hrrr_v4_gsd41.toml` uses this pairing by default. To initialize
from the boundary head instead, remove `initial_source`; RAP also remains a
selectable initial-source row. Explicit `--initial-inputs` wins over automatic
analysis acquisition. All choices use the same mapped preparation operators.

An analysis source selects the input inventory; it does not bypass model
initialization. Declared native-grid identity pairing copies mass cells on an
aligned crop, with adjacent mass winds interpolated to staggered faces and
rotated into the target grid. Specific humidity becomes dry mixing ratio
`q/(1-q)`. Temperature becomes potential temperature on the configured hybrid
coordinate. The existing vertical interpolation and hydrostatic reconstruction
still apply. Static terrain differences adjust surface pressure and soil/skin
temperature; static land masks also select water-column and snow handling.
Consequently, source provenance alone is not proof of bitwise f00 equality.
Compare history against decoded source cells and record residuals, including
float32 rounding and these initialization changes.

Pass `--initial-inputs initial.json` to replace only the initial meteorological
and soil state with a separately decoded analysis. For example:

```json
{
  "schema": "gpuwm-initial-source-v1",
  "source": "hrrr-native",
  "input_files": ["hrrr.t18z.wrfnatf00.grib2"],
  "supplements": {
    "soil_surface_data": ["hrrr.t18z.wrfprsf00.grib2"],
    "vegetation_surface_data": ["hrrr.t18z.wrfsfcf00.grib2"]
  }
}
```

Paths are relative to this JSON file. The native file supplies the hybrid
atmosphere; the same-cycle pressure product supplies the full soil column and
terrain; the surface product supplies vegetation. These same-cycle companions
can post later than the native file, and automatic acquisition waits for them.
This analysis inventory must contain exactly the experiment's start
time. The boundary inventory still includes its own first frame and a closing
frame after the forecast ends. The first boundary coupling is rebased to the
analysis and its tendency still reaches the first subsequent RAP frame.

Use the ordinary `gpuwm fetch --source rap-native` and
`gpuwm fetch --source hrrr-native` commands to acquire these products. Their
generated preparation commands name the primary and supplemental files. Add
`--initial-inputs` to the boundary source's preparation command, then run the
result with `gpuwm sim`.

The model-top interface and eta ladder belong to the experiment. Native HRRR's
interface is 1500 Pa; native RAP's is 1000 Pa. GRIB mass-level pressures have
float32 packing uncertainty, so the vertical operator recognizes a coincident
endpoint within four float32 relative epsilons. It does not extrapolate a
target outside that bound. This endpoint policy is a declared packing-roundoff
correction, not a claim of full-model bitwise agreement with WRF.

For aerosol-aware Thompson, a complete QNWFA/QNIFA pair in the initial analysis
takes priority. The native mapping reads PMTF/PMTC on hybrid levels as these
water-friendly and ice-friendly aerosol numbers. If either is absent,
`use_rap_aero_icbc = true` selects a complete pair from the boundary head,
interpolated on that donor's own pressure column. Otherwise the initializer
applies `aer_init_opt` and `wif_input_opt`, including the required climatology
checks. The initial atmosphere and land state remain the analysis's. Every
lateral frame carries the aerosol fields. The monthly dataset is still needed
for the operational two-dimensional surface emission, as described in
[analyzed aerosol input](ANALYZED-AEROSOL-INPUT.md).

The prepared bundle keeps both mappings, both input manifests and the analysis
composition receipt. Forecast preflight verifies their hashes, source names,
decoder bindings and valid time. Raw GRIB files can be archived after a sealed
bundle is prepared.

This separate-analysis interface prepares a single domain, including an
as-posted boundary window. The analysis evidence travels in its head and seal.
A tree still requires the separate analysis to be applied to every child's
initial state and is refused until that propagation is implemented.
