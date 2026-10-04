# Separate initial analysis and lateral forcing

`gpuwm prep --source rap-native` uses RAP's 13 km hybrid-level product for
the lateral sequence. The older `rap` source retains its 32 km pressure-level
product. Source names select packaged mappings and fetch-table rows.

Pass `--initial-inputs initial.json` to replace only the initial meteorological
and soil state with a separately decoded analysis. For example:

```json
{
  "schema": "gpuwm-initial-source-v1",
  "source": "hrrr-native",
  "input_files": ["hrrr.t18z.wrfnatf00.grib2"],
  "supplements": {
    "soil_surface_data": ["hrrr.t18z.wrfprsf00.grib2"]
  }
}
```

Paths are relative to this JSON file. The native file supplies the hybrid
atmosphere; the same-cycle pressure product supplies the full soil column and
terrain. This analysis inventory must contain exactly the experiment's start
time. The boundary inventory still includes its own first frame and a closing
frame after the forecast ends. A one-hour forecast can therefore use three-hour
RAP forcing without replacing the first RAP boundary with the HRRR analysis.

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

For aerosol-aware Thompson, `use_rap_aero_icbc = true` requires analyzed
water-friendly and ice-friendly aerosol numbers. With a separate initial
analysis, these come from the first boundary analysis and are interpolated on
that donor's own pressure column. The HRRR initial atmosphere is kept. Every
lateral frame carries the aerosol fields. The monthly dataset is still needed
for the operational two-dimensional surface emission, as described in
[analyzed aerosol input](ANALYZED-AEROSOL-INPUT.md).

The prepared bundle keeps both mappings, both input manifests and the analysis
composition receipt. Forecast preflight verifies their hashes, source names,
decoder bindings and valid time. Raw GRIB files can be archived after a sealed
bundle is prepared.

This separate-analysis interface currently prepares a single domain from a
complete forcing window. A tree requires the separate analysis to be applied
to every child's initial state, and an as-posted window requires the donor
binding to travel in each head and seal; these combinations are refused rather
than starting only part of the hierarchy from the selected analysis.
