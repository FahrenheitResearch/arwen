# Event forcing preparation handoff

Software remains Apache-2.0: ngen
`be11a180565a78ff069d855b0b5dbfc23845e7d8`, PET
`6c410f2a8bc843043ad25e13239928031b8c5e64`; actual LICENSE headings read:
“Apache License”, “Version 2.0, January 2004”. Complete notices retained.

The referee owns case selection in
`<scratch>/referee/nextgen/case-c-selection.json`. Its current revision
2026-10-18T01:35Z selects Russian River, 2023-01-05T00Z through 2023-01-17T00Z,
superseding the earlier Winooski message. This is a preparation contract;
there is no native or GPU event result. Confirm any later referee revision
before fetching the event window, including separately chosen warmup hours.

The selected gauges are USGS11462500 (Hopland),11464000 (Healdsburg), and
11467000 (Hacienda Bridge/Guerneville, outlet). Retrospective's existing
January alignment maps these to original NHDPlus/archive COMIDs8269119,
8272379 and8272725 respectively. These are not a model-fabric crosswalk.
The selection bytes are sealed by SHA256
`3fd6740e581728806ef572b611f61758d101613865079c48eb810edf10ae34c5`;
a later referee revision must be read rather than inferred from queued prose.
Dependent July2023 acquisition remains paused. Case(a), case(b), generic
tables and retained artifacts continue; operational output cannot replace
NWM retrospective output.

For this particular12-day `[start,end)` model window there are288 hourly
forcing intervals, requiring endpoint stamps2023-01-05T01Z through
2023-01-17T00Z. Retrospective/observation comparisons may instead include
both window endpoints:289 hourly point samples per gauge. Do not turn that
inclusive point count into289 forcing intervals. Additional warmup intervals
need an explicit referee choice and their own source coverage.

The primary [NOAA NWM archive registry](https://registry.opendata.aws/nwm-archive/)
places v3 CONUS coverage in February1979 through January2023; the retrospective
owner's actual coordinate read closes at2023-02-01T00Z. Its retained January
baseline is separate from the still absent native/GPU event model output.
The registry identifies retrospective meteorology as AORC v1.1; access to
another AORC publication does not establish identical native-provider inputs.
Preserve source publication/version, sample policy and packing provenance.

Hydrofabric supplies the exact ordered catchment IDs and source-grid CSR.
It must also identify the subset's coordinate arrays, CRS, row/column order,
source cell bounds and mask coverage. JSON weights contain `catchment_ids`,
`grid_shape`, `offsets` u64, `indices` u32, `weights` f64. These indices address
the local grid's flattened spatial order, not an assumed global cell index.
No centroid substitution, absent-catchment fill or inferred catchment ordering.
The complete geometry/model-parameter/initial-state requirements in the
referee's event input contract remain independent prerequisites.

For the NOAA AORC hourly profile, a requested interval `[start,end)` needs
source stamps `start+1h` through `end`, inclusive of the last endpoint. The
shipped `aorc-grid` row aligns precipitation totals with the preceding hour.
Co-stamped nonprecipitation endpoint samples are held over that hour; they
are not declared observed hourly means. The exact sampling policy must also
appear in the native realization before claiming native-provider equivalence.
Derived start-aligned sources require a separate explicit table row.

Setup door: `gpuwm.hydro.forcing_inputs.prepare_external(request_path,
library=...)`. The request selects `source="aorc"`, ordered local NetCDF
`inputs`, `output`, UTC `time_origin`, `interval_seconds=3600`, ordered
`catchment_ids`, and `area_weights`. The Rust loader validates timestamps,
masks missing values, unpacks fields, converts interval amounts and regrids.
`prepare_series` then binds the packet through the resident ABI in CONTRACT.md.
No data-path work is added to Python or the per-step host launch sequence.

The official [NOAA AORC registry](https://registry.opendata.aws/noaa-nws-aorc/)
permits public use, requests NOAA attribution and prohibits implied endorsement.
On 2026-10-18 its actual 2023 consolidated archive metadata was fetched from
`https://noaa-nws-aorc-v1-1-1km.s3.amazonaws.com/2023.zarr/.zmetadata`:
11890 bytes, SHA256
`eec224d45eb7621ff8f13621f22c733be19c491389d55ab3ceda62530ef76e09`.
The weather arrays use `[time,latitude,longitude]`, packed int16 and Zstd
chunks `[144,128,256]`; preserve each variable's actual scale, offset and
missing-value metadata. This archive is Zarr, while the connected external
door accepts NetCDF/CSV. A bounded Rust subset/decode/NetCDF handoff is still
needed before claiming direct cloud-to-packet support. Do not stage full-CONUS
hourly grids or use Python/xarray to bridge it.

The setup-only [coordinate export](ZARR-COORDINATES.md) now decodes the actual
sealed latitude/longitude/time chunks in Rust. It preserves source words and
ascending spatial order (4,201 by 8,401 centres), with explicit hashes and
source attributes for hydrofabric. The actual time axis selects `[97,385)`
for the288 forcing endpoints and `[96,385)` for289 inclusive point samples.
No weather payload is read. The metadata declares no spatial bounds or CRS;
optional midpoint bounds are marked inferred, not provider-published. Native
area/sampling policy and geographic CSR qualification remain prerequisites.

NOAA forcing terms do not clear Lynker fabric rights. Shared coordinator
ruling: actual v2.2 bucket notice CC BY-NC-SA4.0, private noncommercial research
only; no public bundle, default download or demo. Independently permitted
government fabric must retain its own provenance. Observation/retrospective
owners retain their event and ID crosswalks; no model skill follows from
acquisition or the analytic forcing-clock regression.
