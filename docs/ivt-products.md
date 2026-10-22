A regional run renders `ivt`, `iwv`, `freezing_level`, and `ar_outline` by default.
They are also catalog products for `gpuwm render --products ...`. All weather
pictures use `gpuwm.rustwx` and the production `rw_wrfbatch` renderer.

`ivt` draws magnitude in kg m-1 s-1, transport direction arrows, and the 250 contour.
The native component planes are `ivt_eastward_1000_300` and
`ivt_northward_1000_300`; positive directions are east and north. The integral is
`(1/g) integral(q*u, q*v) dp`, from min(surface pressure, 1000 hPa) to 300 hPa.
It uses native model pressures and earth-relative model-level winds, converts
WRF water-vapour mixing ratio r to specific humidity q=r/(1+r), integrates the
vector before taking its magnitude, and never resamples onto isobaric levels.
Trapezoids follow native pressure segments, with linear clipping at the two
integration limits. The nearest mass-level value extends to the surface and
model-top interfaces. Invalid or nonmonotonic columns remain missing.

`iwv` integrates q from the surface to the finite model lid in kg m-2.
`freezing_level` is the lowest 0 C crossing above terrain, in metres MSL; a
column without a crossing remains missing. `ar_outline` draws the IVT=250
kg m-1 s-1 contour over IVT magnitude. This threshold alone is an outline of
transport exceeding 250, without length, width or persistence classification.

The native observation scorer accepts CF NetCDF references through its existing
`radar` gridded-observation list. `grid_path` can name the same CF file. Set
`radar_source` to `era5`, and map forecast `fields` for the required quantities.
`ivt_eastward_full_column`, `ivt_northward_full_column` and `ivt_full_column`
are matched to ERA5 162071 (71.162) and 162072 (72.162). ERA5 is a reanalysis
reference, not independent observations of the forecast initialized from it.
These are separate from the bounded operational IVT. The comparison retains
that ERA5 spans the atmosphere and the model has a finite lid.

The CF reference uses `viwve`, `viwvn` (kg m-1 s-1) and optionally `era5_tcwv`
(kg m-2), with `source_variable` attributes naming the original archive
variables and a global `source_store` or `source`. The generic Rust Zarr reader
writes these attributes. The bindings in `gridded-verification.json` and
`cf-products.json` reject wrong quantities and units. Continuous component,
magnitude and water-vapour errors enter the existing
`gpuwm obs scorecard` path; magnitude FSS uses 250 and 500 thresholds.

A public ERA5 acquisition is an explicit research-data choice under CC-BY 4.0.
No ERA5 data ships with these default diagnostic calculations. The archive is
[ARCO ERA5](https://github.com/google-research/arco-era5), and parameter definitions
come from [ECMWF](https://codes.ecmwf.int/grib/param-db/?id=162071).
