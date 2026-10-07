# SFIRE static interpolation control

Build the unmodified WPS v4.6.0 `nearest_neighbor` and `four_pt` routines:

```sh
CUDA_VISIBLE_DEVICES= GPUWM_NO_LOCAL_GPU=1 nice -n 10 python tools/sfire_static_wps_oracle/build_oracle.py --work-dir work/sfire-static-wps-oracle
```

The builder verifies the pinned source SHA-256 before compilation. It requires
`gfortran`, uses scalar REAL arithmetic without FMA contraction, and writes
`native-fixture.json` with source and compiler provenance and 12 point controls.
The terminal interpolation-sequence stub returns the requested missing value.
The native interpolation routines are copied without changes.

The Rust checks are in `static-fields::sfire::tests`. They grade nearest and
four-point samples after actual GeoTIFF decode, including source edge centers
and nearest ties. The fixture is an interpolation control. It does not qualify
full geogrid projection, optional mask interpolation or atmospheric
initialization.
