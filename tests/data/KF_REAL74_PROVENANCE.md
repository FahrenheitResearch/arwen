# KF real74 12Z column provenance

`kf_real74_12z_columns.npz` was extracted by
`tools/extract_kf_real74_columns.py` from the unmodified Phase-3
`prepare_phase3_case()` initial state for 1974-04-03 12:00 UTC. The source
inputs are the local reference bundle's ERA5 analysis, WPS d01 grid, static
fields, 49-level hybrid coordinate, and Phase-3 real-data initialization.
The gpuwm extraction base is commit
`2a90047de259fe8bac1d7741d0bc001e6fa80660`.

The selected d01 mass points are `(j=53, i=113)` for the warm-sector gate and
`(j=194, i=5)` for the stable northern gate. The file records their latitude
and longitude along with the exact FP32 scheme inputs. Selection used a
50,000-column diagnostic scan to locate a strongly unstable warm-sector
column and a northern stable control; activation/no-op status is established
by the independently tested WRF transcription, not asserted as provenance.

SHA-256 (generated 2026-07-15):

    72b7e16140cb317a9a28709c32feccfeffa68f8d8e9303e44abc520448ebf4e9  kf_real74_12z_columns.npz
