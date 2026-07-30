# rustwx-products

Reusable workflow/product orchestration helpers for `rustwx`.

Current scope is intentionally conservative:

- proof cache helpers shared by CLI binaries
- shared HRRR fetch/decode/cache helpers for full `wrfsfc` / `wrfprs` family-file ingest
- shared HRRR direct/native plot batching from recipe metadata + selector-backed extraction
- shared HRRR derived plot batching from one fetched/decoded timestep using
  `rustwx-calc` wrappers
- shared HRRR windowed batching for APCP-based QPF windows, native 2-5 km
  UH run-max windows, native 10 m wind max swaths, and extended-cycle 2 m
  temperature snapshot max/min/range windows, with explicit blockers when GRIB
  time-window metadata is not available cleanly enough yet
- supported-products inventory/reporting helpers that summarize current direct,
  derived, heavy, and windowed product status, including typed maturity/flag
  metadata for operational vs experimental vs proof-oriented and proxy cases
- shared projection/basemap assembly for cropped panel products
- shared Weather two-by-four panel rendering with header text
- a typed HRRR batch request/runner that generates multiple products from one
  fetched/decoded timestep

This crate exists so proof binaries stop owning fetch/decode/prep/render
assembly directly. Product-specific science still lives in `rustwx-calc`.
