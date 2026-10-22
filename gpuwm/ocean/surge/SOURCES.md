# Source and license record

Checked at the source on 2026-10-20. Solver and bridge additions are original
Apache-2.0 project code. No ADCIRC, SCHISM, GeoClaw or ARCO implementation is
copied or linked. GeoClaw was checked as an eligible alternative:

* https://raw.githubusercontent.com/clawpack/geoclaw/master/LICENSE:
  "BSD 3-Clause License"; "Redistribution and use in source and binary forms,
  with or without modification, are permitted".
* NOAA NCEI ETOPO 2022, DOI 10.25921/fd45-gt74:
  https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.ngdc.mgg.dem:etopo_2022
  names "Creative Commons Zero 1.0 Universal Public Domain Dedication (CC0-1.0)."
  Bathymetry in the example is NOAA ETOPO 2022 surface, 30 arc-second subset,
  EGM2008 geoid heights. Attribution: NOAA National Centers for Environmental
  Information (2022), ETOPO 2022 15 Arc-Second Global Relief Model, accessed
  2026-10-20. No GEBCO data are used.
* NOAA CO-OPS observations, astronomical predictions and station metadata:
  https://tidesandcurrents.noaa.gov/disclaimers.html says
  "The information on government servers are in the public domain, unless
  specifically annotated otherwise". Attribution: NOAA/NOS/CO-OPS.
* ERA5: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview
  displays "CC-BY licence". Data DOI 10.24381/cds.adbb2d47. The research example
  explicitly selects CC-BY-4.0 inputs; they are not bundled or a default.
  Attribution: contains modified Copernicus Climate Change Service information
  (2022), processed by ECMWF. Neither the European Commission nor ECMWF is
  responsible for this use. Public archive documentation:
  https://github.com/google-research/arco-era5 (data access, no code copied).

Native readers and writers are seeded from tools/rustwx/vendor/netcrust
(MIT OR Apache-2.0, with its vendored license notices) and
 tools/rustwx/crates/netcdf-writer (MIT workspace license). Cargo's locked
dependency graph also includes Zlib-licensed foldhash and Unicode-3.0 data
under unicode-ident alongside MIT/Apache-2.0 and BSD alternatives. These are
permissive commercial-use terms. The source-notice audit is retained in the
run evidence; their existing upstream notices must accompany
binary redistribution according to the project packaging process. No third
party source is newly vendored here. NumPy is used only for binary memory
views (BSD-3-Clause); the existing Rust renderer supplies map assets and their
project notices.

The incremental device ABI uses the existing CuPy CUDA driver dependency
(MIT). Source checked 2026-10-20:
https://raw.githubusercontent.com/cupy/cupy/main/LICENSE . It grants
"Permission is hereby granted, free of charge" and permits redistribution
and commercial use with its copyright/permission notice. CuPy's installed
distribution retains that notice; no CuPy source is copied into this solver.
