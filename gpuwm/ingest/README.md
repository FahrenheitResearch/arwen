# Real-data ingest inputs

The retained April 1974 ERA5 GRIB omits invariant geopotential/orography,
PMSL, and sea ice.  For this reference case, real initialization reads
`SOILHGT` from the bundle's `met_em` file as an input-data exception.  It
uses surface pressure directly, so PMSL is not required, and initializes
`XICE=0` (the mid-latitude April domain uses `LANDSEA` for land/water).

Fresh cases should request ERA5 invariant geopotential together with the
time-varying fields (for example through `cdsapi`) and horizontally
interpolate it to supply `source_orography`; they must not depend on a
reference `met_em` file.
