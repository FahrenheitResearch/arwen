`era5-eda-ten-t2m.grib` is a real 1300-byte CDS response for ERA5 EDA
2 m temperature at 2013-05-31 18:00 UTC, area [36.5, -98.5, 35.5, -97.5]
in CDS north/west/south/east order. It contains all ten encoded members on
their 0.5-degree grid, ECMWF local definition 36. Retrieved 2026-09-07.
This tiny single-field sample verifies member identity and unchanged bytes;
it is not a complete forecast initialization dataset.

`era5-eda-ten-sst.grib` is a real 1600-byte sea-surface-temperature subset
from the same date/time/area, acquired as part of a full forcing proof.
Its ten members use local definition 17. The current ecCodes source
`definitions/grib1/local.98.17.def` defines perturbationNumber and
numberOfForecastsInEnsemble at PDS octets 50 and 51; the older rendered
ECMWF table still labels those slots zero. No GRIB message was rewritten.

The complete local proof acquired 534,120 bytes for 18Z/21Z, all 37 pressure
levels and the 20 requested surface fields, and selected member 7 into
53,412 unchanged bytes (410 messages). Preparation and forecasts were not run.
