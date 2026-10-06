# Native writer fixture

`native-writer.grib2` contains 17 deterministic messages on 7 by 5 grids.
Messages 1 through 12 cover latitude/longitude, Mercator, polar stereographic,
and Lambert conformal grids, each with simple packing, first spatial
differences, and second spatial differences. Values are exactly
`270 + index * 0.125 K`, with index 6 missing. Message 13 is constant,
message 14 is entirely missing, and messages 15 through 17 are one-hour
accumulation, maximum, and minimum windows ending six hours after initialization.
Every grid specifies the WRF spherical radius of 6,370,000 metres.

The checked-in fixture is decoded by the Rust unit test. Independent tools
also check every value when `GRIB2_WRITER_WGRIB2` names a wgrib2 executable and
`GRIB2_WRITER_ECCODES_DATA` names the ecCodes `grib_get_data` executable.
Run the wx-core `grib2` tests with these environment variables set.
To regenerate the fixture, set `GRIB2_WRITER_GOLDEN_DIR` to an output folder
and run the `external_reader_fixture` test.

Encoding follows the published NCEP descriptions of
[grid templates](https://www.nco.ncep.noaa.gov/pmb/docs/grib2/grib2_doc/grib2_table3-1.shtml),
[statistical intervals](https://www.nco.ncep.noaa.gov/pmb/docs/grib2/grib2_doc/grib2_temp4-8.shtml),
and [complex spatial packing](https://www.nco.ncep.noaa.gov/pmb/docs/grib2/grib2_doc/grib2_temp5-3.shtml).
