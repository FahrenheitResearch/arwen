# Synthetic CAMS-shaped GRIB2 fixtures (test and proof only)

`make_synthetic_cams.py` asks ecCodes (ECMWF's GRIB library, the encoder behind
real CAMS files) to write records of exactly the parameters the `cams-global`
and `cams-oxidants` source rows select, on the IFS L137 hybrid levels with their
coefficients in Section 4, so the Rust decode and the boundary remap are proven
against ECMWF's own encoding of the chemical-constituent template 4.40 and the
ECMWF-local 192/210 and 192/217 records.

The VALUES are synthetic plausible profiles, never a forecast; every output
file name starts with `SYNTHETIC`. ecCodes is a fixture-building dependency
only (`pip install eccodes` in a scratch environment); the engine never imports
it.

`ifs_l137_half_levels.json`: `[n, a_Pa, b]` for the 138 half levels of the
IFS L137 grid, transcribed from ECMWF's published table
(https://confluence.ecmwf.int/display/UDOC/L137+model+level+definitions, read
2026-09-30). Real CAMS records carry these in their own Section 4, and the
decoder reads them from each record; this file exists only so the fixture
writer can put them there.
