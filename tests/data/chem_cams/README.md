# CAMS-shaped GRIB2 test fixture (synthetic values, ECMWF encoding)

Written by `tools/chem_cams_fixtures/make_synthetic_cams.py` with ecCodes 2.48.0,
ECMWF's GRIB library, for the parameters the `cams-global` source row selects:
ozone, NO2, CO and SO2 as WMO chemical-constituent records (discipline 0,
category 20, parameter 2, product definition template 4.40, constituentType
0/5/4/8) and specific humidity (0/1/0), on the 137 IFS hybrid levels (level type
105) with the L137 half-level A/B coefficients in every Section 4, at steps 0 and
3 h of the 2025-07-30 00Z cycle; plus single-level surface pressure (0/3/0). The
grid is a 3 x 3 regular 0.4 degree box, 31.2-30.4 N, 96.0-95.2 W, scanning north
to south.

The VALUES are smooth synthetic profiles, not a forecast. The `.eccodes-meta.json`
and `.eccodes-values.npy` files are ecCodes' own decode of every record (file
order), the independent oracle the Rust decode is compared with bit for bit
(`tests/test_grib2_stack.py`). `ifs_l137_half_levels.json` is ECMWF's published
L137 table. `SHA256SUMS` pins every file.

Command:

    python tools/chem_cams_fixtures/make_synthetic_cams.py tests/data/chem_cams \
        --area 31.2,-96.0,30.4,-95.2 --cycle 2025-07-30T00 --leads 0,3 \
        --params ozone,nitrogen_dioxide,carbon_monoxide,sulphur_dioxide,specific_humidity
