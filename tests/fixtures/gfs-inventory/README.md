# A real GFS `pgrb2.0p25` inventory

`gfs.t12z.pgrb2.0p25.f000.idx` is the unedited `.idx` NOAA publishes
beside the object

    https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.20260730/12/atmos/gfs.t12z.pgrb2.0p25.f000

captured once on 2026-07-30 from the AWS Open Data mirror (**not** from
NOMADS — this is the same index `gpuwm.fetch.gfs_derived_record_bar`
already reads to derive the record bar, and S3 is not the rate-limited
service the NOMADS governor exists for).

    696 records, 31,816 bytes
    sha256 343c68b1af3c85b1594a05d05dc5a21c3355a09cb24241ba0d9221a8201d719b

## Why it is here

The GFS fetch route used to hardcode a 21-level pressure ladder topping
out at 100 hPa, which silently capped the model top any ArWen GFS run
could ask for at 10000 Pa.  Deciding which levels a requested `p_top`
needs is only honest against what the product *actually publishes*, and
"what it publishes" is a fact about NCEP's output, not a constant anyone
should be inventing in a source file.

This file is that fact, frozen. Every level-selection gate in
`tests/test_gfs_levels.py` runs against it offline, so the selection
rule is tested against a real inventory without a single live request.

## What it says

All five fields the 3-D selection takes — `HGT`, `TMP`, `RH`, `UGRD`,
`VGRD` — are published on the same 41 isobaric levels:

    0.01  0.02  0.04  0.07  0.1  0.2  0.4  0.7  1  2  3  5  7  10
    15  20  30  40  50  70  100  150  200  250  300  350  400  450
    500  550  600  650  700  750  800  850  900  925  950  975  1000  (hPa)

The certified ArWen ladder is exactly the last 21 of those (100 → 1000
hPa), so the deeper tops are reached by extending the same ladder
upward, not by replacing it.  The deepest top the product can serve is
0.01 hPa = 1 Pa.

## Refreshing it

Only if NCEP changes the published level set. Re-capture verbatim from
the same S3 path (any cycle), update the digest above, and expect
`test_gfs_levels.py::test_the_certified_ladder_matches_the_captured_inventory`
to say so if the certified constants and the file have drifted apart.
