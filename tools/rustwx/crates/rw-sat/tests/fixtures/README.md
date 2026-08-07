# Cloud-product test fixtures

Real ABI L2 granules from the public `noaa-goes19` bucket, scan
2026-08-04 (day 216) 18:01 UTC. They are **not** committed; fetch them
once with `fetch_fixtures.sh` (or `fetch_fixtures.ps1`) into this
directory, or set `RW_SAT_CLOUD_FIXTURE_DIR` to wherever they live.
Without them the `tests/cloud_products.rs` fixture tests print a loud
SKIPPING line and pass vacuously; the filename-parse test always runs.

| granule | bytes | sha256 |
| --- | --- | --- |
| `OR_ABI-L2-ACHAM1-M6_G19_s20262161801249_e20262161801336_c20262161801594.nc` | 201987 | `433cbe0d5b1d454a895e33bac1f7340909801863f7504c2d12d5c5dd92a973c3` |
| `OR_ABI-L2-ACTPC-M6_G19_s20262161801170_e20262161803545_c20262161804390.nc` | 703425 | `73a1d853a57ec67339ee7c5bd0f726b21e0c19b3db1965a4861d431154c0fbb3` |
| `OR_ABI-L2-CODC-M6_G19_s20262161801170_e20262161803545_c20262161805324.nc` | 3666798 | `cb39c3b459079b94dab231a3a3b72d71c8aa054c6454723613495b2c83660c52` |
| `OR_ABI-L2-CPSC-M6_G19_s20262161801170_e20262161803545_c20262161805325.nc` | 3960766 | `a5edb9293d49d3e032f5f41cc1b8b73dfe3c12135b6ac1ac87847016fd3b5d99` |

Why these four: the ACHAM1 mesoscale granule is the small
decode-and-DQF fixture; the CODC/CPSC/ACTPC trio shares one CONUS scan
start (`s20262161801170`) and one 2 km fixed grid, which is what the CWP
derivation requires (COD publishes no mesoscale sector, so the CWP trio
is necessarily CONUS; the tests read a 160x160 window to stay fast).

Every expected number in `tests/cloud_products.rs` was computed
independently from these granules' raw integers with Python
(netCDF4/numpy), replicating the CF decode (scale/offset, `_FillValue`,
`valid_range`) and the documented DQF rules.
