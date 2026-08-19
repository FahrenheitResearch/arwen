# rw-glm test fixtures

## `OR_GLM-L2-LCFA_G19_s20261620805000_e20261620805200_c20261620805214.nc`

A single, **unmodified** GOES-19 GLM Level-2 Lightning Cluster-Filter Algorithm
(LCFA) granule, used to pin the `decode_granule` mapping in
`tests/granule_decode.rs` against real product bytes.

| | |
|---|---|
| Bucket | `s3://noaa-goes19` (NOAA GOES on AWS, public, anonymous access) |
| Key | `GLM-L2-LCFA/2026/162/08/OR_GLM-L2-LCFA_G19_s20261620805000_e20261620805200_c20261620805214.nc` |
| Size | 245,564 bytes |
| Coverage | 2026-06-11 08:05:00.0Z – 08:05:20.0Z (20 s granule, DOY 162) |
| Platform | G19 (GOES-East, orbital slot GOES-East) |
| Flashes | 107 (`number_of_flashes` dimension) |
| Fetched | 2026-06-11 via anonymous S3 `ListObjectsV2` + GET |

The file is committed **byte-for-byte as published by NOAA** — no re-encoding,
no truncation, no variable edits. NOAA GOES open data is in the public domain
(U.S. Government work); see <https://registry.opendata.aws/noaa-goes/>.

### Why this granule

It is a recent, modestly-sized (~240 KB) G19 granule with a healthy mix of
flashes spanning the GOES-East disk, a non-trivial number of `quality_flag != 0`
(degraded) flashes (3), and `flash_energy` values in the ~`1e-15`..`1e-12` J
range that exercise the f64 scale/offset precision path in `decode_granule`.

The expected decode numbers in `tests/granule_decode.rs` were derived directly
from this granule's raw variables and attributes.
