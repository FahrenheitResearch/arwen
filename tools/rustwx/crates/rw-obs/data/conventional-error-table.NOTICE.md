# Conventional observation error table

`conventional-error-table.r3dv` (492,800 bytes, SHA-256
`9c5ddfd32751c93fb5e8689059f0e68d74360e5060b55d33035da90756f8b4fe`) is
NCEP's regional conventional error table exactly as NCEP Central
Operations publishes it in the fix files of the operational rapid-refresh
analysis, version 4.1.21:
`https://www.nco.ncep.noaa.gov/pmb/codes/nwprod/hrrr.v4.1.21/fix/conus/hrrr_nam_errtable.r3dv`
(Last-Modified 2025-07-15 17:57:09 GMT).  The operational analysis script
copies it to `errtable` in GSI's run directory
(`scripts/conus/exhrrr_analysis.sh:220,233` of the same tag), and GSI then
takes every conventional error from it (`oberrflg` is reset true when the
file is present: `converr.f90`, `converr_read`; a 2026-10-03 12Z run of
that GSI printed `OBERRFLG = T` and `NJQC = F`).  It is a work of the
United States government and in the public domain.

The bytes are unchanged, so the hash above is the hash of NCEP's file.
`rw_obs::errtable` reads it with GSI's own format (`(1x,i3)` for a report
type, then 33 lines of `(1x,6e12.5)`: pressure in hPa, then the error of
temperature in K, of humidity in tenths of saturation, of wind in m/s, of
surface pressure in hPa and of precipitable water in mm; `1e9` where the
type carries no error for that variable).
