# GFS scan-order matched pair

Two views of **the same GFS field, from the same cycle**, published by
the two NCEP outlets in the two row orders.  They exist to prove that
`gfs_grib2_bridge`'s row-order normalization is lossless: the flipped
raw-S3 decode is bit-identical to the NOMADS decode, cell for cell.

| File | Source | Grid | Scan | DRT | Bytes |
|---|---|---|---|---|---|
| `s3-raw-tmp2m-20260729t18z-f000.grib2` | `noaa-gfs-bdp-pds` S3, one message lifted by `.idx` byte range | 1440x721 global, `lat1` = 90 (north) | `0x00` north-to-south | 5.3 complex + spatial differencing | 523,322 |
| `nomads-crop-20260729t18z-f000.grib2` | NOMADS `filter_gfs_0p25.pl` with `subregion` | 41x41 over 30..40N, 260..270E, `lat1` = 30 (south) | `0x40` south-to-north | 5.0 simple | 4,321 |

```
sha256  3cbf77deea57a0f1226c9bff5e3a8651b0e3a07152180c6ac89ea1eabb93bb45  s3-raw-tmp2m-20260729t18z-f000.grib2
sha256  1a68737e6fb53256360e208aada933c5f1381b4968813cb519233b04168c6b6c  nomads-crop-20260729t18z-f000.grib2
```

## Provenance

Cycle `gfs.20260729/18z`, forecast hour f000, retrieved 2026-07-30.

The S3 record was fetched with the vendored fetch backbone, which is
itself the first consumer of these findings:

```bash
tools/rustwx/target/release/rw_fetch fetch --model gfs \
  --date 20260729 --cycle 18 --hours 0 --product pgrb2.0p25 \
  --source aws --mode idx-subset \
  --var-pattern "TMP:2 m above ground" --out DIR
```

The NOMADS counterpart is the grib-filter CGI over the same object:

```
https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl
  ?file=gfs.t18z.pgrb2.0p25.f000&subregion=
  &leftlon=260&rightlon=270&toplat=40&bottomlat=30
  &lev_2_m_above_ground=on&var_TMP=on&dir=/gfs.20260729/18/atmos
```

The S3 file here keeps only the first message of that fetch (TMP at 2 m
above ground); the LAND record it also carried was dropped to keep the
fixture small.  The crop keeps all three messages the CGI returned
(TMP:surface, TMP:2 m, LAND:surface) because it is 4 KB either way.

## Three things these files establish

1. **Row order differs by publisher, not by content.** Flip the raw-S3
   rows and every one of the 1,681 overlapping cells matches the NOMADS
   crop as an exact f64 bit pattern -- no tolerance, no epsilon.
2. **An uncropped grib-filter request is *not* the certified form.**
   Asked for the same fields without `subregion`, the CGI returns the
   raw byte ranges: `scan 0x00`, DRT 5.3, packed lengths identical to
   S3's. Only the `subregion` crop re-encodes to the south-to-north,
   simple-packed form the certified corpus was built from.
3. **Scan order was not the only gate on raw S3.** These objects are
   DRT 5.3, and `gfs_grib2_bridge` admits only 5.0 -- deliberately,
   because complex-packing missing-value semantics have no independent
   real-product proof here yet. Accepting the row order does not open
   that gate, and `a_flipped_raw_s3_decode_is_bit_identical_to_the_nomads_crop`
   asserts it stays shut.

## Licence

NCEP GFS output is a work of the U.S. Government and is in the public
domain.
