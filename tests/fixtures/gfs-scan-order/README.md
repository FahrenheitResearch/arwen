# GFS scan-order and packing matched pairs

Two matched pairs, each two views of **the same GFS field, from the
same cycle**, published by the two NCEP outlets in the two row orders
and the two packings.  Together they are the certification corpus for
`gfs_grib2_bridge`'s raw-S3 (full-file) admission:

1. the TMP pair proves the **row-order normalization** is lossless --
   the flipped raw-S3 decode is bit-identical to the NOMADS decode,
   cell for cell -- and pins the DRT 5.3 value arithmetic on a field
   with no bitmap;
2. the SOILW pair proves the **complex-packing missing-value
   semantics** -- the missing cells of the bitmap-carrying raw record
   land in identical positions to the NOMADS crop's, and every present
   cell decodes to the identical f64 bit pattern.  This is the
   real-product proof the DRT 5.3 gate stayed shut for until it
   existed.

| File | Source | Field | Grid | Scan | DRT | Bitmap | Bytes |
|---|---|---|---|---|---|---|---|
| `s3-raw-tmp2m-20260729t18z-f000.grib2` | `noaa-gfs-bdp-pds` S3, one message lifted by `.idx` byte range | TMP 2 m | 1440x721 global, `lat1` = 90 (north) | `0x00` north-to-south | 5.3 complex + spatial differencing | none | 523,322 |
| `nomads-crop-20260729t18z-f000.grib2` | NOMADS `filter_gfs_0p25.pl` with `subregion` | TMP sfc/2 m, LAND | 41x41 over 30..40N, 260..270E, `lat1` = 30 (south) | `0x40` south-to-north | 5.0 simple | none | 4,321 |
| `s3-raw-soilw-20260729t18z-f000.grib2` | `noaa-gfs-bdp-pds` S3, one message lifted by `.idx` byte range | SOILW 0-0.1 m | 1440x721 global, `lat1` = 90 (north) | `0x00` north-to-south | 5.3 complex + spatial differencing | yes (water masked) | 355,298 |
| `nomads-crop-soilw-20260729t18z-f000.grib2` | NOMADS `filter_gfs_0p25.pl` with `subregion` | SOILW 0-0.1 m | 41x41 over 30..40N, 260..270E, `lat1` = 30 (south) | `0x40` south-to-north | 5.0 simple | yes (water masked) | 2,277 |

```
sha256  3cbf77deea57a0f1226c9bff5e3a8651b0e3a07152180c6ac89ea1eabb93bb45  s3-raw-tmp2m-20260729t18z-f000.grib2
sha256  1a68737e6fb53256360e208aada933c5f1381b4968813cb519233b04168c6b6c  nomads-crop-20260729t18z-f000.grib2
sha256  c854e429091ee95b57878c9c74d6bf780e157c80617b0c0460d0a59aa781db5a  s3-raw-soilw-20260729t18z-f000.grib2
sha256  fe4397ef34206dddd2a7404d0e9274e4549de15577fb1674f83753e6dedfc72e  nomads-crop-soilw-20260729t18z-f000.grib2
```

## Provenance

Cycle `gfs.20260729/18z`, forecast hour f000.  The TMP pair was
retrieved 2026-07-30; the SOILW pair 2026-08-03, from the same
still-published objects.

The first S3 record was fetched with the vendored fetch backbone, which
is itself the first consumer of these findings:

```bash
tools/rustwx/target/release/rw_fetch fetch --model gfs \
  --date 20260729 --cycle 18 --hours 0 --product pgrb2.0p25 \
  --source aws --mode idx-subset \
  --var-pattern "TMP:2 m above ground" --out DIR
```

the SOILW record by the equivalent single `.idx` byte range
(`565:398535237` .. `566:398890534` of the f000 object).  The NOMADS
counterparts are the grib-filter CGI over the same object:

```
https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl
  ?file=gfs.t18z.pgrb2.0p25.f000&subregion=
  &leftlon=260&rightlon=270&toplat=40&bottomlat=30
  &lev_2_m_above_ground=on&var_TMP=on&dir=/gfs.20260729/18/atmos
```

(and `lev_0-0.1_m_below_ground=on&var_SOILW=on` for the soil pair).

The first S3 file keeps only the first message of that fetch (TMP at 2
m above ground); the LAND record it also carried was dropped to keep
the fixture small.  The TMP crop keeps all three messages the CGI
returned (TMP:surface, TMP:2 m, LAND:surface) because it is 4 KB either
way.  The box deliberately spans the Gulf coast so the SOILW pair holds
both land and bitmap-masked water cells.

## Three things these files establish

1. **Row order differs by publisher, not by content.** Flip the raw-S3
   rows and every one of the 1,681 overlapping cells matches the NOMADS
   crop as an exact f64 bit pattern -- no tolerance, no epsilon.
2. **An uncropped grib-filter request is *not* the re-encoded form.**
   Asked for the same fields without `subregion`, the CGI returns the
   raw byte ranges: `scan 0x00`, DRT 5.3, packed lengths identical to
   S3's. Only the `subregion` crop re-encodes to the south-to-north,
   simple-packed form the original certification corpus was built from.
3. **DRT 5.3 admission stands on the SOILW pair.** The raw objects are
   complex-packed with spatial differencing, and `gfs_grib2_bridge`
   refused them until their missing-value semantics had a real-product
   proof.  The SOILW pair is that proof: NCEP carries GFS missing cells
   in the bitmap (Section 5 missing-value management is 0), the raw
   record's missing cells land exactly where the crop's do, and every
   present cell decodes bit-identically
   (`the_raw_53_bitmap_decode_matches_the_nomads_crop_cell_for_cell`).
   Everything outside the proven envelope -- DRT 5.2, embedded
   missing-value management, first-order differencing, any third scan
   mode -- still refuses by name.

## Licence

NCEP GFS output is a work of the U.S. Government and is in the public
domain.
