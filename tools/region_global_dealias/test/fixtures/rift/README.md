# RIFT real-case crops

These compact velocity-only crops lock the visually reviewed local branch
corrections that motivated RIFT-VDA. They are deliberately small enough to run
in every native and WASM test job while retaining the exact fold decision made
in the full sweep.

All arrays are little-endian `f32`. `observed.f32` and `expected.f32` are flat
row-major velocity grids in m/s; `azimuth.f32` and `nyquist.f32` contain one
value per ray.

## El Reno 2013

- Radar/time: KTLX, 2013-05-31 23:23:45Z, 0.9-degree velocity cut
- Public archive object: `KTLX20130531_232345_V06.gz`
- Archive SHA-256: `9C4A7DB6301AC5F72BC5E8F3E3BA99119EF10306550FF59060206D24BEDEAD66`
- Decoded Level-II SHA-256: `C999F140EF61855CD0C0F27348AB3D8D451A1F0D1C458459150F5C2D00BEEEFF`
- Full-sweep crop: rays `[214,239)`, gates `[204,241)`, shape 25 x 37
- Expected refinement: a reviewed 137-gate local branch

## Tuscaloosa 2011

- Radar/time: KBMX, 2011-04-27 22:38:05Z, 0.9-degree velocity cut
- Public archive object: `KBMX20110427_223805_V03.gz`
- Archive SHA-256: `7094F57C3FC01CEC394ADA4A77B45CF690AB4F0C3DB17EB8D14EC14ED019B176`
- Decoded Level-II SHA-256: `DA384CDD42D74980367047CA8C2151F3F49120A3D417FB64312D4771A1EB1109`
- Full-sweep crop: rays `[238,272)`, gates `[139,188)`, shape 34 x 49
- Expected refinement: a reviewed 78-gate local branch

The expected fields are deterministic regression fixtures for the opt-in RIFT
path. The legacy region-global output remains covered separately.
