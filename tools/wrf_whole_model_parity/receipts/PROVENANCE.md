# Run of record (2026-10-12)

`summary.txt` is `summarize.py` over the comparison files of this run.

- Case: the NCAR CONUS 12 km benchmark real.exe pair (WRF v4.4 benchmark set),
  start 2019-11-26 12Z, 72 s step, 15 steps, history every step.
  wrfinput_d01 sha256 e8b86acc703f64c91c02bd5afbcc3ffc514ead80ab6e5dde4af4a43e6344d953,
  wrfbdy_d01 sha256 159365527b4d52beb1ccfcdf4a7d44a58e383d6ad17f8f7687e758234a680c1d.
- Referee and yardstick: stock WRF v4.6.1, serial gfortran 13.3, built scalar and
  vectorised as the README states (glibc 2.39).
- WOOF: one RTX PRO 6000 Blackwell card per job.  `woof-base-default` is the
  published 2.8.9 (cd336e62d6) in default arithmetic; `woof-tip-strict` is
  golden/parity 62dbd8f2e under `GPUWM_WRF_EXACT=1`, which already holds all
  eight scheme ports.  Later commits on the branch change pins, wording,
  verification tooling and Kessler only where saturation vapour pressure
  reaches 0.99 p, which this case's model top never reaches.

## What it says

Every arm is outside WRF's build spread from the first frame, and so is the
cumulus-off control (`cu0`), whose only schemes are YSU, the revised MM5
surface layer, Noah, WSM6, RRTM longwave and Dudhia shortwave.  At frame 0,
before any physics step, `cu0` already differs from WRF in P, GLW and SH2O.
The distance is the same in 2.8.9 default and in the branch's strict build
(`cu0` max ratio 157.01 against 157.0; Kain-Fritsch default root 44.71 against
44.55), so it comes from the whole-model line these runs sit on, not from the
eight schemes.  Whole-model identity is not claimed here.  It needs this
comparison rerun on a line that also carries the strict cold-start and reader
fixes of the exact-chain line, where the strict A071 hour matches stock WRF.

The schemes' standing checks are the column oracles in the battery, graded
word for word against the unmodified WRF v4.6.1 Fortran in both builds.
