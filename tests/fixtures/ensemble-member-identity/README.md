# Ensemble member-identity proof corpus

Real production bytes from the two ensembles whose 2026-08-17 00Z cycles
were byte-measured for the member capability: NCEP GEFS v12
(`noaa-gefs-pds`, `gefs.20260817/00/atmos/pgrb2ap5/`) and NCEP AIGEFS /
Project EAGLE (`noaa-nws-graphcastgfs-pds`,
`EAGLE_ensemble/aigefs.20260817/00/`), retrieved 2026-08-17.

The `aigefs.20260120` and `aigefs.20260410` fixtures are envelopes of the
same mirror's re-encoded archive, retrieved 2026-09-28 from
`EAGLE_ensemble/aigefs.20260120/00/` and `EAGLE_ensemble/aigefs.20260410/00/`
(the mem000 sfc f000 source files were 8,893,895 bytes, SHA-256
`7f741f92bf6c03e839b200dd39b8799c6485140966bd15b86cea5c66cbc5d3b7`, and
8,657,577 bytes, SHA-256
`66eea36fdfd1c97c0677ca46ff688ce4eafbc27154034e5ae39580c52ea8e905`).

The upstream files are too large to commit whole (13-88 MB), so each
fixture is a subset of WHOLE GRIB2 envelopes copied bit-for-bit by
`tools/slice_grib2_envelopes.py` from the recon-staged originals
(`gpuwm-model-gauntlet-staging/{gefs,aigefs}/` under the staging root
the `GPUWM_MODEL_GAUNTLET_STAGING` environment variable names, default
the developer's home directory; SHA-256 manifests there).  A GRIB2 file is a plain concatenation of
self-delimiting envelopes, so every fixture is a valid GRIB2 file of
unmodified production bytes; nothing was re-encoded.

The directory layout is the DECLARED upstream-relative layout of each
feed -- the layout the member grammars resolve -- which is itself part
of what the corpus proves: the AIGEFS leaf filename is byte-identical
for every member (`aigefs.t00z.sfc.f000.grib2`), so only the
`memNNN` path component carries identity there, while GEFS carries its
member token in the leaf name.

What each fixture pins (read back through the Rust `grib2_inventory`):

| ensemble identity | files |
|---|---|
| GEFS control: PDT 1/11, typeOfEnsembleForecast **1**, perturbationNumber 0, encoded size **30** (control EXCLUDED) | `gec00 f000` (2 envelopes: RH sigma 0.995 + soil moisture), `gec00 f003` (1 PDT-1 + 1 PDT-11 envelope) |
| GEFS perturbed: PDT 1, type 3, perturbationNumber = member ordinal | `gep01`, `gep02` (same 2 envelopes) |
| GEFS mean/spread sharing the member namespace: PDT **2**, derivedForecast **0** / **2**, no perturbationNumber | `geavg`, `gespr` |
| AIGEFS unflagged control: PDT 1/11, type **3** (same as perturbed), perturbationNumber 0, encoded size **31** (control INCLUDED) | `mem000 sfc f000`, `mem000 sfc f006` (2t + the PDT-11 `tp` envelope) |
| AIGEFS perturbed | `mem001`, `mem002` (the 2t envelope) |
| AIGEFS control as NOMADS serves it: PDT 1, type **6**, perturbationNumber 0, encoded size 31 | `aigefs.20260927/12/mem000 sfc f000` (the 2t envelope of the NOMADS operational file, `nomads.ncep.noaa.gov/pub/data/nccf/com/aigefs/prod/`, retrieved 2026-09-27) |
| AIGEFS as the AWS mirror re-encoded it, 2026-01-11 00Z to 2026-04-08 06Z: PDT **0** (no ensemble octets), centre 74 (7 on the mean-sea-level pressure record), subcentre 0, master table **4**, local table **0**, generating process 255 | `aigefs.20260120/00/mem000 sfc f000` (the 100 m wind pair and the mean-sea-level pressure envelope), `aigefs.20260120/00/mem001 sfc f000` (the 100 m wind pair, byte-identical to mem000's: only the path names the member) |
| AIGEFS as the AWS mirror re-encoded it, 2026-04-08 12Z to 2026-04-24 12Z: PDT 1, type 3, perturbationNumber and encoded size 31 intact, centre 7, subcentre 0, master table **4**, local table **0**, generating process **255** | `aigefs.20260410/00/mem000 sfc f000`, `aigefs.20260410/00/mem001 sfc f000` (the 100 m wind pair) |

| File | Bytes | SHA-256 |
|---|---:|---|
| `aigefs.20260120/00/mem000/model/atmos/grib2/aigefs.t00z.sfc.f000.grib2` | 873,098 | `cb7d49b33f0c7b3e06e54e1f28110af5c46a7bd445583d2e31457dbabce51e13` |
| `aigefs.20260120/00/mem001/model/atmos/grib2/aigefs.t00z.sfc.f000.grib2` | 420 | `ee29899c7a653135f523131af753f74b448402e93501ea29612a470d15e57056` |
| `aigefs.20260410/00/mem000/model/atmos/grib2/aigefs.t00z.sfc.f000.grib2` | 426 | `3992495c6e5b5756f35a6237ac3121778f1ea569246d127c4ac683f4e1bac79e` |
| `aigefs.20260410/00/mem001/model/atmos/grib2/aigefs.t00z.sfc.f000.grib2` | 426 | `2612f60c6e4a77d3b70fe18c14a3350fc703be840850ddb56469b9691020b6ef` |
| `aigefs.20260817/00/mem000/model/atmos/grib2/aigefs.t00z.sfc.f000.grib2` | 461,206 | `cea236a8624f62192a09ed9d251d71841b1bd0716eb6481bb23bfd02a8c5f4b8` |
| `aigefs.20260817/00/mem000/model/atmos/grib2/aigefs.t00z.sfc.f006.grib2` | 946,926 | `fe11665fd2ec8a5d6bd536ec00155835fda0b06892e3dea76519683070a3928a` |
| `aigefs.20260817/00/mem001/model/atmos/grib2/aigefs.t00z.sfc.f000.grib2` | 471,197 | `aec38766064fe2b28188e0c4fd342ed8a9ba7ecdf336b674873d8c34198e914d` |
| `aigefs.20260817/00/mem002/model/atmos/grib2/aigefs.t00z.sfc.f000.grib2` | 472,212 | `2954985b580fcbfb0166b3c2e50c60d800df233276a88c029068387f101ef7e3` |
| `aigefs.20260927/12/mem000/model/atmos/grib2/aigefs.t12z.sfc.f000.grib2` | 466,388 | `6548e3fec9511078a6f426ee5b2b2d7038105c897c959bd1875603be54eb1963` |
| `gefs.20260817/00/atmos/pgrb2ap5/geavg.t00z.pgrb2a.0p50.f000` | 63,375 | `f62e01ec0b907315338618cd069b873c7ff62d34ad90e3fdea6092d84fec0c51` |
| `gefs.20260817/00/atmos/pgrb2ap5/gec00.t00z.pgrb2a.0p50.f000` | 58,081 | `5d941e244ea072265010ef5de4c5d065f8d0d312bc12ccdd016927064eb7c9f9` |
| `gefs.20260817/00/atmos/pgrb2ap5/gec00.t00z.pgrb2a.0p50.f003` | 20,120 | `db3237e66c2bc7f1a22ee0e7f696df3e9662a5b06a08520fab5a8ef33de9df5d` |
| `gefs.20260817/00/atmos/pgrb2ap5/gep01.t00z.pgrb2a.0p50.f000` | 60,124 | `85b2e10b41e76a2e9995a1f34e3b18a30b6244794e45bfcc506e7b65b4ebcdae` |
| `gefs.20260817/00/atmos/pgrb2ap5/gep02.t00z.pgrb2a.0p50.f000` | 60,276 | `c260919afab5cd1aac06fd2524fc28e3eab21d867f3d1d1fe332b20f13bdb45c` |
| `gefs.20260817/00/atmos/pgrb2ap5/gespr.t00z.pgrb2a.0p50.f000` | 25,569 | `e4419d1198e8bdcb238aabff23f90b29e0a167a2b524e868cd3f642b5e1f9cb1` |

The deterministic counter-case (bytes with NO ensemble identity octets)
reuses the real GDAS corpus at `tests/fixtures/gdas-process-id/`.
