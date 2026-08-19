# Ensemble member-identity proof corpus

Real production bytes from the two ensembles whose 2026-08-17 00Z cycles
were byte-measured for the member capability: NCEP GEFS v12
(`noaa-gefs-pds`, `gefs.20260817/00/atmos/pgrb2ap5/`) and NCEP AIGEFS /
Project EAGLE (`noaa-nws-graphcastgfs-pds`,
`EAGLE_ensemble/aigefs.20260817/00/`), retrieved 2026-08-17.

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

| File | Bytes | SHA-256 |
|---|---:|---|
| `aigefs.20260817/00/mem000/model/atmos/grib2/aigefs.t00z.sfc.f000.grib2` | 461,206 | `cea236a8624f62192a09ed9d251d71841b1bd0716eb6481bb23bfd02a8c5f4b8` |
| `aigefs.20260817/00/mem000/model/atmos/grib2/aigefs.t00z.sfc.f006.grib2` | 946,926 | `fe11665fd2ec8a5d6bd536ec00155835fda0b06892e3dea76519683070a3928a` |
| `aigefs.20260817/00/mem001/model/atmos/grib2/aigefs.t00z.sfc.f000.grib2` | 471,197 | `aec38766064fe2b28188e0c4fd342ed8a9ba7ecdf336b674873d8c34198e914d` |
| `aigefs.20260817/00/mem002/model/atmos/grib2/aigefs.t00z.sfc.f000.grib2` | 472,212 | `2954985b580fcbfb0166b3c2e50c60d800df233276a88c029068387f101ef7e3` |
| `gefs.20260817/00/atmos/pgrb2ap5/geavg.t00z.pgrb2a.0p50.f000` | 63,375 | `f62e01ec0b907315338618cd069b873c7ff62d34ad90e3fdea6092d84fec0c51` |
| `gefs.20260817/00/atmos/pgrb2ap5/gec00.t00z.pgrb2a.0p50.f000` | 58,081 | `5d941e244ea072265010ef5de4c5d065f8d0d312bc12ccdd016927064eb7c9f9` |
| `gefs.20260817/00/atmos/pgrb2ap5/gec00.t00z.pgrb2a.0p50.f003` | 20,120 | `db3237e66c2bc7f1a22ee0e7f696df3e9662a5b06a08520fab5a8ef33de9df5d` |
| `gefs.20260817/00/atmos/pgrb2ap5/gep01.t00z.pgrb2a.0p50.f000` | 60,124 | `85b2e10b41e76a2e9995a1f34e3b18a30b6244794e45bfcc506e7b65b4ebcdae` |
| `gefs.20260817/00/atmos/pgrb2ap5/gep02.t00z.pgrb2a.0p50.f000` | 60,276 | `c260919afab5cd1aac06fd2524fc28e3eab21d867f3d1d1fe332b20f13bdb45c` |
| `gefs.20260817/00/atmos/pgrb2ap5/gespr.t00z.pgrb2a.0p50.f000` | 25,569 | `e4419d1198e8bdcb238aabff23f90b29e0a167a2b524e868cd3f642b5e1f9cb1` |

The deterministic counter-case (bytes with NO ensemble identity octets)
reuses the real GDAS corpus at `tests/fixtures/gdas-process-id/`.
