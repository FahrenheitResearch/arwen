# The model-top default A/B: 100 hPa vs 50 hPa

The evidence behind moving the default `p_top` from 10000 Pa to 5000 Pa
(CHANGELOG "Unreleased", tests/test_ptop_default.py). Two arms of one
real HRRR-forced convective case, identical in every value except
`p_top_requested`; the namelists differ in exactly one line
(configs/ptop_ab_20260824_18z_{control,treatment}.namelist.input).

## The case

- HRRR native-hybrid cycle 2026-08-24 18Z, 18 forecast hours
  (18Z-12Z), one 232 x 184 x 49 Lambert domain at 3 km centred
  44.2N/-101.0W -- the Nebraska Sandhills / South Dakota corridor whose
  SPC-reported hail ran 2143-0130 UTC that evening.
- Physics: thompson-mp8-ysu-mm5-noah-rrtmg-legacy-v1 (the shipped HRRR
  demo profile), certified 49-level eta ladder unchanged in both arms.
- Hardware: RTX 5070 Ti 16 GiB, Linux, gpuwm 2.5.6 editable at the
  lane tip (evaluating commit in each score file).
- Truth: MRMS MergedReflectivityQCComposite_00.50, one frame within
  +/-2 min of each hourly valid time, decoded over the domain box;
  provenance and SHA-256 of every frame in the score files.

## What each arm ran (probes, byte-read from the output)

| | control | treatment |
|---|---|---|
| P_TOP in wrfout | 10000.0 Pa | 5000.0 Pa |
| mean column depth (t0) | 15,874 m | 20,198 m |
| damp_opt=3 sponge base (mean AGL) | 10,874 m | 15,198 m |

## Skill (neighborhood FSS vs MRMS, 30 dBZ, 27 km box, leads 2-18)

| | control | treatment |
|---|---|---|
| primary scalar | 0.0798 | 0.0815 |
| leads won | 5 | 7 |

Five leads tie exactly (both arms carry no scoreable >=30 dBZ echo at
the late overnight leads).

## Structure at the mature hour (00Z, lead 6)

| | control | treatment |
|---|---|---|
| max updraft | 17.4 m/s, plateaued 7-10.5 km, collapse at the sponge base | 22.1 m/s at 8.4 km, natural decay to zero near 14 km |
| comp-refl cells >= 40 dBZ | 191 | 244 |
| cloud-top p90 | 11,762 m AGL | 12,227 m AGL |
| echo-top p90 (18 dBZ) | 10,864 m AGL | 11,425 m AGL |

Across the convective hours (22Z-03Z) the treatment carries more
>=40 dBZ objects during initiation and upscale growth and its cloud-top
p90 runs 200-500 m higher; the control's updraft profile shows the
mechanism directly -- a plateau through the anvil layer with collapse at
its 10.9 km sponge base, which the treatment moves above the storms.

## Cost (4320 steps, identical grids)

| | control | treatment |
|---|---|---|
| forecast wall | 354.9 s | 309.1 s |
| peak sampled VRAM | 3,666 MiB | 3,628 MiB |
| prep+sim peak RSS | 2.37 GiB | 2.31 GiB |

The deeper column costs nothing: the same 49 eta levels span it, and
the 50 hPa lid also shrinks the RRTMGP above-model cap (25 to 13 LW
layers), so the radiation workspace is smaller.

## Files

- `score_{control,treatment}.json` -- the obs-battery score documents
  (registration hash, MRMS provenance and re-hash, per-lead FSS).
- `registration_{control,treatment}.json` -- the single-arm
  registrations the scores were bound to (rule_status
  proposed-unratified; this is an engineering A/B, not a campaign).
- `probes_{control,treatment}.json` -- per-frame positive evidence and
  reductions (P_TOP, column depth, sponge base, w-max profiles,
  reflectivity histograms/objects, echo/cloud tops).
- `ab_summary.json` -- the one-screen arm summary.
