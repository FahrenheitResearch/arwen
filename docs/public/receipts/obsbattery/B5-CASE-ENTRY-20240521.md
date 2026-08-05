# B5 — case entry receipt: the shakedown case, 2024-05-21

2026-08-04, against `0279f0f2`. The battery spec's section 1.1 makes a
data-availability receipt the condition of a case entering the frozen set, and
names what it must answer: cycle files present, MRMS file count for the window,
station count, Stage-IV presence, masked-interior fraction. B4's section 5 item
3 adds one more — the frozen-soil exposure, "as a checked fact, not discovered
on the node". All six are answered below, on real archive bytes.

This receipt admits the case for the **shakedown**. The case *set* is
registered -- owner question Q1 was answered on 2026-08-04 under standing
delegation, with the spec's seven rows standing unchanged and this case as the
shakedown -- but the section 9.2 **freeze** is signed at B5 exit, after the
shakedown has been scored. `B5-REGISTRATION-FREEZE.md` carries the version
series and names what the freeze is still waiting for.

No forecast was run. No skill claim is made here for any model.

---

## 1. The case

| | |
|---|---|
| case id | `case-20240521` |
| config | `configs/battery/case_20240521.toml` |
| menu row | spec section 1.2, **B-04** — daytime cyclic supercell, southwest/central Iowa |
| init | **2024-05-21 12:00 UTC**, the HRRR analysis cycle nearest 12 UTC (spec section 1.3), pre-convective for this domain by roughly eight hours |
| length | 24 h; scored F02-F18, F18-F24 reported, F00-F02 reported and never scored |
| domain | 480 x 400 x 49 = 9.408 M cells, dx 3 km, dt 15 s, Lambert centred 41.30 N / 94.50 W |
| box | 35.542-46.749 N, 103.968-85.032 W |
| suite | Thompson mp8 / YSU / classic MM5 / Noah / cumulus off, `ra_rrtmg_variant = "rrtmg_legacy"` |

**Why this case is the shakedown.** It is the menu's warm-season, discrete-mode,
daytime row: convection is deep and widespread inside the box for most of the
scored window, so every registered FSS threshold has signal to measure, and the
surface network under it is dense. It is also the one case day every upstream
lane already touched with real bytes — B1 settled the MRMS sentinel ruling, the
Stage-IV edition and template questions and the ASOS station freeze against
2024-05-21 objects — so a shakedown here re-uses proven archive routes and any
surprise is the battery's rather than the archive's.

## 2. Route status: ADMITTED, zero refusing gates

`python tools/battery_route_preflight.py --experiment-config configs/battery/case_20240521.toml`

```
  [ADMITS  ] experiment.load                1 domain(s), 24 h from 2024-05-21 12:00:00
  [ADMITS  ] run_config.validate.d01
  [ADMITS  ] hrrr.coverage
  [ADMITS  ] hrrr.route_inputs.physics
  [ADMITS  ] hrrr.root_preparation.profile  prepare with --physics-profile thompson-mp8-ysu-mm5-noah-rrtmg-legacy-v1
  [ADMITS  ] hrrr.hierarchy.slice
  [INFO    ] arwen.device_memory            18.43 GiB peak envelope for 9,408,000 cells (9.144 GiB pool request)
  [INFO    ] arwen.wall_clock_projection    1.4838 s/sim-min -> 0.594 h wall, 16.1 GiB of wrfout
  [INFO    ] hrrr.forcing_inventory         a root prepared for this run seals f00..f24 (25 HRRR analyses)
  status: ADMITTED
```

The geometry gates answer for this case's own centre, not the shape file's:
`hrrr.coverage` admits the Iowa box with its donor margin. Every other line is
byte-identical to the registered composition's, which is the point — a case
supplies an instant and a centre and changes nothing else.

## 3. Data availability

### 3.1 MRMS

| | |
|---|---|
| objects, whole UTC day | **720 of 720** (no archive gap on this day) |
| bytes | 1.216 GB |
| product | `MergedReflectivityQCComposite_00.50`, `noaa-mrms-pds` |
| frames decoded for the scored window | 19 (F00..F18, hourly) |
| worst valid-time offset | **+42 s** (registered window +/- 240 s) |
| observed fraction, every decoded frame | **1.0000** |
| minimum interior valid fraction over the scored leads | **1.0000** |

The spec fails any case with more than 10 % masked interior. This case has
**0.0000** masked interior at every scored lead, under the `-99` = no-echo
ruling (`WAVE-ERRATA-20260804.md` section 2). With `-99` masked instead, B1
measured the same shaped box at 0.238 observed, i.e. 76 % masked — the ruling is
the difference between this case passing and every case failing.

### 3.2 Stage-IV

Present, all three accumulation windows, over `2024-05-21 00 UTC .. 2024-05-22
12 UTC`: **37 hourly + 7 six-hourly + 2 daily = 46 objects, 12.6 MB**. No 404.

**RFC-seam mask: NOT DONE.** Spec section 2.2 makes the River Forecast Center
seam mask "part of each case's entry receipt". It is not in this receipt: no
RFC boundary geometry exists in the tree, and inventing one here would be an
agent-invented registration parameter. Filed as an open item for the section 9.2
registration; the QPF scores are the only thing it touches, and the primary
scalar is reflectivity.

### 3.3 ASOS

| | |
|---|---|
| stations frozen inside the box | **642** (table sha256 `e10ec778da19bf368f146a92e43f4ab2cc94836ed3b344d15039c0ca3efb1ed1`) |
| stations that decoded through the screen over 25 valid times | **557** |
| reports decoded | 13 816 |
| inside the model grid | 589 of 642 (53 outside) |
| inside the interior mask | 489 |
| distinct interior grid cells | 479 (10 stations share a cell with another) |

The spec's admission rules also drop by land mask and by
`|model terrain - station elevation| <= 100 m`, and both need a prepared root's
statics. So **479 is an upper bound on the frozen station set, not the frozen
set**; the set itself is fixed at registration, after the root exists. The spec
asks for at least 40 stations inside the interior mask, and this case clears
that bar by an order of magnitude before the terrain and land screens have even
run.

## 4. Frozen-soil check (B4 section 2 caveat 1, section 5 item 3)

**Answer: the refusal does not apply to this case, and the reason is stronger
than the season.**

The refusal is real and still in the code at this tip
(`gpuwm/wrf_direct.py:1007-1009`):

```
    if np.any(soilt < 273.15):
        raise StockWrfExportUnsupported(
            "direct WRF export does not yet support frozen-soil SH2O setup")
```

Three facts settle its reachability, all read at `0279f0f2`:

1. **It lives on the synthesis branch only.** `_surface_fields`
   (`gpuwm/wrf_direct.py:948-1009`) takes a first branch when the prepared cache
   declares the canonical `surface/*` group and reads `SH2O` straight out of it;
   the frozen-soil check sits after that branch, on the path that synthesizes
   `SH2O` from `SMOIS` because there is no surface group to read. The export
   manifest says so in its own limitations list, conditionally
   (`gpuwm/wrf_direct.py:2039-2041`: `"warm-soil SH2O initialization only"` is
   inserted **only** when no `surface/` array is present).
2. **The tree route's per-domain artifact carries that group.**
   `gpuwm/native_domain_artifacts.py:252-256` calls `write_prepared_cache(...,
   surface=canonical_noah_surface(soil))`, and `canonical_noah_surface`
   (`gpuwm/native_wrf_contract.py:172-186`) includes `SH2O` explicitly. An
   export taken from a tree domain artifact therefore never reaches the check.
3. **The root preparation cache does not.**
   `tools/hrrr_single_domain_benchmark.py:2954` writes the root prepared cache
   with no `surface=` argument. An export taken from the **root** does reach the
   synthesis branch, and is exposed for any domain with one soil node below
   freezing.

So the exposure is a property of **which artifact the exporter is pointed at**,
not only of the season — a sharpening of B4's caveat, which reads as if season
alone decided it. The run book beside this receipt therefore points the WRF-arm
export at the assembled tree, and records the check as a command to run against
the artifact rather than an inference from the calendar:

```
python -c "import numpy as np; from gpuwm.ingest.prepared_cache import PreparedCacheReader; \
  c=PreparedCacheReader(PREPARED); print(sorted(n for n in c._arrays if n.startswith('surface/'))); \
  print('min soil T', float(np.min(c.read_array('surface/TSLB'))))"
```

For this case the season also points the same way — 2024-05-21 over 35.5-46.7 N
is not a frozen-soil day — but the season is the weaker argument of the two and
is recorded as the weaker one.

## 5. Instrument qualification, observation side

Full numbers and per-lead series: `B5-OBS-CONTROLS-20240521.json`. Reproduce
with `tools/obs_precampaign_controls.py`; the scored parameter block hashes to
`9c1ae4a52c1c5c447e70ce134809448b1a03c1baac7e9ff7bdcacd39283260a9`.

The scored interior is **158 400 of 192 000 cells** (5 boundary+relaxation rows
plus the registered 45 km rim = 15 cells excluded on every side).

### 5.1 The persistence floor, measured

`S_refl` = mean over F02-F18 of FSS(30 dBZ, 27 km box), the registered primary
scalar. The MRMS composite at 12 UTC, remapped to the model grid and held:

| | |
|---|---|
| **persistence `S_refl`** | **0.1598** |
| useful-skill line, `0.5 + f_obs/2` | 0.5249 |
| observed base rate at 30 dBZ | 0.0498 |

Per lead it decays the way a persisted radar image should: 0.400 at F02, 0.291
at F04, 0.215 at F07, 0.206 at F12, 0.0003 at F18. This is the floor every model
arm must clear beyond the early hours (spec section 9.3), and it is now a
measured row rather than a promise. The floor **control** — "every arm beats
it" — needs arms, and is B6's.

### 5.2 The wrong-day negative control, on real MRMS

B2 flagged this control as borderline on synthetic fields. **On real MRMS it is
not borderline.** Same box, same masks, same code path, same registered
parameters; only the day's weather changes. The wrong day is another registered
case day, 2024-04-27, read at the same lead offsets from its own 12 UTC.

| forecast | on its own day | on the wrong day | drop |
|---|---|---|---|
| persistence from 12 UTC | 0.1598 | **0.0230** | **0.1368** |
| a **perfect** forecast (the observation itself) | **1.0000** | **0.0281** | **0.9719** |

Read the second row first. A perfect forecast scores exactly 1.0000 on its own
day — which is also a free proof that the masked FSS returns 1 for two identical
fields, the one answer a broken mask would not give — and **0.0281** against
another day. Both wrong-day scores are far below the useful-skill line of
0.5249, so two of the registered control's three clauses (`score_dropped`,
`wrong_day_below_useful_skill`) are already answered YES from observations
alone. The third clause is phrased in the twin band, which does not exist until
the twin pair runs; it holds for **any** twin band below 0.1368, and the twin
band is an FSS difference between two indistinguishable ICs of the same model.

The instrument sees weather, not climatology and not masks.

### 5.3 Regrid sensitivity

The same persistence reference scored under both registered remap operators:

| operator | `S_refl` |
|---|---|
| `nearest` (registered) | 0.15980 |
| `cell_average` (the budget-style alternative) | 0.15250 |
| **delta** | **0.00731** |

The registered choice scores the *higher*, which is the direction the spec's own
reasoning predicted: composite reflectivity is a column maximum, and averaging
source cells into a destination cell smooths a field whose peaks are the signal.

Published now so the section 9.3 comparison — "the registered choice stands
unless the delta exceeds the twin band" — is one subtraction once the band
exists.

### 5.4 Station-shuffle machinery, on the real frozen table

`shuffle_positions` at seed 20260804 over the 479 interior stations with unique
cells:

| | |
|---|---|
| stations deranged | 479 |
| **fixed points** | **0** |
| displacement, min / median / max | 13.3 / **602.4** / 1336.4 km |

And what the derangement destroys, priced as the rise a **perfect** arm would
suffer under it (its baseline is 0 by construction, since a perfect arm reads
its own station's value):

| variable | shuffled RMSE |
|---|---|
| 2 m temperature | **8.712 K** |
| 2 m dewpoint | **5.557 K** |
| 10 m wind speed | **4.464 m/s** |

Those are enormous compared with anything a surface guardrail will measure, so
the mutation is strong enough to be diagnostic. The registered
`station-shuffle-mutation` control compares a real arm with and without the
derangement against the twin band, and is B6's; what is qualified here is the
machinery and the mutation's strength.

### 5.5 Integrity, observation side

Every archive object behind the scored fields was re-hashed at scoring against
the digest taken at fetch: **34 of 34 match**, no stand-in inputs anywhere in
the evidence (`is_stub` false on every provenance). That is the observation half
of promotion-rule clause R4, exercised before a run exists.

### 5.6 What is not qualified here

Registered controls that need a forecast, a twin band, or a `wrfout`, and are
therefore B6's: `persistence-floor` (verdict), `station-shuffle-mutation`
(verdict), `wrong-day-negative` (its twin-band clause),
`reflectivity-operator-crosscheck`, `twin-non-degeneracy`, `determinism`. The
`qualification-summary` refuses on an absent control by design, so none of the
above is quietly assumed passed.
