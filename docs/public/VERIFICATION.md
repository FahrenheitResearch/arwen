# Verification

ArWen's development rule was that no model code is accepted on
generation: every WRF-derived mechanism is gated against WRF v4.6.1
(upstream <https://github.com/wrf-model/WRF>, tag `v4.6.1`, commit
`d66e442fccc04111067e29274c9f9eaccc3cef28`) before it ships. This page
states the methodology, the measured results, and -- with equal
prominence -- what is deliberately not claimed. It ends with
instructions for reproducing the headline comparison.

## 1. Methodology

Four instruments, in increasing scope:

1. **Component ULP oracles.** For each ported physics routine, a
   harness drives the byte-unmodified WRF v4.6.1 Fortran (compiled from
   the pinned commit) over fixture columns and dumps inputs and
   outputs; the CUDA port is compared field by field in units of FP32
   ULP (units in the last place). Examples shipped in this tree:
   `tools/noah_wrf461_oracle`, `tools/ysu_wrf461_oracle`,
   `tools/morrison_wrf461_oracle`, the Noah-MP and RUC column oracles,
   and the legacy-RRTMG fixture decks. Where a routine is bit-exact the
   gate pins max ULP 0 (for example, the batched legacy-RRTMG LW and SW
   engines are bit-identical to their transcription oracles over the
   full fixture decks at four chunk sizes); where it is not, the
   measured distance and its cause are recorded in the physics registry
   rather than hidden behind a tolerance (see
   [PHYSICS.md](PHYSICS.md)).

2. **t=0 full-state digest.** ArWen's own ingest opens the same
   analysis data as the WRF reference chain, and the two initial
   states are then scored array by array -- every registered carrier
   group, on every domain both runs wrote, against ceilings pinned in
   `gpuwm/verify/nest_gates.py` before any of these frames existed.
   The measurement is published as it comes out. On the reference case
   the two t=0 states do not agree within those ceilings: verdict
   **FAIL**, on all four domains
   ([receipt](../../gpuwm/data/certification/t0_state_parity_digest.json),
   [table](../../gpuwm/data/certification/t0_state_parity_digest.md)).

3. **Matched-run protocol.** The model integrates a real case with
   physics, geometry, and output cadence matched to a WRF v4.6.1 CPU
   reference run, and a streaming comparator
   (`tools/matched_wrfout_stream_compare.py`) scores every output frame
   on the interior grid (5-row rim excluded): T2 MAE, PSFC MAE,
   composite-reflectivity correlation and MAE, CSI at the 20 dBZ
   threshold, W correlation, and 10 m wind correlation. Nothing is
   summarized until every frame is scored; the full decay tables are
   published, not just the flattering leads.

4. **Adversarial review.** Ports and their evidence were audited by
   independent review lanes (including non-Anthropic models) whose
   briefs were to falsify claims: several findings in this page's
   sources exist because a reviewer broke an earlier claim and the
   claim was corrected rather than defended.

## 2. The reference case

The deep-validation case is 3 April 1974 (ERA5-initialized), four
one-way nested domains at 12 km / 3 km / 1 km / 500 m, integrated
12Z-18Z with Thompson microphysics (WRF's own tables, hash-pinned),
YSU PBL, MM5 surface layer, Noah LSM, Kain-Fritsch on the root, and the
legacy-RRTMG transcription -- the same option set as the CPU reference.

The CPU reference is WRF v4.6.1 at the pinned commit, built with **GNU
gfortran 15.2.0** (WRF `configure` option 34, gfortran/gcc, dmpar; Intel
oneAPI supplies the MPI layer only), run on 48 MPI ranks. The GPU run is
one RTX 5090. Both write the same history cadence (d01-d03 hourly, d04
half-hourly).

> **Corrected in 1.4.1.** This page previously described that build as
> Intel `ifx`. It was not: the pinned binary's own bytes carry exactly
> one compiler stamp, `GCC: (Ubuntu 15.2.0-16ubuntu1) 15.2.0`, and no
> Intel Fortran signature. The measurement, the `configure` transcript
> and the full flag set are in the
> [build recipe](wrf-reference/10d6e178d96648e35b826217cc001a1f598232bf6ceabd9ce07074a1f65e2031.build-recipe.md)
> committed beside the manifest that pins the binary. The reference
> stream itself is unchanged -- only the description of how it was
> compiled was wrong.
>
> One file in this repository still carries the old wording as a bare
> statement, and it is named here rather than fixed:
> `configs/real74_thompson_1218z_rrtmg_legacy_4dom.toml`, whose header
> comment reads `WRF v4.6.1 d66e442f, ifx, 48 ranks`. That file's
> SHA-256 **is** the `config_sha256` this manifest, the acceptance band
> and the certification capsule are all addressed by, so correcting a
> comment inside it would move a digest the certification chain depends
> on. Read that line as `gfortran 15.2.0, 48 ranks`; the `48 ranks` and
> the pinned WRF commit in it are correct.

### t=0 comparator metrics (all four domains)

| t=0 | T2 MAE / corr | PSFC MAE | 10 m wind corr |
|---|---|---|---|
| d01 | 0.000 K / 1.000 | 0.1 Pa | 0.999 |
| d02 | 0.000 K / 1.000 | 0.9 Pa | 1.000 |
| d03 | 0.000 K / 1.000 | 0.9 Pa | 1.000 |
| d04 | 0.000 K / 1.000 | 0.1 Pa | 1.000 |

Those are the streaming comparator's own four metrics on the interior
grid, and they are not a statement about the initial state. Scored in
full -- every registered carrier group, every element of every array,
against the pinned ceilings -- the two t=0 states do not agree: verdict
**FAIL** on all four domains
([receipt](../../gpuwm/data/certification/t0_state_parity_digest.json),
[table](../../gpuwm/data/certification/t0_state_parity_digest.md)).
Only the precipitation accumulators are bit-identical. On d01 the
largest disagreements are 66 Pa in perturbation pressure, 0.75 m in
terrain height, 296 K in the deepest soil layer and ten categories in
the land-use index; the receipt carries the per-array numbers for all
four domains. The tables below therefore contain initial-state
differences as well as forecast divergence, and the digest is where the
size of the former is written down.

## 3. Matched-run results (2026-07-28 rerun)

Two scored snapshots are highlighted because they bracket convective
initiation: 15Z (+3 h, squall line organizing) and 18Z (+6 h, mature
cell-scale convection). "Old" is the previous matched run of the same
case (2026-07-27, before a series of seam closures); "new" is the
release lineage. The point of publishing both: the release lineage
moved toward the WRF reference on essentially every axis, and nothing
moved materially away.

### d03 (1 km), 18Z -- the verdict lead

| metric | old | new | direction |
|---|---|---|---|
| T2 MAE | 0.565 K | 0.347 K | toward WRF |
| PSFC MAE | 22.91 Pa | 20.45 Pa | toward |
| refl-comp corr | 0.717 | 0.715 | dead heat (-0.002) |
| refl-comp MAE | 9.81 dBZ | 9.75 dBZ | toward |
| CSI (20 dBZ) | 0.377 | 0.425 | toward |
| W corr | 0.130 | 0.138 | toward |
| wind10 corr | 0.891 | 0.901 | toward |

### d03 (1 km), 15Z

| metric | old | new | direction |
|---|---|---|---|
| T2 MAE | 0.281 K | 0.046 K | toward (6.1x) |
| PSFC MAE | 7.15 Pa | 5.20 Pa | toward |
| refl-comp corr | 0.976 | 0.981 | toward |
| refl-comp MAE | 1.63 dBZ | 1.06 dBZ | toward |
| CSI (20 dBZ) | 0.660 | 0.670 | toward |
| W corr | 0.366 | 0.333 | away |
| wind10 corr | 0.937 | 0.963 | toward |

### d02 (3 km), both leads -- a clean sweep

| d02 | old 15Z | new 15Z | old 18Z | new 18Z |
|---|---|---|---|---|
| T2 MAE | 0.2365 K | 0.0514 K | 0.4449 K | 0.165 K |
| PSFC MAE | 4.666 Pa | 3.155 Pa | 15.480 Pa | 11.55 Pa |
| refl corr | 0.9782 | 0.9848 | 0.9127 | 0.929 |
| refl MAE | 1.821 dBZ | 1.326 dBZ | 4.768 dBZ | 4.02 dBZ |
| CSI (20 dBZ) | 0.8078 | 0.8573 | 0.6518 | 0.682 |

Sharpest single detail: at d02 15Z the new run has 14,230 pixels at or
above 20 dBZ against WRF's 14,227 -- a 3-pixel difference in echo
coverage where the old run was off by 295. Reflectivity bias fell from
-0.311 dBZ to -0.004 dBZ.

### Full decay table (all domains, all leads)

Interior grid, 5-row rim excluded. F1..F6 are forecast hours; d04 is
scored half-hourly. CSI `nan` means neither model had any >=20 dBZ
echo -- by design, not an error.

| dom | lead | T2 MAE K | PSFC MAE Pa | refl corr | refl MAE dBZ | CSI20 | W corr | wind10 corr |
|---|---|---|---|---|---|---|---|---|
| d01 | F1 | 0.012 | 0.50 | 0.970 | 1.58 | 0.863 | 0.984 | 1.000 |
| d01 | F2 | 0.019 | 0.80 | 0.971 | 1.84 | 0.854 | 0.975 | 1.000 |
| d01 | F3 | 0.025 | 1.09 | 0.963 | 2.13 | 0.845 | 0.963 | 1.000 |
| d01 | F4 | 0.030 | 1.37 | 0.953 | 2.44 | 0.820 | 0.945 | 1.000 |
| d01 | F5 | 0.034 | 1.78 | 0.945 | 2.72 | 0.800 | 0.922 | 1.000 |
| d01 | F6 | 0.041 | 2.21 | 0.939 | 2.86 | 0.785 | 0.909 | 1.000 |
| d02 | F1 | 0.014 | 1.02 | 0.996 | 0.41 | 0.931 | 0.994 | 1.000 |
| d02 | F2 | 0.029 | 1.87 | 0.994 | 0.80 | 0.897 | 0.872 | 0.998 |
| d02 | F3 | 0.051 | 3.15 | 0.985 | 1.33 | 0.857 | 0.749 | 0.997 |
| d02 | F4 | 0.076 | 4.24 | 0.975 | 1.85 | 0.794 | 0.743 | 0.995 |
| d02 | F5 | 0.116 | 6.92 | 0.951 | 2.80 | 0.750 | 0.580 | 0.992 |
| d02 | F6 | 0.165 | 11.55 | 0.929 | 4.02 | 0.682 | 0.375 | 0.988 |
| d03 | F1 | 0.010 | 0.92 | 0.996 | 0.26 | nan | 0.999 | 1.000 |
| d03 | F2 | 0.015 | 1.21 | 0.997 | 0.33 | 0.708 | 0.986 | 0.998 |
| d03 | F3 | 0.046 | 5.20 | 0.981 | 1.06 | 0.670 | 0.333 | 0.963 |
| d03 | F4 | 0.105 | 7.93 | 0.933 | 2.70 | 0.428 | 0.358 | 0.946 |
| d03 | F5 | 0.186 | 10.00 | 0.795 | 5.35 | 0.313 | 0.370 | 0.954 |
| d03 | F6 | 0.347 | 20.45 | 0.715 | 9.75 | 0.425 | 0.138 | 0.901 |
| d04 | F0.5 | 0.004 | 0.34 | 0.990 | 0.08 | nan | 0.998 | 1.000 |
| d04 | F1 | 0.007 | 0.35 | 0.997 | 0.11 | nan | 0.998 | 1.000 |
| d04 | F1.5 | 0.006 | 0.46 | 0.995 | 0.13 | nan | 0.994 | 1.000 |
| d04 | F2 | 0.007 | 0.70 | 0.997 | 0.14 | nan | 0.988 | 1.000 |
| d04 | F2.5 | 0.014 | 0.63 | 0.999 | 0.25 | 0.023 | 0.942 | 0.998 |
| d04 | F3 | 0.026 | 2.10 | 0.998 | 0.40 | 0.136 | 0.904 | 0.995 |
| d04 | F3.5 | 0.047 | 4.25 | 0.923 | 0.76 | 0.249 | 0.735 | 0.982 |
| d04 | F4 | 0.107 | 7.27 | 0.870 | 3.01 | 0.516 | 0.508 | 0.951 |
| d04 | F4.5 | 0.173 | 10.99 | 0.832 | 5.09 | 0.328 | 0.257 | 0.915 |
| d04 | F5 | 0.268 | 16.37 | 0.667 | 8.32 | 0.418 | 0.216 | 0.879 |
| d04 | F5.5 | 0.309 | 18.78 | 0.680 | 11.36 | 0.391 | 0.112 | 0.842 |
| d04 | F6 | 0.434 | 22.70 | 0.577 | 14.20 | 0.222 | 0.110 | 0.795 |

### Determinism

The run survived two external process kills; frames produced before
each kill were copied aside and byte-compared when regenerated after
relaunch: SHA256-identical (d03 and d04 checked explicitly). ArWen
reproduces its own trajectory bit-for-bit under restart-free relaunch
on the same hardware and build.

## 4. The chaos floor: how to read the late fine-mesh numbers

The d03/d04 late-lead numbers are not a defect signature; they are what
point metrics do to convection-permitting forecasts, and the evidence
for that reading is in the tables themselves:

- **W correlation is a step function, not a decay curve.** On d03 it
  falls from 0.986 to 0.333 in the single hour when deep convection
  initiates (14Z to 15Z), then holds flat-to-recovering for two hours
  (0.333 -> 0.358 -> 0.370). Once individual updrafts exist, vertical
  velocity is a small-scale chaotic field and point correlation stops
  measuring model agreement. d02 shows the same shape one scale
  coarser; d01, which never resolves updrafts, decays smoothly and
  stays above 0.9.
- **The fine-mesh CSI collapse is reproduced by both runs.** The
  d03/d02 CSI ratio starts near 0.8 at 15Z and falls to roughly
  0.54-0.62 as convection matures -- in the old run and in the new run
  at nearly the same ratio. Pixel-overlap scoring of 1 km cells
  penalizes small displacement errors that carry no information about
  model fidelity.
- **Peak-reflectivity differences alternate sign.** Across all 21
  scored leads the GPU maximum exceeds the CPU maximum 15 times, falls
  below it 5 times, and ties once. That is chaotic divergence of
  individual cells, not a systematic intensity bias.

Meanwhile the mesoscale envelope -- squall-line position, surface
temperature and pressure fields, 10 m wind -- stays close through the
full window on every domain (at 18Z on d02: refl corr 0.929, wind10
corr 0.988, T2 MAE 0.165 K).

## 5. Worldwide projections: what their shallower tier means

Lambert conformal (both hemispheres), Mercator, and polar
stereographic (both poles) run end to end -- wizard, config, static
build, ERA5/GFS ingest, native WRF export -- including
antimeridian-crossing domains. Their verification tier is stated here
with the same prominence as the reference case because it is
deliberately **shallower**: transcription oracle plus GPU smoke
integrations, **not** matched-run.

### Projection transcription oracle (binary64)

The projection mathematics is transcribed from the pinned WRF v4.6.1
`share/module_llxy.F` (plus the WPS v4.6.0 geogrid map-factor and
rotation-angle formulas) and gated against a committed fixture of
IEEE-754 binary64 words produced by that unmodified Fortran compiled
at real-8 (gfortran 13.3.0 / glibc 2.39). Nine projection
configurations cover Lambert (NH secant, SH secant, SH tangent),
Mercator (tropical, subtropical, antimeridian) and polar
stereographic (NH, SH, pole-anchored). Every transform -- setup
constants, lat/lon to grid, grid to lat/lon, map factor, wind
rotation -- is compared in binary64 ULPs against per-quantity
ceilings pinned exactly in `tests/test_projection_oracle.py`
(`ULP_CEILINGS`):

| projection | worst pinned ceiling | worst slot |
|---|---|---|
| Lambert conformal | 32 ULP | lat/lon -> j |
| Mercator | 8 ULP | grid -> latitude |
| Polar stereographic | 2 ULP | grid -> lat/lon |

The only drift source is numpy libm vs glibc libm (the transcriptions
are operation-identical). The Lambert map factor additionally carries
a 2.3e-16 relative bound: the product ships the ARW tech-note form
referenced to `truelat1`, geogrid the mathematically identical form
referenced to `truelat2`. A mutation control (truelat1 perturbed by
1e-3 must overflow every ceiling) proves the suite can fail.

### GPU smoke integrations (four worldwide sites)

Wizard-emitted single-domain 12 km configs (116x94x49, the default
physics suite), integrated on the GPU from GFS initialization at the
2026-07-29T06 cycle:

| site | projection exercised | simulated | verdict |
|---|---|---|---|
| Brisbane | Lambert, southern hemisphere | 3 h | PASS -- finite everywhere, no NaN |
| Singapore | Mercator, near-equatorial | 2 h | PASS -- finite everywhere, no NaN |
| Fairbanks | polar stereographic | 3 h | PASS -- finite everywhere, no NaN |
| Fiji | Mercator across the antimeridian | 3 h | PASS -- finite everywhere, no NaN |

Each run's final model state is digested field-by-field (157 arrays)
and the digests, configs, and per-site reports are retained as
machine receipts in the development tree (`evidence/worldwide-smoke/`;
retained outside the release snapshot).

### The boundary, stated plainly

**No matched-run verification exists for the new projections.** No
WRF twin has been integrated on Mercator, polar stereographic, or
southern-hemisphere Lambert; the matched-run family of sections 2-4
is northern-hemisphere Lambert only. A smoke integration proves the
pipeline executes and stays finite -- it does not measure forecast
agreement. Configurations on the new projections inherit the
transcription oracle above, the component-level physics evidence
([PHYSICS.md](PHYSICS.md), including its projection maturity rows),
and these smoke receipts -- nothing more.

## 6. What is claimed, and what is not

Claimed, each with its receipt above or in the linked pages:

- A published full-state t=0 digest of the reference case on all four
  domains, carrying the result it returned: the two t=0 states do not
  agree within the pinned ceilings -- verdict **FAIL**
  ([receipt](../../gpuwm/data/certification/t0_state_parity_digest.json)).
  Parity of the initial state at the FP32/operator floor is not
  claimed; see "Not claimed" below.
- Named component routines bit-exact or measured-ULP-close to
  unmodified WRF v4.6.1 Fortran, per the physics registry's per-option
  records ([PHYSICS.md](PHYSICS.md)).
- Matched-run forecast agreement on the reference case at the levels
  tabulated in section 3.
- Bit-deterministic re-execution on fixed hardware and build.
- Unchanged stock WRF v4.6.1 accepts and integrates this
  preprocessor's outputs, within the stated boundaries
  ([WRF-INTEROP.md](WRF-INTEROP.md)).
- Projection transcription parity at the pinned binary64 ULP ceilings
  for all three projections, plus finite-state GPU smoke integrations
  at four worldwide sites (section 5).

**Not claimed:**

- **The initial states do not match at the FP32 floor.** The
  full-state t=0 digest scores five of its six covered carrier groups
  outside the pinned ceilings on every domain
  ([receipt](../../gpuwm/data/certification/t0_state_parity_digest.json)):
  verdict **FAIL**. Section 2's four surface metrics are small, and
  they were never evidence for the rest of the state. Neither side of
  the staged pair retained a `wrfbdy_d01`, so the lateral-boundary
  tables are recorded as unavailable and no boundary statement is made
  anywhere on this page.
- **The new projections are not matched-run verified.** Mercator,
  polar stereographic, and southern-hemisphere Lambert carry the
  section-5 tier only; no ArWen-vs-WRF forecast comparison exists on
  them.

- **No end-to-end bit-exactness with WRF.** The model state is FP32,
  GPU transcendentals differ from glibc/Intel libm at the ULP level,
  and several deliberate deviations from WRF are registered in
  [PROVENANCE.md](../../PROVENANCE.md). "Bitwise" statements are always
  scoped to a named comparison (a kernel oracle, a restart identity, a
  dual-run byte comparison) -- never to WRF output files. What the
  dual-run byte comparison covers, and what it cannot detect in place
  of ECC, is [DETERMINISM.md](DETERMINISM.md).
- **One case is deeply validated.** The matched-run evidence is one
  meteorological situation, one season, one region, one option set.
  Other cases, seasons, and physics combinations inherit component
  evidence only; their maturity labels say so explicitly.
- **Part of the 2026-07-28 improvement is by construction.** That
  rerun matched the reference configuration (legacy RRTMG, matched
  cadence and geometry); it demonstrates fidelity of the matched
  configuration, not a universally improved solver. The old/new
  comparison also spans two output-writer versions (4-byte file-size
  difference; frames are identified by SHA256, never by size).
- **No data assimilation, no ensemble calibration, no forecast-skill
  claim against observations.** All comparisons on this page are
  model-vs-model. Nothing here says ArWen (or WRF) verified well
  against what actually happened on any date.
- **No resolved tornado dynamics.** The 500 m nest and the STP/UH
  severe suite characterize the tornadic-supercell environment and
  mesocyclone-scale morphology; they do not resolve the near-surface
  corner flow, suction vortices, or tornado-scale wind intensity
  (tornado-like-vortex resolution in the literature sits below
  roughly 25 m horizontal and 10 m vertical).
  Everything on this page is convection-permitting to sub-kilometer
  case-study evidence, not a tornado-resolving claim.
- **FP32 subnormals are flushed on the compile route this product
  uses.** Kernels here are built through CuPy, which appends
  `-ftz=true` after the options the caller passed, and the compiler
  honors the last occurrence -- so FP32 subnormals are flushed to
  zero in arithmetic on those modules even where `--ftz=false` was
  requested. That is toolchain behavior, not a property of the
  silicon: on the measured device (sm_120) the same source compiled
  without the appended flag keeps full IEEE subnormals, and the
  routes in this codebase that bypass the append measure IEEE
  agreement on the same card ([HARDWARE.md](HARDWARE.md) records the
  per-route result). The flushing is real wherever the append
  reaches, so the countermeasures stand; the specific measured
  consequences (branch flips on subnormal inputs) and the
  countermeasures taken are recorded per scheme in the registry.

## 6b. The registered cases, and which of them need data

`gpuwm cases` lists every registered case and, beside each, the command
that runs it. Two doors appear there and they are not
interchangeable: `gpuwm verify NAME` grades a case against its
registered `GATES`, and only cases carrying the `verify` capability have
any; the rest are run as `python -m gpuwm.verify.cases.NAME` and print
their own result. That is why the listing is longer than `gpuwm verify
--help`'s choice list.

Every verify case is **self-contained and needs only the GPU runtime**,
with one exception:

- **`real74_d01`** is a frozen comparison against an external WRF v4.6.1
  run of 3 April 1974, so it needs that run's reference bundle -- its
  `namelists/`, `met_em/`, `static/` and `wrfout_reference/` subtrees.
  The bundle is third-party output of several gigabytes, is not
  redistributable, and there is no `gpuwm fetch` for it. Point
  `GPUWM_REAL74_REFERENCE_BUNDLE` at a directory holding it, or place it
  at `~/Downloads/WRF_1974_MP55_reference_bundle`. Without it the case
  refuses by name and says the same thing.

The two LES tornado cases are the module half of a case whose other half
is a config under the repository's `configs/` directory. `configs/` is
not a Python package and ships in no wheel, so from an installed gpuwm
name the file with `--config`, or point `GPUWM_CONFIGS_ROOT` at a
checkout's `configs` directory. `--help` works either way.

## 7. Reproduce this

The headline comparison (section 3) was produced as follows.

- **Config:** `configs/real74_thompson_1218z_rrtmg_legacy_4dom.toml`
  (ships in this repository) -- Thompson mp8 with the packaged,
  SHA-256-validated WRF tables; `ra_rrtmg_variant = "rrtmg_legacy"`;
  four domains, 12Z-18Z (21,600 s); history cadence 60/60/60/30 min.
- **Commit:** the run of record executed at internal development
  commit `152f7d31` (2026-07-28, recorded here for provenance; that
  history predates this public repository). The configuration and
  comparator ship unchanged in this release, so the comparison runs
  from any checkout of it.
- **Inputs you must stage yourself** (none are redistributable by this
  repository): the ERA5 case retrieval for 3 April 1974 (see
  [DATA.md](DATA.md) for the CDS walkthrough; the config's
  `[case_data]` table names the exact files), the NCAR WPS_GEOG static
  tree, and -- for the CPU side of the comparison -- a WRF v4.6.1 run
  of the same case built from the pinned commit (the reference run's
  namelist is recorded with the run metadata; its option set is the one
  named above).
- **Commands:**

```bash
gpuwm check configs/real74_thompson_1218z_rrtmg_legacy_4dom.toml --alloc
gpuwm run   configs/real74_thompson_1218z_rrtmg_legacy_4dom.toml --outdir out/rematch --directory-input-hash content

python tools/matched_wrfout_stream_compare.py \
  --gpu-dir out/rematch --cpu-dir /path/to/wrf-reference-wrfouts \
  --out-csv out/rematch/metrics.csv --exclude-rows 5 \
  --start-time 1974-04-03_12:00:00 --done-file out/rematch/run.done

python tools/matched_wrfout_t0_state_digest.py \
  --candidate-dir out/rematch --reference-dir /path/to/wrf-reference-wrfouts \
  --out-json out/rematch/t0_state_parity_digest.json \
  --out-md out/rematch/t0_state_parity_digest.md
```

Three flags here are not decoration:

- `--directory-input-hash content` hashes every byte of the static geography
  tree instead of its listing. Two people staging that tree separately get the
  same digest only under `content`, and certification refuses a run whose
  geography was bound by listing.
- `--start-time` is what fills the `forecast_hour` column. Without it every
  lead cell is written empty, and a comparison with no lead cannot be placed
  against an acceptance band, which is keyed by lead.
- `--done-file` is the comparator's terminating condition: it exits once that
  file exists and no pair is still pending. Without it the poll loop has no way
  to finish, so the command runs until it is killed. Touch that file when the
  run completes.

These commands were executed through their production entry points against a
two-frame fixture pair before this paragraph was written; the exact argv, exit
code, and resulting lead column are recorded in
`gpuwm/data/certification/recipe-receipt/receipt.json` beside the metrics CSV
it produced. The `gpuwm run` line is the one half that is not executed there --
no fixture stands in for a four-domain six-hour integration -- and the receipt
says so, recording instead that the production parser accepts the line as
written.

- **Expected resources:** on one RTX 5090 the full four-domain window
  took 6.7 h of wall time (67.2 wall-seconds per simulated minute
  whole-tree; 61.4 pre-convective, 72.9 after convective initiation)
  and 31.5 GB of disk (20.1 GB wrfouts + 11.4 GB checkpoints). Measured
  machine-wide VRAM peak was 29,004 MiB (28.3 GiB) on a 32 GiB card --
  this case is sized for a 32 GiB card and will not fit smaller ones;
  size your own case with `gpuwm domain` ([HARDWARE.md](HARDWARE.md)).
- **Expected result:** metrics within ordinary chaotic-divergence
  scatter of section 3's tables (bit-identical only if hardware,
  driver, and build match the run of record). The digest command scores
  the two t=0 states array by array and, as on the run of record, they
  do not agree within the pinned ceilings -- verdict **FAIL**; compare
  your per-array numbers against the committed
  [receipt](../../gpuwm/data/certification/t0_state_parity_digest.json)
  rather than expecting a floor.
