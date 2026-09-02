# Phase 2 baselines — pre-extension reference

Captured **before** `CumulusResult` is touched, which is the one
irreversible ordering constraint in Phase 2: once the momentum extension
is in the tree there is no clean pre-extension state to compare against
without a checkout in a working tree carrying 45 uncommitted paths.

Launch commit **0871b2fb** (the runner records it itself, and prints
`(dirty)` — that flag is the owner's uncommitted campaign work, not the
port's; everything of mine is landed).

## Determinism, established FIRST

Two runs of each scheme *before* the extension, on review's argument:
if the post-extension comparison failed, we could not otherwise tell
whether the extension changed something or whether this configuration
was never bitwise reproducible. These configs are new and `prepared_myj`
had never been run this way.

| | runs | wrfout files | identical | peak VRAM |
| --- | --- | ---: | ---: | ---: |
| GF, `cu_physics = 3` | `nt_base_gf`, `nt_base_gf2` | 9 | **9 of 9** | 10.33 / 10.36 GiB |
| KF, `cu_physics = 1` | `nt_base_kf`, `nt_base_kf2` | 9 | **9 of 9** | 8.84 / 8.73 GiB |

**And GF differs from KF on all 9 files.** That is the guard against the
gate passing for the wrong reason: it proves the wrfout content actually
reflects the cumulus scheme, so "identical" across two runs of one
scheme is a statement about determinism rather than about a file that
would be identical whatever ran.

## Peak VRAM's resolution — measured, and it is not a noise floor

review saw two peaks that differed on byte-identical output and read it
as the 50 ms sampler's noise floor, which would have put a **0.11 GiB**
threshold under any later comparison. A third run of each scheme, 59 s
each, says otherwise:

| run | GF peak | KF peak | wall |
| --- | ---: | ---: | ---: |
| 1st | 10.328 | 8.836 | 128 s / 52 s |
| 2nd | **10.361** | **8.734** | 59 s / 52 s |
| 3rd | **10.361** | **8.734** | 58 s / 52 s |

**The second and third runs agree exactly, for both schemes.** The spread is
entirely first-run-versus-warm — the first run pays NVRTC compilation (128 s
against 59 s for GF) and that changes what is resident when the sampler
fires. It is a *cause*, not a floor.

So the rule for Phase 3 is not a tolerance band. It is: **discard the first
run of any fresh kernel set, and compare warm runs, which reproduce peak
VRAM exactly.** Stated now, before the NT number exists, so the threshold
cannot be chosen after seeing it.

If a later comparison does land close, the honest reading is still
"indistinguishable" rather than "equal" — but on this evidence the
instrument is sharper than a 0.11 GiB band would have allowed, and treating
0.11 as irreducible would have thrown away real resolution.

**All 9 wrfout files are identical across all three runs of each scheme**,
so determinism is now three-way rather than two-way.

## A third baseline, because the first two missed the changed code

The momentum extension factored the A-grid-to-C-grid face interpolation
out of `couple_ysu_tendencies`. **Neither the GF nor the KF baseline ever
calls that function**: both carry `bl_pbl_physics = 2` (MYJ), inherited
from the configs they copy. So "9 of 9 identical for both schemes" is a
statement about a path the refactor does not touch (review).

And the two chosen are the unrepresentative ones — **19 of the 24 configs
in that directory use `bl_pbl_physics = 1`**, so the refactored path is
the one nearly all of the campaign runs on.

`_nt_baseline_ysu_30min.toml`, a 1,800 s truncation of `tc_hafs.toml`
against `prepared_hafs`, closes it:

| | physics.py | wrfout files | identical |
| --- | --- | ---: | ---: |
| `nt_ysu_pre` | `d86a3d48`, pre-refactor | 20 | — |
| `nt_ysu_post` | `HEAD`, refactored | 20 | **20 of 20** |

3 domains, 20 frames, 570 steps. Done by `git checkout <commit> -- <one
file>`, run, restore, run — no stash, nothing near the 45 uncommitted
paths.

**Why a unit test could not have done this.** Once YSU delegates, any test
comparing the two routes compares the same code reached two ways. The
extraction's fidelity is only visible against output produced *before* the
extraction existed, which is a run-level artifact by construction. The
unit test in `test_cumulus_momentum_extension.py` has been corrected to
claim only what it proves — the `chm`-before-helper ordering — and to
point here for the rest.

## Why 30 minutes is enough

The gate is **bitwise**, and the failure modes the momentum extension
could introduce — a changed buffer shape, a perturbed optional-components
set, an altered accumulation order — diverge in the **first frame or not
at all**. A longer run cannot expose a bitwise difference that 30 minutes
hides; it can only cost more. Written down because the next person
shortening a gate will want the argument and there would not be one.

What it does **not** cover: a divergence that needs more than 30 minutes
of forecast to become visible *at output precision*. No such mode is
known for this change, and if one is proposed the run length is the
first thing to revisit.

## Frame count

**9 files per run**, 2 domains — d01 at `history_interval_s = 720` over
1,800 s and d02 at 360. Not 33: that number was a property of
`_profile_1h.toml`, whose whole cycle directory is no longer on disk.
The gate is stated over what the run produces.

## A measurement worth keeping

| | preflight estimate | observed peak | ratio |
| --- | ---: | ---: | ---: |
| GF | 4.201 GiB | **10.33 GiB** | 2.46x |
| KF | 4.173 GiB | **8.84 GiB** | 2.12x |

Same tree, same 30 minutes, only `cu_physics` differing: **GF costs
1.49 GiB more observed peak than KF**. That is a like-for-like measured
number of exactly the shape §35 hands the NT-vs-GF margin to, and it
arrives free with the baselines.

Two cautions before anyone quotes it. `preflight_alloc_estimate_bytes`
is the **allocation** estimate, not a peak prediction — the estimator
carries run-time pool retention as a separate term — so the 2.46x is not
by itself an under-budget finding. And GF-minus-KF is not GF's cost: KF
allocates too.

## What survives the estimator caution — the differential

The 2.46x ratio is confounded: `preflight_alloc_estimate_bytes` is an
allocation estimate and the observed peak is a peak, so they are different
quantities. That does **not** confound the *difference of differences*
(review):

```
estimator   GF 4.201   KF 4.173     GF - KF = +0.028 GiB
observed    GF 10.36   KF  8.73     GF - KF = +1.63  GiB
```

Every term common to both runs cancels — the CUDA context, the base pool,
the same tree, the same 30 minutes, the same everything but `cu_physics` —
and the allocation-versus-peak confusion cancels with them, because it is
common-mode: same estimator, same box, same run shape.

**The estimator says GF and KF cost the same to within 28 MiB. The
measurement says GF costs 1.63 GiB more.** That is the same order as the
~1.16 GiB of full-domain per-call staging measured in `gf.py`'s `__call__`
that `gf_column_workspace_bytes` does not price (§34). Two independent
routes landing near 1.2–1.6 GiB is not proof — GF and KF differ in other
ways, kernel frames among them — but it is the form of the question that
survives the caution, and it is one step from closed: itemise what the
estimator counts for each scheme and see whether the gap is the staging.

Not Phase 2 work. Recorded so it is not filed under "confounded" on the
strength of the ratio being confounded — the ratio is; the differential
is not.

## A pre-existing test failure, measured and not touched

`tests/test_prepared_cache.py::test_a_v101_shape_header_still_binds_after_
the_upgrade` pins the domain identity at 13 top-level keys. It is 18.

Measured rather than assumed: HEAD's module loaded alongside the working
one reports **18 at HEAD and 18 with the tolerance change**, so it is not
caused by this work. The 2.5.6 cache carries all 18 keys, so the pin is
stale for a reason unrelated to the two `run.` fields, and it is left
alone.

## Digests

sha256, identical across both runs of each scheme by construction —
that identity IS the determinism result above.

### GF

```
097830585ec5f48ab2d21b105c7efc86301f134bc19cb0d45201ac5b78c5d5a6  wrfout_d01_2025-10-25_18_00_00
f8353f2022754350cc2be36d730a720b7005bb37c0addba07db083eb72970d43  wrfout_d01_2025-10-25_18_12_00
318e651619ec85e82c299234f69c89febded036bc7cc15be2691e87312bddbde  wrfout_d01_2025-10-25_18_24_00
3a27832ce9cada8be7e86e9eadf75ee6b874ec442d5c61e845f8e0e1c38480c8  wrfout_d02_2025-10-25_18_00_00
27628fe0d5be019981f92e2d12d40afae833df18b2d5c568a1a473c12d818c09  wrfout_d02_2025-10-25_18_06_00
b837b9c27ebad1b516638972ac01ab107991b8ed407741c381d6546046ec6fd2  wrfout_d02_2025-10-25_18_12_00
fe5dbae2fb7d0de25f30376f4a3d3c5279157f5e215fea4d62f8c0388c9a5227  wrfout_d02_2025-10-25_18_18_00
a620de379d9b71097dbb7305bcd979ad581f1bfdffb823d9fb3bd95e91b34d22  wrfout_d02_2025-10-25_18_24_00
9e785f51eefbed4a4955c2202fbe87a5657e122d5fbf0a38c5b76682b7860f63  wrfout_d02_2025-10-25_18_30_00
```

### KF

```
2114eda547b57cb634c526fc8f787afd32ed0b8ea24fd1b10b031466ca4d7e4e  wrfout_d01_2025-10-25_18_00_00
71435073784ecaa24ddeca3b73099e878f04db96830e21166220f463e8cea10b  wrfout_d01_2025-10-25_18_12_00
ad83224d64ad754b8181a512525b467798b220d12d23bb64888c0886ee16e0fe  wrfout_d01_2025-10-25_18_24_00
5c05ecd281efa1aa13c7f8ee9eeba4f146f81b4efa37b578aefd18c0ad0f0a31  wrfout_d02_2025-10-25_18_00_00
8e28decfaf6032262465b6932a73b7672beed9c1200e7134746481a7f5df164e  wrfout_d02_2025-10-25_18_06_00
8bf28f41e8afdc615c0b119a33e90e931c510dbc754d9a3a20bcbbb085f18bed  wrfout_d02_2025-10-25_18_12_00
5c17abd74f69822d71b0dab75b2909c20952110ed558809a3bbe00b7a4bf9c3d  wrfout_d02_2025-10-25_18_18_00
193b859e2dbaffcec7dd909bd9560122c24d825e1e8da496daeec775d7b8a30f  wrfout_d02_2025-10-25_18_24_00
f500f68f63967dc0565ac6bf3f8e0d8c90b5a2ff686f755fb2b205f07aae654b  wrfout_d02_2025-10-25_18_30_00
```


---

## Phase 2 end condition: the first `cu_physics = 16` forecast

Met on 2026-08-29.  `E:/GPUWRF/runs/2025-10-24_18/output/nt16_run1`, 2
domains, 120 steps, 9 wrfout frames, exit 0.

It took **four attempts**, and the three refusals are the point: every one
was a fail-closed gate naming its own cause, none was a wrong number.

| # | refusal | site |
|---|---|---|
| 1 | `no measured local frame for kernel module(s) ntiedtke` | the frame table |
| 2 | `cu_physics must be 0 (off), 1 (Kain-Fritsch) or 3 (Grell-Freitas)` | `initialize_physics` — a **second** whitelist |
| 3 | `cumulus pratec requires nca_seconds` | the driver's no-hold contract |

Site 2 is the one worth remembering.  `validate_run_config` gates the
config and reads `CU_SCHEMES`; `initialize_physics` gates the driver and
restated `(0, 1, 3)` as a literal.  So a `cu_physics = 16` config was
valid, priced, scheduled, compiled and frame-recorded, and *then* refused
at driver construction, after the whole preflight had passed.  The search
that found the nine sites of the Phase 2 group looked for `cu_physics == 3`
**dispatch** sites; this is a **membership** test, and reading never found
it.  Running the model found it in two seconds — which is the argument for
running one early rather than reading harder.

### The scheme does work, and that is a separate question from completing

A run that finishes proves plumbing.  Convective rain at the last frame:

| scheme | dom | RAINC max | RAINC mean | wet cells | RAINNC max |
|---|---|---|---|---|---|
| GF | d01 | 4.013 | 0.0848 | 30.8% | 22.54 |
| GF | d02 | 0.923 | 0.0163 | 49.7% | 61.01 |
| KF | d01 | 4.875 | 0.2058 | 25.8% | 21.70 |
| KF | d02 | 7.230 | 0.3828 | 30.8% | 59.18 |
| NT | d01 | 6.771 | 0.0695 | 70.5% |  8.10 |
| NT | d02 | 0.810 | 0.0346 | 69.2% | 31.57 |

New Tiedtke is **broader but weaker**: it rains a little in many places
rather than a lot in a few.  On the 4.5 km d02 nest its area-mean
convective rain is 2.1x GF's and 0.09x KF's, and its wet fraction is the
highest of the three.  Its RAINNC is correspondingly the lowest — a more
active, more diffuse convective scheme removes instability before the
microphysics can condense it out, which is coherent rather than suspicious.

**This does not yet say anything about the intensity gap.**  These are
30-minute runs dominated by spin-up, one per scheme, and GF's `sig` clamp
was not instrumented here.  That NT is active where `TC-INTENSITY.md`
predicts GF switches itself off is consistent with the diagnosis; it is
not a test of it.  That is Phase 5.

### Determinism (Phase 3, first half)

Two runs of unchanged code, `nt16_run1` vs `nt16_run2`, `cmp` on **every
file the run produces** — 25 files, not just wrfout:

- **identical: 10** — all 9 wrfout frames, and `track.csv`.
- **differ: 15**, and after normalising instance keys (`path`,
  `experiment_fingerprint`, `*_seconds`, `*_ms`, pids) only three carry a
  residual, each explained: `certification-capsule.json` (a `git_commit`
  that moved because commits landed between the runs),
  `evidence/run-receipt.json` (VRAM sampler maxima and sample counts, 721
  vs 697 — the 50 ms poller), and `progress.jsonl` (pid and timing text).

No physics-bearing file differs.

### Cost, against the two baselines it was built to be compared with

`_nt16_myj_30min.toml` differs from `_nt_baseline_gf_30min.toml` in
`cu_physics` (3 -> 16) and the removed `ishallow` **and nothing else** —
verified by diff, not by intent.  That is deliberate: it is the GF
baseline's own `tc_hafs_myj.toml` lineage.  *"It was smaller"* and
*"it is the baseline's own lineage"* are different reasons, and only the
second makes the comparison mean anything.

| scheme | peak GiB (warm) | wall s (warm) |
|---|---|---|
| KF | 8.734, 8.734 | 51.9, 51.9 |
| GF | 10.361, 10.361 | 59.0, 58.5 |
| NT | 11.060 -> **10.992** | 49.2, 49.5 |

**New Tiedtke is the fastest of the three** — 49.4 s median against KF's
51.9 and GF's 58.8, about 16% faster than GF on an otherwise identical
config.

It also costs **+0.63 GiB of peak VRAM over GF**, and *why* is not
established.  What can be said: NT's CuPy pool peak is 140 MiB **lower**
than GF's (7.889 vs 8.029), so the excess is not pool growth.  What cannot
be said is where it is instead — `peak(device) - peak(pool)` subtracts two
independently sampled maxima that need not be simultaneous, and treating
that difference as a decomposition is exactly the trap CLAUDE.md warns
about.  Doing it anyway produced an "implied frame" of 9,233 B against
YSU's recorded 9,232 B, a one-byte agreement that is pure artifact: all 21
ntiedtke kernels compile to a **0 B** frame under the model's own
`load_module`, and loading either cumulus module leaves
`cudaLimitStackSize` at its 1,024 B default.  Attributing the rest needs a
timeline probe, which is its own piece of work.

### One thing the model corrected about itself

The adapter held **two** chunk workspaces live across `_for`'s
constructor — 857.0 MiB where 433.2 was needed — every cumulus step on
every domain, because the tile (17,920) divides neither domain's column
count (38,870 and 71,289) and the single-slot cache thrashed between a
full width and a remainder.

Releasing the old before building the new is bitwise inert (run 3 vs run
1: 9/9 wrfout and `track.csv` identical under `cmp`).  **In isolation it
removes a 424 MiB transient; against the model the peak moves 70 MiB**
(11.066/11.060 -> 10.992).  The cumulus double-workspace is not where the
run's global peak sits, and a peak is a maximum over a timeline rather
than a sum of parts.  70 MiB still clears the >50 MiB bar and costs
nothing, so it stays — but the 424 was never the number.

---

## The determinism gate, re-run clean — and two numbers medianed

review caught that the first determinism pair was run with a commit
between the two runs, which is why `certification-capsule.json` carried a
`git_commit` residual.  Not a correctness problem, but a residual I
caused.  Re-run as a clean pair, `nt16_run4` vs `nt16_run5`, nothing
committed between them:

- **25 files produced, 11 identical** — all 9 wrfout frames, `track.csv`,
  **and `evidence/microphysics-transitions.json`**, which the first pair
  showed as differing.
- **14 differ; only 2 carry a residual** after normalising instance and
  time keys: `evidence/run-receipt.json` (VRAM sampler maxima and sample
  counts) and `progress.jsonl` (pid and timing text).

The eleventh identical file is the interesting one and it confirms the
point rather than just tidying it.  `microphysics-transitions.json`
differed in the first pair *only* by `experiment_fingerprint`, and that
fingerprint tracks the config and source hashes — not the output path.  So
it was my commits that moved it, exactly as diagnosed, and with the source
held still it is byte-identical across runs in different directories.

### Two numbers restated as medians of three

Standing rule 2 asks for at least three runs and the median.  Two figures
above were quoted from single runs and are corrected here.

| | quoted before | median of 3 |
|---|---|---|
| peak VRAM, post-fix | 10.992 (run 3 alone) | **11.002** (10.992 / 11.002 / 11.015) |
| saving from the pipeline fix | ~70 MiB | **~61 MiB** |
| gap over the GF baseline | +0.63 GiB | **+0.641 GiB** |

The fix still clears the >50 MiB bar and still costs nothing, so the
disposition is unchanged — but 70 MiB was one run's number and 61 MiB is
the config's.

One thing the third run makes visible that two could not: NT's post-fix
peak spreads 23 MiB across three runs (10.992–11.015), where GF's runs 2
and 3 agreed **exactly** at 10.361.  That is a difference in behaviour,
not just in luck, and it is a second thread for whoever takes the
timeline probe — a peak that moves run to run is a peak sitting near
something that varies.

Wall time is unaffected by the fix and unchanged in the ranking: NT
49.2–49.7 s across five runs, against KF 51.9 and GF 58.5–59.0.

---

## The hypothesis test: the cycle owned the spread, not the gap

review proposed that the two open threads — the +0.641 GiB over GF and
the 23 MiB run-to-run spread where GF reproduces exactly — might be one
phenomenon, both caused by the allocate/free cycle that GF does not have.
Flagged as a hypothesis, with the discriminating measurement named: remove
the cycle, and see whether the spread collapses **and** the gap shrinks.

Ablate-first says test the hypothesis before building the cure, so the
first attempt was the cheap instrument — cache a pipeline **per width**
instead of one slot, which removes the cycle in three lines.

**It was the wrong instrument, and that is worth recording.**

| config | peaks (GiB) | median | spread |
|---|---|---|---|
| current (rebuild per chunk) | 10.992 / 11.002 / 11.015 | 11.002 | 24.0 MiB |
| per-width cache | 12.835 / 14.593 / 14.595 | 14.593 | **1802.0 MiB** |
| GF baseline | 10.361 / 10.361 | 10.361 | 0.0 MiB |

+3.59 GiB and a spread 75× **worse**.  The cache holds three workspaces
resident where one was needed and fragments the pool around them, so it
removes the cycle while adding residency — it confounds the two things the
experiment was built to separate.  Read literally it says the hypothesis is
refuted; read honestly it says nothing about the hypothesis at all, because
the intervention changed two variables.

### The instrument that does isolate

Allocate ONE pipeline at the tile width and pad the short last chunk.
Same 433.2 MiB resident as the current build — one variable moved, the
cycle — and reachable without touching a kernel.

| config | peaks (GiB) | median | spread | wall |
|---|---|---|---|---|
| current (rebuild per chunk) | 10.992 / 11.002 / 11.015 | 11.002 | 24.0 MiB | 49.5 |
| **padded, zeros** | 10.937 / 10.939 / 10.939 | 10.939 | 2.0 MiB | 49.4 |
| **padded, replicated column** | 10.937 / 10.943 / 10.943 | 10.943 | 6.0 MiB | 49.8 |
| GF baseline | 10.361 / 10.361 | 10.361 | 0.0 MiB | 58.8 |

Both pad variants are bitwise identical to the current build (9/9 wrfout
under `cmp` against `nt16_run4`).  **The shipped variant pads with a copy
of the chunk's first column rather than zeros**, because only that carries
an argument: the sole cross-column operation is `llo3`'s chunk-wide OR,
and OR over a multiset that repeats a member it already contains cannot
change its value.  Zeros happen to measure identical; replication cannot
do otherwise.

### The verdict, split

**The spread was the cycle.**  24.0 → 6.0 MiB, and GF — which allocates
once — reproduces its peak exactly.  That half of the hypothesis is
confirmed, and the mechanism is the one proposed: a repeating transient
that the 50 ms sampler catches at a different phase each run.

**The gap was not.**  +0.641 → +0.582 GiB.  The cycle accounted for
60 MiB of 641, about 9%.  **Roughly 90% of the gap over Grell-Freitas
remains unexplained**, is not the allocate/free cycle, is not the CuPy
pool (NT's pool peak is *lower* than GF's), and is not a local-memory
reservation (all 21 kernels compile to a 0 B frame and neither cumulus
module moves `cudaLimitStackSize` off its 1,024 B default).

So the timeline probe still has one target rather than two, and it now has
three eliminated explanations to start from instead of none.  That is what
the hypothesis bought even though it was half wrong — which is the case
for stating hypotheses precisely enough to be halved.

---

## The timeline probe: the gap grows with run length, and Phase 5 cannot ignore it

review asked which pool figure the elimination rested on — `used_bytes`
or `total_bytes` — before spending a probe on the strength of it.  It was
`total_bytes`, the right one.  **But the elimination was wrong anyway, and
checking the field is what led to finding out why.**

### The instrument

`peak(device) − peak(pool)` is three independent maxima from a 50 ms
poller, and subtracting them is what produced the spurious 9,233 B frame.
So the probe samples all three series **from one thread at one instant**,
and reads `cudaLimitStackSize` directly rather than inferring a frame from
a VRAM delta.  (Script kept in the session scratchpad, not the tree.)

Two by-products, both of which correct earlier statements here:

* **`pool_total` is effectively monotone**, so for that pair the maxima
  subtraction happens to be valid — measured, `2.927` both ways.  The
  earlier arithmetic was unsound *reasoning* that landed on a sound number.
* **The 9,232 B "coincidence" was real, and coincidental.** Both runs
  reach `cudaLimitStackSize = 9,232 B` and both therefore carry the same
  0.822 GiB reservation.  It cancels in the difference and cannot explain
  the gap.  The earlier inference matched YSU's row to one byte because
  the *difference* happened to equal a term present in *both*.

### The 30-minute picture, decomposed at one instant

| | NT | GF |
|---|---|---|
| device at peak | 10.937 | 10.492 |
| pool_total at peak | 8.009 | 8.166 |
| **non-pool** | **2.928** | **2.326** |
| stack limit | 9,232 B | 9,232 B |

At 30 minutes the whole gap is non-pool, and the pool looks eliminated.
Tracing *when* it appears: GF's non-pool is flat after the reservation
lands; NT's climbs +476 MiB in three bursts.  Deleting the relocation
block isolates it — NT with no relocation climbs +129 MiB once at startup
and is then flat, so about **115 MiB per nest relocation**, where GF
relocates identically and pays +29 MiB total.

A minimal reproduction — build an `NtPipeline`, run it, drop it, five
times — shows **zero** growth, so it is not the pipeline rebuild, and
`load_module` is `lru_cache`d so it is not recompilation.  The mechanism
is still open.

### Then the 2-hour run, which changes the conclusion

`run_seconds` 1800 → 7200 gives ~20 relocation events instead of ~4, to
ask whether that per-event cost saturates or accumulates.

| | NT 30 min | NT 2 h | GF 2 h |
|---|---|---|---|
| device peak | 10.937 | **13.310** | 11.138 |
| pool_total at peak | 8.009 | 10.095 | 8.800 |
| non-pool at peak | 2.928 | 3.215 | 2.339 |

* **The per-relocation non-pool growth SATURATES.**  Non-pool goes 2.93 →
  3.22 GiB across four times the run and five times the relocations.  Not
  a leak, and not the blocker it looked like.
* **The gap over Grell-Freitas WIDENS with run length**: +0.58 GiB at 30
  minutes, **+2.17 GiB at two hours**.
* **The larger half is now the pool** — +1.30 GiB of the 2.17 — which is
  the term the 30-minute run appeared to eliminate.  It did not; a
  30-minute run is simply too short to show it.

The device peak is not a monotone climb but a series of **excursions**:
steady state near 11.0 GiB with spikes to 12.08 and then 13.31.

### What this means for Phase 5, stated plainly

13.310 GiB of 15.92 leaves 2.6 GiB of margin **at two hours**, and Phase 5
is a twelve-hour the reference tropical cyclone forecast at `f012`.  The gap is still widening at
the point the probe stops looking.  On this evidence a Phase 5 run of New
Tiedtke is **at risk of exhausting the card**, and that has to be resolved
before the run rather than discovered during it — which is the same
argument that put arm 2 into the acceptance condition.

It also retires a claim made three messages ago.  "Not the pool" was true
of the 30-minute run and false in general, and the honest form is:
**the reservation is eliminated by direct measurement; the pool is not
eliminated at all.**

---

## The 4-hour pair: it saturates, and the "widening gap" was a two-point line

review's first instruction was to read `pool_total` as a **series**
before running anything longer, since two points cannot distinguish a
saturating curve from a linear one.  That was right, and the series turned
out not to settle it either: `pool_total` climbs as a **staircase** and was
still stepping at t=129 s of a 158 s run — in **both** schemes.  So the
cheap analysis was worth doing and did not answer the question.

Two 4-hour runs did.

| | 30 min | 2 h | 4 h | 2h → 4h |
|---|---|---|---|---|
| NT device peak | 10.937 | 13.310 | **13.499** | **+0.19** |
| GF device peak | 10.492 | 11.138 | **12.019** | +0.88 |
| NT `pool_total` | 8.009 | 10.095 | 10.278 | +0.18 |
| GF `pool_total` | 8.166 | 8.800 | 9.668 | +0.87 |
| **gap** | +0.45 | **+2.17** | **+1.48** | — |

**New Tiedtke saturates.**  Doubling the forecast from two hours to four
adds 0.19 GiB, against 2.37 GiB for the previous doubling.  It is not a
leak and it does not run away.

**And the gap NARROWS between 2 and 4 hours**, because NT has flattened
and Grell-Freitas has not — GF is still climbing at +0.88 GiB per
doubling where NT is at +0.19.  The two schemes are converging, not
diverging.

### The claim this retires is mine, from two messages ago

> "The gap over Grell-Freitas WIDENS with run length: +0.58 GiB at 30
> minutes, +2.17 GiB at two hours."

That was a line drawn through two points, and it is the same error the
probe exists to prevent — one scale up from the maxima-subtraction it
replaced.  The 30-minute and 2-hour numbers are both correct; the
**extrapolation** was not, and at four hours the trend reverses.  Stated
plainly: **New Tiedtke is not on a path to exhaust the card.**

### What does NOT transfer, and it is the thing that matters for Phase 5

These probes are **2-domain**.  Phase 5's the reference tropical cyclone forecast is a
**3-domain** tree.  So what has been established is the *shape* — New
Tiedtke's VRAM saturates within about two forecast hours rather than
growing with run length — and **not the level**.  13.499 GiB of 15.92
leaves 2.4 GiB here, and that headroom figure does not carry to a
configuration with another domain in it.

The residual anchor is also weaker than it looked: GF's ~11.3 GiB recorded
operating peak is the **3-domain profiling tree**, and GF measures 12.019
on this 2-domain 4-hour probe — above it.  The two are not the same
quantity and should not be compared.

**So Phase 5 is not blocked on a runaway, and the go/no-go is a headroom
measurement on Phase 5's own domain configuration** — which is a short run
of the real tree, not another extrapolation from this one.  That is the
same conclusion arm 2 reached by a different route: measure the thing you
are actually going to run.

### What remains open

The mechanism is still unidentified, and two candidate correlations were
tested and **failed against their control**:

* **Relocations.**  On the NT run alone it looked exact — 5 `relocated`
  events, 5 excursion clusters, and the 14 `held` events producing none.
  The GF control has **3** relocations and **4** clusters, so the
  correspondence is not established.
* **History writes.**  11 d01 writes against 4–5 clusters, with offsets of
  ±3 to 7 s.  No alignment.

Both are recorded because the first one looked clean, and a correlation
that survives only the case it was found in is the thing this port keeps
learning to distrust.

---

## The 14-hour run: it does not saturate, and I made the error I had just written up

The run **completed** — 14 forecast hours, 3360 steps, 212 frames, SUCCESS
— and the completion is the least interesting thing about it.

| run | scheme | forecast h | peak GiB | wall s | forecast-s per wall-s | status |
|---|---|---|---|---|---|---|
| `run_kf` | KF | 14.0 | 8.902 | 956 | 52.7 | complete |
| **`nt_14h`** | **NT** | **14.0** | **15.920** | **4418** | **11.4** | complete |
| `run_myj` | GF | 17.2 | 15.920 | 2107 | 29.4 | RUNNING |

Card total: **15.920 GiB**.

### It pins the card at two forecast hours and stays there

`device_used` first reaches 15.920 GiB at **t = 614 s, 13.9% of the way
in**, and does not come back down for the remaining 86% of the run.

It completed anyway because **WDDM pages GPU memory to host RAM**.  The
proof is in the pool counter: `pool_total` **exceeds the physical card**,
peaking at **19.831 GiB** with 19,363 samples above 15.92.  A pool cannot
hold more than the device unless the device is not really holding it.

And the cost of paging is the wall clock.  **At 30 minutes New Tiedtke is
the fastest of the three schemes** (49.4 s against KF 51.9 and GF 58.8).
**At 14 hours it is 4.6× slower than Kain-Fritsch.**  A scheme does not
become 4.6× more expensive with run length for physics reasons; that
inversion is the signature of memory pressure, and it is the clearest
evidence that the peak is real rather than an artifact of how it is
sampled.

### "It saturates" was wrong, and §39 says why

Two entries ago this file recorded that New Tiedtke plateaus: 2 h → 4 h
added 0.19 GiB against 2.37 for the previous doubling.  At 14 hours it is
at the card.  **The 4-hour reading was another tread of the staircase.**

§39 of `docs/ntiedtke/PORT-RECORD.md` states this failure mode in as many words — *"a
staircase whose treads are long relative to the observation window is
exactly as ambiguous as two endpoints; the last flat stretch looks like
saturation and is only the next tread"* — and I wrote it, then concluded
saturation from the two points either side of one tread. Writing a trap
down does not stop one walking into it, and this is the third time in this
investigation that an inference outran its instrument.

### The maxima-subtraction rehabilitation also fails at this scale

This file recorded that `pool_total` is effectively monotone, so
subtracting independent maxima happens to be valid for that pair —
measured 2.927 both ways at 30 minutes.  At 14 hours the same subtraction
gives **−3.911 GiB** against a true 3.320, because `device_used` is
clamped at the card while `pool_total` keeps climbing past it.  A negative
non-pool term is nonsense on its face, which is the useful part: the
arithmetic announces its own failure here, and at 30 minutes it did not.

**The rule stands unconditionally after all: never subtract independent
maxima.**  The rehabilitation was a coincidence of the short run.

### What it means for Phase 5

`cu_physics = 16` **cannot run Phase 5 on this card as it stands.**  Not
"at risk" — measured.  It reaches the ceiling at two forecast hours of a
twelve-hour requirement and finishes only by paging, at 4.6× the wall
cost.

But the finding is **not specific to New Tiedtke**, and that is the part
that matters beyond this port: `run_myj` shows **Grell-Freitas at the same
15.920 GiB ceiling**, and its 29.4 forecast-s per wall-s against KF's 52.7
is the same paging signature at a smaller multiple.  On this 2-domain
the reference tropical cyclone tree, **both cumulus schemes exceed the card at length and only
Kain-Fritsch fits.**  New Tiedtke is worse, and Grell-Freitas — the
baseline every intensity number in `TC-INTENSITY.md` is graded against —
is already over.

That reframes the work: the target is not "make New Tiedtke fit where
Grell-Freitas fits", because Grell-Freitas does not fit either.

---

## The control that fits: Kain-Fritsch stops growing at 12% of the run

Raised by review, and it is the observation that reframes the search:
**every VRAM comparison so far has been New Tiedtke against Grell-Freitas,
and both fail.**  Kain-Fritsch completes the same 14 forecast hours on the
same tree at 8.902 GiB — seven under the card, no paging, 52.7
forecast-s per wall-s.  That is not a third data point, it is a **working
control**, and the investigation had been diffing two failing
configurations against each other.

Run at four hours with the same paired probe, `pool_total` running max:

| fraction of wall | KF | GF | NT |
|---|---|---|---|
| 0.1 | 6.191 | 7.360 | 6.976 |
| 0.2 | **6.490** | 8.240 | 8.080 |
| 0.4 | 6.490 | 8.597 | 9.068 |
| 0.6 | 6.490 | 9.392 | 10.144 |
| 0.8 | 6.490 | 9.668 | 10.278 |
| 1.0 | 6.490 | 9.668 | 10.278 |

| | last step >150 MiB | of run | final pool |
|---|---|---|---|
| **KF** | t = 34.8 s | **12%** | 6.490 GiB |
| GF | t = 196.6 s | 68% | 9.668 GiB |
| NT | t = 130.2 s | 43% | 10.278 GiB |

**Kain-Fritsch reaches its final pool in the first eighth of the run and
never grows again.**  By §40's criterion its window spans roughly eight
tread-periods, so "KF saturates" is a claim that is actually entitled to
be made — which is what makes it a control rather than another endpoint.

### Two explanations eliminated, for free

* **Cadence.**  The obvious difference — KF holds for `cudt_minutes`
  while GF and New Tiedtke run every step — is not present here.  All
  three baseline configs carry `cudt_minutes = 0.0`, so all three run the
  scheme on every model step.  Checked before it cost a run.
* **The static workspace term.**  Preflight prices KF 526.6 MiB, New
  Tiedtke 1052.0, Grell-Freitas 1464.3 — which does **not** order like the
  peaks (KF 8.73, GF 10.36, NT 10.94).  Grell-Freitas has the largest
  workspace and the *smaller* peak of the two failing schemes, so a fixed
  per-scheme allocation is not the explanation.

### A candidate, named as a candidate

Both failing adapters allocate fresh device arrays on **every** cumulus
call — 7 allocation sites in `NewTiedtke.__call__` and 13 in
`GrellFreitas.__call__`, several inside the chunk loop — where
Kain-Fritsch's native path returns `_NativeKFCumulusResult`, which
`physics.py:2395-2408` treats as reusable launch output rather than
copying.

That difference has the right **shape**: it follows scheme membership
rather than scheme arithmetic, and it produces growth *over time* rather
than a fixed cost, which is what distinguishes a staircase from a step.

**It is not established, and I am not going to assert it.**  Three
mechanisms have been proposed in this investigation and two came back
wrong; the one that looked cleanest — five relocations, five excursion
clusters, fourteen `held` events producing none — died against its GF
control.  Reading allocation sites out of source is the same class of
argument.

**The test it wants** is an ablation, not more reading: give
`NewTiedtke.__call__` persistent output buffers, so the per-call
allocation count goes to zero, and re-run the 4-hour probe.  If the
staircase flattens to KF's shape the candidate holds; if it does not, the
mechanism is elsewhere and one more explanation is eliminated.  A
microbenchmark will not settle it — CuPy reuses same-size blocks happily,
so an isolated allocation loop would show flat and prove nothing, which is
this campaign's own 424-MiB lesson.
