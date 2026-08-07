# Tornado-LES attempt #2 — what changed, why, and what it is graded on

**Registered before the run.** Config
`configs/les_tornado_100m_mayfield_20211210_attempt2.toml`, lane
`lane/les-tornado-attempt1`. Placement receipt
`docs/les/attempt2/corridor-placement.json`.

Attempt #1's screens in `docs/les/ATTEMPT1-EXPECTATIONS.md` still define
**what** is measured. This document defines **where**, and records the
evidence for moving it. Every screen is committed whatever it reads.

---

## 1. Attempt #1 did not fail. It measured that the box was misplaced.

Both arms of attempt #1's dual-run pair integrated 21600 s clean and the
byte comparison adjudicated **PASS, bit-identical** — 112 wrfout files
compared, 112 identical; 5053 final-state digests compared, 0 differing
(`scratchpad les-attempt1/receipts/dual_run_screen-adjudicated.txt`).

Its d04 was centred on 36.72 N 88.70 W, about 6 km southwest of Mayfield,
because that is where the **historical** storm went. Its own certified d02
output says the **model's** storm went somewhere else.

2–5 km updraft helicity maxima from arm a's hourly d02 frames, m² s⁻²
(`scratchpad les-attempt1/uh_track_arm_a.py`, re-run 2026-08-06 against
the certified output):

| frame | strongest cells (UH @ position) |
|---|---|
| 00Z | none ≥ 100 |
| 01Z | 268 @ 36.323 N 88.853 W · 240 @ 35.044 N 88.338 W · 165 @ 36.085 N 90.808 W |
| 02Z | **877 @ 36.461 N 90.046 W** · 819 @ 35.515 N 87.752 W · 463 @ 38.573 N 91.112 W |
| 03Z | 710 @ 35.757 N 90.720 W · **691 @ 36.897 N 89.185 W** · 518 @ 36.202 N 90.290 W |
| 04Z | 614 @ 36.117 N 89.791 W · 529 @ 37.371 N 87.947 W · 506 @ 37.117 N 88.545 W · **494 @ 36.439 N 89.398 W** |
| 05Z | 747 @ 35.118 N 90.715 W · **610 @ 36.303 N 89.204 W** · **594 @ 36.597 N 88.876 W** |
| 06Z | 565 @ 37.830 N 86.636 W · **357 @ 36.706 N 88.375 W** · 321 @ 35.920 N 89.236 W |

Bold entries fall in the ruled corridor (36.25–36.95 N, 89.45–88.85 W).
They describe a southwest-to-northeast train of violent cells **40–90 km
southwest of attempt #1's d04 and one to two hours later than history**,
with the strongest corridor crossings between 04Z and 06Z.

Attempt #1's d04 contained **none** of them. It contained Mayfield, which
the model's storm did not hit.

---

## 2. The new placement

Derived by `scratchpad les-attempt2/solve_placement.py`, which registers
the domains through `ProjectedGrid.nest` — the same call the runner makes
(`gpuwm/static/projection.py:592-595`) — rather than re-deriving the
geometry. The receipt was cross-checked against the config as loaded by
`gpuwm.experiment.load_experiment`: centres and starts MATCH.

| domain | attempt #1 start | attempt #2 start | attempt #2 centre | extent |
|---|---|---|---|---|
| d01 3 km 306×244 | (1, 1) | **unchanged** | 36.7200 N 88.7000 W | 33.270–39.941 N, 94.163–83.237 W |
| d02 1 km 450×450 | (79, 48) | **unchanged** | 36.7200 N 88.7000 W | 34.639–38.746 N, 91.334–86.066 W |
| d03 500 m 300×300 | (151, 151) | **(111, 127)** | 36.4999 N 89.1545 W | 35.809–37.185 N, 90.018–88.299 W |
| d04 100 m 200×200 | (131, 131) | **(132, 120)** | 36.4497 N 89.1485 W | 36.358–36.541 N, 89.263–89.035 W |

d04's centre is the ruled 36.45 N 89.15 W to within 0.14 km; d03's is the
ruled 36.50 N 89.15 W to within 0.40 km. **No shift was needed for
containment** — the ruled centres are legal as ruled.

### Containment proof

`gpuwm/experiment.py:1604-1632` requires, per axis, with
`span = nx // parent_grid_ratio`:

```
i_parent_start - 1                       >= spec_bdy_width + blend_width
parent_nx - (i_parent_start + span - 1)  >= spec_bdy_width + blend_width
```

For this experiment that is **10 parent rows on every side** (5 + 5).

| child in parent | W | E | S | N | required | verdict |
|---|---|---|---|---|---|---|
| d02 in d01 | 78 | 78 | 47 | 47 | 10 | PASS |
| **d03 in d02** | **110** | **190** | **126** | **174** | 10 | **PASS** (110–190 km) |
| **d04 in d03** | **131** | **129** | **119** | **141** | 10 | **PASS** (59.5–70.5 km) |

d04 ⊂ d03 ⊂ d02 ⊂ d01, transitively, with the nearest edge 59.5 km inside
its parent. The margins are enormous because the corridor sits well inside
d02; the ruling's worry that containment might force a shift did not
materialise.

### What the new box holds, and what it gives up

| landmark | inside |
|---|---|
| every bold corridor cell above | d01, d02, **d03** |
| Cayce KY (36.508 N 89.039 W) | d01, d02, d03, **d04** |
| attempt #1's d04 centre (36.720 N 88.700 W) | d01, d02, **d03** |
| Mayfield KY (36.742 N 88.637 W) | d01, d02, **d03** |
| initiation, NE Arkansas (35.891 N 90.344 W) | d01, d02 only |
| Princeton KY (37.109 N 87.881 W) | d01, d02 only |
| Bremen KY (37.362 N 87.231 W) | d01, d02 only |

d03 still holds the historical track's Mayfield segment *and* the model
corridor, so the two can be compared on one 500 m grid. Princeton and
Bremen, which attempt #1's d03 held, now fall outside it. **That is the
cost of following the model instead of the history and it is deliberate.**
Initiation stays outside d03 in both attempts, so genesis remains a
parent's job and nothing here grades it.

---

## 3. The route taken: full re-run from 00Z, not a restart

The ruling asked for a restart from attempt #1's 02:00 checkpoint set if
the machinery allowed it, a d01+d02 restart with cold-started d03/d04 if
not, and a full re-run only as a last resort. **The full re-run is the only
supported route.** Evidence, all from the lane worktree:

**A restart cannot tolerate a re-placed nest — three independent refusals.**

1. *Preflight, before the card is touched.* `preflight_prepared_tree`
   compares each domain's prepared-cache identity against the live
   experiment (`gpuwm/prepared_domain_tree_forecast.py:992-1003`).
   `prepared_domain_config_identity` is `asdict(DomainConfig)`
   (`gpuwm/ingest/prepared_cache.py:115-123`), which **includes
   `i_parent_start`**; the only tolerated field is `start_time`. Exit 2.
2. *The tree fingerprint.* `gpuwm/io/restart.py:2674-2677` refuses unless
   `header["experiment_fingerprint"]` matches. That fingerprint's
   components (`gpuwm/prepared_domain_tree_forecast.py:527-561`) include
   `experiment_identity` (the whole `ExperimentConfig`, `DomainConfig`s
   included), `execution_plan` (whose `_domain_rows` carries
   `i_parent_start`/`j_parent_start` explicitly, `:482-506`), and
   `domain_cache_content_sha256`. The fingerprint is **tree-wide**, so
   re-placing d03 also blocks resuming d01.
3. *The setup fingerprint.* Even with the above bypassed,
   `gpuwm/io/restart.py:3202-3206` refuses on the base-state/map-factor
   digest, which moves with the nest (`ht`, `msft`, `f`, `sina`, `cosa`
   and the derived `thb/pb/alb/phb/mub2d`,
   `gpuwm/state_serialization_contract.py:56-61`).

The published tolerance is explicit
(`gpuwm/io/restart.py:2473-2504`): *"A restart may change the forecast
length and the output/restart cadence; everything else — geometry,
timestep, physics, nesting, prepared inputs — must be the run that wrote
the checkpoint."*

**A subset restart does not exist.** `_tree_restart_paths`
(`gpuwm/io/restart.py:2604-2625`) refuses a partial domain set by name;
`domain_ids` equality (`:2688-2690`) and a single `checkpoint_set_id`
(`:2762-2764`) reinforce it; the restore loop applies every node
unconditionally (`:2778-2793`). There is no per-domain `restart` flag and
no `--restart-domains`.

**Mid-run cold-start of a nest from its parent is not on this route.** The
primitive exists — `nest_init.parent_only_init`
(`gpuwm/ingest/nest_init.py:1065-1204`) SINTs a child's full prognostic
state from the parent's *live* state — but its only consumer is
`gpuwm/da/nested_forecast.py:610`, which is marked EXPERIMENTAL, is not on
a default route, and documents that it is **not a mid-run introduction**:
it rebuilds a whole model per leg. The delayed-nest-start hook
(`gpuwm/core/model.py:631-673`) cold-starts from the *input catalog*, not
from the parent, and the prepared-tree runner cannot reach it at all —
`run_prepared_tree` sets `model._input_catalog = None`
(`gpuwm/prepared_domain_tree_forecast.py:1441`) and never sets
`_activation_context`, so a delayed-start nest on this route would raise
`AttributeError` at `gpuwm/core/model.py:642`.

**Cost of the chosen route:** one extra sim-hour per arm over a 02Z
restart, plus the 00Z–02Z re-integration. At attempt #1's measured
3881 s / 4262 s per 6 sim-h arm, ~11 min per sim-hour, a 7 sim-h arm is
roughly 75–85 min and the pair roughly 2.6–2.9 h. Accepted.

**What is reused, and what is rebuilt.** d01 and d02 did not move, so the
HRRR source files attempt #1 fetched (18 grib2, 9.6 GB, manifest
`318d2ba0…`) serve both attempts unchanged and nothing is refetched. The
route inputs, root preparation and hierarchy preparation **are** rebuilt,
because `i_parent_start` reaches the emitted namelists and the per-domain
prepared caches. That is ~5 minutes of CPU, off the card.

---

## 4. The other change: d02's cadence

`history_interval_s` on d02 goes **3600 s → 900 s**.

This is the instrument fix, and it is the reason the placement above
carries a caveat. An hourly frame of a 13–23 m s⁻¹ storm is a 50–90 km
sample interval — **wider than the 20 km box it is used to place**. So:

* consecutive hourly maxima cannot be attributed to a cell without
  guessing (the 04Z→05Z pairing has two plausible readings, implying
  13.9 and 15.7 m s⁻¹ respectively), and
* the storm can cross d04 entirely between two samples, which is exactly
  what the table in §1 shows: **no hourly maximum lands inside the new
  d04 either**, and none could be expected to.

Straight-line interpolation between the 04Z and 05Z maxima of the
strongest corridor cell puts it inside d04 from about **04:16Z to 04:39Z,
a 19.3 km chord, leaving through the north edge**. That is the crudest
possible track model. It was **not** used to move the box — the centre is
held at the ruled 36.45 N 89.15 W rather than nudged ~7 km north onto the
chord's midpoint, because optimising against a track sampled more coarsely
than the box is fitting to interpolation noise.

At 900 s, d02 samples the storm every 6–14 km. That is what makes the next
placement a measurement instead of an inference.

Everything else is unchanged: d01 3600 s, d03 900 s, d04 300 s,
`restart_interval_s` 3600 s.

---

## 5. Registered expectations — committed whatever they read

**E2.1 — the corridor reproduces.** d02's 15-minute frames should show a
violent cell (UH ≥ 300 m² s⁻²) traversing the corridor between 03Z and
06Z. If attempt #2's d01/d02 evolve differently from attempt #1's, the
placement derivation is invalidated at the root and that is the finding.
d01 and d02 are geometrically identical to attempt #1 and start from the
same analysis, but the window is longer and d02's write cadence differs,
so **bit-identity with attempt #1's parents is not expected and is not
claimed**.

**E2.2 — does the cell enter d04 at all.** The pass/fail this attempt
exists to answer. Graded from d04's 300 s frames: peak 10 m AGL wind, peak
vertical vorticity, and whether any of it is interior rather than on the
inflow face. A miss is a FINDING about fixed-box downscaling of a
50–90 km-sampled track, not a failure of the model.

**E2.3 — the interpolated transit window.** The hourly model predicts
04:16Z–04:39Z, 19.3 km, exiting north. Graded against what the 15-minute
d02 frames and the d04 output actually show. This is the falsifiable claim
of the placement derivation.

**E2.4 — spin-up fetch, carried from attempt #1's screen E1.** The storm
enters d03's west edge and gets ~66 km before reaching d04 — comparable to
attempt #1's 50–75 km, so the risk registered there is unchanged. The D90
fetch meter runs on d04's output and the scored footprint is the
fetch-clean interior only. If D90 approaches 20 km there is no
fetch-clean interior and the tornado-scale screens are unscoreable on this
geometry, which was registered as a live risk before attempt #1 and
remains one.

**E2.5 — the inflow ruling holds.** `inflow_perturbation` ON at d03, OFF
at d04, unchanged from e754bc77. If d04's D90 shows the resolved eddies do
NOT arrive developed from its LES parent, that is what revisits the
ruling.

**E2.6 — dual-run byte identity.** The pair must compare bit-identical
under the patched `dual_run_screen.py` (which excludes `path`, `samples`,
`last_checkpoint` and `timing_seconds` as run-local metadata). A mismatch
is a corruption finding on a card with no ECC and stops the claim.

**E2.7 — VRAM.** The shape is identical to attempt #1's in every dimension
that costs memory, so the measured device peak should reproduce attempt
#1's 15.91 GiB against the 28.4 GiB budget. The G5 smoke measures it at
the exact shape and is the GO/NO-GO.

---

## 6. Known defect in a neighbouring instrument — not used here

`gpuwm/verify/cases/les_tornado_mayfield_20211210.py:130-145`
(`domain_grids`) builds every domain centred on the projection reference
and **never reads `i_parent_start`**. For attempt #1 that was harmless
because attempt #1's domains genuinely were concentric. It is wrong for
any off-centre placement, so **nothing in this document is derived from
that module**; the placement receipt uses
`ProjectedGrid.nest` directly, as the runner does.

That module and its test remain bound to attempt #1's config by
`CONFIG = configs/les_tornado_100m_mayfield_20211210.toml`, so they are
unaffected by this attempt and still pass. Fixing `domain_grids` to honour
the real nest registration is a separate, safe change and is **not** made
here, so that attempt #2's launch does not perturb attempt #1's passing
audit on the same night.

---

## 7. Revision b — the lid, raised, and what it cost

**Added after revision a ran and tripped.** Revision a is not withdrawn:
its arm a is the record of both the crash and the science that motivated
this change.

### What happened

Arm a integrated cleanly to ~04:51 sim and then tripped the health gate:

```
post-step.d04: w(64, 159, 80) = -215.017 m/s  violates lower bound >= -200
```

d04, z = 10,843 m, **interior** — 80 cells from the nearest edge.
Reproduced deterministically from the clean 04:00 checkpoint with the
per-step gates on. The cell held **w = +0.01 m/s one frame earlier**, so
this is a numerical explosion out of still air, not a physical evolution.

The production gate had reported something else entirely — a NaN in
`nest.scratch.nest_child_field` at flat index 201 during
`post-d01-sync.d02`. That index is *deterministically* the lowest flat
index in d04 the specified zone does not overwrite (j=0 is south-specified,
(j=1,i=0) is west-specified, (j=1,i=1) is the first cell that is neither),
so it carries **no information about origin** — it is simply the
lowest-indexed survivor of a contaminated field, reported against d02's
validator because every domain's view of that shared arena starts at
element 0.

### The mechanism

The old lid sat at 16,296 m, so the top-5 km Rayleigh sponge began at
**11,296 m — below the anvil of a violent supercell**. Measured:

| time | max\|w\| inside the sponge | max\|w\| at k=64 |
|---|---|---|
| 04:20 | 5.67 | 10.46 |
| 04:30 | 8.46 | 15.75 |
| 04:40 | 10.44 | 15.54 |
| 04:45 | 9.38 | 17.21 |
| **04:50** | **17.36** | **21.01** |

The updraft was still 21 m/s at the failing level and 17.4 m/s *inside*
the sponge, with in-sponge energy nearly doubling in the five minutes
before the trip. A Rayleigh sponge absorbs gravity waves in quiescent air;
it is not built to have live convection driven through it. The failing
level sits 452 m below the sponge base, in a band protected by **neither**
the sponge nor `w_damp` — the latter is a *Courant* limiter and at
dz = 609 m, dt = 0.5 s would need w ≈ 1218 m/s to fire.

Eliminated first, by measurement or by reading shipped code: horizontal
and vertical CFL (0.42 / 0.21 on d04, a 2.2× margin at the vortex peak),
vertical stretching (1.035–1.058, well under 1.2), FP32 subnormal
flush-to-zero (nothing at the cell within 33 orders of magnitude of the
subnormal range), the lateral boundary (rim second-differences 6× *smoother*
than the interior), the relaxation coefficients (dt-invariant across all
four domains, 10× margin), and any concentric-nest assumption on the run
path (none exists).

### The trade, priced

`p_top` 10000 → **5000 Pa**, `nz` unchanged at **72**. Ladder regenerated by
`tools/build_stretched_eta_ladder.py --nz 72 --dz0 20 --dz-max 650
--p-top 5000 --bl-top 1700 --require-below 30`; receipt
`docs/les/attempt2/eta72-raisedlid.json`.

**Every ratified vertical bound was re-verified at the new lid rather than
assumed, and every one still passes:**

| | revision a | revision b | bound | verdict |
|---|---|---|---|---|
| levels below 1.7 km | 33 | **31** | floor ≥ 30 | PASS |
| levels below 2.0 km | 35 | **33** | target 33–38 | PASS |
| effective BL dz | 51.52 m | **54.84 m** | target 50–60 m | PASS |
| first half level | 10.00 m | 10.00 m | — | PASS |
| top half level | 15,326 m | **18,519 m** | — | — |

So the cost is **two boundary-layer levels and 3.3 m of effective BL dz**,
bought against a model top a violent supercell cannot punch through. This
is a configuration moving *within* the ratified bounds, not a change to
them — which is why it did not require a new ruling. An earlier estimate
that the raised lid would need nz ≈ 78–79 and brush WSM6's 80-level
ceiling was **wrong**, and is corrected here.

### E2.8 — registered before revision b runs

The raised lid must carry the full window with no bound violation. A trip
would mean the lid height is not the whole story and the hunt resumes with
one hypothesis dead. **Note what the diagnostic that motivated this cannot
prove:** a changed `p_top` cannot restart from revision a's checkpoints
(the restart tolerates only run length and cadence), so it is a full 00Z
re-run, and a different vertical grid is a **different realization** — after
nearly five hours of chaotic convection it is not the same storm. Passing
04:51 is *supporting evidence*, not proof, and a trip would not by itself
exonerate the lid.
