# Moving nests: what shipped, and the ruling that closed it

*Lane: `lane/moving-nest-regrid`, branched from `feature/da-scorecard` @
`ff5898c8f` — the same base the continental-cycling architecture branched
from, so every line reference below resolves on both.*

*The two companion documents, `docs/continental-cycling-architecture.md`
and `docs/continental-cycling-build-plan.md`, live on
`lane/continental-cycling` @ `4bd624ad4`. That lane has not landed, so
they are NOT in this release and the references to them below are
deliberately not links. Read them with
`git show 4bd624ad4:docs/continental-cycling-architecture.md`.*

This is the first leg of the moving-nest program: **the mechanism**, built
and proven.

**Status: closed, and one answer has since moved.** The one open question
— what a restart across a move guarantees — was ruled by Drew on
2026-08-06: *nothing*. See §3. That answer removed work rather than
adding it: no identity site had to change, and Stage 6 turned out not to
be blocked by Stage 2 after all.

**Superseded in 2.5.4 for the guarantee only.** A restart across a move
now reproduces the run that wrote the checkpoint bit for bit, because the
three things a resume needs are carried across one: the placement and the
tracker's hysteresis in the relocation header, the acoustic Omega in
`CHECKPOINT_ONLY_STATE`, and the consultation window in
`restart.CARRIED_SCRATCH_SLOTS`. The identity findings in this document
are unchanged and are what made that possible — the fingerprint still
binds the placement, a move still invalidates it against a FRESH build,
and the resume is permitted by replaying the move chain rather than by
relaxing the gate. `gpuwm.core.nest_relocation.RESTART_ACROSS_MOVE_POSTURE`
is the current wording.

---

## 1. What shipped

A moving nest here is what the architecture ruled it is
(`continental-cycling-architecture.md` §4.2, see the note above):
**a sequence of static nests joined by a re-grid at cycle boundaries.**
WRF's continuous per-step motion is not adopted and its namelist keys stay
refused.

- `gpuwm/core/nest_relocation.py` — the primitive. `relocate_child(node,
  i_parent_start=, j_parent_start=)` rebuilds the child from the live
  parent at the new placement, stamps the overlap bitwise from the
  outgoing child, and rebuilds the SINT tables.
- `gpuwm/core/nest.py` — `NestCoupler.relocate()`, which rebuilds the
  three stagger registrations **once per placement generation**. No
  reallocation: extents and ratio do not change, so the audited `nest_*`
  manifest slots are identical and the new tables are re-uploaded into
  them.
- `gpuwm/experiment.py` — a gated `[relocation]` table, default off, every
  key refused while it is off. The WRF continuous-motion keys stay
  refused; their message now points at the discrete mechanism instead of
  saying "non-goal".
- `gpuwm/verify/cases/nest_relocate.py` — the proof, on the idealized
  ratio-3 tree.

The arithmetic that makes it exact: a placement is a whole number of
**parent** cells, so a move of `di` parent cells is a shift of `di * ratio`
**child** cells. Substituting into WRF's donor pickup
(`interp_fcn.F:975-985`) shows the donor index `ci` and sub-cell offset
`ip` are *identical* for an overlapped cell before and after. The overlap
is therefore a pure index-space copy, and anything the child derives from
its parent by SINT is bitwise unchanged on it.

### Measured, on the RTX 5090

Idealized WK82 ratio-3 tree: parent 168×168×60 at 1 km, child 120×120×60
at 333 m, moved 4 parent cells (12 child cells, 90 % overlap). One 60 s
leg before the move, one after.

| claim | result |
|---|---|
| null move (same placement) leaves the child bitwise identical | child state sha256 `94b62f4a…` before **and** after, 25 fields stamped |
| …and reproduces what it *rebuilds* rather than copies | 8 fields (SINT base state, map factors), 58,081 cells, **0** mismatches |
| …and re-establishes WRF `start_domain`'s RK post-condition | 16 fields, 13,017,600 cells, **0** mismatches |
| parent bitwise unchanged, both moves | parent sha256 `2ad8c25a…` unchanged across the null move and the real move |
| overlap equals the shifted outgoing child, bitwise | **0** mismatches over **18,714,960** cells across **25** restart-contract fields |
| donor alignment (SINT base state on the overlap) | 0 mismatches, 12,960 cells on `mub2d` |
| **treatment fired** (not two identical arms) | **10,497,036 of 18,714,960** overlap cells (56 %) differ from the cold start; `u` 98.5 %, `v` 98.6 %, `w` 96.3 %, `mup` 97.6 % |
| post-move integration | 40 steps / 10 forces, health validator armed, finite everywhere, boundary-blowup metric under the same frozen N2c threshold (0.5) |

The treatment row is the one that makes the rest mean anything. Without
it, "the overlap matches" is satisfied just as well by a transplant that
never ran. The nine fields showing 0 % change are the hydrometeor species,
which are still identically zero 60 s into a WK82 run — physics, not a
gap.

The second and third rows exist because the state digest alone is partly
circular: at zero shift the stamp covers exactly the fields the digest
covers. They were also the rows that **caught something**. The first
version of this check asserted that the RK time-t copies (`u0`, `v0`, …)
survive a null move unchanged, and they do not. Establishing which side
was wrong needed evidence rather than a guess:
`gpuwm/core/dycore.py:2230-2238` rewrites every `*0` array from its
current field at the top of `step()`, before anything reads it, so its
inter-step value is dead scratch — which is also exactly why
`STATE_SERIALIZED_ATTRS` excludes them and a checkpoint does not carry
them. The assertion was wrong, not the code, and the check now holds the
property that is actually defined (`*0 == current` after the move), which
still catches a reseed that never ran or one that ran before the stamp.

Both directions are proven: `tests/test_nest_relocation.py` includes
negative controls that must FAIL — a wrong-sign shift breaks the overlap
equality, a one-cell-wrong plan breaks donor alignment, and an off-parent
placement is refused by `register_nest`'s ±2 stencil rule.

---

## 2. What did NOT change, and why I stopped

Nest position is bound into identity in three places, and **none of them
was touched**:

| site | what it binds |
|---|---|
| `gpuwm/ingest/prepared_cache.py:115-123` | `prepared_domain_config_identity` is an unfiltered `asdict(DomainConfig)`, so it contains `i_parent_start`/`j_parent_start` |
| `gpuwm/prepared_domain_tree_forecast.py:482-506` | `_domain_rows` puts both fields in the experiment identity, which feeds `TREE_RESTART_IDENTITY_COMPONENTS` |
| `gpuwm/core/model.py` `restart_identity_payload` | the same fields again, through `tree_restart_identity_components` (`:527-561`) |

So today, **moving a nest one cell invalidates every prepared cache in the
tree and breaks the tree restart fingerprint.** `gpuwm/io/restart.py:2460`
states the rule being enforced: *"a checkpoint resumes only into the run
that wrote it."* Under the current definition of "the run", a relocated
tree is a different run.

Under the 2026-08-06 ruling (§3) **that is the correct and final
behaviour**, not a limitation to be lifted. Two pinned tests hold it —
`test_the_full_prepared_identity_still_differs_across_a_move` and
`test_a_move_invalidates_the_tree_restart_fingerprint` — so if anything
ever starts treating a relocation as resumable it shows up in a diff.

**One fingerprint change was necessary, and it is the conservative one.**
Adding `relocation` to `ExperimentConfig` put it inside
`restart_identity_payload` (`gpuwm/core/model.py:305`), which would have
moved the tree restart fingerprint of **every** experiment — including
ones with no nest at all — and refused every checkpoint written before
this lane existed. `relocation` is therefore in
`RESTART_TOLERATED_EXPERIMENT_FIELDS`, which restores the payload to
byte-for-byte what it was. This is not a relaxation: the bounds are
admissibility policy and reach no computed value, and a move is an
explicit recorded event rather than something the bounds derive. Three
tests hold it, including a control proving the fingerprint is still
sensitive to a real one-cell placement change.

What I built instead is placement bookkeeping — which, after the ruling,
is all it ever needed to be. It is **not** a reproducibility contract and
nothing consumes it as one:

- `Placement(grid_id, i_parent_start, j_parent_start, generation)` —
  architecture §5 Layer 2, verbatim.
- `placement_independent_identity(dc)` — architecture §5 Layer 1, computed
  by *subtracting* the placement fields from the existing prepared-cache
  identity document rather than by restating the field list, so the two
  cannot drift apart. Answers "are these the same domain in different
  places?", which is a comparability question, not a resume question.
- `RelocationSegment` — a base preparation digest plus an append-only
  chain of records, each naming its predecessor, proven non-commutative so
  it records a history and not a set. Its job is receipts and
  diagnostics: naming which moves produced the state in front of you. A
  matching `segment_id` is explicitly **not** licence to resume across a
  placement boundary, and the docstrings say so.

---

## 3. The ruling

**Context.** The build plan's Stage 2
(`continental-cycling-build-plan.md`, see the note above)
prescribes moving `i_parent_start`/`j_parent_start` out of Layer 1 and
dropping `domain_cache_content_sha256` from the tree restart fingerprint,
and lists Stage 6 (the moving nest) as **blocked by Stage 2**. I did not
implement that here, because shipping a restart-semantics change as a
side effect of a re-grid is exactly the quiet relaxation that should not
happen. I asked instead what a resume across a move should *guarantee*,
and offered three options.

### RULED — 2026-08-06, Drew

> *"a restart across a move promises nothing imo its a pure efficiency
> experiment"*

**Adopted semantics: a relocation invalidates any restart claim
outright.** Not "bit-exact within a segment", not "replayable from the
chain" — nothing. A moving nest is an efficiency experiment: its purpose
is that a low-VRAM card can run a *smaller* nest that follows the weather
at a resolution a static nest of the same cost could never reach. That
value does not depend on a resume story, and inventing one would have
bought a contract nobody needs at the price of a real constraint on every
future change.

I had recommended segment-scoped bit-exactness. The ruling is simpler and
better, and it retires the question rather than answering it.

What this means concretely:

- **No segment contract is promised.** `RelocationSegment` /
  `RelocationRecord` stay, but as **internal bookkeeping for receipts and
  diagnostics only** — "which moves produced the state in front of you".
  A matching `segment_id` is never licence to resume across a placement
  boundary. Their docstrings say so, so the next reader cannot mistake
  addressability for a guarantee.
- **The three binding identity sites stay exactly as they are.** The
  prepared-cache key and the tree restart fingerprint still bind the
  placement and still invalidate on a move. That is now the *intended
  behaviour*, not a blocker awaiting Stage 2. Pinned by
  `test_the_full_prepared_identity_still_differs_across_a_move` and
  `test_a_move_invalidates_the_tree_restart_fingerprint`.
- **Determinism within a placement is untouched and still holds** — as a
  plain consequence of the existing machinery, which this lane does not
  modify, not as anything relocation provides. The dual-run byte
  comparison keeps working on any run that does not move. We simply
  promise nothing across the boundary.
- **Stage 2 is decoupled from Stage 6.** The build plan lists Stage 6 as
  blocked by Stage 2 because the moving nest was assumed to need a
  relaxed identity. Under this ruling it does not: relocation works with
  the identity exactly as it is today. Stage 2 remains worth doing for
  the *cycling* system (an ensemble crossing prepared cases), but it is
  no longer on the moving nest's critical path.

The one thing this does forbid: nothing may quietly start treating a
relocation as resumable. If a later lane wants that, it is a new ruling,
not an implementation detail.

---

## 4. Parked, deliberately

- **FIRST ITEM OF LEG 2 — stage the transplant through host memory.**
  Promoted out of the memory note below by the ruling itself. If the
  target user is a **low-VRAM card**, then allocating the incoming child
  before releasing the outgoing one — briefly doubling child residency, up
  to ~16.3 GiB at 3 km — undercuts the entire value proposition at exactly
  the moment it is supposed to pay off. A card that cannot hold two
  children cannot move a nest, which would leave the feature working only
  for the users who least need it. Leg 2 should stage the overlap through
  host memory (or refill in place) so peak device usage stays ≈ one child.
  Not built now; recorded here as the thing leg 2 opens with.
- **The placement policy.** No tracker, no scoring, no hysteresis, no
  "when to move". Architecture §4.3 specifies it; this leg supplies only
  the bounds a caller's proposal is judged against
  (`max_move_parent_cells`, `min_overlap_fraction`).
- **Physics continuation state.** What a relocation carries is the restart
  layer's serialised-state contract. Accumulators, scheme timers and
  driver-held surface fields are re-initialised at the new placement —
  exactly as they already are at every leg boundary of the DA route
  (`tools/da_cycle_prepared.py:34-43`). A caller with its own continuation
  inventory can move it with a second `transplant_overlap` against the
  same plan. Stage 3's resident-state loop is where this stops being
  necessary.
- **Transient memory** — see the first item above, which this became. The
  doubling buys reuse of the certified cold-start path instead of a
  bespoke in-place refill, and it is a cycle-boundary cost rather than a
  per-step one. That trade was right for proving the mechanism and is
  wrong for the user the ruling names.
- **No gate-ledger row.** The verify case carries its own verdict rule
  (every component bitwise-zero) and imports the N2c boundary threshold
  rather than restating it. Adding a `NEST_GATES` row is a release-facing
  act; say the word and it lands with its pin.

## 5. Leg 2 (2026-08-06): what landed on lane/moving-nest-leg2

- **Host-staged transplant.** The transient doubling in §4 is closed:
  `relocate_child(staging="host")` snapshots the serialised state (plus
  the donor-alignment base fields) to pinned host buffers, RELEASES the
  outgoing device allocation, rebuilds at the new placement into the
  freed bytes, and uploads the overlap. Bitwise-identical to the
  in-device transplant (tests/test_nest_relocation_staging.py); the
  receipt's `staging` block samples the live pool around each phase so
  the peak claim is measured. The off-parent refusal moved BEFORE the
  release; failures after the release are documented as run-ending.
- **Runner.** `gpuwm.core.relocation_runner.RelocationRunner` consults a
  follow source at complete cycle boundaries (optionally gated by
  `cadence_seconds`) and executes admissible moves. The follow source is
  the storm tracker's published plan-provider seam
  (`gpuwm.core.storm_tracking`, adopted verbatim):
  `provider(parent_state, nest_footprint, t) -> (di, dj) | None`.
  `[[relocation.move]]` rows are the manual itinerary through the same
  contract. Every move writes a receipt (time, requested AND executed
  offsets, overlap fraction, fields transplanted, fill counts and
  provenance, staging samples) into the run receipts.
- **Restart across a move, marked and loud.** Every executed move chains
  its record into the live `experiment_fingerprint`
  (`mark_fingerprint_across_move`), so a post-move checkpoint refuses to
  resume into a fresh build BY CONSTRUCTION; the checkpoint header
  carries the promises-nothing posture (`RESTART_ACROSS_MOVE_POSTURE`)
  and both the refusal and any tolerated restore state it by name.
  Still no certification, per the §3 ruling.

## 6. Statics on relocation (2026-08-06 requirement) — NAMED FOLLOW-UP

Drew's requirement via the WRF moving-nest discussion: a relocated
nest's STATIC fields (terrain, landuse, soil categories) must be
REBUILT for the new footprint from the nest's own static source at nest
resolution (30s baseline or [static.highres], both footprint-parametric
and cached), not inherited parent-interpolated — over-land
storm-following at 2 km lives on resolved terrain.

What this leg ships (the agreed minimum, not the end state):

- the `initializer` parameter of `relocate_child` is the per-footprint
  rebuild seam; a custom initializer must state its `static_provenance`
  or the move refuses;
- the default (`parent_only_init`) records the explicit
  parent-interpolated fallback on every receipt
  (`PARENT_INTERPOLATED_STATICS_FALLBACK`) — never silent;
- the overlap claim is TESTED: footprint-parametric statics rebuilt at
  the new placement agree bitwise with the outgoing child's on shared
  ground, and the transplant never writes statics
  (test_footprint_rebuilt_statics_survive_and_match_on_the_overlap);
- the real-data routes (`gpuwm run`, prepared domain tree) REFUSE
  follow-source configs at the front door until the rebuild lands — a
  "moving" arm that integrates statically is a void treatment arm.

FOLLOW-UP (leg 3): a real-data relocation initializer that builds
own-source statics for the new footprint (build_static_for_domain is
already footprint-parametric), runs the t=0 nest cold-start terrain
adjustment on the parent-interpolated strip (reuse nest_init, do not
invent), donor-fills land-surface state on the strip with recorded
provenance, and rebuilds the physics driver against the new statics.
That is what lifts the two front-door refusals and lets the 2011-04-27
demo's moving arm actually move.

## 7. Leg 3 (2026-08-06): real-data relocation, on lane/moving-nest-leg3

The §6 follow-up, built complete:

- **Footprint-rebuilt statics, with the overlap equality ENFORCED.**
  `gpuwm.ingest.relocation_init.real_relocation_initializer` invokes
  `build_static_for_domain` for every new footprint (plus the
  `[static.highres]` overlay when the case enables it).  The footprint's
  grid is `ProjectedGrid.translated` -- the reference grid with an exact
  integer index offset whose transforms DELEGATE to the reference (and
  whose float32 WPS sampling twin delegates likewise,
  `gpuwm/static/build.py:_TranslatedWps32`) -- which is what makes
  "identical source + identical cells = identical bytes" hold
  unconditionally rather than modulo pole re-rounding.  Drew's design
  ruling is asserted twice: the route preparer refuses any move whose
  rebuilt statics differ bitwise from the outgoing child's on shared
  ground, and `test_overlap_statics_equality_on_the_real_static_source`
  proves the equality against the real 30-arc-second WPS_GEOG tree.
- **Fresh-strip atmosphere by the t=0 machinery.**  Full-parent SINT
  (`parent_only_init`, now grid-overridable), then the exact t=0 child
  sequence reused verbatim: analytic fine base (`_make_real_base`),
  three-operand `blend_terrain` against `_capture_parent_blend_fields`,
  and `_adjust_and_rederive` (adjust_tempqv + start_domain re-derivation
  + press_adj).  On the doubly-interior overlap this reproduces the t=0
  base bitwise, so `donor_alignment_check` stays armed there -- scoped
  by the initializer's declared `donor_alignment_frame_width`
  (spec_bdy_width + blend_width), where the invariant is genuinely
  undefined.  Inside the frames the transplanted perturbations are
  REBASED (`rebase_transplanted_perturbations`, the initializer's
  `post_transplant` hook): totals preserved on exactly the cells whose
  base bytes differ, bitwise stamp untouched everywhere else, EOS
  diagnostics re-derived after.
- **Land state by donor fill, per leg 1's contract.**
  `RealRelocationChildPreparer` snapshots the driver-held land-surface
  continuation fields before the release (`capture_outgoing`), moves the
  overlap by index-space transplant, fills the strip from
  nearest same-landmask-class donors (`donor_fill_plan`; class fallback
  COUNTED), and rebuilds the physics driver against the new statics
  through the same `initialize_landuse`/`initialize_physics`/radiation
  wiring the t=0 preparer uses.  Accumulators re-initialise, stated on
  the receipt.  Post-move, wrfout metadata/global attributes refresh so
  frames describe the footprint that produced them.
- **Refusals: lifted where the machinery exists, kept where it cannot.**
  `gpuwm run` wires `gpuwm.runtime.build_real_relocation_runner` and no
  longer refuses follow-source configs (a single-domain follow config
  still refuses -- no nest to move).  The prepared domain-tree route
  kept its refusal at this leg: a prepared tree runs without the case's
  GEOG source, so no footprint could be rebuilt there.  §8 lifts it for
  bundles prepared WITH a statics corridor; corridor-less bundles still
  refuse, with the remedy named.  `execute_experiment` still refuses a
  follow source on any route that wired no runner.
- Receipts per move now carry: static source and rebuild timings, the
  placement translation, the terrain adjustment applied, the overlap-
  statics verdict, donor-fill provenance counts, rebase counts, and the
  driver rebuild timing.

Both 2011-04-27 demo arms preflight PASS on this tree (`gpuwm check`,
input catalog + memory).  16 GiB note: the affine estimates are 8.9 /
9.4 GiB (follow / static arm), but the conservative WDDM-floor peak
envelopes are 14.9 / 15.7 GiB -- above a 16 GiB card's usable budget --
so the first 4080 run should re-measure the 1.746x floor rather than
trust it across cards.

## 8. Leg 4 (2026-08-08): the statics corridor — prepared routes move

The §7 refusal on the prepared (GFS/ERA5/mapped tree) route existed
because relocation rebuilds a child's statics from the GEOG source and
a prepared tree deliberately runs without its ingest inputs.  The
corridor moves the rebuild to PREPARATION time, when the source IS on
hand, and keeps the run fully sealed (`gpuwm/static/corridor.py`):

- **Emission (opt-in).** `--statics-corridor` on the GFS tree
  preparation (`rw-wps --source gfs`, `python -m gpuwm.gfs_direct`;
  bare flag = every child domain, or comma-separated child grid ids)
  builds child-resolution statics over each child's WHOLE parent extent
  -- the chase cannot leave the parent -- through the same
  `build_static` the domain statics use, on the child reference grid
  `translated` and re-extented on the SAME lattice.  Sealed as
  `hierarchy-artifacts/statics-corridor/dNN.npz` (byte-deterministic
  writer) plus a receipt whose copy is embedded in `proof.json`, so the
  corridor digests are covered by `--preparation-receipt-sha256` like
  every other sealed artifact.  Flag absent: the bundle is byte-for-byte
  unchanged.  `gpuwm fetch --author-front-door-manifest` prints the
  flag, and the go/run-plan prepare stage passes it, whenever the
  experiment config declares a follow source (same predicate both
  sides, so the pasted and driven lines cannot drift).
- **Acceptance.** `prepared_domain_tree_forecast` accepts
  `[relocation]` follow/moves when the bundle carries a corridor
  covering the follow child: receipt-vs-proof equality, cache SHA-256,
  geometry-vs-experiment equality, and exact float64 grid-arithmetic
  probes are all verified at preflight; the corridor is re-hashed after
  the run.  ANY verification failure refuses loudly -- a follow config
  never degrades to a silently static nest.  Corridor-less bundles keep
  the §7 refusal, now naming `--statics-corridor` as the remedy.
- **Consumption is the §7 machinery, not a fork.**
  `real_relocation_initializer` grew a `statics_builder` seam (the one
  route-owned input); the corridor supplies a crop-backed builder and
  everything after the statics -- full-parent SINT, the t=0 terrain
  adjustment, blend-frame rebase, donor-fill land movement, driver
  rebuild (`rebuild_child_driver_from_land_state`, with the prepared
  route's native land-use identity and t=0 radiation wiring) -- is the
  same implementation, wired by
  `gpuwm.runtime.build_prepared_tree_relocation_runner` into the same
  `RelocationRunner`.  The overlap-statics bitwise assertion stays
  armed on every move.
- **Why a crop is exact.** The corridor grid delegates per-cell
  transforms to the child reference grid (float32 sampling twin
  included), the static build is per-cell on those coordinates, and the
  terrain smoother's dependency cone lies inside the shared halo -- so
  a footprint cropped from the corridor equals the direct footprint
  build BITWISE.  tests/test_statics_corridor.py proves it across
  placements on a synthetic WPS_GEOG tree spanning the build's gcell /
  categorical-count / interpolation / smoother paths, validates the
  instrument with a planted one-ULP perturbation, and re-proves it
  against the real 30-arc-second tree when the case bundle is staged.
- **Cost, stated where it is paid.** The corridor is parent-extent at
  child resolution: 97 float64 planes = 776 bytes per corridor cell.
  A 3 km parent of 300x300 cells at ratio 3 is a 900x900 corridor
  (~629 MB on disk, same when loaded to host at run time); 450x450
  cells at ratio 3 is ~1.41 GB.  The preparation prints the numbers per
  corridor, the run receipt carries `statics_corridor_host_bytes`, and
  it is HOST memory only -- crops happen on the CPU and a rebuilt child
  re-occupies the same device footprint, so the GPU preflight estimate
  is unchanged.
- HRRR trees inherit all of this when their preparation emits the same
  sealed artifact: the acceptance, verification and consumption live at
  the source-agnostic tree-runner/hierarchy seam.
