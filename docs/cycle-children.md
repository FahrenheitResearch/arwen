# Cycle children: arbitrary geographic spawn, across cycles

This is the child half of the cycling spine. It answers one question --
*"put a child domain on that storm, keep it there while the storm lives,
and take it away when the storm dies"* -- and it answers it in a way that
works on any parent, at any projection, without a case name anywhere in
the mechanism.

Modules: `gpuwm/cycle/placement.py`, `gpuwm/cycle/children.py`, front door
`tools/cycle_child_leg.py`. The shared seam is `gpuwm/cycle/contracts.py`.

---

## 1. Why a placement request is geographic

A `PlacementRequest` carries `lat`, `lon`, `dx_m`, `nx`, `ny` -- and no
parent index. **A parent index is a resolution result, never a request.**

The reason is not tidiness. Between two cycle boundaries the parent may
have been re-initialised, the analysis may have shifted its grid, or a
different parent may be running entirely. An index request means a
different place on each of those and *nothing reports it*. A geographic
request means the same place on all of them, and if it cannot be honoured
it is refused by name.

Resolution goes through primitives that already exist on this line:

| step | primitive | why this one |
|---|---|---|
| lat/lon -> `(j, i)` | `gpuwm.downscale._nearest_parent_index` | projection-agnostic, cos-lat weighted, already proven to re-derive a reference run's real nest geometry from nothing but a point |
| `(j, i)` -> footprint | `gpuwm.downscale._centered_placement` | round-half-up centring, and its `__post_init__` runs WRF's own +-2 SINT stencil admission (`nest_interp.register_nest`) |

The centring arithmetic is transcribed in `_centered_start` for the
refusal path only, because `_centered_placement` raises a bare
`ValueError` for a footprint that hangs off the parent -- which is exactly
the case the spine has to answer with its own named refusal. Every
*accepted* placement is handed back to the primitive
(`_confirm_with_primitive`), and a disagreement between the two is itself
a refusal. The transcription cannot drift silently.

---

## 2. The slot pool, and its honest cost

A child **cannot be allocated mid-run** on this codebase, and that is a
design decision rather than a gap:

- `gpuwm.experiment.active_experiment` refuses any `grid_id` not declared
  dormant;
- `build_experiment` prices the full experiment at t=0;
- `gpuwm.core.clock.build_schedule` bakes activation ticks into
  precomputed op tables.

Together those make a spawn an **activation of a reservation**, not an
allocation that can OOM after three hours of integration. VRAM is
deterministic from the first second of the run.

So the spine makes the *reservation* fixed and the *placement* arbitrary,
and uses the cycle boundary as the allocation point -- which is free,
because a cycle boundary is already a process boundary.

> **State this to the user before they choose `--child-slots N`:
> N slots cost N nests' worth of VRAM for the whole run, filled or not.**
> An unfilled slot costs its memory-plan reservation and zero compute.
> That is the price of determinism, and it is not hidden.

`SlotPool` is deliberately not a bare set of integers: every `release`
carries a **reason**, and the reason is what lets a receipt distinguish
*"its storm decayed"* from *"it blew up"* from *"it left the parent"*
months later.

---

## 3. The three providers

One signature: `provider(*, cycle_index, valid_time, parent_geometry,
signal) -> list[PlacementRequest]`.

- **`TrackerPlacementProvider`** -- centroids off the parent's own UH or
  composite-reflectivity plane. Uses `weighted_centroid` from
  `gpuwm.core.storm_tracking` inside a discovered footprint window
  (argmax, take a window, consume it, argmax again), so **two storms in
  one search box produce two nests rather than one planted between them,
  on neither**.
- **`SchedulePlacementProvider`** -- a static `(valid_time, lat, lon,
  dx_m, nx, ny)` list from the run plan. Ranked at `+inf` strength: a
  child a person asked for is never silently outranked by a trigger.
- **`ObsPlacementProvider`** -- centroids read straight off the radar-grid
  observation file's `z_obs` / `z_max` plane.

The obs provider is **genuinely new capability, not a re-spelling**. Every
existing spawn trigger reads a *model* field, so a storm the model has not
yet produced can never trigger a nest -- and that is precisely the storm a
forecaster most wants a nest on. `ObsPlacementProvider` places on the
storm the **radar** sees, with a dead-flat model plane, and the test
`test_obs_provider_places_on_radar_echo_without_a_model_field` asserts
both arms: the obs provider finds it, the tracker on the same parent finds
nothing.

---

## 4. Refusal, never a silent clamp

The keepout is `spec_bdy_width (5) + blend_width (5)` = 10 parent rows,
matching `gpuwm.experiment.validate_spawn_placement` so that a placement
this module accepts is one the experiment loader will also accept -- the
plan does not die at materialization hours later.

A footprint that leaves fewer rows on any side is REFUSED, with this
exact shape:

```
requested child at 32.55N/-98.00W maps to parent index (j=48, i=37); the
north edge keeps 3 parent rows of the required 10 (spec_bdy_width 5 +
blend_width 5); placement REFUSED
```

Every number there is load-bearing: the geographic ask, the index it
mapped to, the rows kept, the rows required, and the two widths that add
up to the requirement.

`allow_clamp=True` (`--allow-placement-clamp` on the front door) clamps
instead, and sets `clamped=True` on the record. **That flag is a
workaround and is labelled as one**, in the help text and here: *a clamped
nest keeps integrating while it has stopped following its storm, which
looks exactly like a working nest.* Refusal is the default. The clamp is
the opt-in, not the other way round.

Other named refusals:

| refusal | when |
|---|---|
| `slot pool exhausted` | more storms than reservations |
| `within min_separation_km ... of an already-assigned child at ...` | two requests on one storm |
| `child spacing does not divide the parent spacing exactly` | a ratio that is not an integer |
| `child extent is not a whole number of parent cells` | `nx % ratio` |
| `child step does not divide the parent step exactly` | via `TickRatio`, at **plan** time |

**A request never vanishes.** Every input to `rank_and_assign` and
`plan_children` appears in the output -- placed, or REFUSED with the
reason named. A planner that dropped the fourth storm because there were
three slots would write a receipt indistinguishable from one where the
fourth storm was never detected, and those two mean opposite things.

---

## 5. Retire and reclaim, across the boundary

`plan_children` runs at every cycle boundary and does existing children
first, so a moving storm never costs a second reservation:

- a live child whose storm still exceeds `retire_below_strength` is
  **re-placed** at the new centroid, keeps its slot and its anchor state,
  and records `moved_parent_cells`;
- a live child whose signal has fallen below the threshold is **RETIRED**,
  releasing its slot, with the observed strength on the record -- this is
  the despawn, and at a cycle boundary it costs nothing because the
  experiment is rebuilt anyway;
- a live child whose storm walked **out of the parent** is retired rather
  than clamped, because clamping is opt-in here too;
- a new request claims a free slot (`PLANNED`);
- a request with no slot is `REFUSED`.

`emit_dormant_domains` then renders the pool as `[[domain]]` blocks with a
`spawn` table, ready for `gpuwm.core.nest_spawn.build_spawn_config`. That
is the hinge of the whole design: **an arbitrary geographic plan becomes a
declared experiment, and every piece of spawn machinery downstream
(`SpawnController` -> `SpawnRunner.on_leg_boundary` -> `walk_spawn_legs`)
runs unchanged.**

The declared trigger is `time` with an `at_s` the supervisor supplies,
because the *spine* decides placement at the boundary. A field threshold
there would put the decision back inside one leg, where it cannot see the
analysis that just landed.

---

## 6. A child may die; the cycle may not

`run_child_leg` catches:

- a **non-finite state** -- returns `DIVERGED` with the field, the cell
  index and the value:
  `d03 diverged at cycle 0: u is nan at cell [1]; the child is retired and
  the cycle continues`
- a **`CycleRefusal` from the runner** (a vertical-velocity ceiling, for
  instance) -- returns `DIVERGED` carrying the refusal's observations.

It catches **nothing else**. An allocator failure or a supervisor bug is
the *run's* problem, and swallowing it here would turn a halt into a
silently thinner forecast. `test_a_parent_refusal_is_not_swallowed`
asserts that asymmetry directly.

That asymmetry is what makes unattended arbitrary spawning safe: a bad
placement or a blown-up 500 m child costs **one slot**, not the run.

---

## 7. The daemon argv-contract test, and why it exists

`tests/test_da_nowcast_daemon_argv.py` is not about children. It is here
because it is the same class of failure.

`tools/da_nowcast.py:obs_cmd` is keyword-only and requires a
`DealiasChoice`. The continuous DA daemon's call site never learned about
it, so **every cycle of the daemon died on its first observation stage**:

```
TypeError: obs_cmd() missing 1 required keyword-only argument: 'dealias'
```

It escaped because the three existing tests that name `Daemon` read its
source as *text* (`inspect.getsource`) rather than binding the call. Text
matching cannot tell a call site from a signature.

The new test binds. It reads each argv builder's call site out of the
daemon's own AST and offers the keyword names to the real
`inspect.signature`, for `obs_cmd`, `cycle_cmd`, `render_cmd` and
`bootstrap_cmd`. A missing required keyword-only argument, or a keyword
the callee does not accept, fails there rather than in the field. A call
site may only opt out of the AST check by routing through a named
`*_kwargs` builder, whose mapping is then bound live.

The dealias choice is wired the whole way -- the parser flags (same names
and same shipped default as the front door), the bootstrap argv, the obs
argv, and the **re-exec argv the daemon rebuilds at every epoch roll**. A
choice that reached the obs stage but not `loop_argv` would unfold
velocity until the first roll and then quietly stop, with every gallery
after it still saying the run was dealiased because the first one was.
`test_epoch_reexec_preserves_dealias_choice` holds that.

---

## 8. Running it

```
python -m tools.cycle_child_leg --plan PLAN.json --out RECEIPT.json \
    --retire-below-strength 40 --min-separation-km 20
```

`--retire-below-strength` is required and has no default: its units are
the trigger field's, and a default would be a hardcoded threshold for
somebody else's field. The front door plans; the model advance is
injected (`run_children(advance=...)`), and calling the front door without
a driver **refuses** rather than writing a healthy-looking receipt for a
leg it never ran.
