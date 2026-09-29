# Automatic nest lifecycle

The lifecycle surface is additive. Existing one-shot `spawn` and tree-level
`[relocation]` configurations keep their old meaning.

A trigger-spawned child may additionally declare:

```toml
retire = { trigger = "uh", threshold = 60.0, sustained_s = 900.0, min_lifetime_s = 1800.0 }
rearm = { max_firings = 4, cooldown_s = 1800.0 }
follow = { field = "uh", threshold = 100.0, fallback_threshold = 35.0, search_margin_cells = 12, min_shift_cells = 2, max_shift_cells = 8, cooldown_seconds = 600.0, cadence_seconds = 300.0, max_move_parent_cells = 8, min_overlap_fraction = 0.70 }
```

### The two movement bounds are one number and its consequence

`min_overlap_fraction` states the physics: how much of the child a move
keeps, so the rest is strip the child has to spin up.
`max_move_parent_cells` and the follower's own `max_shift_cells` are
DERIVED from it and are not free to disagree with it. Overlap is
separable, so a shift of `(m, n)` parent cells keeps
`(1 - m*r/nx) * (1 - n*r/ny)` of the child and the binding case is the
DIAGONAL move: a floor `f` admits a per-axis magnitude of only
`1 - sqrt(f)` of the nest's own width in parent cells.
`gpuwm.core.nest_relocation.max_parent_cells_for_overlap` is that
derivation, and the example above therefore needs a nest at least 49
parent cells wide before its `max_shift_cells = 8` and
`max_move_parent_cells = 8` are reachable at
a floor of 0.70. Declaring a maximum above the implied bound does not
widen anything: the move passes the per-axis check, the floor refuses
it, and the largest move the configuration appears to offer is one no
storm can ever be followed with. Every configuration under `configs/` is
checked against its own bound by
`tests/test_relocation_overlap_clamp.py`.

The bound is a fraction of the NEST's width, so a door that authors a
nest whose dimensions it chooses derives the maximums after it has
chosen them, not before. `gpuwm cyclone-setup` proposes a smaller
layout on a card that cannot hold the requested one, and grows the nest
to `--nest-budget-gib` on a card that holds more, and
`gpuwm.cyclone_setup.follow_table_for_nest` is where its `[domain.follow]`
table and its `cyclone.json` receipt both get their two maximums, on the
dimensions that proposal settled on, in either direction: a
36-parent-cell reduction admits 5 where the preset's 40 admits 6, and a
60-parent-cell nest grown to a budget admits 9. A fitted proposal
therefore shows those two numbers among its reviewed changes.

A scheduled follow source never fails on this: the runner clamps a
proposal that would breach the floor to the largest move in the same
direction that clears it, names `min_overlap_fraction` in the receipt's
`clamped_by` beside the requested and executed shifts, and the nest
makes up the rest at the next cadence. A containment slide is clamped
and receipted the same way, on the sliding ancestor's own extent: its
row carries `overlap_fraction`, and `clamped_by` names
`containment.max_move_parent_cells`, `min_overlap_fraction`,
`parent_edge` or `mover_compensated_placement` -- whichever bound moved
the number. `tools/relocation_ledger_audit.py` reads `overlap_fraction`
off every executed move and every slide and checks it against the
declared floor, which the refusal used to make unnecessary.

`reach_speed_m_s` bounds the whole run rather than one move: at model time
`t` the nest may be at most `reach_speed_m_s * t` plus one move from where it
was declared, along each grid axis. It defaults to 40 m/s, faster than any
tropical cyclone or supercell on record, and a prepared run's statics corridor is
sized to it, so a smaller value for a storm known to move slowly buys a
smaller preparation. A move past it is clamped and `clamped_by` names
`reach_speed_m_s`.

`retire` takes the same trigger vocabulary `spawn` does -- `"uh"`,
`"reflectivity"`, `"pressure"`, `"time"`. A field trigger retires a nest
when the signal under its live footprint stays QUIET continuously for
`sustained_s`, once `min_lifetime_s` of that episode has elapsed. Quiet
is the trigger's own sense of decay:

| trigger | quiet when | `threshold` |
| --- | --- | --- |
| `uh`, `reflectivity` | the footprint MAXIMUM is at or below `threshold` | m2 s-2 / dBZ |
| `pressure`, `level_hpa = 0` | the footprint MINIMUM has FILLED to or above `threshold` | hPa (800-1100) |
| `pressure`, `level_hpa = 850` (default) | the vortex DEPTH under the footprint -- the height field's own span -- has fallen below `threshold` | m (1-500) |

A dying cyclone is a rising minimum, which is why the pressure row is
the maximum test inverted rather than the same test with a different
number. Under `level_hpa` the depth is used rather than an absolute
height because an 850 hPa surface is ~1500 m in the deep tropics and
~1350 m in a cold airmass: a fixed height would retire the nest on the
airmass instead of on the storm. `level_hpa` is refused on a maximum
trigger and on `"time"`, and the two threshold bands are disjoint, so a
units error refuses at load.

`"time"` is the deterministic form, and it is the only one whose
boundaries can be written down before the run starts:

```toml
spawn  = { trigger = "time", at_s = 300.0 }
retire = { trigger = "time", at_s = 900.0, min_lifetime_s = 0.0, sustained_s = 0.0 }
rearm  = { max_firings = 2, cooldown_s = 300.0 }
```

`retire`'s `at_s` is EPISODE AGE in seconds, not model time: the slot
above is born 300 s in, retires 900 s later at t = 1200, re-arms after
its cooldown at t = 1500 and retires again at t = 2400. `spawn`'s `at_s`
is model time and must land on a whole number of parent steps. A `time`
trigger refuses `threshold` (it reads no field) and a field trigger
refuses `at_s` (the field chooses its own instant); every key is honored
or refused, never ignored.

Prefer the time form for any test, gate or proof. A field trigger's
instant moves with the physics, so what lands on disk cannot be checked
against a timetable -- which is what makes it right in production and
useless as evidence.

Retirement is evaluated only at completed spawn-leg boundaries. It therefore
changes the domain set used to build the next schedule; it never skips an op in
an already-running schedule. Retiring a parent removes its live subtree.

A retired slot re-arms only while its parent is a domain the model is
integrating: the root or a permanent intermediate domain always is, and a
spawned parent is during its own episode. A nest below a spawned parent that
retired waits for that parent's next episode, even once its own cooldown is over.

A re-armed slot is a new episode. History is written under `dNN/episode-NNN/`
for a domain that DECLARES `retire` and/or `rearm`, from its first episode, so
one slot's episodes share a layout. A domain without those tables keeps the
flat `wrfout_dNN_*` pathname it has always written: a plain one-shot `spawn` is
not a lifecycle episode, and `follow` relocates a nest within one episode
rather than starting another.

Publication refuses a valid time this run already wrote, which is the duplicate
a lifecycle or restart boundary can produce. A frame left at that pathname by a
PREVIOUS run is replaced as it always has been; re-running into an existing
output directory is not the defect the refusal exists to prevent.

Per-domain followers each own a `uh_follow_window.dNN` accumulator and a
separate `StormTracker`, so cadence and cooldown state cannot cross-talk.

A streamed parent carries each declared follower window through the same tile
buffers and canonical store as the fixed spawn/follow windows. Its configured
children determine the inventory: each follower adds one FP32 horizontal plane
on its parent, in every tile compute window and in the full-domain store. These
bytes enter resident and streaming admission. Dormant consumers reserve their
slots before tile attachment and begin a fresh window when their episode starts;
another follower's consultation or birth does not reset their siblings.

Lifecycle checkpoints preserve these generated windows with each follower's
own cadence, segment and cooldown state. Ordinary checkpoints without the
lifecycle opt-in still omit and clear consumer windows. The shared transport is
verified with two different follower cadences over a streamed parent and with
resident/streamed checkpoint continuation. Relocating a streamed child itself
still requires reconstruction of its canonical store, geography and live tile
stepper; that separate operation remains guarded. The current follower-parent
path also retains the existing resident donor projection at consultation, so
this transport proof is not a bounded-memory relocation proof.

Legacy tree-level `[relocation]` remains supported unchanged. A single child
cannot select both authorities.

## Shipped configs

`configs/nest_lifecycle_20240521_4km.toml` is the retire -> re-arm proof: one
slot, two episodes, `time` triggers throughout so the episode boundaries are
arithmetic and the frames they produce can be checked against a timetable
written before the run. `configs/nest_spawn_oneshot_20240521_4km.toml` is the
same geometry with the lifecycle tables removed, and must keep writing flat
`wrfout_d02_*` paths -- the default-path promise stated as two runs.

`configs/cyclone_nest_slots_12km.toml` is the tropical shape: **three** dormant
4 km slots on a 12 km GFS parent, each opening itself on a pressure minimum,
riding it, and closing when it fills. Every trigger in it is the same signal
used three ways -- `spawn` on a low deep enough to be worth 4 km, `follow` on
the same low as it moves, `retire` when it fills -- with `level_hpa = 850` on
all three, so the threshold is metres of geopotential height and means the same
thing in any basin or season. The three slots are kept off one storm by the
exclusion rule rather than by their windows.

## Restart

Restart with per-domain lifecycle tables is admitted. The checkpoint persists
the policy state, not only the arrays: which slots have fired and which are
spent, each live episode's number, fired placement and birth time, each retired
slot's retirement time, the sustained-decay quiet timers, and every follower's
segment, generation, move count and cooldown timestamps. The consumer tracking
windows (`uh_spawn_window`, `uh_follow_window`, `uh_follow_window.dNN`) ride
their own domain's member, so the next spawn/retire/follow decision reads the
same fold an unbroken run would have read.

The resume rebuilds the tree the checkpoint describes -- spawned children
materialized at their FIRED placement through the same seams a leg boundary
uses, then moved in one hop to their persisted CURRENT placement -- before a
single array lands. A checkpoint taken exactly on the leg lattice was taken
before that boundary was evaluated, so the resume replays that one boundary
pass; a resumed episode-2 domain writes `dNN/episode-002/` from its first frame.

The contract is bit-identity: a run split at a checkpoint and resumed produces
the same wrfout frames and the same final state as the unbroken run, for any
number of segments. Receipts (`spawn_receipts.jsonl`, relocation receipts) are
outside it -- each segment owns its own ledger.

## Cost over a long forecast

Two things used to grow with the length of the run rather than with the size
of the config, and both are bounded now.

`spawn_receipts.jsonl` is **appended**, one complete JSON object per line,
flushed as each boundary is decided. It was previously one JSON document
re-serialised whole at every boundary, so the bytes a run wrote to it grew as
the square of its length. Every line carries its own `contract`, so a killed
process loses nothing earlier and a reader never parses a truncated array. The
in-memory ledger keeps every decision and a window of the most recent held
boundaries; the file keeps all of them.

Leg boundaries are taken at **decision points**, not at every history interval.
Each one costs a full schedule rebuild, and a re-armable slot used to keep
asking for them for the whole run -- ~4,600 rebuilds at 384 h to re-read one
cooldown clock. A cooldown that has not elapsed, a window that has not opened,
a manual trigger's `at_s` and a minimum lifetime that has not run out are all
known instants, so the walk runs straight to the earliest of them. Anything
that reads the live field (a field retirement past its `min_lifetime_s`, a
field spawn inside its window) keeps every boundary, and so does a run with a
mounted relocation runner, which stays on its follower's cadence.

A `uh` or `reflectivity` trigger anywhere in the tree forfeits this entirely,
and that is not conservatism -- it is the one place skipping a boundary would
change an ANSWER rather than a cost. Those two signals are consumer-owned
windows the runner **zeroes at every boundary it takes**, so what a watch reads
is "the strongest since I last looked". Skip six hours of boundaries and the
next look sees six hours of accumulation: a slot coming off its cooldown would
fire immediately on rotation that happened while it was spent. `pressure` is
exempt because it is reduced from the live prognostic column and carries no
window at all (the line `STASH_BACKED_FIELDS` already draws for the follow
cadence).

The block states contract `gpuwm-nest-lifecycle-restart.v1` and is honored whole
or refused by name. A restore refuses when: the checkpoint carries no lifecycle
block but the experiment declares one (it predates persistence and cannot say
which slots fired); the checkpoint carries one but the experiment declares none
(every fired slot and move history would be dropped on the floor); the contract
is unknown or the key set is not the one this build reads; the block names a
domain the member set does not carry, or names a retired domain that still owns
a member; or the leg cadence differs from the one the resuming run stops on,
which would evaluate the same policy at different instants.
