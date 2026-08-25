# Automatic nest lifecycle

The lifecycle surface is additive. Existing one-shot `spawn` and tree-level
`[relocation]` configurations keep their old meaning.

A trigger-spawned child may additionally declare:

```toml
retire = { trigger = "uh", threshold = 60.0, sustained_s = 900.0, min_lifetime_s = 1800.0 }
rearm = { max_firings = 4, cooldown_s = 1800.0 }
follow = { field = "uh", threshold = 100.0, fallback_threshold = 35.0, search_margin_cells = 12, min_shift_cells = 2, max_shift_cells = 10, cooldown_seconds = 600.0, cadence_seconds = 300.0, max_move_parent_cells = 8, min_overlap_fraction = 0.70 }
```

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
