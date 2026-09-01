# The cycle spine

`gpuwm cycle` drives a cycling weather-model system: a parent forecast
advancing in cycles, a data assimilation analysis landing at every cycle
boundary, and child domains spawned at arbitrary locations that advance
in sync with both.

The spine is **file-mediated and ledger-driven**. It is a separate
process that talks to the forecast and the analysis through files on
disk with versioned schemas -- never as a library either side imports,
because the MPAS port pins gpuwm by SHA and re-verifies that pin before
and after every run. That constraint is a gift: it makes the cycle
crash-recoverable, lets a Rust DA engine or the renderer read the same
artifacts, and kills the pickle-as-restart-format problem on day one.

Three planes:

- **Clock plane** -- one authority, the `CycleClock`. Integer ticks.
  Everything derives from it, nothing derives from a peer.
- **State plane** -- the anchor, a versioned on-disk directory at every
  cycle boundary. NetCDF plus a JSON manifest. Never pickle.
- **Control plane** -- the ledger, an append-only JSONL of transitions.
  Crash recovery is a replay.

---

## 1. Clock ownership

**There is one `datetime` in the whole spine: `epoch_anchor`.** It is the
parent init's `config_start_time`, which the MPAS port already treats as
authoritative. Everything below it is an integer number of ticks, and one
tick is one millisecond of **model** time (`TICK_HZ = 1000`).

Why an integer: a `dt` of 2.5 s is 2500 ticks and a `dt` of 120 s is
120000 ticks, both exact. A `dt` that is *not* a whole number of
milliseconds is **refused by name**, not rounded. A rounded `dt` is
exactly how a child drifts from its parent over a long cycle with
nothing reporting it -- the drift is invisible per step and fatal per
hour.

`CycleClock` owns two things and no others:

- **cycle boundaries** -- `boundary_ticks(i)` is `i * cycle_ticks`, and
  `valid_time(i)` derives the wall time from the anchor. `build()`
  refuses a `cycle_seconds` that is not a whole number of parent steps,
  naming the remainder in ticks.
- **analysis times** -- `snap(when)` returns `(ticks_on_lattice,
  offset_ticks)` and **never snaps silently downstream**. The caller
  either accepts the observed offset by name
  (`--accept-snap-offset-seconds`) or takes a refusal from
  `require_on_lattice()` naming the requested time, the nearest lattice
  time and the offset in seconds. The DA daemon's `snap_to_step` snaps
  and receipts; that is fine inside one model, and it is how a
  parent/child tree drifts. The spine does not inherit it.

A child's step must **divide** the parent's exactly. `TickRatio` refuses
otherwise, naming the remainder, and it refuses at *plan* time -- in
`--dry-run`, before any allocation -- not at hour three.

A child spawned mid-flight takes `birth_ticks` from
`CycleClock.birth_ticks(cycle_index)`, **never** copied from the parent's
`DomainClock`. That is the generalization of the streamed-LBC-clock burn:
*derive from the authority, never from a peer.*

**What the spine does not own:** within-leg parent/child stepping stays
with `gpuwm.core.clock.build_schedule`, which is transcribed op-for-op
from WRF `module_integrate.F` and already asserts tick-exact sync. The
spine hands it a boundary and checks the answer came back on it.

### The arming triple

At every cycle boundary `CycleSupervisor._assert_armed` asserts, and
writes into the receipt:

```
parent.ticks  == clock.boundary_ticks(cycle_index)
anchor.ticks  == clock.boundary_ticks(cycle_index)
child.ticks * TickRatio(parent_step, child_step).child_ticks
              == parent.ticks                       (each live child)
```

Any mismatch halts the run with `CLOCK_UNARMED` and a refusal naming the
grid id, the observed child ticks, those ticks scaled into parent space,
the expected tick and the drift in seconds.

Deleting the child branch of `_assert_armed` turns
`tests/test_cycle_supervisor.py::test_arming_detects_child_drift` red.
That is what makes it an arming test rather than a comment.

---

## 2. The ledger: three-phase commit, one supported reader

`cycle_ledger.jsonl` is append-only, one object per transition, schema
`gpuwm-cycle.ledger/v1`. It is written the way
`gpuwm/ensemble/cycle.py` publishes an analysis, because that is the one
place in this program that already gets atomicity right:

1. the record is serialised into `cycle_ledger.jsonl.staged` and
   **fsynced**, so the bytes exist before anything claims they do;
2. the staged line is appended to the live log and the log is fsynced;
3. the staged file is removed.

A crash between 1 and 2 leaves a staged file whose `seq` is not live:
`settle()` rolls it forward, which is safe because phase one proved the
bytes. A crash between 2 and 3 leaves a staged file whose `seq` *is*
live: `settle()` drops it rather than double-counting. Neither direction
is a guess.

**Reading a directory listing, or the JSONL itself, to discover a
cycle's state is OUT OF CONTRACT.** `CycleLedger.settle()` is the only
supported reader: it settles any in-flight publication first, and a raw
read cannot, so a raw reader takes a half-published ledger for a whole
one. `CycleLedger.state()` folds the settled records into
`{last_completed_cycle, halted, halt_reason, forecast_only_streak,
live_children}` -- that fold is what `--resume` resumes from.

`halt()` validates its reason against `HALT_REASONS` **before** the log
is touched, so a typo cannot poison a lineage.

Per cycle, `RECEIPT.json` (`gpuwm-cycle.transition/v1`) lands under
`<root>/cycle_<NNN>/` through a staged write and an atomic rename. It
carries the arming triple, the three-hash ingestion block, every
placement decision with its geographic request, every child state
transition, and every refusal.

---

## 3. Proving the analysis was eaten

An analysis that was silently dropped and an analysis that worked are
otherwise indistinguishable, so the ingestion receipt
(`gpuwm-cycle.analysis-ingestion/v1`) carries three hashes:

1. `background_sha256` -- the parent state the spine wrote into the
   anchor, hashed by the spine;
2. `increment_sha256`, with `increment_nonzero_cells` and per-field
   `increment_l2`;
3. `analysis_sha256` -- the state *after* application, hashed by the
   forecast process off its own rehydrated device state.

The gate runs **both arms**, because an exact-zero delta means the
experiment never ran:

| increment | state moved? | verdict |
|---|---|---|
| `nonzero_cells > 0` | no | halt `ANALYSIS_NOT_INGESTED` -- the increment reached the anchor but not the device |
| `nonzero_cells == 0` (NULL_ARM) | yes | halt `ANALYSIS_NOT_INGESTED`, `arm="NULL_ARM"` -- rehydration itself perturbed the state |
| `nonzero_cells > 0` | yes | applied |
| `nonzero_cells == 0` | no | applied, bit-stable |

Both arms are exercised in
`tests/test_cycle_supervisor.py::test_null_arm_requires_bit_stability`.

---

## 4. Failure behaviour: fail closed, name what was observed

`CycleRefusal.__init__` **raises if you construct it without
observations**. You cannot emit a nameless refusal from the spine.

| Class | Trigger | Cycle action |
|---|---|---|
| **DATA** | no radar volume in the window; feed late | advance **forecast-only**; ledger records `analysis: SKIPPED_NO_OBS` with the window searched; the staleness counter increments; past `--max-forecast-only-cycles` (default 3) → halt `STALE_ANALYSIS_BUDGET_EXHAUSTED` |
| **PLACEMENT** | child leaves the parent; slot pool exhausted; keepout violated | **slot stays unfilled**, named refusal in the ledger, **cycle continues**. A refused placement never silently becomes a clamped one -- clamping requires `--allow-placement-clamp` and is always receipted |
| **ANALYSIS** | LETKF rejects; the ingestion gate fails | **HALT.** An unusable analysis poisons the lineage. Anchor N-1 is preserved intact; the ledger says `ANALYSIS_REJECTED` / `ANALYSIS_NOT_INGESTED` with the observed numbers |
| **INTEGRATION** | child NaN / vertical-velocity refusal | **retire the child**, keep the parent and the cycle, record `DIVERGED` with the field, cell and value |
| | parent refuses or diverges | **HALT: `PARENT_DIVERGED`** |

**The deliberate asymmetry: a child may die without killing the cycle;
the parent may not.** That is what makes unattended arbitrary child
spawning safe -- a bad placement or a blown-up 500 m child costs one
slot, not the run.

A root whose ledger records a halt refuses to run again. Resuming past a
halt would build a lineage on an analysis the run already refused.

---

## 5. Arbitrary spawn

Mid-run creation of an *undeclared* domain is not possible on this line:
`active_experiment` refuses a grid id not declared dormant and
`build_schedule` bakes activation ticks into precomputed op tables. So
the spine makes the **reservation** the fixed thing and the
**placement** the arbitrary thing, and uses the cycle boundary as the
allocation point -- which is free, because a cycle boundary is already a
process boundary.

- *Within a cycle*, spawn is activation of a pre-declared dormant slot.
  `--child-slots N` declares N identically shaped nests; VRAM stays
  deterministic, which is the entire point of that design.
- *Across cycles*, the placement plan is recomputed at every boundary
  from the analysis. A live child is re-declared at its (possibly moved)
  placement and restored from its anchor state; a child whose storm died
  is retired and its slot returns to the pool; a new storm claims a free
  slot.

Placement providers (`gpuwm-cycle.placement/v1`) return desired children
in **geographic** coordinates -- `(lat, lon, dx_m, nx, ny)`, never parent
indices. They live in `gpuwm.cycle.placement` and are imported lazily, so
a tree without that module still plans, dry-runs, and cycles the parent
alone; asking for one that is not present refuses by name and says who
owns it.

---

## 6. Flags, defaults, and why

| Flag | Default | Why that default |
|---|---|---|
| `--root DIR` | required | the ledger, the anchors and the receipts are one lineage; the spine will not guess where it lives |
| `--epoch-anchor ISO8601` | required | the only `datetime` in the spine. A default would be a second authority |
| `--parent-dt-seconds` | `120.0` | a coarse-mesh MPAS step. Must be a whole number of ticks; `--cycle-seconds` must be a whole number of these |
| `--cycle-seconds` | required | the DA cycle length is the run's shape; there is no safe default |
| `--cycles N` | required | how far the lineage runs |
| `--parent-kind` | required, from `PARENT_KINDS` | which engine advances the parent. `replay` advances the clock alone and is how you rehearse a lattice; a kind with no engine adapter in this tree refuses at plan time rather than pretending to integrate an atmosphere |
| `--child-slots N` | `0` | reservations are priced at t=0. Zero means parent-only, which is the honest default for a tree with no placement provider |
| `--child-dt-seconds` | the parent step | a child step must divide the parent's exactly; defaulting to the parent's is the one value that always does |
| `--placement-provider` | `none` | placement is a capability, not an assumption. `none` cycles the parent alone |
| `--max-forecast-only-cycles` | `3` | forecast-only is legitimate when a feed is late; forecast-only *forever* is a run that quietly stopped being a DA cycle. Three cycles is enough to ride out one missed volume without hiding a dead feed |
| `--allow-placement-clamp` | off | a nest that silently moved is worse than a nest that did not fly. Off means a placement needing a clamp is REFUSED and receipted; on means it is clamped and receipted |
| `--accept-snap-offset-seconds` | `0.0` | an analysis time must land on the parent-step lattice. Accepting an offset is a decision an operator makes by name, in the receipt |
| `--resume` / `--no-resume` | resume | crash recovery is the normal case; a cycling run that restarts from cycle 1 by default would silently discard hours |
| `--dry-run` | off | prints the boundary lattice, the resolved child ratios, and refuses invalid combinations **before** a byte of state is written |

### Parent state — what the cycle actually cycles

Until these landed, `gpuwm cycle` handed the supervisor a closure that
returned two integers. It cycled a clock over an **empty root**: no anchor
was written, so the three-hash ingestion gate had nothing to compare, the
hydrostatic instrument never fired, and every receipt said
`"ingestion": null` about the most important gate in the system. The
engine that does all three existed and was reachable only from
`tools/cycle_demo_*.py`.

| Flag | Default | Why that default |
|---|---|---|
| `--parent-state GLOB` | required for a real run | the parent's recorded npz frames, in the same flat on-disk shape `tools/cycle_mpas_leg.py --history` reads. `--dry-run` does not need it; a run that writes receipts does |
| `--parent-mesh-id ID` | required for a real run | every anchor carries a mesh identity. No default: an identity the spine guessed is one no downstream reader can trust |
| `--analysis-increment CYCLE=PATH` | none | the increment applied at `CYCLE`, an npz keyed by **prognostic field name**. Repeatable. `CYCLE=null` is the explicit **null arm** — a gate with only the applied arm proves nothing, because a hash that moved could have been moved by rehydration. With no increments at all the run is forecast-only *by construction*, which the supervisor distinguishes from a DA cycle that stopped receiving observations |

### The model parent — reaching the port's dycore

**Disposition (2026-08-31).** This leg was written and proven against the
MPAS port as a *frozen checkout* (`lane/mesh-binding@12fe6a3`), reached
through `--port-root`. The port has since shipped standalone as the
`gpuwm-hex` package on PyPI, which does not pin gpuwm and does not need
the import-guard dance below. The leg lands as proven: it still runs
against a frozen port checkout, and every refusal on this path names that
checkout. Teaching the worker to drive an installed `gpuwm-hex` instead
of a checkout is a named follow-up (the process boundary, the anchor
schema and the receipt grading all carry over unchanged; only
`mpas_cycle_bridge.portbind`'s loading seam changes).

`--parent-kind mpas-cuda` (and `mpas-cuda-frames`) run the **real frozen CUDA
dycore**, and they run it **out of process**. That is not a style choice: the
port pins its own gpuwm checkout and installs an import guard, so a spine that
constructs the port in-process hits an import wall on a real card. Every leg
spawns `mpas_cycle_bridge.worker` in a fresh interpreter and the two sides talk
only through anchors and segments on disk.

**The stamp is earned, not requested.** `--parent-kind` says which engine you
asked for; what lands on the anchor is what the evidence grades out at. A leg
that cannot show step receipts matching `--port-steps`, rehydration through the
port's state seam, and a device-readback hash equal to the analysed state is
stamped `mpas-cuda-frames` with its gaps listed on the receipt. Asking louder
does not upgrade the evidence.

| Flag | Default | Why that default |
|---|---|---|
| `--port-root PATH` | required for a model parent | the MPAS port checkout the forecast worker runs from |
| `--port-config PATH` | required for a model parent | the port's case configuration JSON |
| `--port-steps N` | required for a model parent | dycore steps per cycle boundary. The step **receipts** the worker returns are counted against this number — a leg that ran fewer steps than asked cannot earn the `mpas-cuda` stamp, which is what stops a crashed arm from being graded as a closed loop |
| `--port-timeout SECONDS` | no timeout | how long to wait for one forecast segment |

A model parent does **not** take `--parent-state`: its state comes off the
device, not out of a recorded series.

### Placement — where the children go

| Flag | Default | Why that default |
|---|---|---|
| `--parent-geo-file PATH` | the first `--placement-obs-file` | XLAT/XLONG on the parent's mass grid, from a radar-grid NetCDF or an npz. The radar-grid file's grid **is** the target model grid by contract, so a run placing on observations already holds its parent's geography |
| `--parent-dx-m METRES` | required with a placement provider | the child/parent refinement ratio is derived from it, and no file on this path declares it. A guessed spacing silently changes every placement |
| `--placement-obs-file PATH` | none | the radar-grid file the obs provider places on. **Repeatable, one per cycle in order**; a single file is reused for every cycle. A storm that never moves between cycles is what hid child retirement, slot reclaim and the out-of-parent refusal for a week |
| `--placement-obs-field NAME` | `z_obs` | `gpuwm-obs.radar-grid.v1` ships `z_obs`, `z_max` and `z_mean` side by side and states the choice between them is the **consumer's**, so it is configured rather than fixed or sniffed — every canonical file carries all three, which makes "discover it" a hardcode with extra steps. The default is the file's own observation variable: what every DA lane reads and what the real three-volume KTLX series was placed against. It was `z_max` in the factory and `z_obs` in every producer, a name only hand-built test data ever satisfied |
| `--placement-tracker-field NAME` | `composite_reflectivity` | same rule for the parent plane the tracker reads; an absent name is refused naming what the anchor does carry, rather than returning an empty list that reads as "no storms" |
| `--placement-threshold VALUE` | `40.0` | the trigger value a peak must reach, in the placement field's own units |
| `--retire-below-strength VALUE` | required with a placement provider | a child under this much signal is retired and its reservation returns to the pool. Deliberately **no default**: its units are the trigger field's, so a default would be a hardcoded threshold for somebody else's field |
| `--min-separation-km KM` | `40.0` | two children are never planted on one storm. A request inside this radius of an assigned child is REFUSED by name, never silently dropped |
| `--child-nx` / `--child-ny` | `199` | child grid points; must be a whole number of parent cells |
| `--child-dx-m METRES` | `1000.0` | child spacing; must divide `--parent-dx-m` exactly |

### A note on `--cycle-seconds` and `--parent-dt-seconds`

`--cycle-seconds 900` with `--parent-dt-seconds 120` is 7.5 parent steps
and is **refused**, naming `remainder_ticks=60000`. A 15-minute DA cycle
needs a parent step that divides 900 s (30 s, 60 s, 180 s, 300 s); a
120 s parent step needs a cycle that is a multiple of 120 s, such as
960 s or 1200 s. The refusal is the invariant working.

---

## 7. Worked example

```
$ gpuwm cycle --root runs/cycle-demo \
      --epoch-anchor 2026-08-14T18:00:00Z \
      --parent-dt-seconds 120 --cycle-seconds 960 --cycles 3 \
      --parent-kind mpas-cuda --child-slots 2 --child-dt-seconds 30 \
      --dry-run
cycle: root runs\cycle-demo
cycle: parent-kind mpas-cuda  epoch-anchor 2026-08-14T18:00:00+00:00
cycle: parent step 120000 ticks (120 s)  cycle 960000 ticks (960 s) = 8 parent steps
cycle: boundary lattice
  index         ticks  valid time
      0             0  2026-08-14T18:00:00+00:00
      1        960000  2026-08-14T18:16:00+00:00
      2       1920000  2026-08-14T18:32:00+00:00
      3       2880000  2026-08-14T18:48:00+00:00
cycle: 2 child slot(s), step 30000 ticks (30 s), 4 child steps per parent step
cycle: placement-provider none
cycle: forecast-only budget 3 cycles; placement clamp refused; snap offset accepted up to 0 s
```

Nothing was written: `--dry-run` refuses before it allocates.
