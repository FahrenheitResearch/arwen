# Reconciling the 8x4090 box fork with `base`

The rescued tree `bowecho-dea/arwen-boxdelta/` holds seven files that also
exist in `arwen-tilestream`, all of them divergent. `HANDOFF.md` calls them a
three-way fork and warns that **`scp` in either direction destroys real work**.
This document is what a merge has to know instead: for every file, where the
fork point actually is, what each side genuinely contributed, and — hunk by
hunk — what the merged file keeps.

Only `tilestream/driver.py` is merged here. The other six are classified, and
the box copies of all seven are preserved verbatim under
`tilestream/rescued-tools/boxdelta-forks/` so every claim below can be checked
against the artefact rather than against this summary.

---

## Method, so the classification can be re-run

The handoff names the sides "box", "local" and "integration" but not the commit
any of them forked from. It was recovered by search: for each path, every
distinct blob it has ever had on any ref was diffed against the box copy, and
the nearest one taken as the fork point.

For `tilestream/driver.py` that is blob `88c89ede67`, 1469 lines, differing
from the box copy by **exactly 64 lines** — the figure the handoff quotes,
which confirms the identification. It is still the tip content on six
branches (`mgpu-real-init`, `tilestream-ingest`, `tilestream-mgphys`,
`tilestream-mgstream`, `tilestream-mgstream-skeptic`, `tilestream-physrungs`)
and is an **ancestor of `base`**, so a real three-way merge base exists.

`base` and `integration` hold the identical blob for all seven paths
(`fdf7b8da…` for `driver.py`), so "local" and "integration" are one side today,
not two. The merge is box-versus-base over a known ancestor.

Two mechanical tests were then run per file, and their counts are quoted in
the table:

* of the lines the box **added** since the fork point, how many already exist
  verbatim in `base` — the test of whether `base` is a superset;
* of the lines the box **removed**, how many `base` still keeps — the test of
  whether the box copy is simply older.

---

## The correction to the handoff

The handoff lists all seven as three-way, "no side a superset". Measured, that
is true of `driver.py` and of nothing else.

| file | fork point | box | base | box-added not in base | box-removed base still keeps | verdict |
|---|---|---|---|---|---|---|
| `driver.py` | `88c89ede67` (1469 L) | 1515 L | 2219 L | **48** | 8 | **genuine three-way** |
| `bench.py` | `ec9cfe30c3` (819 L) | 891 L | 1102 L | 4 (re-wrapping only) | 1 | base is a superset |
| `test_gate.py` | `4c9d670613` (2292 L) | 2262 L | 2583 L | 4 (a downgraded assertion) | 33 | box is behind |
| `RESULTS.md` | `6aa54e18ec` = **base** | 1326 L | 1651 L | 16 (one real table) | 284 | two-way; box behind, one genuine contribution |
| `make_plots.py` | `9f288c5a98` = **base** | 427 L | 628 L | 1 (stray import) | 176 | two-way; box behind |
| `rings.py` | `8344476aea` = **base** | 972 L | 995 L | 16 (superseded text) | 37 | two-way; box behind |
| `bench_ooc_output.py` | `24e23b2068` = **base** | 1084 L | 1126 L | 3 (re-wrapping) | 40 | two-way; box behind |

For the four files whose fork point **is** `base`'s current blob, `base` has
not touched them since the box was taken. They are not three-way at all: only
the box moved, and for three of the four it moved backwards.

**This matters for the warning it inverts.** The handoff says `scp` in either
direction destroys work, and for `driver.py` that is exactly right. For
`rings.py`, `test_gate.py`, `make_plots.py`, `bench_ooc_output.py` and
`RESULTS.md` the danger is one-directional: copying box → local would delete
284 lines of `RESULTS.md`, 176 lines of `make_plots.py`, a whole benchmark mode,
and the measured contention finding in three separate places. Copying
local → box would lose only the one `RESULTS.md` table named below.

---

## `tilestream/driver.py` — the merge, hunk by hunk

`base` refactored the monolithic `run_tiled` (ancestor lines 953-1435) into a
`TiledRun` class: `__init__` does the setup, a nested `_sweep` closure runs the
loop, and `run_tiled` is now a thin build-and-sweep wrapper. Every box hunk had
to be re-seated into that shape; none could be applied as a patch.

**A plain three-way merge is wrong here and would have silently deleted a
measurement.** Hunk 4 below is a case where the box edited a line that `base`
left untouched, so `git merge-file` takes the box side by construction — and
the box side is the pre-measurement guess that `base` had already replaced with
the measured answer.

| # | hunk | side | disposition |
|---|---|---|---|
| 1 | module docstring, ring cost | both, conflicting | **both kept, each with its box** |
| 2 | `import time as _time` | box | kept |
| 3 | `tile_states=` parameter | box | kept, re-seated on `TiledRun` |
| 4 | `ring_ordering` docstring | both, conflicting | **base kept, box rejected** |
| 5 | setup-is-not-a-per-step-cost comment + `_t_call` | box | kept, reworded for `__init__` |
| 6 | `nbuffers = min(nbuffers, len(specs))` | box | kept |
| 7 | tile-buffer build + `deviceSynchronize` + `_t_factory` | box | kept, extended |
| 8 | `setup/factory/sweep_seconds` into the report | box | kept, re-seated |

### 1. Ring cost in the module docstring — both kept

The two trees measured the same quantity and **disagree at the one plan they
share**:

| plan | base / local | 8x4090 box |
|---|---|---|
| 1950², tile 650 | 4.97% | 5.16% (5.2% in its docstring) |
| 3276², tile 546 | 5.84% | 5.77% (5.8%) |
| 5412², tile 1353 | 2.39% | 2.39% |

Neither is a dry number and neither supersedes the other, so the merged
docstring carries both and names the box each came from. A ring-cost figure
quoted without its box is not a number — `NO-DRY-NUMBERS.md` applied to a case
it did not anticipate. **Open**: 4.97 against 5.16 at an identical plan is not
noise; it wants a re-measure to close, and until then neither should be quoted
as *the* ring cost.

### 2-3. `import time as _time`, and `tile_states=`

**Said first, because it qualifies everything in this section: neither
`tile_states=` nor any of the timing keys has a caller.** `grep` across both
repositories finds `tile_states` only in the box's own `driver.py` — the box
added the parameter and never called it — and finds `setup_seconds`,
`factory_seconds` and `sweep_seconds` nowhere but there. `bench_window.py`,
which names the setup-timing defect in its docstring, solves it structurally
inside its own harness rather than by reading the driver's report.

They are merged anyway, and the reason is narrow: this is measured work from a
box that no longer exists, the cost of carrying it is one optional keyword and
five report keys, and the cost of dropping it is that it cannot be recovered.
It is a provision, not a dependency, and the follow-up that would make it earn
its place is wiring `bench_window` to the driver's own numbers.

The box's headline contribution. `run_tiled` builds a fresh tile buffer per
call, which at a physics rung is a whole `initialize_physics` plus a
radiation-firing warmup step per buffer — tens of seconds against a step of
tenths. `tile_states=` lends pre-built buffers instead.

Kept, on both `TiledRun.__init__` and the `run_tiled` wrapper, with the box's
buffer-count and compute-window shape checks intact — a buffer built for
another plan's window gathers the wrong rectangle, which is a silently wrong
forecast rather than a crash.

Three changes against the box version:

* `base`'s `TiledRun` already solves most of this: a caller that holds the
  object and calls `sweep()` reuses the buffers, the transfer plans **and** the
  ring arena. The docstring now says so and points callers there first;
  `tile_states=` is the escape hatch for callers that cannot hold one.
* Supplying `tile_states=` **and** `tile_state_factory=` now refuses instead of
  silently ignoring the factory.
* `tile_states` is materialised once. The box wrote
  `len(tile_states)` in the error message after `list(tile_states)` had already
  consumed it, so a generator that supplied too few buffers would have been
  reported as supplying zero.

### 4. `ring_ordering` docstring — base kept, box rejected

The box text asks the question:

> It is exposed so the assumption can be MEASURED -- both must give the same
> digest here, and if they ever do not, this card's documented
> `asyncEngineCount == 1` is not what is actually happening.

`base` carries the answer:

> MEASURED, and the answer is why the default is what it is: `"submission"` is
> bit-exact on an IDLE card and WRONG BY 3.7e+02 on the same 3x3 plan while
> another process shares the GPU. It is kept only as that measurement. Never
> use it for a forecast.

The box's wording is box-original — it appears in no blob in the repo's
history, so it is not merely a stale copy — but it is a hypothesis that the
local side went on to test and refute. Taking the box side here would delete
the refutation. Rejected, and the same rejection applies to the two other
places the box reverted the finding (`rings.py`, `test_gate.py`, below).

### 5-8. The instrumentation

Kept in full, re-seated on the class:

* `setup_seconds` / `factory_seconds` are computed in `__init__` and exposed as
  attributes, because in `base`'s shape the setup **is** the constructor. The
  buffer build is followed by a full `deviceSynchronize()` so an asynchronous
  allocation cannot be billed to the first sweep.
* `sweep_seconds` / `sweep_clocks` are per-`sweep()`-call lists, appended after
  the sweep's device synchronization and clock epilogue and before the shadow
  swap — the same point the box measured.
* All five (`setup_seconds`, `factory_seconds`, `loop_seconds`,
  `sweep_seconds`, `sweep_clocks`) go into `report`, so the box's report
  contract survives for callers driving `run_tiled(report=...)`.

The architecture strengthens the box's finding rather than merely preserving
it: setup was a per-`run_tiled`-call cost on the box and is a per-`TiledRun`
cost here, so a caller that sweeps many times amortises it to nothing. The
comment says which is which so the two cannot be added together.

`nbuffers = min(nbuffers, len(specs))` is placed after `validate_plan`, where
`specs` first exists and before the tile buffers, the geography slots and the
graph steppers are sized from it. Arithmetic-neutral: `b = itile % nbuffers`
cannot select a buffer past the tile count, and the clock epilogue already
slices to the same bound. `self.nbuffers` and `report["nbuffers"]` now record
the clamped value, which is the number of buffers that were actually built.

**Not taken from the box:** its `tiles = list(...)` line and its
`f"tile_states supplied {len(tile_states)}…"` message, superseded above; and
the three docstring lines of hunk 4.

---

## The other six

### `bench.py` — base is a superset; nothing to merge

The box added ring-mode support to the bench's own `TiledRunner`
(`write_mode=`, `ring_ordering=`, the save/patch inside the gather bracket, the
WAR waits, `ring_bytes()`, `_dst_home()`, the widened digest matrix).
**`base` already contains all of it**: 75 of the box's 79 added lines are
present verbatim, and the four that are not are the same code re-wrapped —
`base` continues the parameter list past `ring_ordering` where the box closed
it, and wraps `field_geometry(self.home,` differently. Of the 12 lines the box
removed, `base` removed 11 of the same.

`base` additionally carries `physics_rung`,
`selftest_physics_runner_matches_driver`, `manifest_digest` and `seeded`.

Merged `bench.py` = `base`, unchanged.

### `test_gate.py` — box is behind, and its change would break under contention

The ancestor and `base` carry a `RING_OBSERVATIONS` list, described in the file
as *"configurations whose outcome is a MEASUREMENT of the machine, not a
property of the code, so the gate reports them and never fails on them"*, with
`ring_ordering="submission"` as its only member and both outcomes recorded as a
pass.

The box deleted that list and moved the case into `RING_CONTROLS` with
`expect=True` — a hard assertion that dropping the cross-stream events is
bit-exact. That is precisely the assertion the local side measured to be false
on a shared card. On the box's own gate it is a **false PASS on an idle card
and a spurious FAIL under contention**, and it is the same regression as
driver hunk 4.

`base` also has 309 lines of integration work on top.

Merged `test_gate.py` = `base`, unchanged. The box's 4 lines are rejected.

### `rings.py` — box is behind; one numeric conflict worth recording

The box copy predates three findings `base` states:

* `margin_mode="halo"` leaves **192 cells unsaved on a 192² 3x3 plan and 9,344
  on a ragged 200x192 one** — *and is bit-exact anyway* at halo 14, 15 and 16,
  because the dropped cells sit outside the reading tile's influence cone. So
  the hash gate cannot see this failure and `assert_ring_covers_reads` is the
  only instrument that can. The box text says only that the run "stops being
  bit-exact", which is the opposite of what was measured.
* the `submission`-ordering refutation, again.
* the ragged-tile cost cliff.

One conflict is worth a note rather than a silent overwrite. On the same plan,
3276² at tile 512, halo 16:

* `base`: **24.46%**, explained — 13 ragged tiles, each narrower than its
  neighbours' windows and so read right through, against 5.84% at tile 546
  which divides the domain exactly.
* box `rings.py`: **6.3%**, with no mention of raggedness.

The box's own `RESULTS.md` uses `base`'s methodology (it reports 22.7% for
5182² at tile 512 with 21 ragged tiles), so the box's `rings.py` figure is
inconsistent with the box's own results document and is treated as superseded.

Merged `rings.py` = `base`, unchanged.

### `bench_ooc_output.py` — box deleted a mode

The box removed `mode_steps` ("solver step time per domain size — the
denominator of every cadence") and its two dispatch entries. That is a removal,
not a contribution.

Merged `bench_ooc_output.py` = `base`, unchanged.

### `make_plots.py` — box deleted four functions

The box copy is `base` minus `_output_data`, `_src`, `plot_output_scaling` and
`plot_output_breakdown` (176 lines), plus one stray
`from matplotlib.ticker import FuncFormatter`.

Merged `make_plots.py` = `base`, unchanged.

### `RESULTS.md` — box is 284 lines behind, but owns one real table

The box copy predates all of section 12 ("Output: does streaming make writing
history cheaper?", 12.1-12.7, including *"Controls, and a false result this
lane produced and caught"*) and the "Allocated and integrated, not
extrapolated" subsection.

It does hold **one genuine contribution `base` has nowhere** — the box's own
ring-cost table, with two rows measured at plans the local side never ran:

| plan | tiles | ring, as a fraction of ONE domain |
|---|---|---|
| 1950², tile 650 | 9 | **5.16%** |
| 3276², tile 546 | 36 | 5.77% |
| 5412², tile 1353 | 16 | **2.39%** |
| 192², tile 64 (the gate size) | 9 | **45.4%** |
| 5182², tile 512 (21 ragged tiles) | 121 | **22.7%** |

with the guidance that at 5182² a tile of 512 leaves 21 ragged tiles and costs
22.7%, 768 leaves 13 and costs 11.7%, and a size that divides the domain leaves
none and costs 2-6%.

The 192² row is the gate's own geometry and explains why the gate's arena
fractions look alarming (a 64-cell tile with a 16-cell halo is nearly all
ring); the 5182² rows are a second, independent confirmation of the ragged-tile
cliff at a domain the local side never measured.

**Prescription:** merged `RESULTS.md` = `base` plus this table appended as a
clearly-labelled 8x4090 section. It is **not** appended in this commit, because
two of its rows disagree with `base`'s prose for the same plans (5.16 against
4.97; 5.77 against 5.84) and dropping a conflicting pair of measurements into a
results document without resolving them is how a number loses its condition.
The table is preserved verbatim at
`tilestream/rescued-tools/boxdelta-forks/RESULTS.md` and quoted here in full,
so nothing is lost while the re-measure is outstanding.

---

## What is still open after this merge

1. Ring cost disagrees between the two boxes at two shared plans (4.97/5.16 at
   1950²/650, 5.84/5.77 at 3276²/546). Needs one re-measure to close.
2. `rings.py` and the box disagree 4x at 3276²/tile 512 (24.46% against 6.3%).
   Resolved in `base`'s favour on the evidence above, not by measurement.
3. The box's `RESULTS.md` table is preserved but not folded into `RESULTS.md`,
   pending 1.
4. `LINUX-RESULTS.md`, now under `rescued-tools/`, leads with a capacity
   multiplier the handoff bans by value. See that directory's README.
