# The fast-fix lane

**Target: defect reported to fix published, under 90 minutes.**

Designed by the 2026-08-13 test-estate audit, built here. This document is the
procedure; the pieces it names are in this directory.

The single structural idea is this: **the full battery never blocks a publish.**
It runs beside the cut and becomes a post-publish alarm. That is what gets a fix
out in under 90 minutes, and it is only safe because the blocking lane carries
the user door.

---

## The shape

```
  T+00  branch from the LAST TAG, not from integration
  T+05  write the fix and its covering test
        (the test comes first, and it must be RED before the fix)
  T+15  ---- fork ----
        LANE A (blocking, budget 25 min)      LANE B (parallel, budget 35 min)
          A1  user-door gate                    the FULL battery, every tier,
          A2  fastfix selection                 on the fix branch
          A3  the ALWAYS list
  T+40  Lane A green  ->  tag, build, publish
  T+50  published
  T+75  Lane B lands.  Green: nothing happens.
        Red: the post-publish alarm names the file and the assertion.
```

### Why branch from the last tag

Because `integration/*` carries whatever landed since, and a fast fix must not
inherit an unrelated red. The audit reproduced nine reds on the release line,
and **not one of them was in a file any battery leg ran** — a fix branched off
integration would have been asked to explain all nine.

---

## Lane A, step by step

### A0. Write the covering test first, and watch it fail

Not ceremony. A fix whose test has never been red is a fix with no evidence
that it fixes anything, and the fast lane deliberately removes the wide net
that would otherwise catch that.

```
pytest tests/test_<the_area>.py::<the_new_test>     # MUST be red here
<apply the fix>
pytest tests/test_<the_area>.py::<the_new_test>     # green
```

### A1. The user-door gate

A release proves its user doors. `gpuwm` registers 28 subcommands; `stream` is
one of them, and the streaming miss that ordered the audit shipped through it.

Run the door sweep, plus the two permanent matrix rows (`run` on the smallest
shipped single-domain config, `stream` on the smallest `[tiles]` tree), plus
any row the fix touches.

> **Status.** The door sweep and the door matrix are the one piece of the
> audit's five that this lane does not yet ship. Until they exist, Lane A is
> not complete and a fast fix must run the touched door by hand and say so in
> the release notes. This is stated rather than papered over: an undocumented
> gap in a lane whose whole purpose is speed is how the next miss ships.

### A2. The changed-area selection

```
python tools/battery/fastfix.py --base <last-tag> --head HEAD
```

prints the test files the change can break, **cheapest first**, each with the
reason it was selected, and an estimated wall time. To run them:

```
pytest $(python tools/battery/fastfix.py --base <last-tag> --format pytest) \
       -q -m "not slow"
```

Selection is by **direct imports only**. That is the audit's measurement, not
a preference:

| | mean closure | selects, per touched product file |
|---|---|---|
| transitive closure | 186 modules | **441 of 591 test files (75%)** |
| direct (depth 0) | 5.6 modules | 1% to 12% |

and both were validated against a real defect rather than argued about.
Commit `6e9c690f0` touched `gpuwm/core/moist.py` and `gpuwm/core/dycore.py`
and broke two things. Depth 0 catches it. Depth 1 (186 files) and depth 2
(276 files) catch nothing extra for three times the cost.
`tests/test_fastfix_selector.py` pins that commit so the setting cannot drift
on a hunch.

### A3. The ALWAYS list

`tools/battery/always_files.txt`, unconditionally, every time.

The selector **structurally cannot see repo-scanning gates**. A citation
checker, a line-ending gate, a release-note gate, a receipt regenerator has no
import edge to anything. The same commit `6e9c690f0` also left
`tools/ftz_receipt/receipt/route_inventory.json` stale, and the gate that
catches that is a miss at every import depth.

`fastfix.py` adds the list itself, so A2 and A3 are one command. A **non-Python
change** — a `.cu`, a config, a receipt — selects the ALWAYS list and nothing
else, because import analysis has nothing to say about those files and
pretending otherwise would be worse than admitting it.

### Lane A budget

| step | seconds |
|---|---|
| index build | 9 |
| door sweep | 2 *(to build)* |
| door matrix, 2 permanent rows + touched rows | 120 to 300 *(to build)* |
| ALWAYS list, 19 files | 37 |
| fastfix selection, typical fix | 60 to 400 |
| **total** | **under 12 min typical, 25 min worst case** |

That leaves 65 minutes for writing the fix, cutting and publishing, against
the 90 minute target.

---

## Lane B: the full battery as an alarm

Start it at the fork, on the fix branch, and let it run past the publish:

```
pytest $(grep -v '^\s*#' tools/battery/stage1_files.txt | grep .) -q
```

plus the cargo leg, the tiles leg, and the GPU shard
(`tools/battery/gpu_shard_files.txt`, on a designated GPU node, with
`GPUWM_NO_LOCAL_GPU` **unset**).

**A red in Lane B is an alarm, not a rollback.** It names the file and the
assertion and it pages. It rolls back only when it names a **user-door row** —
that is, when the thing it caught is something a user's first command would
hit. Anything else is a fix on the next branch.

This is the whole trade, and it is worth writing down plainly: the lane accepts
that a defect Lane A does not cover can reach PyPI for the length of one battery
run. What it buys is that the defect being fixed right now reaches users in
under 90 minutes instead of over a working day. That trade is only defensible
while Lane A carries the user door, which is why A1 is the one piece that may
not be skipped once it exists.

---

## Tiers, and what "slow" means here

`slow` is a **cadence, not a deletion**. Nothing marked `slow` is excluded from
a cut.

| tier | when | what |
|---|---|---|
| A | every merge / the fast lane | door sweep, ALWAYS list, fastfix selection |
| B | every cut | `stage1_files.txt`, `cargo_gates.txt`, `tiles_gates.txt`, `gpu_shard_files.txt` shard 1 |
| C | weekly | `-m slow`: the exhaustive census sweep, the frozen-oracle sweeps, `gpu_shard_files.txt` shard 2 |
| D | post-publish alarm | everything, on the published commit, in parallel with the cut |

Run tier A and B with `-m "not slow"`. Run tier C with `-m slow`.

---

## Durations, and why they are tracked

`tools/battery/durations.json` holds measured per-file seconds.
`fastfix.py --record` regenerates it, and the selector uses it to sort
cheapest-first so a red arrives early. Files with no measurement are printed
with a `~` so an estimate is never read as a measurement.

This exists because of audit finding G11: `test_health_field_census.py`
documented its own cost as "about 65 s" and measured **541 s**. Nothing in the
project measured test cost, so the fast/slow split was being maintained from
stale prose. A number in a tracked file is harder to be wrong about for a year
than a sentence in a docstring.

---

## What is built and what is not

**Built and in this directory**

- `fastfix.py` — the selector, with `--format report|pytest|json` and `--record`
- `always_files.txt` — the unconditional repo-scanning gates
- `gpu_shard_files.txt` — the GPU pytest shard (gap G2), shard 1 per cut, shard 2 weekly
- `stage1_files.txt`, `cargo_gates.txt`, `tiles_gates.txt` — the existing legs
- gates: `tests/test_fastfix_selector.py`, `tests/test_gpu_shard_manifest.py`,
  `tests/test_stage1_manifest.py`, `tests/test_cargo_gate_manifest.py`,
  `tests/test_tiles_gate_manifest.py`

**Not built**

1. **The user-door sweep and the door matrix (A1).** The largest remaining
   item and the highest value: it is the gate that answers the miss that
   ordered the audit.
2. **The parallel-battery runner and the post-publish alarm.** Mechanically
   small. The discipline is the part that matters, and it is written above.

**A prerequisite, not an optional**

The assembly venv must be built with `pip install -e .[dev,render]` **and must
have setuptools**. Today `[dev]` is `pytest` and `psutil` only, and the `wrf`
package lives in `[render]`. A venv without them silently skips 52+ tests, and
`tests/test_package_data_coverage.py` was measured 56% dark inside stage 1 for
exactly that reason. It now fails rather than skipping — but a fast lane running
in a venv that cannot execute its own gates is just a faster way to ship the
same class of miss.
