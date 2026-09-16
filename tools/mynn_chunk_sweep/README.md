# MYNN column-chunk sweep

Measures what the MYNN column width is worth on one card, and proves the
forecast does not depend on it.

## Why there is a sweep at all

`mynn_dmp_mf_columns` integrates eight rising plumes up a column, one CUDA
thread per column, and it is **52% of all GPU time in a quiet root step** of
the 2.7 km + 900 m pair. It was launched 128 blocks of 128 threads on a
170-SM card: 42 SMs given nothing, one block on every SM that got any, 4.7%
of the card's resident thread slots, against a register file that admits
three blocks per SM. The 16,384-column width 2.7.4 shipped came from a sweep
taken at `nz = 49` on a card with fewer SMs than the chunk had blocks.

That reasoning produced a derivation
(`gpuwm/core/mynn_pbl_scratch.py::derive_mynn_column_chunk`): the largest
chunk whose blocks fill the card in one wave, clipped by what the card's
memory admits, never below the 16,384 columns 2.7.4 shipped. On a 170-SM
card with a 65,536-register file that is **65,280 columns, 510 blocks**, and
3,919 MiB of workspace at `nz = 59` against 984 MiB at the floor.

**Then this sweep ran, and it refuted the reasoning.** Ten arms on an
RTX 5090 at `nz = 59` on 2026-09-15, sole tenant: the quiet root cycle is
monotonic in the width, with no knee anywhere in the range.

| columns | 16,384 | 32,768 | 45,056 | 65,536 | 90,112 |
|---|---|---|---|---|---|
| quiet root cycle | **0.5263 s** | 0.5819 s | 0.6170 s | 0.6252 s | 0.6395 s |
| pass spread | 0.0006 s | 0.0017 s | 0.0002 s | 0.0007 s | 0.0016 s |
| workspace | 984 MiB | 1,967 MiB | 2,705 MiB | 3,935 MiB | 5,410 MiB |

So the shipped width is a measured one and the derivation is kept as a
receipt term (`memory.mynn_column_chunk.would_have_derived`) so the next
card's sweep has something to argue with.

**But that sweep had only looked one way, and the other way moved the
answer.** It started at the width 2.7.4 shipped and only ever widened, so
"narrower is faster" was the whole of what it could find and it ranked its
own narrowest arm first. Five widths at and below the shipped one, same
card, same prepared leg, same evening, same protocol:

| columns | 4,096 | **8,192** | 12,288 | 16,384 | 24,576 |
|---|---|---|---|---|---|
| quiet root cycle | 0.5039 s | **0.4450 s** | 0.4819 s | 0.5268 s | 0.5353 s |
| workspace | 246 MiB | **492 MiB** | 738 MiB | 984 MiB | 1,475 MiB |

**The optimum is 8,192 columns and it is interior**: 15.5% cheaper per quiet
root cycle than 16,384 on half the workspace, leading the runner-up by
0.0369 s against a worst pass-to-pass spread of 0.0037 s anywhere in the
sweep, and with the cost turning back up by 13.2% at 4,096, where the launch
count starts to dominate. A minimum with a slower arm on each side is a
measurement; a winner at the end of a ladder is only a direction. The two
sweeps overlap at 16,384 and agree there to 0.0005 s five hours apart, which
is what lets the nine widths be read as one curve, and all twenty arms wrote
one digest. Confirmed on the full 1,500-step pair as well: 2,877.5 s to
2,390.5 s with all 244 frames byte-identical.

`MYNN_PBL_COLUMN_CHUNK_DEFAULT` is therefore 8,192 columns, and
`MYNN_PBL_COLUMN_CHUNK_FLOOR` stays at 16,384 as the lower bound of the
derivation that rides the receipt. What this harness is now for is answering
the derivation's argument on another card, and answering the open question on
this one: **why** narrow wins. The hypothesis is a per-chunk cost that scales
with the ALLOCATED width rather than the columns used -- the outer domain's
89,401 columns waste 9% of the width at 16,384 and 32% at 65,280, the nest's
36,378 waste 26% and 44% -- which the turn at 4,096 bounds from below without
explaining.

## Running it

It is a long job on the card, so it runs detached and looks after the mutex
itself: it waits for an idle card and an absent `OWNER`, claims `OWNER`,
runs, writes `DONE` or `FAILED`, and releases `OWNER`.

```sh
# on the card's host, from a checkout of this lane
tmux new -d -s mynn-chunk 'bash tools/mynn_chunk_sweep/sweep.sh'

# an explicit ladder: the widths are the question, so name them
tmux new -d -s mynn-chunk \
  'bash tools/mynn_chunk_sweep/sweep.sh --chunks "4096 8192 12288 16384 24576"'

# smoke test first: two widths, ten root steps, its own work dir
bash tools/mynn_chunk_sweep/sweep.sh --dry
```

`--chunks` outranks `CHUNKS` in the environment and the `--dry` shorthand. A
width that is not a positive column count, or one listed twice, is refused
before the card is claimed: the list sizes every arm's workspace and names
every row of the ranking, and two arms of one pass at the same width would
overwrite each other's run directory and digest list.

Wait on `$W/DONE` (or `$W/FAILED`), never on the process.

### What it needs

| | default | what it is |
|---|---|---|
| `PREPARED` | `$ROOT/baseline/pair/ctrl/prepared` | a prepared control leg, read-only; the sweep never prepares one |
| `BASECONF` | `$ROOT/baseline/pair/ctrl/experiment.toml` | that leg's config; the sweep copies it and rewrites `run_seconds` only |
| `ENGINE` | `$HOME/gpuwm-venv` | the installed engine to time |
| `W` | `$ROOT/mynn-chunk-sweep` | where everything it writes goes |
| `CHUNKS` | `4096 8192 12288 16384 24576` | the widths, in columns, bracketing the shipped one; `--chunks` outranks it |
| `PASSES` | `2` | how many times to run the ladder; even passes run it in reverse |
| `STEPS` | `200` | root steps per arm (`run_seconds = STEPS x DT`) |
| `WARM` | `5` | root cycles dropped before timing, for the first-step compile |
| `KEEP_FRAMES` | `0` | `1` keeps each arm's frames after hashing them |
| `FREE_GIB` | `60` | refuses to start below this, rather than filling the disk mid-sweep |

Two passes is the default on purpose. The baseline run of record was
contaminated by another lane's job from root step 146 onward, and an arm that
won only because it ran while the card was cool or alone must be visible as a
disagreement between passes rather than as a result.

### Foreign jobs

Another lane's jobs use this card **without** the mutex. `cardwatch.tsv`
samples `nvidia-smi` every 5 s for the whole sweep, and every arm's row
carries two counts over the samples inside that arm's window: `foreign_seen`,
samples that saw a compute process from outside `ENGINE`, and `own_seen`,
samples that saw the arm's own forecast. Nothing is ever killed or reniced.

`foreign_seen` used to count every sample whose process list was non-empty,
which includes the arm's own run, so it read 32 to 42 for arms that were sole
tenant and could never have read zero while a forecast was running. A column
that cannot take the value it is read for says nothing; it is now tested
against the engine install's own path prefix. `own_seen` is the other half of
the same check: an arm with `own_seen` at 0 means the sampler was idle, not
the card.

## What comes out

| file | what it is |
|---|---|
| `sweep.tsv` | one row per arm, 24 columns, header included |
| `digests/c<width>-p<pass>.txt` | `sha256sum` of every frame that arm wrote |
| `summary.txt` | the table, the ranking, and the identity verdict |
| `IDENTITY` line in `summary.txt` | `EQUAL`, `MOVED` (with the first frame that disagreed), or `UNPROVEN` |
| `DONE` / `FAILED` | the marker to wait on; `FAILED` carries the log tail |
| `cardwatch.tsv`, `stages.tsv`, `logs/` | card samples, stage boundaries, one log per arm |

The timing column that matters is `quiet_median_s`: the median wall time of a
root **cycle** (the parent's step plus the nest sub-steps that follow it),
over the cycles in the quiet population. `summarise.py` splits the cycles the
way the profile of record splits them, using each arm's own median rather
than a threshold measured on another card, and reports the two radiation
populations separately.

`receipt_chunk` and `receipt_source` come from the run's own receipt
(`memory.mynn_column_chunk`). The collect step **refuses to rank** if any arm
ran a width other than the one it was given: an override that did not reach
the run would otherwise produce a timing and a width that do not belong to
each other.

## The identity contract

Every arm must write byte-identical frames. Not close: identical.

Every MYNN column kernel gives one thread one whole column, reads no
neighbour, and holds no shared memory, no atomic and no per-chunk seed
(`gpuwm/core/kernels/mynn_pbl.cu`), and the walk writes each chunk's results
into its own slice of the output fields. The width is therefore workspace
shape and nothing else.

So `IDENTITY: MOVED` is a defect to find, not a tolerance to widen. Stop, and
look for what in the walk is not column-local. `tests/test_mynn_pbl_scratch.py
::test_the_column_chunk_is_not_a_seam` makes the same claim on a small
carried forecast; this sweep makes it on the real leg at every width in the ladder.

## Reading the result

The number to take back is the width whose median is lowest **and whose lead
over the next width is larger than the worst pass-to-pass spread** -- the
collect step says so explicitly when it is not. Then:

- If the winner is `MYNN_PBL_COLUMN_CHUNK_DEFAULT`, the shipped width stands
  and the run says so -- but only if the ladder had a slower arm on BOTH
  sides of it. A winner at the end of the ladder settles nothing except
  which way to run the next one.
- If the winner is another width, **the default moves to it** in a commit
  that cites this sweep by date and by the path of its `summary.txt`, and the
  test in `tests/test_mynn_column_chunk_derivation.py` that pins the default
  to the measured optimum moves with it. A width that wins a sweep and is
  left as an override is a tuned value living in an escape hatch.
- Either way the run's receipt keeps `would_have_derived`, so the derivation
  this card would have used is on the record beside the width that ran.

`GPUWM_MYNN_COLUMN_CHUNK` is the sweep's handle and an operator's escape
hatch. It is not where a tuned value lives: whatever a sweep wins ships as
the default, which is what "fixed means default" requires of a measurement.
