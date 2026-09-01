# Advancing ensemble members concurrently

> **STATUS on this release line, 2026-08-31: the machinery is here, the
> driver flag is NOT.** `tools/da_member_leg.py` (the worker, the
> `MemberPool` scheduler and the VRAM refusal) and
> `tools/da_identity_check.py` are folded and tested --
> `tests/test_da_member_pool_dispatch.py` exercises the scheduling
> against real fake workers over the real stdin/stdout protocol. What is
> not folded is the `--member-workers` flag on
> `tools/da_cycle_prepared.py`.
>
> The reason is specific rather than general. This lane extracted the
> member leg out of an 851-line driver. That driver has since grown to
> 1831 lines inside a single `main()`, and gained 31 flags the extracted
> worker knows nothing about -- surface observations, GOES CWP, the
> nested forecast, ensemble save/resume, clear-air reflectivity,
> treatment verification. Wiring the flag without porting those would
> give a concurrent run that silently skipped them, which is precisely
> the bit-identity claim below inverted: the two arms would differ, and
> nothing would say so. So the flag waits for the port, and the numbers
> below describe the arms as this lane measured them, not as this line's
> driver runs them.
>
> `tests/test_da_member_leg.py` is held back with the flag, because it
> asserts a concurrent arm this driver does not have. It must land WITH
> the port, so the identity claim is tested where it is made.

The cycling radar-DA driver advanced its trajectories one at a time:
eleven fresh models built and torn down per leg for a ten-member run.
`--member-workers N` advances N of them at once. The default is 1 --
the serial path, unchanged.

    tools/da_cycle_prepared.py ... --member-workers 3

## What the measurements said, and why the design is not streams

The expectation going in was CUDA streams: give each member a
non-blocking stream, remove the device-wide barriers, let the members
overlap. That expectation was wrong, and it was wrong for a reason worth
writing down.

A member leg was never split into host setup and GPU integration. It is
now (`--member-workers` aside, every run reports `phase_seconds` per
trajectory). On the live-fire-3 shape, 132x132x49 with dt 15 s:

| phase | seconds |
|---|---|
| `wire` -- prepared-cache read, per-array SHA-256, land-use, physics init | 0.10 |
| `integrate` -- 60 GPU steps | 1.7 - 5.0 |

So hoisting the cache restore out of the member loop, the cheap win that
was expected to dominate, is worth 3-5% and no more.

But the integration is not GPU-bound either. `tools/da_probe_phase.py`
brackets `execute_experiment` with CUDA events and cProfile at the same
time:

    wall 5.02 s | event span 5.02 s | profiler total 5.02 s

Those being one number means the interpreter is busy for essentially the
whole leg while the device waits. The costs, per leg:

| what | seconds |
|---|---|
| Dudhia shortwave (a 49-level Python loop issuing thousands of tiny CuPy ops) | 1.94 |
| 120 blocking `.item()` calls | 0.77 |
| 120 `free_all_blocks` calls from the per-step pool trim | 0.71 |

A workload like that does not overlap on streams, because what needs
overlapping is Python. The probes confirm it:

| arrangement | width 2 | width 4 |
|---|---|---|
| threads on non-blocking streams | **0.97x** | **0.83x** |
| separate processes | **1.40x** | **1.46x** |

Threads lose outright -- the GIL serializes the dispatch and contention
makes it worse. Processes overlap but saturate almost immediately:
0.287, 0.400, 0.414, 0.419, 0.414 legs/s at widths 1, 2, 3, 4, 6. The
box has 32 cores and they are not the limit; separate CUDA contexts
time-slice under WDDM, and Windows has no MPS. **The ceiling is ~1.45x
and width 2 buys 1.40x of it.**

This is also why nothing here folds a member axis into the kernels.
That rewrite touches 100 `__global__` entry points across 26 modules,
moves a `source_sha256` the kernel manifest and the FTZ receipt pin, and
optimises the half of the machine that is already idle.

## Bit-identical, and how that is known

Members are mathematically independent, so any difference between the
arms would be a defect -- in shared state, allocator ordering,
handle-to-stream association, or a cross-member reduction -- never
acceptable noise. Three things make the claim structural rather than
hopeful:

- **One body, two arms.** `tools/da_member_leg.py::run_member_leg` is
  the driver's old loop body. The serial path and the worker call that
  same function; the concurrent arm is not a second implementation.
- **Order cannot reach the analysis.** Results are keyed by trajectory
  name and consumed in `trajectories` order, and the LETKF still reads
  its backgrounds from disk in `sorted()` index order.
- **Isolation is asserted.** `run_member_leg` refuses a model carrying a
  shared scratch arena or the shared dycore-state workspace. That
  failure mode is silent cross-member corruption, not an exception, so
  it is checked rather than assumed.

Proven with `tools/da_identity_check.py`, which compares raw bytes
(`np.array_equal`, NaN-aware) and fails on one ULP. On the smoke case
(100x80x49, five trajectories, two legs, a real LETKF analysis applied
between them), 727 arrays and report leaves per comparison:

| arms | result |
|---|---|
| serial vs `--member-workers 2` | BYTE-IDENTICAL |
| serial vs `--member-workers 3` (uneven split) | BYTE-IDENTICAL |
| serial vs `--no-pool-trim` | BYTE-IDENTICAL |
| pre-refactor driver vs post-refactor serial | identical on all 537 shared leaves |

Run it yourself with `--keep-member-snapshots` on both arms, which
writes every trajectory's leg-end state -- the thing the next leg
restores -- to `<out>/snapshots/legNNN/<name>.npz`.

## The device budget

The DA driver hand-builds its `ExperimentState` and so bypasses
`estimate_experiment`; this route has never had a VRAM gate at all.
`--member-workers > 1` now prices the one thing width changes -- a CUDA
context and a local-memory backing store per process -- and refuses
before spawning:

    --member-workers 40 does not fit this card: each worker process holds
    its own CUDA context and local-memory backing store, priced at 3038
    MiB, so 40 of them need 121516 MiB against 30994 MiB free. Use
    --member-workers 10 or fewer.

Measured per-process cost is 3151-3236 MiB against 3038 estimated, so
the model and the card agree to about 3%.

## `--no-pool-trim`

Separate from concurrency, and a serial win: the per-step pool trim
costs 0.87 s of a 5.0 s leg (4.99 -> 4.12 s) and is documented
byte-inert -- `free_all_blocks` releases only unused cached blocks. It
stays **on** by default, because it was added for a measured reason on a
different shape (4-domain real74, where pool churn drove WDDM page
demotion and cost 32% wall time). This single-domain route does not have
that problem, but the default is not this route's to change.

## Adoption by the nowcast front door

`tools/da_nowcast.py` is owned elsewhere and is untouched here. When it
wants this, it is one argument passed through to the cycle driver:

    cycle_cmd += ["--member-workers", str(args.member_workers)]

with a matching `--member-workers` on its own parser defaulting to 1.
Nothing else changes: the report's `execution` block records the width
and the trim setting, so the receipt says which arm produced it.

## Honest limits

- The end-to-end A/B at the full twelve-leg nowcast scale has not been
  measured on an idle card; the card was taken by another job partway
  through. The per-leg throughput figures above are from an idle card
  and are what the speedup claim rests on.
- The identity proof is on the five-trajectory smoke case, not the full
  nowcast. A longer run is a stronger receipt and has not been taken.
- Single-run measurements on a box with no ECC. These are engineering
  numbers, not certification evidence.
- The 1.45x ceiling is a property of this card and WDDM. A Linux box
  with MPS may behave differently and has not been tested.
- **The biggest lever is not in this branch.** Dudhia shortwave is 1.94 s
  of host-side CuPy dispatch, about half a leg. Making that one routine
  issue fewer, larger kernels would beat every arrangement of members
  here. It is a physics-code change needing its own certification.
