# ArWen profile-first performance survey

Date: 2026-07-31

Branch: `perf/profile-first-20260731`

Numerical-change base: `789f61181fb0b198ace10775f3ea184eb5e786a3`

Final model commit: `5953fb49902550896a6488efc3f4e10f3f630010`

## Result

The representative full-physics trace improved from 41.345 to 37.514
ms/model-step, a reduction of 3.832 ms or 9.27%. The trace contains the same
five model steps and cadence event in both arms.

The main measured changes are:

| Metric | Baseline | Final | Delta |
|---|---:|---:|---:|
| Projected wall, mean | 41.345 ms/step | 37.514 ms/step | -3.832 ms (-9.27%) |
| Projected wall, median | 40.509 ms/step | 37.101 ms/step | -3.408 ms (-8.41%) |
| GPU kernel time | 29.309 ms/step | 28.972 ms/step | -0.337 ms (-1.15%) |
| GPU memory-operation time | 1.185 ms/step | 0.910 ms/step | -0.275 ms (-23.17%) |
| GPU-empty/non-operation gap | 10.852 ms/step | 7.631 ms/step | -3.220 ms (-29.68%) |
| Kernel launches | 1,281.4/step | 1,203.2/step | -78.2 (-6.10%) |
| All GPU operations | 1,590.0/step | 1,408.6/step | -181.4 (-11.41%) |
| Scalar D2H operations | 27.6/step | 3.2/step | -24.4 (-88.41%) |
| Pool requests, same 10-step lane | 563.1/step | 298.7/step | -264.4 (-46.96%) |
| Pool-requested bytes, same lane | 3,099.207 MB/step | 1,730.841 MB/step | -1,368.366 MB (-44.15%) |
| Driver allocations in timed lane | 0 | 0 | 0 |
| Solver I/O (disabled in benchmark config) | 0 ms/step | 0 ms/step | 0 |

The improvements are not additive per commit: paired timings were noisy, and
each commit changed the operating point for the next. The cumulative
baseline/final trace above is the whole-pass result.

## Hardware and software receipt

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5090, 32,607 MiB, compute capability 12.0 |
| OS | Windows 11, build 26200 |
| NVIDIA driver | 610.74; CUDA driver API 13.3 |
| CUDA runtime used by CuPy | 12.9 |
| NVCC | 13.0.48 |
| Python | 3.13.7 |
| CuPy | 14.0.1 (`cupy-cuda12x`) |
| NumPy | 2.2.6 |
| Nsight Systems | 2025.3.2 |
| Nsight Compute | 2025.3.0 |

No dependency, vendor-tree, certified-output, tolerance, or test-gate change
was made.

## Benchmark method and sizing

### Repeatable small lane

`tools/benchmark_seeded_step.py` fixes seed `20260731`, dimensions
64 x 64 x 32, configuration, and step counts. Each receipt runs 10 warmup
steps per lane, 30 timed steps, and five section-attribution steps. Warmup,
NVRTC compilation, output hashing, and final reporting are outside the timed
region.

Before launch, the harness estimated 320 MiB of device memory, 55 executed
steps, no trace output for an ordinary receipt, and less than ten seconds.
Observed receipt size was about 4.7 KiB.

Three baseline receipts and three final receipts produced the same timed-lane
SHA-256:

```text
baseline: fc2ae8dfe3bc154a9b78f1af2f901da91ab7c3f046a67b10a151fd1e5d6a9338
final:    fc2ae8dfe3bc154a9b78f1af2f901da91ab7c3f046a67b10a151fd1e5d6a9338
```

The small dry lane was neutral, as expected because it does not execute the
full-physics seams changed in this pass:

| Metric, median of 3 | Baseline | Final | Delta |
|---|---:|---:|---:|
| Wall | 2.7709 ms/step | 2.7778 ms/step | +0.0069 ms (+0.25%) |
| CUDA-event GPU time | 2.7707 ms/step | 2.7776 ms/step | +0.0069 ms (+0.25%) |
| Pool requests | 111.0/step | 111.0/step | 0 |
| Driver allocations | 0 | 0 | 0 |

The five-step attribution lane is intentionally shorter than the 30-step
timing lane, so those two lanes are not hash-comparable to one another. Each
lane is deterministic across its three receipts.

Final small-lane section breakdown, median of three receipts:

| Section | Calls/step | GPU ms/step | Host submission ms/step |
|---|---:|---:|---:|
| Stage fluxes | 3 | 1.4354 | 1.5744 |
| Acoustic substeps | 7 | 0.6403 | 0.1677 |
| Boundary and damping | 17 | 0.3496 | 0.5021 |
| Slow tendencies | 3 | 0.3247 | 0.3043 |
| RK bookkeeping | 6 | 0.1967 | 0.0943 |
| Diagnostics | 4 | 0.0897 | 0.0553 |
| Acoustic setup | 3 | 0.0686 | 0.0407 |
| Held tendencies | 3 | 0.0151 | 0.0018 |
| Unattributed | 1 | 0.4077 | 0.7998 |
| Whole attribution lane | n/a | 3.5487 | 3.5472 |

Section host times are range-local enqueue times. They overlap queued GPU
execution and are not added to GPU time.

### Representative full-physics lane

The representative state is 250 x 200 x 49, or 2.45 million mass cells, with
the production physics bundle active. Output writing is disabled so I/O is
measured as zero rather than hidden in another category.

Each Nsight capture used two warmup steps and five captured steps. The
pre-launch estimate was 7.31 GiB, about 45 seconds, and less than 2 MiB of
trace data. The final report is 340,464 bytes. One cadence-first module load
appears in both comparable captures; there was no recurring per-step module
load or driver allocation.

The final projected wall decomposes as:

| Mutually exclusive component | Final ms/step | Share |
|---|---:|---:|
| GPU kernels | 28.972 | 77.23% |
| GPU copies and memsets | 0.910 | 2.43% |
| Non-GPU projected gaps: host submission and blocking boundaries | 7.631 | 20.34% |
| Solver I/O | 0.000 | 0.00% |
| Total | 37.514 | 100.00% |

Host Python was not independently isolated because `py-spy` was unavailable.
It is contained within, but is not equal to, the 7.631 ms non-GPU gap. The
7.258 ms/step of final CUDA API call time overlaps device execution. Solver
I/O is zero because output was disabled for this benchmark config; this is not
an estimate of enabled output-writing cost.

Transfer and API counts:

| Metric | Baseline | Final |
|---|---:|---:|
| D2D copies | 156.0/step | 75.0/step |
| Memsets | 121.0/step | 123.2/step |
| H2D copies | 4.0/step | 4.0/step |
| D2H copies / stream synchronizations | 27.6/step | 3.2/step |
| Summed `cuLaunchKernel` API time | 8.320 ms/step | 7.258 ms/step |

API time is summed CPU call time and overlaps device work. It is not added to
the mutually exclusive wall decomposition.

The final top kernels by total time are:

| Kernel family | Final ms/step | Share of kernel time |
|---|---:|---:|
| Morrison sedimentation | 4.032 | 13.92% |
| CuPy FP32 add kernels | 1.991 | 6.87% |
| Smagorinsky scalar flux | 1.916 | 6.61% |
| Acoustic `advance_w_phi_msf` | 1.851 | 6.39% |
| Smagorinsky W diffusion | 1.635 | 5.64% |
| CuPy FP32 multiply kernels | 1.124 | 3.88% |

This pass did not run a whole-step Nsight Compute roofline. The evidence
supports "device-execution-bound," not a stronger "bandwidth-bound" claim.

### Health-check lane

Health validation is outside the representative solver capture, so it was
measured separately on the same prepared state:

| Metric | Value |
|---|---:|
| Repeated validation wall | 7.529 ms/check |
| `validate_full_state` GPU time | 6.723 ms/check |
| Kernel share | 89.30% |
| Metadata upload | 49,168 bytes/check over 8 H2Ds |
| Result readback | 16 bytes/check |
| Production cadence used for amortization | 1 check / 4 steps |
| Amortized cost | 1.882 ms/model-step |
| Share of a 36.5 ms step | 5.16% |

An exact per-table metadata cache was prototyped. Production-like stepping
showed eight PBL pointers change every step and eight cumulus pointers change
on due steps. Three balanced repeated-check pairs improved by 0.044, 0.009,
and 0.001 ms/check; median 0.009 ms/check, or 0.002 ms/model-step at normal
cadence. The patch was reverted because the scan kernel, not descriptor
upload, is the measured cost.

## Tier A changes landed

All code changes below preserve model-update floating-point arithmetic and
ordering. Validation kernel topology changed, but those kernels are read-only.
Every model commit was compared with its direct parent in a detached worktree.

The exact gates were:

- `AB-137`: SHA-256 equality for 114 forecast, 9 dry-dynamics, and 14
  moist-dynamics fields.
- `WK82-29`: 29-field combined pair
  `b98f8dd7e18694e45d580969387922331d1057fa170c9039a03bfd0ed1d9044b`
  / the same digest.
- `YSU-REP-16`: direct-parent representative identity pair for `1b72431` /
  `8082cd1`,
  `1c67e6ac8325e8f279ca0b22356f35c757a6ffcee0a5439f14e20f96e2244142`
  / the same digest. This was an identity-only five-step rerun; its timing was
  not used for the performance claim.
- `REP-16`: representative 16-field combined pair
  `20fc2ccc9221ef7b7a416a9bbe0b2f304f01577c09d0cab79dabf36212a35282`
  / the same digest.
- `SMALL-9`: small deterministic pair
  `fc2ae8dfe3bc154a9b78f1af2f901da91ab7c3f046a67b10a151fd1e5d6a9338`
  / the same digest.

| Commit | Direct parent | Change | Measured direct-parent delta | Byte proof |
|---|---|---|---|---|
| `1b72431` | `789f611` | Add seeded GPU step benchmark | Baseline measurement only: 2.7709 ms/step | `WK82-29` |
| `8082cd1` | `1b72431` | Batch YSU output validation into one read-only status kernel | -1.095 ms/step traced (-2.65%); -44 launches, -14 D2Hs, -45 pool requests, -20.26 MB requested per step | `AB-137`, `YSU-REP-16` |
| `2f482aa` | `8082cd1` | Batch canonical microphysics validation | -1.466 ms/step traced (-3.64%); -26 launches, -8 D2Hs, -33 total GPU operations per step | `AB-137`, `WK82-29` |
| `5d015dd` | `2f482aa` | Skip known-clean KF expiry recovery probe | -0.019 ms/step traced (-0.05%, noise); -2 launches and -1 D2H per step | `AB-137`, `WK82-29` |
| `b9e32b6` | `5d015dd` | Write the hydrostatic recurrence into its destination | Paired median -0.594 ms/step (-1.57%); -49 requests and -9.80 MB/step | `AB-137`, `REP-16` |
| `bb76d4d` | `b9e32b6` | Reuse dead scalar-update scratch while preserving every ufunc boundary | Paired median -0.372 ms/step (-1.01%); -130 requests and -1.274 GB requested/step | `AB-137`, `WK82-29`, `REP-16` |
| `6fda9f4` | `bb76d4d` | Batch exact native KF validation, with strict production provenance and unchanged custom fallback | Paired median -0.371 ms/step (-1.00%); -7 D2Hs and -22 validation launches per due call; -4.8 requests and -3.006 MB/step at measured cadence | `AB-137`, `WK82-29`, `REP-16` |
| `a083785` | `6fda9f4` | Copy strict native KF outputs directly into persistent driver storage | Paired median -0.297 ms/step (-0.81%); -1.6 requests and -11.84 MB/step | `AB-137`, `WK82-29`, `REP-16` |
| `5953fb4` | `a083785` | Reuse `h_diabatic` for the consumed microphysics heating workspace | Paired median -0.225 ms/step (-0.62%); -5 requests and -49.0 MB/step | `AB-137`, `WK82-29`, `REP-16` |

The paired medians above come from three order-balanced parent/candidate
pairs. The individual wall deltas are retained below where noise affected the
decision:

| Commit | Candidate minus parent, ms/step |
|---|---|
| `b9e32b6` | -0.673, -0.594, +0.211; median -0.594 |
| `bb76d4d` | -0.372, +1.206, -0.659; median -0.372 |
| `6fda9f4` | -0.920, +0.455, -0.371; median -0.371 |
| `a083785` | -0.297, -0.928, +0.065; median -0.297 |
| `5953fb4` | -0.368, -0.225, +0.496; median -0.225 |

Negative values mean the candidate was faster. Deterministic allocation and
operation-count deltas were used alongside the noisy wall readings.

## Measured Tier A rejections and deferrals

No rejected patch remains in the worktree.

| Candidate | Measured result | Decision |
|---|---|---|
| Write stage-flux expressions directly to scratch | Byte-identical; -18 requests and -177.194 MB/step. Six balanced deltas: -0.304, -0.280, -0.008, +0.291, +0.483, +0.738 ms; median +0.141 ms regression | Reverted |
| Write Morrison preparation expressions directly to scratch | Byte-identical; -6 requests and -59.2 MB/step. Deltas +0.215, -0.656, +0.603 ms; median +0.215 ms regression | Reverted |
| Reuse `_pd_fold_sources` storage | Isolated path 0.613 -> 0.621 ms, +0.008 ms, with an additional alias-lifetime risk | Not implemented |
| Cache health descriptor tables | Median gain 0.009 ms/check = 0.002 ms/step at cadence; kernel itself is 6.723 ms/check | Reverted |
| CUDA Graph for clean RK acoustic tails | Direct six-node replay was byte-identical across 13 arrays. Three stage tails contain 10, 16, and 28 nodes; 54 launches could become 3. The measured GPU-empty ceiling is only 0.111922 ms/step (0.287%), despite 0.306 ms/step of overlapped CPU API time | Deferred |

The graph audit found no repository graph abstraction. Broader capture still
crosses pointer-changing CuPy expressions, due-mask topology, first-use module
loading, and required failure boundaries. Adding a parameter block, graph
cache, restart invalidation, and direct-path fallback for a measured
0.112 ms/step clean ceiling did not clear the engineering/risk gate.

The remaining ordinary-step scalar reads are one YSU status, one
microphysics status, and one KF expiry probe. They enforce current failure or
state-transition semantics. The open-boundary terrain probe and RUC ice probe
were inactive in this measured configuration, so no performance claim or
generic cache was made for them.

## Tier B proposals: report only

These were not implemented or committed because each can change floating
point behavior.

| Rank | Proposal | Numbered gain bound/estimate | Exact numerical impact | Risk |
|---:|---|---|---|---|
| 1 | Optimize or fuse Morrison sedimentation | Current cost 4.032 ms/step. A 10-25% kernel improvement would save 0.403-1.008 ms/step (1.07-2.69% of final wall) | Changes instruction scheduling, branch paths, and potentially contraction/rounding inside a long microphysics kernel | High |
| 2 | Fuse selected FP32 add/multiply chains | Current add + multiply cost is 3.115 ms/step. A 10-30% chain improvement would save 0.312-0.935 ms/step (0.83-2.49%) | Removes store/load rounding boundaries and may introduce FMA or a different evaluation order | High |
| 3 | Replace the Python hydrostatic level loop with one column recurrence kernel | The measured in-place recurrence costs about 0.679 ms/step, which is the hard upper bound before the new kernel's own work; maximum wall share 1.81% | A new compiler context can retain intermediates, contract operations, or alter the former per-level global-store rounding boundary | Medium-high |
| 4 | Reorder or change precision/math modes for reductions and exact transcendental paths | No recurring FP64/exact-math hotspot was measured in this five-step window, so the supported gain estimate is 0 ms for this profile | Reordered reductions, relaxed math, precision changes, and global contraction flags directly change certified results | High; do not pursue without a validating profile |

The upper bounds are not promises and are not additive.

## Structural limit

The largest structural limit after Tier A is device execution:

- 28.972 ms/step, or 77.23% of final wall, is kernel execution.
- The six largest kernel families account for 12.549 ms/step.
- The remaining non-GPU gap is 7.631 ms/step.
- Only 0.112 ms/step of that gap belongs to the largest currently clean,
  fixed-topology graph slice.

Therefore the next large improvement cannot come from graphing only the
already-clean acoustic tails. It needs either:

1. validation and pointer-stability work that makes a much larger topology
   capturable, with a new measured ceiling; or
2. Tier B kernel work under a numerical validation framework.

The profile does not justify a persistent whole-step kernel.

## Verification record and limits

- Relevant KF/physics/preflight suite: 151 passed.
- Affected KF/physics suite after native copy removal: 72 passed.
- Cross-scheme microphysics suite: 125 passed.
- Focused health/corruption prototype suite: 32 passed before that prototype
  was reverted.
- Direct-parent `AB-137` and `WK82-29` gates passed after each late model
  commit; earlier commits were gated before the next forward commit.
- No tolerance or test was widened.
- The full repository suite was not run. A marker-discipline test invocation
  did not finish within a 120-second diagnostic timeout at the harness
  commit; it was not changed or represented as passing.
- `py-spy` was not installed. Host attribution therefore uses Nsight CUDA API
  correlation and projected GPU gaps, not a Python-bytecode sample.

All benchmark and profile launches were sized before execution. No individual
GPU benchmark approached the 30-minute stop boundary; the longest balanced
six-arm set completed in 178.2 seconds.
