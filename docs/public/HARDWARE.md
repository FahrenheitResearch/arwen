# Hardware and VRAM sizing

ArWen runs on one NVIDIA GPU with CUDA 12.x. This page explains how
the sizing model works, where its safety factor comes from, and what
we measured on real hardware -- including the run where the estimator
was wrong and what changed because of it.

## The short version

Tell the wizard your card; it sizes the grids:

```bash
gpuwm domain --point 35.3,-97.5 --card 24gb ...    # or --vram-gib N
```

| card tier | flat reserve | working budget | what fits (measured examples) |
|---|---|---|---|
| 16 GiB | 3 GiB | 13 GiB | full 12-3-1-0.5 km four-domain ladder at ~3.3 GiB alloc estimate |
| 24 GiB | 4 GiB | 20 GiB | four domains: 170x136 (12 km), 336x272 (3 km), 360x294 (1 km), 288x236 (500 m) |
| 32 GiB | 6 GiB | 26 GiB | the reference-class case: 4 domains to 500 m at 400x400+ |

A first single-domain forecast is far below any of these: the
acceptance run (250x200x49 at 12 km, full physics) used ~6.3 GiB of
device memory and completed 6 simulated hours in 3.6 min on an
RTX 5090 ([FIRST-LIGHT.md](FIRST-LIGHT.md)).

## How the estimate is built

`gpuwm check` (and the wizard, which calls the same estimator
in-process) prices a run in three layers:

1. **Itemized alloc estimate** -- every persistent field, scratch
   arena, and kernel workspace, summed per domain. This is the number
   the hard pass/fail gate compares against your measured free VRAM
   minus `--reserve-gib`.
2. **Footprint projection** -- the alloc estimate plus transient
   call-peak envelopes (radiation chunk workspaces are the largest).
3. **Projected machine peak = footprint projection x 1.75**, compared
   against the card budget (capacity minus the flat reserve). The
   wizard bisects grid sizes until this fits; a wizard-emitted config
   passes `gpuwm check` with zero warnings by construction.

Example (24 GiB tier, printed by the wizard):

```
  itemized alloc estimate 7.17 GiB; footprint projection 11.29 GiB
    x 1.75 observed peak envelope = 19.75 GiB
  budget 20.00 GiB (24 GiB card - 4 GiB reserve); headroom 0.25 GiB
```

## Where 1.75 comes from (the honest part)

The 1.75 factor is not a safety margin picked to look prudent; it is a
measurement of the estimator being wrong, kept visible.

On the four-domain reference run (2026-07-28, RTX 5090 32 GiB,
Windows), the preflight projected a 16.22 GiB footprint. The measured
machine-wide peak was 29,004 MiB -- 1.75x the projection -- and it
finished 57.1 MiB (0.2%) under the 30,472,743,936-byte Windows WDDM
budget the gate had checked. The gate passed for the wrong reason: it compared the smaller
alloc estimate against the budget. Rather than quietly retune the
estimator, the measured peak-to-projection ratio became a mandatory
multiplier on every sizing decision, and the wizard's fit criterion is
`footprint x 1.75 <= budget`. On single-domain runs the projection is
conservative in the other direction (the acceptance run projected
12.78 GiB; the model used ~6.3 GiB) -- the envelope factor costs you
grid points, not correctness.

If you hand-build a config, run `gpuwm check CONFIG --alloc` before
the first long run: `--alloc` actually allocates the estimate on the
device and verifies the three-way inequality (measured pool peak <=
estimate <= budget) instead of trusting arithmetic.

## Windows / WDDM notes

- On Windows the display driver (WDDM) owns device memory; the
  usable budget is what the driver grants, not the sticker capacity.
  The preflight reads the real budget and prices against it (measured
  30,472,743,936 B granted on a 32 GiB card).
- Desktop compositing holds VRAM (~3.2 GiB on the acceptance
  machine's desktop). `gpuwm check` measures *free* VRAM at check
  time; close what you can before a big run.
- Consumer GeForce cards have no ECC. We treat sustained operation
  near the WDDM budget as a reliability risk, not an achievement:
  the flat tier reserves (3/4/6 GiB) exist so routine runs never
  operate there. The reference run that peaked 57.1 MiB under budget
  completed cleanly and bit-deterministically -- and is exactly the
  margin the sizing model now prevents.
- Redirected stdout is block-buffered on Windows; watch
  `run-progress.json`, not the log tail
  ([FIRST-LIGHT.md](FIRST-LIGHT.md#5-run-measured-6-h-forecast-in-36-min)).

## Linux notes

- No WDDM: the budget is the CUDA-reported free memory minus your
  `--reserve-gib`. The same estimator and envelope factor apply.
- CUDA preprocessing and the deterministic Rust CPU preprocessing
  backend are both exercised on Linux; the sealed Linux runtime
  archive bundles the bridges and CPU library
  ([docs/install.md](../install.md)).
- The stock-WRF interoperability receipts (serial and 12/24-rank MPI)
  were produced on Linux nodes ([WRF-INTEROP.md](WRF-INTEROP.md)).

## Throughput reference points (all measured, RTX 5090)

| workload | rate |
|---|---|
| 250x200x49 single domain, full certified physics, dt 60 s | ~0.55 s/step incl. output; 6 h in 3.6 min |
| four domains 12/3/1/0.5 km to 400x400, matched-run configuration | 67.2 wall-s per simulated minute whole-tree (61.4 pre-convective, 72.9 convective) |
| 500 m offline downscaled child, 400x400x49, dt 2.5 s | 0.91 s/step warm; 3 h in 66 min |
| legacy RRTMG vs RTE+RRTMGP (same 3-domain stack, radt 12/3/1) | 34.8 vs 18.7 wall-s per simulated minute |

Absolute numbers are properties of that machine (they vary up to ~30%
between sessions on the same box); ratios travel better than absolutes.

## FP32, subnormals, and GPU-model caveats

The model state is FP32 throughout, matching WRF's default REAL. Two
hardware-level facts worth knowing:

- Kernel compilation flushes FP32 subnormals to zero (CuPy compiles
  with FTZ), and on some architectures subnormal flushing is
  unconditional in hardware. The measured consequences are branch
  flips on physically negligible inputs; each known instance is
  recorded in the physics registry ([PHYSICS.md](PHYSICS.md)), and the
  radiation preparation path routes one subnormal-sensitive block
  through the host by design.
- Determinism holds per build and hardware: the reference run
  reproduced output frames SHA256-identically across a mid-run kill
  and relaunch. No cross-GPU or cross-driver bit-identity is claimed.
