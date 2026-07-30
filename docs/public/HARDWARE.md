# Hardware and VRAM sizing

ArWen runs on one NVIDIA GPU with CUDA 12.x or 13.x, field-verified
through 13.2 driver stacks on sm_89 by two independent nodes. This
page explains how
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
| 12 GiB | 3 GiB | 9 GiB | Four domains: 156x126 / 312x256 / 336x276 / 268x220, or 398x318 single-domain at 12 km. **Windows: experimental** (see below) |
| 16 GiB | 3 GiB | 13 GiB | full 12-3-1-0.5 km four-domain ladder at ~3.3 GiB alloc estimate |
| 24 GiB | 4 GiB | 20 GiB | four domains: 170x136 (12 km), 336x272 (3 km), 360x294 (1 km), 288x236 (500 m) |
| 32 GiB | 6 GiB | 26 GiB | the reference-class case: 4 domains to 500 m at 400x400+ |

### 12 GiB on Windows is an EXPERIMENTAL tier

The 12 GiB tier is fully measured on Linux. On Windows it is a pioneer
tier, and the wizard says so on every sizing it prints.

Windows/WDDM accounting in gpuwm comes from exactly ONE machine: a
32 GiB RTX 5090 running campaign-scale multi-domain forecasts. Applied
literally, its two fixed pool constants are 4.12 GiB -- a third of a
12 GiB card before a single grid cell exists -- and the 1.75 envelope
on top of them exceeded a 9 GiB budget at the *smallest layout the
wizard can build*. Every ladder was refused, for an accounting term
measured somewhere else.

Windows cards at or below 12 GiB are therefore sized like Linux -- the
itemized alloc estimate under the 1.45 envelope -- plus a single
reduced 1.5 GiB fixed reserve standing in for the WDDM residency the
CuPy pool never sees. Windows cards of 16 GiB and up are unchanged.

What that risks, plainly: the layout may be optimistic. The worst case
is paging (slow) or a clean out-of-memory failure before or during the
run. **Neither corrupts a forecast and neither damages anything** --
which is why sizing optimistically is the better failure here than
refusing a card gpuwm can probably run. `gpuwm check` is unchanged: it
still warns rather than blocks.

**Please send the calibration back.** One measured peak from a real
Windows small-card run is worth more than every estimate on this page.
Run the forecast, then report the peak line from
`gpuwm check <config>` together with the config file and your card
model -- that is the measurement that turns 1.5 GiB from a guess into
a number.

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
3. **Projected machine peak = footprint projection x the envelope
   factor for your platform** (1.75 on Windows, 1.45 on Linux, where
   the projection also drops two Windows-only constants -- see below),
   compared against the card budget (capacity minus the
   flat reserve). The wizard bisects grid sizes until this fits; a
   wizard-emitted config passes `gpuwm check` with zero warnings by
   construction.

Example (24 GiB tier, printed by the wizard on Windows):

```
  itemized alloc estimate 7.17 GiB; footprint projection 11.29 GiB
    x 1.75 observed peak envelope = 19.75 GiB
    envelope factor: windows (measured, 1 WDDM run)
  budget 20.00 GiB (24 GiB card - 4 GiB reserve); headroom 0.25 GiB
```

Both the wizard and `gpuwm check` name the factor and its platform on
every sizing report, so you can always see which evidence priced your
grid. `gpuwm check --json` carries the same three fields
(`observed_peak_envelope_platform`, `..._factor`, `..._basis`).

## Where the envelope factor comes from (the honest part)

The factor is not a safety margin picked to look prudent; it is a
measurement of the estimator being wrong, kept visible. It differs by
platform because what it models -- WDDM's accounting of a run's true
machine-wide footprint -- only exists on one of them.

### Windows / WDDM: 1.75 (measured, 1 run)

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

### Linux: 1.45 over the alloc estimate (measured-preliminary, 3 runs)

Three independent first-run pilots (2026-07-30) instrumented the
machine-wide peak with `nvidia-smi` sampling across whole forecasts:

| node | card | grid | alloc estimate | footprint projection | machine peak | peak / alloc |
|---|---|---|---|---|---|---|
| 1 | 4090 | 224x178 (12 km) + 448x352 (3 km) | 7.20 GiB | 11.31 GiB | 9.54 GiB | **1.32** |
| 2 | 4090 | 438x352 (12 km) | 7.29 GiB | 11.39 GiB | 8.99 GiB | **1.23** |
| 3 | 4070 | 342x272 (12 km) | 3.51 GiB | 4.90 GiB | 4.04 GiB | **1.15** |

All three were 6 h GFS-initialised forecasts. Two things follow, and
the second matters more than the first.

**The peak lands at 0.79-0.82x the footprint projection, not 1.75x.**
Applying the Windows envelope predicted 19.80 and 19.94 GiB against a
20.00 GiB budget on the 4090s -- so the wizard stopped growing the grid
on cards that finished 37-42% used.

**The footprint projection itself is wrong on Linux.** It adds two
grid-independent constants to the alloc estimate --
`pool_retention_residual_bytes` (2.73 GiB) and
`PROBE_DEVICE_OVERHEAD_BYTES` (1.39 GiB) -- both calibrated on one
Windows/5090 fixture, and neither visible in any of the three
measurements. At the wizard's smallest possible layout those constants
are **4.12 GiB of a 5.38 GiB projection: 77% of the floor**. That is
why a 12 GiB card could not be sized at *any* ladder depth while its
GPU sat 66% idle -- shrinking the grid could not touch the part that
did not fit.  (The same reasoning is what the experimental Windows
small-card tier above applies on Windows, with a reduced fixed reserve
in place of the two constants and no measurements behind it yet.)

So on Linux the projection **is** the itemized alloc estimate, and the
envelope factor over it is **1.45** -- 10% clear of the worst of the
three observations (1.32). Three runs on two card models are still not
a calibration; it is labelled *measured-preliminary* wherever it prints.
Re-derive it from further instrumented Linux runs rather than tuning it.

The flat tier reserve (3/4/6 GiB) is unchanged on both platforms, and
neither change moves a gate: the enforced numbers remain the itemized
estimate and the measured `--alloc` legs.

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
- Redirected stdout is block-buffered on Windows; watch the run's
  progress file, not the log tail: `run-progress.json` in `--outdir`
  for the config-driven `gpuwm run` route, `evidence/progress.json` for
  the domain-tree tool route, and `progress.json` for the
  single-domain tools
  ([FIRST-LIGHT.md](FIRST-LIGHT.md#5-run-measured-6-h-forecast-in-36-min)).

## Linux notes

- No WDDM: the budget is the CUDA-reported free memory minus your
  `--reserve-gib`. The same estimator applies, but the projection drops
  the two Windows-pool constants and the envelope factor is the
  measured-preliminary 1.45 rather than Windows' 1.75 (see above), so
  the same card sizes a much larger grid -- roughly one card tier's
  worth. A 12 GiB Linux card sizes more cells at every ladder depth than
  the 16 GiB Windows tier delivers -- and unlike the experimental
  Windows small-card tier, this one is measured.
- Throughput is better than the Windows numbers below suggest.
  Node 2's 438x352x49 single domain at dt 60 s, Morrison + RTE-RRTMGP +
  YSU + Noah + KF, ran 6 simulated hours in 400 s on a 4090 --
  **0.147 wall-s per simulated minute per Mcell**, against 0.229 for the
  same physics and time step on the Windows/WDDM 5090. Normalised for
  grid size that is ~1.56x faster per cell on the weaker card; the gap
  is the platform.
- Output volume, not VRAM, is the binding constraint on a long Linux
  run: node 1's 6 h two-domain 12/3 km forecast wrote 24 GB of
  `wrfout` (32 frames at 15-minute cadence), and node 2's 438x352x49
  frames were 651 MB each.
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
