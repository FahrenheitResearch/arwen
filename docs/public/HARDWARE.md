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
# --card, --cycle and --out are the required trio; --vram-gib N
# replaces --card for a capacity between the named tiers.
gpuwm domain --point 35.3,-97.5 --card 24gb \
  --cycle 1999-05-03T12 --hours 6 --out configs/myarea.toml
```

| card tier | flat reserve | working budget | what fits (measured examples) |
|---|---|---|---|
| 12 GiB | 4 GiB | 8 GiB | Four domains: 156x126 / 312x256 / 336x276 / 268x220, or 398x318 single-domain at 12 km. **Windows: experimental** (see below) |
| 16 GiB | 4 GiB | 12 GiB | full 12-3-1-0.5 km four-domain ladder at ~3.3 GiB alloc estimate |
| 24 GiB | 4 GiB | 20 GiB | four domains: 170x136 (12 km), 336x272 (3 km), 360x294 (1 km), 288x236 (500 m) |
| 32 GiB | 6 GiB | 26 GiB | the reference-class case: 4 domains to 500 m at 400x400+ |

## Independent runs on multiple GPUs

One forecast remains a one-GPU process. On a host with several cards,
`gpuwm multi-run` starts any number of independent production runners at
once, with one unique physical GPU per process. The convenient form invokes
`gpuwm run`:

```toml
schema = "gpuwm.multi-run-plan/v1"
summary = "runs/production-summary.json"
preflight = "estimate"  # estimate (default), alloc, or off

[[run]]
name = "coastal"
device = 0              # physical nvidia-smi index or full GPU UUID
config = "configs/coastal.toml"
outdir = "runs/coastal"
scratch = "scratch/coastal"
cache = "cache/coastal"

[[run]]
name = "inland"
device = 1
config = "configs/inland.toml"
outdir = "runs/inland"
scratch = "scratch/inland"
cache = "cache/inland"
```

```bash
gpuwm multi-run plan.toml
```

Independent entries may carry different domains, forcing, and physics. They
need not be variants of one configuration.

Prepared single-domain and domain-tree forecast routes use the shell-free
module form (`gpuwm.prepared_single_domain_forecast` and
`gpuwm.prepared_domain_tree_forecast`). `args` is an argv array, never a
shell command; `{outdir}`, `{scratch}`, `{cache}`, and
`{input0}`, `{input1}`, ... expand to the plan's validated absolute paths:

```toml
[[run]]
name = "prepared-tree"
device = "GPU-..."
module = "gpuwm.prepared_domain_tree_forecast"
inputs = ["prepared/tree", "configs/tree.toml"]
args = [
  "--prepared-root", "{input0}",
  "--preparation-receipt-sha256", "<sha256>",
  "--experiment-config", "{input1}",
  "--experiment-config-sha256", "<sha256>",
  "--io-mode", "history",
  "--outdir", "{outdir}",
]
outdir = "runs/prepared-tree"
scratch = "scratch/prepared-tree"
cache = "cache/prepared-tree"
```

Those two prepared forecast modules are the complete module allowlist for
this schema version. The argv must contain exactly one separate
`"--outdir", "{outdir}"` pair. Abbreviated, `--outdir=...`, duplicate,
fixed, or embedded output forms are refused before launch, so a target cannot
ignore the validated root and write elsewhere. Each runner's path-valued
options are also schema-known: required options appear exactly once, optional
ones at most once, and every value must be one exact declared `{inputN}`.
Literal, abbreviated, duplicate, or undeclared input paths are refused,
including a second run trying to name the first run's output as an input.
The same path cannot be declared twice within one run.

All plan paths are relative to the plan file. Configs, prepared trees, and
other declared `inputs` are read-only and may be shared, including the same
config on two GPUs for a cross-device numerical comparison. The device UUID
is intentionally part of each production capsule, so cross-device capsules
differ in that identity field even when frame bytes and numerical results
match. Every mutable outdir,
scratch, and cache root must be distinct and non-overlapping across the whole
plan, and must be new. Config-run output directories are claimed before
launch; prepared runners retain their own atomic absent-outdir claim. This
preserves earlier outputs and prevents two GPUs from writing the same fixed
filenames. Each run captures top-level output in
`scratch/gpuwm-run.log`; supervised worker logs and progress remain in its
outdir as usual. `CUPY_CACHE_DIR` and NVIDIA's `CUDA_CACHE_PATH` point to
separate `cupy/` and `cuda/` children of each run's new cache root.

Before launching forecasts, `preflight = "estimate"` runs `gpuwm check` once
per selected GPU. `"alloc"` adds the allocation measurement, and `"off"`
skips the checks with a warning. A nonzero check is reported and retained in
the summary but does not block `gpuwm run`: sizing remains advice, consistent
with the single-run command. Missing inputs or an unsafe configuration still
fail in the run's normal validation path and contribute a nonzero aggregate
status. Module-form runners use their production entry point's own input and
memory preflight instead of pretending a config-driven check applies.
Each config-form check runs in a fresh UUID-masked child while holding the
same machine-wide physical-GPU lock used by its forecast, before importing
the check command and any transitively loaded CUDA runtime.

The parent resolves each selector with `nvidia-smi`, rejects aliases to the
same physical UUID, and sets `CUDA_VISIBLE_DEVICES` before starting the child.
It also passes the physical UUID to the existing supervisor lock; module-form
runners acquire that same lock before importing their target module.
The parent resolves one exact machine-wide lock root before replacing each
child's temporary-directory variables and proves that root is outside the
plan, summary, every input, and every output/scratch/cache tree. Set
`GPUWM_GPU_LOCK_ROOT` to an explicit stable host directory when the platform's
ordinary temporary root would overlap a run path.
CUDA-facing code that names `Device(0)` or `getDeviceProperties(0)` is
therefore using the only *logical* device visible inside that process, not
hard-coding physical card zero. This process boundary is intentional: there
is no in-process `--device` switch that could run after a CUDA context already
exists.

Before launching children, the orchestrator creates the summary parent and
probes that its filesystem supports the same atomic create-only hard-link
publication used for the final JSON. A summary that appears during a later
race is preserved and the orchestrator refuses to overwrite it. The receipt
contains the raw plan SHA-256, SHA-256 for every declared file input and
config, and an explicit delegated-to-runner-receipt marker for directory
authorities rather than implying their contents were hashed. It also contains
UTC start/completion times,
durations, exact child exit codes, resolved device identity, all isolation
paths, logs, and PIDs. It records the sum of child durations, the concurrent
execution window, and their overlap ratio only when every forecast succeeds.
That ratio measures process overlap; it is not a performance speedup, which
would require a separately measured serial baseline. Failed or interrupted
runs retain observed timing but do not report the overlap ratio. The
command exits zero only when every forecast exits zero; otherwise it exits one
after all children finish. Ctrl-C writes an
`interrupted` summary and returns 130. It does not kill children or unrelated
processes; unobserved child PIDs are printed and recorded so a supervised
worker is never orphaned by terminating only its parent supervisor. Once the
create-only publication probe succeeds, interruptions during directory claims,
authority monitoring, checks, forecasts, or final publication also attempt an
honest create-only receipt naming the stage and every claimed directory and
known child PID. Unexpected failures stop the non-daemon monitor, attempt a
`failed` receipt, and then propagate a group error with the original exception
preserved as its cause.

A sticky periodic monitor starts before the first child and remains active
continuously across config checks and forecasts. It watches the raw plan and
every declared file authority, records the capture SHA-256 plus timestamped
signature observations and errors, and performs a final forced signature
observation before publication. Original config bytes are not reopened after
capture.
A pre-launch mismatch prevents any child from starting; a mutation observed
during checks prevents forecast launch; and a mutation during forecasts makes
the aggregate receipt fail. Restoring the old bytes does not clear a prior
observation. Prepared directories remain delegated because those production
runners bind their directory authorities in their own receipts.

For config-form runs, multi-run reads each experiment config once while loading
the plan, hashes those exact bytes, and durably creates a per-run payload.
CLI route detection, both `gpuwm check` layers, supervisor input discovery,
and every fresh worker consume a payload validated against that SHA-256; the
original path remains only the source identity and relative-path base. The
supervisor also makes its usual create-only worker payload in the run outdir.
Changing the original path after plan capture cannot change any parsed phase.

After input discovery, each multi-run supervisor snapshots forcing, Vtable,
WPS namelist, and declared source-orography files into one plan-wide
content-addressed store beside the summary. Publication is create-only and
locked per SHA-256, so identical inputs shared by any number of runs occupy one
full copy. Workers validate every snapshot against the capsule hash and remap
all file opens to those read-only bytes; provenance and certification retain
the original declared paths. Before remapping or runtime, the worker's parsed
role, absolute original path, and slot-detail multiset must exactly equal the
parent SHA inventory, including duplicate identities. A forcing glob that
temporarily loses, gains, or renames a match therefore publishes a failed
heartbeat and failure capsule even if the original directory is restored
before validation. The store path is collision/isolation checked and its SHA
entries are listed in the summary. Ordinary single `gpuwm run` invocations
keep their historical direct-input behavior and do not create this store.

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
refusing a card gpuwm can probably run. `gpuwm check` does not stop a
later `gpuwm run` -- nothing prevents you from starting a forecast it
warned about -- but since v1.1.0 an observed peak above the WDDM budget
is an exit code 4, not a green exit with a warning in it. A script that
reads the exit status is blocked; a person reading the output is
advised. (Reserves moved to a flat 4 GiB through 24 GiB in v1.1.0; the
12/16 GiB rows above were 3 GiB in v1.0.1.)

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
3. **Projected machine peak = the alloc estimate PLUS the non-pool
   residency PLUS a measured constant.** It is a sum, not a multiple:

   ```
   peak envelope = alloc estimate
                 + CUDA context + local-memory backing store
                 + 0.50 GiB unmodelled
                 + 5% of the estimate per nest beyond the root
   ```

   The middle term scales with the DEVICE (its SM count) and the kernel
   set your physics selects -- not with the grid. That is why the model
   has to have an intercept, and why the version that did not
   [got the sign of its own error wrong](#where-the-envelope-comes-from-the-honest-part).

   The wizard bisects grid sizes until this fits the budget with
   headroom to spare; a wizard-emitted config passes `gpuwm check` on a
   real card of the tier it was sized for.

Example (24 GiB tier, printed by the wizard on Windows, where the retained WDDM floor is the branch that binds):

```
  domain    dx        mass grid      dt         resident
  d01     12.000 km   164 x 130        60 s     0.59 GiB
  d02      3.000 km   328 x 256        15 s     2.09 GiB
  d03      1.000 km   354 x 276         5 s     2.44 GiB
  d04      0.500 km   284 x 220       5/2 s     1.58 GiB
  peak envelope: footprint 10.52 x 1.75 WDDM floor = 18.41 GiB, which is above the affine form (estimate 6.40 + non-pool 2.30 (CUDA context + local-memory backing store) + 0.50 unmodelled + 5% of the estimate x 3 nest(s) = 10.16 GiB) and therefore binds
    envelope basis: windows; measured, 1 WDDM run
  ingest (preprocessing): root 2 forcing times x 0.22 GiB each, 2 resident at a time + 3 nest initial state(s) 2.30 GiB, all resident for the single export transaction = 2.75 GiB resident; peak envelope 5.72 GiB
    ingest envelope basis: measured, CONUS 12 km 414x330x49 x 9 GFS times, RTX 5090 / Linux: itemization + 0.65x one forcing time of transients, x1.15 headroom, + CUDA context
  BINDING PHASE: the forecast is the memory-binding phase at 18.41 GiB peak envelope (forecast 18.41 GiB, ingest 5.72 GiB); it fits the 19.57 GiB budget with 1.16 GiB to spare
  budget 19.57 GiB (24 GiB card presents about 22.56 GiB free, minus this suite's 2.99 GiB reserve); headroom 1.16 GiB
```

Both the wizard and `gpuwm check` show every term, so you can always
see what priced your grid. `gpuwm check --json` carries them as
`alloc_estimate_bytes`, `non_pool_device_bytes`,
`envelope_unmodelled_bytes`, `envelope_per_nest_fraction` and
`envelope_basis`.

**A card never hands over its nameplate capacity.** A real RTX 4080
(16,376 MiB physical) presents 15.33 GiB free to a fresh CUDA context;
a 32 GiB card with a desktop on it presented 30.27 of 31.84. The
`--card` tiers therefore size against nominal capacity minus the
larger of 0.75 GiB and 6%, so the tier is conservative against the
cards of its class rather than equal to the best one. Before
2026-08-01 they assumed the nameplate, and every ladder the 16 GiB
tier emitted failed `gpuwm check` on a real 16 GB card minutes after
the wizard printed PASS.

## Where the envelope comes from (the honest part)

The envelope is not a safety margin picked to look prudent; it is a
measurement of the estimator being wrong, kept visible.

### Why it is a sum and not a multiplier (2026-08-01)

Until 2026-08-01 the envelope was `factor x projection` with no
intercept. A model with no intercept cannot describe a cost that has a
large fixed term, and this one does: a CUDA context plus the
launch-time local-memory backing store is 1.5-2.9 GiB before a single
grid cell exists, and it does not move when the grid does.

A 16 GiB fleet node (RTX 4080, Linux, driver 595.58.03) instrumented
whole forecasts machine-wide with `nvidia-smi` at 250 ms across a 6.6x
span of grid size, on an otherwise idle card:

| grid | domains | itemized estimate | old x1.45 envelope | measured peak |
|---|---|---|---|---|
| 170x136 | 1 | 2.07 GiB | 3.00 GiB | 3.65 GiB |
| 224x180 | 1 | 2.75 GiB | 3.99 GiB | **4.38 GiB** |
| 340x272 | 1 | 4.82 GiB | 7.00 GiB | 5.95 GiB |
| 448x360 | 1 | 7.56 GiB | 10.96 GiB | 8.75 GiB |
| 474x378 | 1 | 8.27 GiB | 11.99 GiB | 9.25 GiB |
| 594x476 | 1 | 12.38 GiB | 17.95 GiB | 12.59 GiB |
| 630x504 | 1 | 13.76 GiB | 19.95 GiB | 13.88 GiB |
| 242x194 + 480x384 | 2 | 8.22 GiB | 11.91 GiB | 10.09 GiB |

Read the bold row: a 224x180 domain -- the *cautious first run* the
wizard's own advisory tells you to make -- was declared 3.99 GiB and
peaked at 4.38. **Below about 3.5 GiB of estimate the old envelope was
optimistic**, and above it, increasingly pessimistic, reaching +44% at
the top of the table. That also reconciles two fleet reports that
looked contradictory: a 5090 measuring ~19% *under*-prediction and this
4080 measuring 25-30% *over* are the same model read at different grid
sizes.

Fitting `peak = a x subtotal + b` over the single-domain rows returns
`a = 0.98`. The itemization predicts the pool essentially 1:1; the
residue is a constant. So the model is a sum of the three things this
estimator already knows how to compute, and only one small term is
fitted:

* the **itemized alloc estimate** -- the pool side;
* the **non-pool residency** -- CUDA context plus the local-memory
  backing store of the widest kernel your physics launches, scaled by
  the device's resident-thread capacity;
* **0.50 GiB unmodelled**, plus **5% of the estimate per nest**. The
  worst residual measured over the whole table is +0.10 GiB for a
  single domain and 4.3% of the estimate per nest for a tree, both
  rounded up.

Against every measured run above and the three 2026-07-30 Linux pilots
below, the new envelope is conservative by 5% to 26% and never lands
under a measured peak. The old one was optimistic by 10% at the bottom
of its range.

**The device matters.** The backing store is
`(frame - default stack) x SMs x threads per SM`, so the 170-SM RTX
5090 this module was calibrated on carries 2.2x what a 76-SM 4080 does.
`gpuwm check` reads the SM count off your card whenever it is measuring
your card. When you size for a card that is not in the machine
(`--card`, `--vram-gib`, `--budget-gib`), it uses the largest SM count
sold at that capacity, which over-prices every other card in the class
rather than under-pricing any.

### Windows / WDDM: the 1.75 multiplier, kept as a floor (measured, 1 run)

On the four-domain reference run (2026-07-28, RTX 5090 32 GiB,
Windows), the preflight projected a 16.22 GiB footprint. The measured
machine-wide peak was 29,004 MiB -- 1.75x the projection -- and it
finished 57.1 MiB (0.2%) under the 30,472,743,936-byte Windows WDDM
budget the gate had checked. The gate passed for the wrong reason: it
compared the smaller alloc estimate against the budget.

That is a real measurement over a real projection, and it stays. On
Windows the envelope is the LARGER of `footprint x 1.75` and the affine
form above -- the affine form is a floor under it, never a discount.
Nobody has instrumented a Windows run small enough to show where the
two cross, so the multiplier is not retired; it is bounded from below
by a model that cannot be optimistic about small configurations.

### Linux: the three 2026-07-30 pilots, re-read

Three independent first-run pilots (2026-07-30) instrumented the
machine-wide peak with `nvidia-smi` sampling across whole forecasts:

| node | card | grid | alloc estimate | footprint projection | machine peak | peak / alloc |
|---|---|---|---|---|---|---|
| 1 | 4090 | 224x178 (12 km) + 448x352 (3 km) | 7.20 GiB | 11.31 GiB | 9.54 GiB | **1.32** |
| 2 | 4090 | 438x352 (12 km) | 7.29 GiB | 11.39 GiB | 8.99 GiB | **1.23** |
| 3 | 4070 | 342x272 (12 km) | 3.51 GiB | 4.90 GiB | 4.04 GiB | **1.15** |

All three were 6 h GFS-initialised forecasts. Under the affine model
their non-pool terms are 2.30 GiB (4090, 128 SMs) and 1.10 GiB (4070,
46 SMs), which puts the envelope at 10.00, 10.09 and 4.86 GiB against
measured peaks of 9.54, 8.99 and 4.04 -- conservative by 5%, 12% and
20%, on three cards none of which is the one the model was fitted on.

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
envelope over it is the affine form described above.

**The reserve is not flat either.** It carries the same local-memory
backing store, which is a property of the SELECTED KERNEL SET: 1.93 GiB
for WSM6 + MYNN, 2.91 for the Thompson default, 3.94 for NSSL2
double-moment, all on the reference 5090 profile. The wizard used to
size against a flat 4.0 GiB and then verify against the real figure, so
both NSSL2 physics profiles emitted a config that failed their own
`gpuwm check` at every card size. The fit loop prices the reserve from
the candidate experiment now -- the same call `gpuwm check` makes -- so
the two cannot disagree about the same file.

None of this moves a gate: the enforced numbers remain the itemized
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
  the reserve, and the headroom the fit loop leaves on top of it,
  exist so routine runs never operate there. The reference run that peaked 57.1 MiB under budget
  completed cleanly and bit-deterministically -- and is exactly the
  margin the sizing model now prevents. What running the forecast
  twice and comparing bytes does and does not detect in place of ECC
  is stated precisely in [DETERMINISM.md](DETERMINISM.md).
- Redirected stdout is block-buffered on Windows; watch the run's
  progress file, not the log tail: `run-progress.json` in `--outdir`
  for the config-driven `gpuwm run` route, `evidence/progress.json` for
  the domain-tree tool route, and `progress.json` for the
  single-domain tools
  ([FIRST-LIGHT.md](FIRST-LIGHT.md#5-run-measured-6-h-forecast-in-36-min)).

## Linux notes

- No WDDM: the budget is the CUDA-reported free memory minus your
  `--reserve-gib`. The same estimator applies, but the projection drops
  the two Windows-pool constants and the envelope is the affine form
  rather than Windows' 1.75 multiplier (see above), so the same card
  sizes a much larger grid -- roughly one card tier's worth. A 12 GiB Linux card sizes more cells at every ladder depth than
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
machine-level facts worth knowing:

<!-- BEGIN GENERATED ftz-statement: hardware-fp32-subnormals (tools/ftz_receipt/render_statement.py) -->
- FP32 subnormal handling was measured on this machine's GPU
  (NVIDIA GeForce RTX 5090, compute capability 12.0, driver
  13.3 (13030)) across the 6 compile routes the model
  uses, crossed with 6 arithmetic mechanisms.  The answer
  depends on the route:
  - `R1` loader RawModule (`gpuwm/core/kernels/__init__.py:78`,
    gpuwm.core.kernels.load_module), effective NVRTC options `-std=c++17`
    `-ftz=true`: `flush-to-zero` on 6 of 6 mechanisms [disassembly
    `tools/ftz_receipt/receipt/sass/r1.sass`]
  - `R1-ftztrue` loader RawModule + explicit --ftz=true (control)
    (`gpuwm/core/kernels/__init__.py:78 + control flag`, cupy.RawModule),
    effective NVRTC options `-std=c++17` `--ftz=true` `-ftz=true`:
    `flush-to-zero` on 6 of 6 mechanisms [disassembly
    `tools/ftz_receipt/receipt/sass/r1_ftztrue.sass`]
  - `R2` RawModule with the shortwave option tuple
    (`gpuwm/core/rrtmg_sw.py:2890`, cupy.RawModule), effective NVRTC options
    `-std=c++17` `--ftz=false` `-ftz=true`: `flush-to-zero` on 6 of 6
    mechanisms [disassembly `tools/ftz_receipt/receipt/sass/r2.sass`]
  - `R3` direct NVRTC + cuda.function.Module (`gpuwm/core/rrtmg_lw.py:3723`,
    cupy.cuda.compiler.compile_using_nvrtc), effective NVRTC options
    `-std=c++17` `--ftz=false` `-arch=compute_120`: `ieee-agreement` on 6 of
    6 mechanisms [disassembly `tools/ftz_receipt/receipt/sass/r3.sass`]
  - `R4` CuPy-generated ReductionKernel (`gpuwm/core/mynn_pbl_gpu.py:290`,
    cupy.ReductionKernel), effective NVRTC options `--std=c++17`
    `-ftz=true`: `flush-to-zero` on 6 of 6 mechanisms [disassembly of 6
    objects, `tools/ftz_receipt/receipt/sass/r4_0.sass` and siblings]
  - `R5` inline PTX without .ftz (`gpuwm/core/kernels/__init__.py:78`,
    gpuwm.core.kernels.load_module), effective NVRTC options `-std=c++17`
    `-ftz=true` (this route rides `R1`'s compile): `ieee-agreement` on 4 of
    6 mechanisms; `not-applicable` on 2 of 6 mechanisms [disassembly
    `tools/ftz_receipt/receipt/sass/r1.sass`, the object `R1` compiled]
  `R5` and `R1` are kernels inside ONE compiled object -- same device, same
  flags, one compile -- and they did not measure alike, so on this device
  the outcome follows the instruction the compiler emitted rather than the
  hardware alone.
  The control arm is load-bearing: the 3 distinct bit tables among the 6 arms
  are what shows the pipeline responds to the flag at all.
  The consequences reach physics: each known instance is recorded in the
  physics registry ([PHYSICS.md](PHYSICS.md)), and the radiation preparation
  path routes one subnormal-sensitive block through the host by design.
  Evidence: the device bit table `tools/ftz_receipt/receipt/bitpatterns.csv` (both
  passes byte-identical: true), the objects the routes themselves compiled,
  `tools/ftz_receipt/receipt/cubin/`, and their disassembly,
  `tools/ftz_receipt/receipt/sass/` (Cuda compilation tools, release 13.0,
  V13.0.39), all recorded in `tools/ftz_receipt/receipt/receipt.json`.
<!-- END GENERATED ftz-statement: hardware-fp32-subnormals -->
- The consequence that reaches the science is a branch flip on
  physically negligible inputs, not a change in a resolved quantity.
- Determinism holds per build and hardware: the reference run
  reproduced output frames SHA256-identically across a mid-run kill
  and relaunch. No cross-GPU or cross-driver bit-identity is claimed.
  The pin set that "per build and hardware" actually names, and the
  three mechanisms that make it necessary, are in
  [DETERMINISM.md](DETERMINISM.md).
