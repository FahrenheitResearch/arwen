# Noah-MP device-column report

## Result

`VEGE_FLUX`, formerly 63% of the measured host-column call, now runs as one
physical-argument CUDA batch across all vegetated columns. Its measured
**max ULP is 0** against the unmodified WRF v4.6.1 leaf, and the complete
host/device trajectory is bitwise equal.

At 352 land columns the paired forced-host authority measured 6.18
ms/column; the device-leaf runtime measured 4.94--4.95 ms/column, a **1.25x
speedup**. The earlier standalone baseline was 6.16 ms/column, consistent with
the paired measurement.

The whole-column cost is **still flat at width**. This is not an assembled
device column: ENERGY outside `VEGE_FLUX`, `BARE_FLUX`, and the rest of
`NOAHMP_SFLX` remain in CPython. Vegetated columns also replay the pre-VEGE
ENERGY prefix once after the batch returns. Noah-MP is therefore **not yet
viable at production width**, and the registry template should remain
expert-only.

## What moved

`gpuwm.core.noahmp_vegeflux_gpu` converts the physical Python call to the
existing `dev_vege_flux` CUDA input, launches all captured columns together,
and reconstructs `VegeFluxState`. The wrapper uploads the authoritative glibc,
physics, and saturation coefficients to `__constant__` memory; this also keeps
ptxas constant folding away from the FP32 tie values. It does not ship
translation-unit-wide `-fmad=false`.

ENERGY now exposes a narrow call override. The runtime first executes each
column until a vegetated column reaches VEGE_FLUX. Bare columns complete on
the original path. After one device evaluation, each deferred vegetated
column restarts from its pristine entry state and receives its corresponding
device result at the same call site. Noah-MP has no horizontal coupling, so
column order cannot alter this continuation.

That replay is intentionally conservative and is also the main avoidable cost
in this first conversion. It avoids retaining a partially mutated Python
ENERGY frame while establishing the device seam. A runtime tripwire replaces
the host leaf with an exception and proves no CPython VEGE_FLUX executes on
the device path.

No CUDA arithmetic was rewritten for this change. The already oracle-matched
`gpuwm/core/kernels/noahmp_vegeflux.cu` leaf is reused; the new work is the
runtime batching and continuation boundary.

## Bitwise evidence

### Baseline before the conversion

I re-derived both requested six-step baselines from an archive of the
pre-conversion authority at commit `fbd2ad6`; I did not copy the reported
strings into the test:

| domain | six-step SHA-256 |
|---|---|
| bare | `35badccff04bcbd4bac74adfd126c2e203ff11f799a4d5649657805cdbecfeaf` |
| snowpack | `23de48f07fc6d230ee3069c9935a5a2b8a329deacbf65e17a98dc4ac4e608a88` |

The historical hash procedure sorts the carried names and hashes each name,
dtype spelling, and contiguous array bytes. The carried tuple contains **45
arrays, not 47**; the older prose had miscounted it. Shared surface-layer work
in the live tree has since changed the absolute bare digest, so the durable
runtime gate runs the original host VEGE_FLUX beside the device batch on the
same current inputs and compares every byte of all 45 arrays. The snowpack
absolute digest remains the historical value.

Both the bare and snowpack device runs equal their paired host authority after
six steps, array by array. The four whole-column `noahmp-sflx.csv` cases also
remain bitwise:

- `veg_warm_day_dry`
- `veg_warm_night_rain`
- `snowpack_frozen_soil`
- `bare_thin_snow_melt`

### Leaf oracle and falsification

The collected CUDA test captures the physical Python argument list for every
unmodified-WRF VEGE_FLUX fixture case. It compares 300 output values and
observes no differing bits: **max ULP 0**.

The same test can compile a negative-control form using CUDA device libm. That
form produces at least one mismatch against the same 300-value gate. The gate
therefore demonstrably observes the constant-table glibc transcription that
the production kernel depends on.

## Width measurement

Each entry below is the range from two independent warmed runs on the rented
GPU. Device and forced-host modes use the same committed runtime and inputs;
five repetitions were used through 48 columns and three at the larger widths.

| land columns | forced host (ms/column) | device VEGE_FLUX (ms/column) | speedup at width |
|---:|---:|---:|---:|
| 2 | 15.92--16.30 | 15.10--15.23 | 1.05--1.07x |
| 8 | 8.41--8.42 | 7.16--7.18 | 1.17x |
| 48 | 6.53--6.58 | 5.23--5.30 | 1.24--1.25x |
| 160 | 6.24--6.25 | 4.94--4.99 | 1.25--1.26x |
| 352 | 6.18 | 4.94--4.95 | **1.25x** |

The 48-to-352 device figures only fall from 5.23--5.30 to 4.94--4.95
ms/column. The remaining flat term is per-column host work. This does not
mean the VEGE work failed to reach the GPU: the batch oracle and host
tripwire establish execution location independently. It means the requested
leaf conversion is incomplete as a whole-column conversion.

The width process's CuPy pool reached 16.4 MB. The rented card reported 17 MiB
total device use before and after the run.

## One-hour trajectory

The trajectory comparison used a 16x10x40 domain, 130 land columns, 300 RK3
steps, and the same initial state in forced-host and device-leaf modes. At
every 50 steps it compared all 45 carried arrays plus `u`, `v`, `w`, `thp`,
`qv`, and `mup` byte for byte before recording the digest.

| step | identical host/device SHA-256 |
|---:|---|
| 50 | `6e808591cb21451a84259818257ebc2c81a958d90a47243d9c77ff68f6893811` |
| 100 | `f3533464b58dc89e1f59cc739565afc34ba6a2633e6412da1d222ef84f3e372b` |
| 150 | `50b9ddec7b63a29ca3ff52440f9efe5fd0be4a2ed17be2a0216b90e95de47336` |
| 200 | `ba6bbac3a0d96598636268be008821a1bfcfeb5f08dd045913d7e943f6061b79` |
| 250 | `4c387a6f02d153e24255f5ffb9832e8ea0e5599bba9f5673f511230bbe105066` |
| 300 | `507bd84fec4dae0251cf6594cc2d366e2b06609a9afa1839455bb6afde889ca0` |

The trajectory's measured max ULP is **0**. Forced host took 254.490 s
(6.525 ms/land-column); device VEGE_FLUX took 207.301 s (5.315
ms/land-column), a **1.23x speedup**.

The forecast process's CuPy pool reached 5.75 MB, and `nvidia-smi` reported
17 MiB total device use before and after. These observed allocations are far
below the 29,500 MiB rail; no process was killed, stopped, or suspended.

## Final verification

The final code archive at `0887bc0`, extracted without a remote git checkout,
passes the complete collected GPU Noah-MP suite: **960 passed**. Polling
`nvidia-smi` every 0.2 s for the whole suite observed a peak of **2,585 MiB**,
under 9% of the 29,500 MiB rail. The complete CPU-only run with
`GPUWM_NO_LOCAL_GPU=1` reports **690 passed, 270 skipped**; the skips are the
GPU-marked tests.

## What remains

The next answer-preserving step is an explicit post-VEGE ENERGY continuation
so vegetated columns do not repeat the prefix. Then:

1. move `BARE_FLUX`, about 7% of the original call;
2. move the remaining ENERGY path and its thermal/radiative leaves;
3. compose the existing snow, phase-change, soil-water, and soil-thermal
   device leaves behind physical argument lists;
4. merge only the device libm copies already proved identical, and settle the
   guarded `powf` variants separately with subnormal and signed-zero evidence;
5. repeat the full leaf oracle, six-step bare/snowpack state gate, width sweep,
   and 50-step trajectory hashes after every leaf.

Until those steps remove the flat host term, the honest registry status is
unchanged: implemented, expert-only, and not production-width viable.

---

# Second conversion: the ENERGY continuation, BARE_FLUX and WATER

*The section above is the record of the first device leaf and is left as it
stands. Everything below is a separate lane on a different machine; where the
two report milliseconds they are not comparable, and this section says so
rather than merging them.*

## Result

Three leaves now execute as CUDA batches -- `VEGE_FLUX` (as before), plus
**`BARE_FLUX`** and **`WATER`** -- and, more importantly, the column no longer
restarts to reach them.

The first seam paused a vegetated column by *rewinding* it: the column was
restored to its pristine entry state and replayed the whole pre-`VEGE_FLUX`
prefix once the batch returned. That is why the whole-column cost barely fell
when 63% of the call moved to the device. `ENERGY`, `NOAHMP_SFLX` and
`NOAHMP_SFLX`'s post-ENERGY half now have generator forms that suspend at the
physical leaf call and **resume in the same frame**. Each land column runs
exactly once.

Measured on Drew's local RTX 5090 (Windows, CUDA 13.0, CuPy 14.1.1), paired in
one process against the same tree's host authority:

| land columns | host authority ms/col | device leaves ms/col | speedup |
|---:|---:|---:|---:|
| 2 | 13.37 | 10.95 | 1.22x |
| 8 | 5.13 | 3.45 | 1.49x |
| 48 | 2.94 | 1.34 | 2.20x |
| 160 | 2.80 | 1.07 | 2.62x |
| 352 | 2.62 | 1.00 | **2.63x** |

The same sweep run against the **pre-lane committed tree (`3fef784`) on this
same box**, so old and new differ by nothing but the code:

| land columns | pre-lane host authority | pre-lane device `VEGE_FLUX` | speedup |
|---:|---:|---:|---:|
| 2 | 12.37 | 12.46 | 0.99x |
| 8 | 4.98 | 4.46 | 1.12x |
| 48 | 2.92 | 2.34 | 1.24x |
| 160 | 2.66 | 2.09 | 1.28x |
| 352 | 2.60 | **2.05** | 1.27x |

Three things to read off those two tables.

1. **The host authority is unchanged** -- 2.60 versus 2.62 ms/column at 352.
   The restructuring costs nothing on the host path, so the two device columns
   are being compared against the same quantity.
2. The pre-lane device path reproduces the previously published **1.25x** on
   this machine (1.27x), which is what says the old measurement was sound and
   this box is simply faster in absolute terms. **Do not compare 1.00 ms here
   with 4.94 ms in the section above.**
3. The device column is **2.05x faster than the previous device column** on the
   same machine, and 2.63x faster than running the leaves on the host.

## Measured ULP

**0**, everywhere it was checked, on this card:

- `BARE_FLUX` -- the runtime batch reproduces every output of every
  `noahmp-bareflux.csv` row bitwise, through the physical keyword call rather
  than a hand-built flat row.
- `WATER` -- the runtime batch reproduces every `output` row of
  `noahmp-water.csv` bitwise, including the four `probe` locals that never
  reach a WRF output but decide a branch, and including the nine snow/soil
  arrays it mutates in place.
- `VEGE_FLUX` -- unchanged, still 0 over its 300 physical output values, with
  its CUDA-device-libm negative control still failing the same gate.
- Whole column: all four `noahmp-sflx.csv` columns are bitwise **when run
  interleaved through the staging loop**, which is the control flow the device
  path uses, not one at a time.
- Whole forecast: see the digests below.

The complete collected Noah-MP suite is **980 passed, 1 skipped** on the GPU
and **600 passed, 109 skipped** under `GPUWM_NO_LOCAL_GPU=1`.

## Bitwise evidence

### The six-step carried state, re-derived rather than trusted

The historical bare and snowpack digests in the section above **no longer
reproduce, and neither of them did before this lane started.** That is a
correction to the record: the earlier text said only the bare digest had moved
with shared surface-layer work. It was both.

This was settled by extracting the pre-lane committed tree at `3fef784` to a
scratch directory and running the same six steps there. The digests over all 45
carried arrays:

| domain | `3fef784` (pre-lane) | this lane, device | this lane, host authority |
|---|---|---|---|
| bare | `b8463904221a6db3ccd29175d83da9685f06a9c743f050ca25681c5ab97d4526` | same | same |
| snowpack | `364c088b65a23d9ceeb48ee3370b77a1bdce054e86035b7d3bbdfd65c40bb3ca` | same | same |

So the conversion is **bitwise against the previous device path over six steps
on both a bare and a snowpack domain**, and bitwise against its own host
authority. The pre-lane run used committed content only; the live tree carries
other lanes' uncommitted surface-layer edits and produces the same digest, so
those edits do not reach this configuration.

### The staging itself, checked without a GPU

A device-versus-host comparison cannot see a bug in the *staging*, because both
modes stage identically. Two CPU-only gates cover that gap against the WRF
oracle, which is an authority entirely outside this change:

- nine `noahmp-energy.csv` columns driven **interleaved** through the staging
  loop are bitwise equal to the unstaged call across every field of the ENERGY
  state; and
- all four `noahmp-sflx.csv` whole columns, interleaved with their `VEGE_FLUX`,
  `BARE_FLUX` and `WATER` calls collected and answered in batches, are bitwise
  against the fixture.

### The falsifications, and two that were wrong first

Every gate above is shown failing before it is shown passing. Two of those
attempts failed to fail, and the measurement that replaced each is more
interesting than the gate:

- **A one-ULP nudge of `BARE_FLUX`'s entry `TGB` changes nothing.** Five Newton
  iterations converge the seed away. Measured across every live scalar on one
  fixture column: `SFCTMP` moves ten of the thirteen outputs, `RHSUR` seven,
  `UR` and `RHOAIR` four, `EAIR` and `PSFC` two, `LWDN`/`UU`/`VV`/`RSURF`/
  `GAMMA` one each -- and `TGB`, `CM`, `CH` and `QSFC` move none. The last
  three are never read at all under `opt_sfc=1`.
- **A one-ULP nudge of `WATER`'s entry `SMC` changes nothing**, because WATER
  rebuilds `SMC(k) = SH2O(k) + SICE(k)` at `:6212` and SOILWATER works on
  `SH2O`/`SICE`. `ZSOIL` moves five emitted columns and is the gate now.
- At the **forecast** level, one ULP of `BARE_FLUX`'s output `TGB` is also
  invisible -- the domain is grassland at `SHDMAX` 60%, so `:2289` forms
  `FVEG*TGV + (1-FVEG)*TGB` and one ULP scaled by 0.4 falls below half an ULP
  of a 300 K sum. Measured: 1 ULP absorbed, 2 through 64 visible. The six-step
  gate now uses one perturbation per leaf at the smallest magnitude each is
  measurably observable at, with a companion test asserting the 1-ULP case
  really is absorbed so the `2` cannot be quietly tidied back to a `1`.
- Misrouting one column's `WATER` answer to another column is rejected by
  `NOAHMP_SFLX`'s own `ERRWAT` check at 115 kg/m2 -- before any bit comparison
  runs. That is a stronger rejection than the test was written to expect and it
  is recorded as such.
- Transposing two live slots in each packer (`sag`/`lwdn` for BARE_FLUX,
  `SH2O`/`SICE` for WATER) is rejected by comparison against the oracle
  generators' own column order.

## Is it still flat?

**Less, but yes, and that is still the whole diagnosis.**

From 48 to 352 land columns the device figure falls from 1.34 to 1.00
ms/column, a 25% fall, where the pre-lane device figure fell only 12% (2.34 to
2.05) and the host authority falls 11% (2.94 to 2.62). More of the cost is now
amortising across columns, which is what moving work into batches looks like.
But 1.00 ms/column at 352 is still overwhelmingly a per-column host term: it is
not a launch cost and not a transfer cost, because it does not fall when the
launches are shared over seven times as many columns.

The profile says exactly what the term is. `cProfile` over the whole-column
fixtures, re-taken after the conversion rather than carried over -- the
generator seam conveniently lifts the batched leaves out of ENERGY's own
cumulative time:

| subtree | share of the `sflx` tree | status |
|---|---:|---|
| `VEGE_FLUX` | 45.2% | on the device |
| `WATER` | 13.3% | on the device |
| `BARE_FLUX` | 11.2% | on the device |
| ENERGY minus its leaves | 16.4% | **host** |
| -- of which `RADIATION` (`ALBEDO` 4.6, `SURRAD` 0.4) | 5.4% | host |
| -- of which `THERMOPROP` | 3.5% | host |
| -- of which `TSNOSOI` | 1.2% | host |
| -- of which `PHASECHANGE` | 0.8% | host |
| `SFLX_PRE` (`ATM`, `PHENOLOGY`, `PRECIP_HEAT`) | 3.3% | host |

About **70% of the column tree is now batched**. `RADIATION` is the largest
single thing left, and `noahmp_radiation.cu` already exists and is
oracle-matched; after it the remainder is ENERGY's own composition arithmetic,
which is not a leaf and would need the whole of ENERGY on the device.

## What a 360,000-column d04 call now costs

At the measured 1.00 ms per land column, a 360,000-land-column nest costs about
**360 s -- six minutes -- per land-surface call**, against about 15.7 minutes
for the same call with the leaves on the host. `bldt = 0.0` means that call
happens every step.

That is a 2.6x improvement on an unusable number. **It does not change the
expert-only status and the column-budget gate should not move.** Six minutes
per step is not a forecast. The registry template should stay expert-only and
`GPUWM_NOAHMP_EXPERT_COLUMN_BUDGET` should keep its current default. What
should change is the published per-column figure and the sentence saying the
cost is flat, because both are now wrong in the scheme's favour, and a stale
pessimistic number is still a false number. **That is a lead decision, not this
lane's -- nothing in `gpuwm/config.py` or the registry was touched.**

For the record, the remaining host term bounds what further leaf conversions
can buy: if `RADIATION`, `THERMOPROP`, `TSNOSOI` and `PHASECHANGE` all moved
and cost nothing, the column would still carry ENERGY's own composition and
`SFLX_PRE`, roughly 14% of the original tree -- call it 0.4 ms/column, or about
2.5 minutes for that nest. Getting below that needs ENERGY itself on the
device, not another leaf.

## VRAM and machine hygiene

Whole-machine `nvidia-smi` was polled every 0.2 s for the entire GPU session
(1,467 samples): peak **9,932 MiB**, of which roughly 3,300 MiB is the user's
desktop. That is 34% of the 29,500 MiB rail. The GPU lock was taken once for
the whole batch and released with `rm -f gpu.lock/owner` followed by
`rmdir gpu.lock`. No process was killed, stopped or suspended.

## What remains

1. **`RADIATION`** -- 5.4% of the tree and the largest remaining leaf.
   `noahmp_radiation.cu` exists and is oracle-matched; this is a seam and a
   packing job, not new arithmetic. Note its `glibc_powf` is one of the five
   narrow copies that return NaN for a subnormal or negative base, so unlike
   `BARE_FLUX` and `WATER` this conversion has to answer the domain-guard
   question before it can be wired in.
2. `THERMOPROP`, `TSNOSOI`, `PHASECHANGE` -- 5.5% between them, all with
   existing `__device__` cores.
3. ENERGY's own composition and `SFLX_PRE`, which are not leaves.
4. The ten proved-identical `expf`/`logf`/`atanf` device copies are still ten
   copies; merging them remains free and unclaimed.
5. `powf`'s five narrow domain guards are still unsettled, and item 1 is the
   first conversion that actually depends on the answer.

## The registry strings this lane did not change

The task said not to move the registry template or the column-budget gate, so
neither was touched. For the lead deciding what to do with them, here is
exactly what is now stale and what the measurement says instead. All three
strings quote the **pre-VEGE_FLUX** host figures, so they were already one
conversion out of date before this lane started.

| where | what it says now | what is now measured |
|---|---|---|
| `land_surface.noah-mp` scaling warning | "17.47, 9.61, 7.66, 7.31 and 7.24 ms per land column at 2, 8, 48, 160 and 352 columns, i.e. 7.18 ms/column asymptotic and 97% Python interpreter time inside NOAHMP_SFLX. A 360,000-land-column nest is therefore about 43 minutes of wall clock PER LAND-SURFACE CALL" | 10.95, 3.45, 1.34, 1.07 and 1.00 ms at the same widths on one RTX 5090; about **six minutes** for that nest |
| the same warning's flatness claim | "The cost is FLAT in the column count" | still substantially flat, but it now falls 25% from 48 to 352 columns rather than 6%; the flat term is `RADIATION`, `THERMOPROP`, `TSNOSOI`, `PHASECHANGE`, ENERGY's own composition and the SFLX prefix |
| the expert-template acknowledgement | "Noah-MP's host column costs 7.18 ms per land column and does not get cheaper with more columns" | 1.00 ms/column at 352; the *reason* for expert-only is unchanged, the number is not |
| `NOAHMP_RUNTIME_RESTRICTIONS["column_solver_location"]` | already updated by this lane (it is `gpuwm/core/noahmp_*` and therefore in scope) | — |

**The recommendation is that the expert-only status and
`GPUWM_NOAHMP_EXPERT_COLUMN_BUDGET` both stay exactly as they are.** Six
minutes per land-surface call at `bldt = 0.0` is not a forecast, and 352
columns is still the right order for the default budget. Only the quoted
numbers should change: a warning that overstates the cost by 7x is as much a
false statement as one that understates it, and the next person to read it will
either disbelieve the rest of the warning or re-derive the figure themselves.

---

# Third conversion: RADIATION, THERMOPROP, TSNOSOI, PHASECHANGE

*Same machine as the second section -- Drew's local RTX 5090, Windows, CUDA
13.0, CuPy 14.1.1 -- so the millisecond figures below **are** comparable with
that section's and are **not** comparable with the first section's, which came
off a rented Linux card.*

## Result, in the unit that matters

Seven leaves now execute as CUDA batches. At 352 land columns, measured twice
independently and paired in one process against this same tree's host
authority:

| land columns | host authority ms/col | device leaves ms/col | speedup |
|---:|---:|---:|---:|
| 2 | 11.72 -- 13.28 | 10.79 -- 11.18 | 1.09 -- 1.19x |
| 8 | 5.07 -- 5.08 | 3.16 -- 3.19 | 1.59 -- 1.61x |
| 48 | 3.00 -- 3.16 | 0.88 -- 0.92 | 3.28 -- 3.60x |
| 160 | 2.71 -- 2.72 | 0.55 -- 0.64 | 4.25 -- 4.96x |
| 352 | 2.65 -- 2.66 | **0.535 -- 0.581** | **4.6 -- 5.0x** |

Those are whole-RK3-step figures divided by land columns, which is what the two
sections above quote. This lane also timed `noahmp_lsm_step` itself, which the
earlier sweeps could not separate from the dycore:

| land columns | host authority LSM ms/col | device LSM ms/col | speedup |
|---:|---:|---:|---:|
| 48 | 2.67 -- 2.77 | 0.542 -- 0.579 | 4.78 -- 4.94x |
| 160 | 2.61 -- 2.62 | 0.450 -- 0.544 | 4.82 -- 5.79x |
| 352 | 2.60 -- 2.61 | **0.486 -- 0.532** | **4.90 -- 5.35x** |

The previous device column measured **1.00 ms/col** at 352 on this same box, so
the device path is **1.7 -- 1.9x faster than it was**, and the host authority is
unchanged at 2.65 versus 2.62 ms -- the two device columns are being compared
against the same quantity.

**In the user's unit.** A 360,000-land-column d04 call is now about **193 -- 209
seconds, 3.2 -- 3.5 minutes**, against six minutes before and about 15.9 minutes
with the leaves on the host. At `bldt = 0.0` that call happens every step; at
`dt = 1.667 s` -- the 36-steps-per-simulated-minute the six-minute figure was
quoted against -- that is

> **1.9 -- 2.1 hours of wall clock per simulated minute**, down from 3.6.

A 15-minute forecast goes from 54 hours to **29 hours**. That is a real
improvement and it is **still not survivable**, and "what it would take to be
survivable" below says what the remaining cost actually is rather than
promising that another leaf will fix it.

## Measured ULP

**0**, everywhere it was checked.

- All four new leaves reproduce the CPython leaf the WRF fixtures pin, bit for
  bit, on every physical call the four whole-column fixtures pause at --
  including the **in-place** half, because WATER and the thermal leaves write
  into the arrays they are handed as much as into what they return.
- Every radiation leaf oracle -- SNOW_AGE, SNOWALB_CLASS, GROUNDALB, TWOSTREAM,
  SURRAD, ALBEDO -- is max_ulp 0 through the **shipped** compilation, which is
  `-std=c++17` and nothing else. See "the --fmad=false question".
- The composed `d_radiation` reproduces ALBEDO's own outputs bit for bit and
  matches SURRAD evaluated on the host from those outputs through :2787-2790.
- Six steps of the whole runtime on a bare and on a snowpack domain, all 45
  carried arrays: bitwise against the pre-conversion tree **and** against this
  tree's own host authority.
- The complete collected Noah-MP suite is **1058 passed, 2 skipped** on the GPU.

## Bitwise evidence

### The six-step carried state, re-derived rather than trusted

The section above records that both historical digests had stopped reproducing
before the previous lane. They reproduce now, and this lane re-derived them
rather than reading them: the pre-change tree was extracted from commit
`004a539` to a scratch directory with `git archive` and run there.

| domain | `004a539` (pre-change), device | this lane, device | this lane, host authority |
|---|---|---|---|
| bare | `b8463904221a6db3ccd29175d83da9685f06a9c743f050ca25681c5ab97d4526` | same | same |
| snowpack | `364c088b65a23d9ceeb48ee3370b77a1bdce054e86035b7d3bbdfd65c40bb3ca` | same | same |

So four more leaves moved to the device and **not one bit of a six-step
trajectory moved with them**, on a domain with no snow and on a domain with a
45 mm pack.

### The packing, which is the only new code

The four kernels are not new. THERMOPROP, TSNOSOI and PHASECHANGE are reached
through the *existing* `noahmp_leaf_thermoprop`, `noahmp_thermal_tsnosoi` and
`noahmp_thermal_phasechange` entry points at their existing flat layouts, and
no line of their arithmetic was touched. RADIATION needed one composition,
`d_radiation` = `d_albedo` + the five assignments at :2787-2790 + `d_surrad`,
because the alternative -- two launches with those five statements on the host
-- puts per-column CPython back on the path, which is the cost this whole seam
exists to remove.

That leaves the *packing* as the only thing that can be wrong, so it has its
own file. `tests/test_noahmp_leaf_batches_cuda.py` runs the four unmodified-WRF
whole-column fixtures through `sflx_steps`, records every physical call each
column pauses at, and answers each one both ways. It compares the returned
record **and** the arguments afterwards, because a packer that unpacked into
the wrong slice would leave the record right and the column wrong.

Three falsifications, each shown failing before any of it counted:

- **Misrouting one column's answer** is rejected for all seven leaves.
- **Transposing SOLAD and SOLAI** in the RADIATION pack -- adjacent slots
  carrying different numbers into different SURRAD terms, which is the defect
  this seam is most likely to produce -- is rejected on every daylight column.
- **Dropping an output** is rejected: the key sets must differ by exactly the
  declared `UNRECONSTRUCTED` set and nothing else. That set has one member,
  WATER's `ETRANI`, which WRF computes as a local that `NOAHMP_SFLX` never
  reads.

One harness defect worth recording because it looks exactly like a physics
defect: WATER is INOUT and mutates its arguments, so the first version of that
file answered on the host and then handed the *same*, already-mutated tuples to
the device and reported an SMC difference. Deep-copying each call before the
host sees it is what makes the comparison a comparison. A leaf-batch harness
that does not do this will invent a bug.

## The --fmad=false question, settled

`tests/test_noahmp_radiation_cuda.py` has always compiled the kernel with
`--fmad=false`, with a comment saying the flag was belt-and-braces. This
project's standing rule is that `-fmad=false` is a diagnostic and never a
shipped fix, and the runtime compiles through
`gpuwm.core.kernels.load_module`, which passes `-std=c++17` and nothing else.
So the comment was an untested claim about the source rather than a measurement
of the translation unit that actually runs.

It is now measured. Every radiation leaf oracle is **max_ulp 0 in the shipped
compilation, with FMA contraction enabled**. The intrinsics really are all
there.

## The glibc_powf domain guard, settled

`noahmp_radiation.cu`'s `glibc_powf` is one of the five narrow copies: it
returns NaN for any base outside `[FLT_MIN, inf)` -- both zeros, every
subnormal, every negative -- and for a zero exponent, instead of taking glibc's
sign/zero/inf path. The section above flagged this as the question RADIATION
would be the first conversion to depend on.

It does not depend on it, and that is a measurement rather than an argument.

**There is exactly one live `powf` call site in the file**, in GROUNDALB:

```
FADD(glibc_powf(f_max(K_P01, cosz), K_P17), K_P15)
```

The base is `MAX(0.01, COSZ)` -- **WRF's own MAX**, in the pinned source. It is
at or above 0.01 for every possible COSZ, including both zeros, every
subnormal, every negative and NaN (`f_max` returns `(b > a) ? b : a`, so a NaN
COSZ yields 0.01). The guard is unreachable there.

Three things now hold that down, because a guard nobody has driven proves
nothing:

1. a new `noahmp_rad_libm_probe` kernel drives `glibc_powf` with both zeros,
   the smallest subnormal, a negative subnormal, `FLT_MIN`, three negative
   bases, both infinities and NaN, and the guard is **observed** returning NaN
   for all of them -- and observed on a zero exponent as well;
2. the same test records that the authoritative host transcription does **not**
   answer NaN for all of those, so the guard is a real narrowing and not a
   restatement of the reference behaviour; and
3. GROUNDALB itself is swept over that entire neighbourhood of COSZ plus the
   0.01 boundary, and produces no NaN at all.

Item 3 is the one that would fail if the analysis were wrong, and it is the one
to re-run if RADIATION ever gains a second `powf`.

Separately, on the domain the guard admits -- 4,098 bases across `[0.01, 1.0]`
including the boundary and its neighbours -- the device `glibc_powf` is
**max_ulp 0 against `gpuwm.core.noahmp_libm.powf`**. That is the half of the
nine-copies question RADIATION actually depends on: not whether the copies
agree at zero, but whether they agree where GROUNDALB evaluates them.

## Is it still flat?

**Yes -- and it is now the only thing left to say about the cost.**

From 48 to 352 land columns the device LSM figure falls from 0.542--0.579 to
0.486--0.532 ms/column: between 2% and 16%, against 11% for the host authority
over the same range. Adding seven times the columns buys almost nothing, which
means the remaining cost is not a launch cost, not a transfer cost and not an
under-occupied kernel. It is per-column host work, and there is now a
measurement of exactly what.

`cProfile` over one device-path `noahmp_lsm_step` at 352 land columns.
(Profiling inflates the absolute figure to 0.99 ms/column; the *shares* are the
result.)

| | share of the LSM call |
|---|---:|
| the seven device batches, launches and packing included | **13.8%** |
| residual host physics inside the column | 40.5% |
| the driver's own per-column Python and write-back | **45.7%** |

and within that residual: ENERGY's own composition 36%, the SFLX prefix (ATM,
PHENOLOGY, PRECIP_HEAT) 22%, `sflx_post`'s own arithmetic 10%. The single
largest named function in the whole call is `gpuwm.core.noahmp_libm.f32` at
**811,038 calls in three LSM calls** -- a quarter of the call is FP32 rounding,
in Python, one value at a time.

**Packing is not the flat term, and this measurement is why converting these
four leaves was worth doing at all.** Measured directly on the exact call
tuples the runtime produces, the three pre-existing packers cost 24.2
microseconds per column *together* at 352 columns -- VEGE_FLUX 7.8, BARE_FLUX
4.8, WATER 11.7. That is about 2% of the previous 1.00 ms/column figure. A new
leaf therefore costs 5--10 us of staging and buys its whole profile share,
which is what the 1.7--1.9x above is made of. The worry that leaf conversion
would eventually cost more in packing than it saved was wrong, and it was wrong
by an order of magnitude.

## What it would take to be survivable

The batched share of the *host* column tree, measured on the four whole-column
fixtures, is now **87.5%**, up from 76.4%:

| batched leaf | share of the sflx tree |
|---|---:|
| VEGE_FLUX | 49.40% |
| WATER | 14.67% |
| BARE_FLUX | 12.13% |
| RADIATION | 5.48% |
| THERMOPROP | 3.67% |
| TSNOSOI | 1.25% |
| PHASECHANGE | 0.87% |
| **total** | **87.47%** |
| residual host | 12.53% |

**Leaf conversion is now essentially exhausted.** What is left is not a leaf:
ENERGY's own composition arithmetic, the SFLX prefix, NOAHMP_SFLX's own
marshalling, and the driver's per-column loop. Scaling the measured shares
against the measured 0.532 ms/column:

- move **all** the remaining physics -- ENERGY assembled on the device, the
  SFLX prefix, `sflx_post` -- and the call falls to about **0.32 ms/column**,
  1.9 minutes for that nest, **1.1 hours per simulated minute**. Better, and
  still not a forecast.
- remove the driver's per-column Python as well, leaving only the batches, and
  it falls to about **0.07 ms/column**, 26 seconds for that nest, **16 minutes
  per simulated minute**. That is the first genuinely different regime.

So the honest statement is: **Noah-MP cannot reach production width by moving
more leaves. It needs the per-column Python loop itself to go** -- the
`for j: for i:` in `noahmp_lsm_step` that builds a kwargs dictionary, a
`SnowColumn`, a `ficeold` vector and a generator frame per column, and the
`_write_back` that assigns about sixty fields per column afterwards. That is
45.7% of the call before any physics runs, and no leaf conversion touches it.

The two things that would have to happen, in order:

1. **Assemble ENERGY on the device.** The obstacle is recorded verbatim in
   `noahmp_energy.cu`, `noahmp_sflx.cu` and `noahmp_energy_gpu.py`: several
   `.cu` files each define their own `glibc_logf`/`glibc_expf`/`glibc_powf`/
   `f_min`/`f_max`, so any two in one translation unit are duplicate
   definitions. All of those files are in this lane's ownership, so the
   refactor is available; the low-risk form is a C++ namespace per source,
   which keeps every leaf's *validated* arithmetic and its own libm copy
   exactly as measured and makes the composition the only new thing to gate.
2. **Vectorise the driver loop and the write-back**, which are pure array
   shuffling, and stage the column state as arrays rather than as one Python
   object per column.

Neither is a bitwise risk in the way a new kernel is. Both are large.

The one cheap thing left, worth about 3.5% of the tree, is the SFLX prefix:
`sflx_pre` would have to become a generator so ATM, PHENOLOGY and PRECIP_HEAT
can pause the same way, and `noahmp_fluxprep.cu` and `noahmp_vegprecip.cu`
already carry oracle-matched kernels for them.

## What is now pinned that was not

`gpuwm/core/noahmp_kernel_sources.py` and `tests/test_noahmp_kernel_sources.py`
are new, and they exist because the MYNN lane's VRAM census hit something that
looks like three broken kernels:

```
noahmp_driver.cu   noahmp_energy.cu   noahmp_thermal.cu
    NVRTC: identifier "r_pow" is undefined
```

They are not broken. They are **fragments**. glibc 2.39's `powf`/`expf`/`logf`
are transcribed exactly once in this tree, in `noahmp_leaves.cu`, because two
copies of a 32-entry constant table can drift and only one of them would be
audited against `glibc-libm-fp32.csv`. Those three files borrow that single
copy and are compiled after it, which is what `thermal_source()` and
`energy_source()` have always done -- so they compile and launch perfectly well
on the real path, including on the new TSNOSOI / PHASECHANGE / THERMOPROP
runtime path this lane added.

The trap is that the opposite rule is also wrong: `noahmp_fluxprep.cu` carries
its own libm tables, so prepending `noahmp_leaves.cu` to *it* is a
duplicate-definition error. Neither "compile each file alone" nor "compile
everything after the libm" is right, and nothing in the directory listing says
so. `NOAHMP_TRANSLATION_UNITS` now says it in a form a tool can read, with a
test that pins both halves -- every listed unit compiles, and every composed
unit still genuinely fails alone.

**A whole-project sweep should build its source text from
`translation_unit_source(name)` rather than from `read_text()` per `.cu`.**

### Per-thread local frames

The same lane found that `kf_column` reserves 5.7 GiB of non-pool device memory
on first launch, invisible to the CuPy pool, because it declares about 48
per-thread arrays dimensioned on `KF_KMAX = 128`, and derived the law
`reservation = (max_local_size_bytes - 1024) * 1536 * 170` on this card.

Every Noah-MP kernel was measured against it, reading `local_size_bytes` off
the loaded module rather than guessing from PTX. The largest per-thread frame
in the whole scheme is **368 bytes** (`noahmp_thermal_tsnosoi`); most units
report 0. **Noah-MP reserves nothing.** Its arrays are dimensioned on
`NMP_NLAY = NSNOW + NSOIL = 7`, a physical constant of the pinned identity
rather than a compile-time ceiling. `test_the_per_thread_local_frame_is_small`
now holds every unit under 4 KiB so a future conversion cannot quietly
introduce one.

## VRAM and machine hygiene

Whole-machine `nvidia-smi` was polled every 0.2 s for the entire GPU session:
peak **6,037 MiB**, of which about 3,400 MiB is the user's desktop. That is 20%
of the 29,500 MiB rail. The lock was taken once, after polling from 18:10 to
19:00 while another lane held it, and released with `rm -f gpu.lock/owner`
followed by `rmdir gpu.lock`. No process was killed, stopped or suspended, and
no file this lane did not create was deleted.

## The registry strings this lane did not change

Unchanged instruction, unchanged compliance: the registry template and
`GPUWM_NOAHMP_EXPERT_COLUMN_BUDGET` were not touched. The recommendation is
also unchanged -- **expert-only stays, and the column budget stays** -- because
3.2 minutes per land-surface call at `bldt = 0.0` is not a forecast either.

What is stale is the same three strings the previous lane listed, now stale by
a further factor of two. For the lead:

| where | what it says now | what is now measured |
|---|---|---|
| `land_surface.noah-mp` scaling warning | 17.47 / 9.61 / 7.66 / 7.31 / 7.24 ms per land column at 2 / 8 / 48 / 160 / 352, "7.18 ms/column asymptotic", "about 43 minutes ... PER LAND-SURFACE CALL" | 10.8--11.2 / 3.16--3.19 / 0.88--0.92 / 0.55--0.64 / **0.535--0.581** ms at the same widths on one RTX 5090; about **3.2--3.5 minutes** for a 360,000-column nest |
| the same warning's flatness claim | "The cost is FLAT in the column count" | still flat, and now correctly so for a different reason: the flat term is no longer physics leaves but ENERGY's composition, the SFLX prefix and the driver's own per-column Python, which is 46% of the call |
| the expert-template acknowledgement | "7.18 ms per land column and does not get cheaper with more columns" | 0.535--0.581 ms/column at 352; the reason for expert-only is unchanged, the number is 12x out |
| `NOAHMP_RUNTIME_RESTRICTIONS["column_solver_location"]` | updated by this lane (it is `gpuwm/core/noahmp_*` and in scope) | -- |

## Defects found outside this lane's scope, not fixed

- `tests/test_physics_dispatch.py::test_noahmp_is_admitted_at_four_soil_layers_and_only_there`
  fails in the shared worktree with *"sf_surface_physics=4 requires a surface
  layer (sf_sfclay_physics != 0)"*. It **passes at committed HEAD `004a539`**;
  the failure comes from an uncommitted `gpuwm/config.py` edit in another lane
  (the message names Noah, Noah-MP and RUC together). Reported, not touched.

---

# Fourth conversion: the SFLX prefix, the driver loop, and a correction to the diagnosis above

*Same machine as the second and third sections -- Drew's local RTX 5090,
Windows, CUDA 13.0, CuPy 14.1.1.  Every millisecond below was taken with **one
script, run back to back on the pre-lane tree and on this one**, because the
absolute figures on this box move by up to 30% between harnesses and only a
paired measurement is worth anything.  See "the numbers above do not
reproduce", which is a correction and not a quibble.*

## Result, in the user's unit

`NOAHMP_SFLX`'s whole first half -- ATM, the DZSNSO/TROOT/BEG_WB marshalling,
PHENOLOGY, the FVEG selection and PRECIP_HEAT -- now runs as one device batch,
and `module_sf_noahmpdrv.F`'s 2-D-to-1-D marshalling and its write-back are
vectorised over the slab instead of looping over columns.  Eight leaves are
batched, and every remaining piece of Noah-MP that has an oracle-matched CUDA
kernel in this tree is now reached from the runtime.

`noahmp_lsm_step` itself, paired in one process against this same tree's host
authority.  Three independent runs of the pre-lane tree (`085a020`, extracted
with `git archive` to a scratch directory) and four of this one:

| land columns | host authority ms/col | `085a020` device ms/col | this lane, device ms/col |
|---:|---:|---:|---:|
| 2 | 6.80 -- 7.02 | 5.24 -- 6.48 | 6.14 -- 6.52 |
| 8 | 3.51 -- 3.62 | 1.57 -- 1.82 | 1.72 -- 1.76 |
| 48 | 2.59 -- 2.64 | 0.533 -- 0.597 | 0.521 -- 0.536 |
| 160 | 2.45 -- 2.51 | 0.359 -- 0.462 | 0.357 -- 0.378 |
| 352 | 2.45 -- 2.51 | **0.431 -- 0.454** | **0.339 -- 0.365** |

The host authority is unchanged -- 2.45--2.51 ms/column in both trees -- so the
two device columns are being compared against the same quantity, and the
restructuring costs nothing on the host path.

**In the user's unit.**  A 360,000-land-column d04 call at `bldt = 0`,
`dt = 1.667 s`, 35.99 steps per simulated minute:

| | seconds per land-surface call | **wall hours per simulated minute** |
|---|---:|---:|
| `085a020` (before this lane) | 155 -- 163 | **1.55 -- 1.63** |
| this lane | 122 -- 131 | **1.22 -- 1.31** |
| the same leaves forced on the host | 882 -- 903 | 8.8 -- 9.0 |

**1.2 -- 1.3 hours of wall clock per simulated minute, from 1.55 -- 1.63.**  A
15-minute forecast goes from about 24 hours to about 19.  It is a real
improvement and it is **still not survivable**, and the corrected ceiling
analysis below says why more honestly than the previous section did.

## The numbers above do not reproduce, and that matters

The third section published **0.535 -- 0.581 ms/column** at 352 for commit
`085a020`.  Measured again today on the same box, same commit, with this lane's
harness -- `noahmp_lsm_step` wrapped, four repetitions, host authority run
first in the same process -- it measures **0.431 -- 0.454**.  A different
harness in this lane's first hour, three repetitions and a fresh domain per
width, measured **0.478**.

So the absolute figure on this machine depends on the harness and on the hour
by up to 30%, and no single published millisecond from any of these four
sections should be compared with any other.  What is safe is the **paired**
comparison: two trees, one script, back to back, minutes apart.  That is how
the table above was taken and it is the only claim this section makes about
speed.

## Measured ULP

**0**, everywhere it was checked.

- **The prefix batch** reproduces the CPython `sflx_pre` -- the routine the WRF
  fixtures pin -- on every physical call the four whole-column
  `noahmp-sflx.csv` fixtures pause at, over the whole `PreEnergy` including its
  nested `EnergyCall`: 143 compared words per column, 572 in all, no differing
  bits.
- **Six steps of the whole runtime**, bare and snowpack, all 45 carried arrays,
  reproduce the digests this lane re-derived from this tree before touching
  anything:

  | domain | six-step SHA-256 |
  |---|---|
  | bare | `b8463904221a6db3ccd29175d83da9685f06a9c743f050ca25681c5ab97d4526` |
  | snowpack | `364c088b65a23d9ceeb48ee3370b77a1bdce054e86035b7d3bbdfd65c40bb3ca` |

  and equal this tree's own host authority, which is now literally `sflx_pre`
  itself: `register_host_leaf("sflx_pre", sflx_pre)`.
- **The write-back was checked over a wider field set than any gate this lane
  inherited.**  `_CARRIED` is 45 arrays; the write-back also fills 26
  diagnostics, and `RS`, `SOILENERGY`, `SNOWENERGY`, `Q2MV`/`Q2MB` and
  `PONDING` -- the branchy half of `:1223-1400` -- are in **none** of the
  digests above.  The pre-change tree was extracted from `50dcf50` with
  `git archive` and run in a scratch directory; six steps on a bare and on a
  snowpack domain agree on **all 73 fields the driver owns**, field by field
  and digest by digest.
- The complete collected Noah-MP suite is **1069 passed, 2 skipped** on the
  GPU and **720 passed, 351 skipped** under `GPUWM_NO_LOCAL_GPU=1`.

## What moved, and what deliberately did not

`gpuwm/core/noahmp_sflx_pre_gpu.py` is new.  It answers the physical
`sflx_pre` call for a whole batch with four kernels that were **already in this
tree and already oracle-matched**: `noahmp_leaf_atm` (`noahmp_leaves.cu`),
`k_sflx_marshal` (`noahmp_sflx.cu`), `noahmp_phenology` and
`noahmp_precip_heat` (`noahmp_vegprecip.cu`).

**No arithmetic was transcribed.**  The only new code is the packing, the FVEG
block -- four comparisons and one float32 add, over the column axis -- and the
reconstruction of `PreEnergy`.  A composition lane that re-transcribes a leaf
has thrown away the leaf's gate, so the shape of this conversion is the same as
the third section's: a seam and a packing job.

The seam itself is one line.  `sflx_steps` now yields
`LeafRequest("sflx_pre", ...)` where it used to call `sflx_pre`, and
`register_host_leaf` binds `sflx_pre` as the host answer.  Everything else --
`_drive_staged_columns`, `evaluate_leaf_batch_on_host` as the paired authority,
`GPUWM_NOAHMP_HOST_LEAVES=1`, the generic misroute falsification -- applied to
it without modification.  That is what the generator seam was built for.

On the driver side, `Q_ML`, `Z_ML`, `P_ML`, `PRCP`/`PRCPSNOW`, `FVGMAX`,
`CO2AIR`/`O2AIR`, `FICEOLD`, the water-soil-category override, the urban
vegetation remap, the `PLAI` bare override, the two `CYCLE ILOOP` skips and the
glacier refusal are evaluated once for the slab; `_write_back_batch` replaces
`_write_back` and writes about sixty carriers with one advanced-index
assignment each instead of about sixty NumPy scalar stores per column.
Measured: the write-back went from about 9 ms to **2.96 ms** per land-surface
call at 352 columns.

Two places where the obvious vectorisation is wrong, and both are written out
rather than called:

1. gfortran's `MAX(x, 0.0)` at `:1305-1307`.  **Python's `max` and
   `numpy.maximum` disagree on a negative zero** -- `max(-0.0, 0.0)` is `-0.0`,
   `numpy.maximum(-0.0, 0.0)` is `+0.0` -- so the `where(b > a, b, a)` form is
   used, which is what `max` does.
2. The two column-energy integrals at `:1381-1394` accumulate **in layer
   order** and FP32 addition is not associative, so the layer loop is still a
   loop.  Only the column axis inside it is vectorised.

## The falsifications, and the one that caught a real defect first

**The prefix gate found a live packing bug on its first run.**  CuPy's
`atm_y[:, slot]` is a *strided* view -- 17 floats apart -- and a raw kernel
argument carries a pointer, not a stride.  The first version therefore fed
column 0's neighbouring output slots to PRECIP_HEAT on every column.  Measured
signature: `PAHV`/`PAHG`/`PAHB` left at zero and `CANLIQ` untouched on the one
fixture column that has rain, 34 differing words out of 572.  This is recorded
because it is exactly the defect class this report has been warning about for
three sections, it is invisible to every leaf's own oracle CSV, and the
whole-column gate caught it immediately.

Shown failing before any of it counted:

- transposing `sfcprs`/`sfctmp` in the flat 11-slot ATM **input** row is
  rejected;
- transposing `rain`/`snow` in the flat 17-slot ATM **output** row is rejected;
- misrouting one column's prefix answer is rejected by the generic batch gate;
- the four guards that make the crop and irrigation regions dead are driven and
  observed refusing;
- for the write-back: **rolling every column's result onto its neighbour moves
  the 73-field digest**, which is the defect a batched write-back actually
  produces -- index arrays and value arrays one apart, every field the right
  shape and every column carrying its neighbour's surface.

The durable write-back gate is a claim that needs no reference implementation:
`COLUMN_BATCH` decides the shape of every advanced-index assignment, Noah-MP
has no horizontal coupling, so `1`, `7` and `2048` must produce the same
slab -- and at `COLUMN_BATCH = 1` the batched write-back degenerates to one
column at a time, which is the shape the per-column version had.

## Is it still flat?

**Less flat than it has ever been, and still flat.**  From 48 to 352 land
columns the device figure falls 1.24--1.57x depending on the run, against
1.24--1.57x at `085a020`; the two ranges overlap and the 48-column point is
where the run-to-run noise is.  The host authority falls 1.04--1.07x over the
same range.  More of the cost amortises across columns than before, but 0.35
ms/column at 352 is still overwhelmingly a per-column host term.

## The correction: what the 45.7% actually was

The third section reported this decomposition of one 352-column call:

> device batches 13.8% / residual host physics 40.5% / **the driver's own
> per-column Python and write-back 45.7%**

and concluded

> **removing the driver's per-column Python as well ... falls to about 0.07
> ms/column ... 16 minutes per simulated minute.  That is the first genuinely
> different regime.**

**The 45.7% was not the driver.**  Re-measured at `085a020` with `cProfile`
over 352-column calls, the cumulative shares are:

| | share of the call at `085a020` |
|---|---:|
| the seven device batches | 13.5% |
| `energy_steps` -- ENERGY's own composition | 36% |
| **`sflx_pre` -- ATM, PHENOLOGY, PRECIP_HEAT and NOAHMP_SFLX's own marshalling** | **21.8%** |
| **`sflx_post_steps` -- NOAHMP_SFLX's own second half and ERROR** | **10%** |
| `_write_back` | 6.5% |
| the driver's own marshalling loop | 2.9% |

So the "driver's own per-column Python and write-back" was `sflx_pre` +
`sflx_post` + the write-back + the marshalling loop.  Three quarters of it is
**physics** -- ATM, PHENOLOGY, PRECIP_HEAT, ERROR and NOAHMP_SFLX's own
arithmetic -- and no amount of loop restructuring removes physics.  The driver
itself, the `for j: for i:` and `_write_back`, was about **9%**, and this
lane's vectorisation of it bought about 4% of the call, which is what a 9% term
can buy.

That does not make the 0.07 ms/column figure wrong; it makes the *route* to it
wrong.  `0.07` is what is left when **only the device batches remain**, and
reaching it needs ENERGY's composition, NOAHMP_SFLX's prefix and postfix and
the driver all on the device.  This lane moved the prefix and the driver.
Recomputed against the measured 0.339--0.365 ms/column with the post-change
shares:

| | share of the call now | ms/column if it alone went | h per simulated minute |
|---|---:|---:|---:|
| everything host-side, leaving only the eight device batches (22.5%) | -- | 0.079 | **0.29** (17 minutes) |
| ENERGY's own composition (`energy_steps`) | **49.1%** | 0.18 | 0.64 |
| NOAHMP_SFLX's second half and ERROR (`sflx_post_steps`) | 13.9% | 0.30 | 1.1 |
| the driver's marshalling and write-back | about 5% | 0.33 | 1.2 |

`gpuwm.core.noahmp_libm.f32` is still **21% of the call on its own** -- FP32
rounding, in Python, one value at a time -- and it is spread across ENERGY and
`sflx_post`, not concentrated anywhere a seam can cut.  It is measured at 64 ns
per call against 46 ns for the same `struct` round trip written inline, so
there is no version of it that is materially cheaper; the only way to stop
paying it is to stop evaluating FP32 arithmetic in CPython.

## What remains, in the order it is worth doing

1. **ENERGY's own composition on the device -- 49% of the call, and the only
   remaining item that changes the regime.**  `noahmp_energy.cu` already
   carries `noahmp_energy_assembly`, bitwise against `noahmp-energy.csv`, but
   it takes the six subsystems' outputs as *inputs* from the fixture, and in a
   running column those subsystems need the composition's own intermediate
   results.  Two routes, and they are not equally good:
   - split the assembly into the six segments between the leaf calls and batch
     each, which needs six flat layouts, six packers and six new gates derived
     from a fixture that pins only the whole; or
   - **compose the whole column into one kernel**, which is the refactor
     `noahmp_energy.cu` and `noahmp_sflx.cu` have both been recording as
     blocked since they were written: several `.cu` files each define their own
     `glibc_logf`/`glibc_expf`/`glibc_powf`/`f_min`/`f_max`, so any two in one
     translation unit are duplicate definitions, and `noahmp_thermal.cu`
     exposes TSNOSOI and PHASECHANGE only as `extern "C" __global__` entry
     points with no reusable `__device__` core.  Every one of those files is in
     this lane's ownership.  The second route is larger and is the one that
     reaches 0.079 ms/column; the first cannot, because it leaves the packers
     and the generator frame in place.
2. **`sflx_post` and ERROR -- 13.9%.**  `k_sflx_error` exists and is
   oracle-matched, so ERROR is the same seam-and-packer job the prefix just
   was, worth about 3% net after its own packing.  The rest of `sflx_post` is
   NOAHMP_SFLX's own arithmetic and has no kernel.
3. The prefix batch's own reconstruction, 9.4 ms per 352-column call, of which
   about 1 ms is eight `ndarray.copy()` per column and about 1.6 ms is
   re-wrapping already-binary32 keywords in `np.float32`.  Both are avoidable
   by handing `EnergyCall` contiguous rows of the batch's own arrays; worth
   about 2%, and it is bookkeeping rather than physics.
4. The ten proved-identical `expf`/`logf`/`atanf` device copies are still ten
   copies, and merging them is now not merely free -- it is **the first step of
   item 1's second route**.

## VRAM and machine hygiene

Whole-machine `nvidia-smi` was polled continuously for the full GPU Noah-MP
suite (600 samples): peak **3,019 MiB**, minimum 2,496 MiB, of which about
2,500 MiB is the user's desktop.  That is 10% of the 29,500 MiB rail, and the
suite's own footprint is about 500 MiB.  The width sweeps ran inside the same
lock.  The GPU lock was taken once for the whole session and released with
`rm -f gpu.lock/owner` followed by `rmdir gpu.lock`.  **No process was killed,
stopped or suspended, and no file this lane did not create was deleted.**

The largest per-thread local frame in the scheme is unchanged: the four kernels
this conversion reaches were already loaded by the leaf tests, and
`test_the_per_thread_local_frame_is_small` still holds every unit under 4 KiB.

## The registry strings this lane did not change

Unchanged instruction, unchanged compliance: the registry template,
`GPUWM_NOAHMP_EXPERT_COLUMN_BUDGET` and the maturity grade were not touched,
and the recommendation is unchanged -- **expert-only stays and the column
budget stays** -- because two minutes per land-surface call at `bldt = 0` is
not a forecast either.

`NOAHMP_RUNTIME_RESTRICTIONS["column_solver_location"]` is in this lane's
ownership and is **stale as of this section**: it quotes seven leaves, 0.55
ms/column, 3.2--3.5 minutes per nest, and "the driver's per-column Python and
write-back, which is 46% of the call on its own".  The last of those is the
claim this section corrects.  It was left alone rather than edited because the
correction belongs in one place first, and because the previous two sections
both recorded their stale strings here rather than changing them mid-lane; the
lead should move all four strings together.

## Defects found outside this lane's scope, not fixed

- `tests/test_physics_dispatch.py::test_noahmp_is_admitted_at_four_soil_layers_and_only_there`
  still fails in the shared worktree with *"sf_surface_physics=4 requires a
  surface layer (sf_sfclay_physics != 0)"*, from another lane's uncommitted
  `gpuwm/config.py` edit.  Unchanged from the previous section.  Reported, not
  touched.

---

# A surface-call interval that is opt-in, and what it actually costs

Every section above measures the same thing: the wall clock of one Noah-MP
land-surface call.  This one is about how *often* that call happens, which is
the larger number and was never this lane's to change until now.

At d04 -- 600x600, 360,000 columns, `dt = 5/3 s` -- `bldt = 0.0` schedules a
land-surface call on **every dynamics step**, 36 per simulated minute.  That is
the 1.22--1.31 hours of wall clock per simulated minute this report has been
recording, and the ceiling analysis two sections up puts full device conversion
near 15 minutes per simulated minute, still 33x the entire four-domain MYNN
stack.  Optimisation alone does not reach usable.  A 60-second interval is a
straight **36x**, an order of magnitude larger than anything this lane has
landed by converting arithmetic.

It was authorised for Noah-MP with a constraint: it must not become permanent,
and no future configuration -- or future reader -- may come to treat it as
normal.  So the whole of the implementation is about scope.

## What WRF's own call structure permits, which is less than one would like

The obvious refinement is to give the *land surface* a long interval and leave
the *surface layer* on every step, because soil moisture and soil temperature
evolve on hours-to-days timescales while the surface layer at 333 m does not.
That would be the honest version of this change rather than the cheap one.

**WRF does not offer it.**  In `phys/module_surface_driver.F` the predicate is
computed once, into `run_param`, and `run_param_if:` opens at **:1895** and
closes at **:4500**.  Inside that single block are both

    sfclay_select   :2003 -- :2469        the surface layer
    sfc_select      :2541 -- :4361        the land surface model
                                          (NOAHMPSCHEME at :3043)

and the interval is handed to both as the same `DTBL = DT * STEPBL` (:1970,
:1974).  `phys/module_pbl_driver.F` repeats the identical predicate verbatim at
:904--:941, so the PBL is on that clock too.  Searching
`Registry/Registry.EM_COMMON` for a second boundary-layer cadence finds none:
`BLDT` (:2487) is the only one.  `noahmp_acc_dt` (:2702) is an accumulator
bucket reset, not a call interval.

gpuwm mirrors this exactly -- one `_surface_pbl_step_due` gate at
`gpuwm/core/physics.py:1615` wrapping the surface-layer, land-surface and PBL
dispatch, with `bldt_seconds` handed to each as its physics `dt`.

So the decoupled version is not a configuration choice.  It would be a
**structural divergence from the reference**, which in a port whose acceptance
bar is bitwise against unmodified WRF is a much larger decision than a namelist
value, and not one this lane should take unilaterally.  Recorded here so the
option is understood rather than quietly forgotten: it is available, it is not
free, and it costs the parity argument.

The consequence for the measurement below is that a longer `bldt` does not buy
a slower soil column alone.  It buys a **surface layer and a PBL updated once
per simulated minute at 333 m grid spacing**, and that is what has to be
priced.

## How the setting is scoped so it cannot spread

The failure mode worth guarding is inheritance, not authorship.  `bldt` is in
`allowed_parameter_keys` for the `tools.prepared_domain_tree_forecast` route,
and every registry template carries a `parameters` dict that routes merge in.
One `"bldt": 1.0` in a template would reach every configuration built from that
template, silently, with no author having typed it.

* `RunConfig.bldt` is **still `0.0`** and `gpuwm/config.py` was not edited.
* The registry's generic `parameters["bldt"]["default"]` is **still `0.0`** and
  `gpuwm/physics_registry_v2.json` was not edited.
* The interval lives in exactly one file,
  `configs/real74_4dom_noahmp_surface_interval.toml`, as a `[[domain]]`
  override on **d04 only**.  `[shared]` keeps `bldt = 0.0`, so d01--d03 call
  the surface every step.  `resolve_clock` confirms it: STEPBL is 36 on d04 and
  1 on the other three.
* The justification travels with the assignment.  The gate requires the exact
  sentence `COST MITIGATION, NOT A PHYSICS RECOMMENDATION` in the comment block
  *directly above* the `bldt =` line; a copy elsewhere in the file does not
  license the value.

`tests/test_noahmp_surface_interval.py` is the gate, and it was shown failing
before it passed -- first with no config on disk, and with the `RunConfig`
default read off an instance rather than off the dataclass field.  It refuses a
`bldt` **key** in a template rather than a non-zero value, because a template
is the inheritance vector and `"bldt": 0.0` there is one character from not
being zero.  Eight negative controls inject each violation into a scratch copy
and prove it is reported.

**Residual surface, stated rather than fixed.**  `bldt` remains in
`allowed_parameter_keys` for the domain-tree route, so a plan may still pass it
as an explicit per-run override for any template on that route, including Noah
rather than Noah-MP ones.  That is the registry's designed explicit-override
mechanism and narrowing it would mean editing another lane's file, so it is
recorded here instead of changed.

## A defect the opt-in exposed

`gpuwm/runtime.py:958` assigned

    state.physics.bldt_seconds = integration_cfg.dt
    state.physics.stepbl = 1

**unconditionally**, after `initialize_physics` had already derived both from
`cfg.bldt`.  Any configured interval was silently discarded on the
single-domain path.  At `bldt = 0` the overwrite is a no-op, which is why
nothing in the suite ever caught it -- every config in the tree is `bldt = 0.0`.

It is now guarded by `if integration_cfg.bldt == 0.0`, which is bit-identical
for every case that has ever run through it.  The multi-domain path never had
the defect: it takes its cadence from `resolve_clock`
(`gpuwm/core/clock.py:556-589`), which is why the four-domain config above
works.  This is the only file outside this lane's ownership that was touched,
and the diff is those two lines plus the comment explaining them.


## What the interval actually costs the forecast, measured

Two arms, identical in every respect except `bldt`, both in one process on the
local RTX 5090:

* the geometry is the **d04 regime itself** -- `dx = 333.33 m`, `dt = 5/3 s` --
  so `bldt = 1.0` is exactly `STEPBL = 36`, the same decimation the real nest
  would see;
* 24x14 = 336 columns, 308 of them land, deliberately **under the 352-column
  measured ceiling** so no expert budget is acknowledged merely to measure;
* 8,640 steps = **4 simulated hours**, from 12:00Z (05:20 local solar);
* the shortwave forcing is an imposed analytic diurnal cycle, 133 -> 787 W/m2
  across the window, **not** a constant.  That matters: a steady-forcing
  experiment lets the surface equilibrate and would understate the cost,
  because tracking fast change is the entire purpose of a short interval.
  `gpuwm/core/noahmp_runtime.py` rebuilds its host slab from `driver.fields`
  on every call, so the imposed forcing genuinely reaches Noah-MP.

### Wall clock

| | surface calls | wall | wall s per simulated minute |
|---|---:|---:|---:|
| `bldt = 0` | 8,640 | 1,162.43 s | **4.84** |
| `bldt = 60 s` | 241 | 85.96 s | **0.36** |

**13.5x**, not 36x, because the cheap arm still integrates all 8,640 dynamics
steps.  The land-surface cost itself divides out cleanly from the paired
difference -- the arms differ only in how many surface calls they made:

    d(wall) = 1076.47 s over d(calls) = 8399  ->  128.17 ms per call
                                              ->  0.4161 ms per land column

Projected to d04 at 360,000 land columns (the all-land upper bound for a
600x600 nest, which is the figure this report has been using; a real nest with
water is proportionally cheaper):

| | per land-surface call | **wall per simulated minute at d04** |
|---|---:|---:|
| `bldt = 0` | 149.8 s | **1.50 h** |
| `bldt = 60 s` | 149.8 s | **2.5 min** |

The `bldt = 0` figure independently reproduces the 1.22--1.31 h this report has
been recording from a different harness, which is the only reason it is quoted
across harnesses at all.

### The forecast impact, field by field

Worst absolute difference over the whole four hours, across the 308 land
columns:

| field | max abs diff | rms at that time | when |
|---|---:|---:|---:|
| 2 m temperature | 0.236 K | 0.096 K | 220 min |
| 2 m dewpoint | 0.723 K | 0.276 K | 220 min |
| sensible heat flux | 36.4 W m-2 | 14.2 W m-2 | 240 min |
| latent heat flux | 23.1 W m-2 | 8.4 W m-2 | 220 min |
| **PBL height** | **1243 m** | **849 m** | **100 min** |
| skin temperature | 0.214 K | 0.090 K | 220 min |
| soil temperature | **0.029 K** | -- | 220 min |
| soil moisture | **0.000222 m3 m-3** | -- | 240 min |

**The soil is essentially untouched and the PBL is not, which is exactly the
split the call-structure section predicted.** Over four hours the deepest soil
layers move by hundredths of a kelvin and two ten-thousandths of a volumetric
fraction -- a longer interval is, for the soil column alone, very nearly free,
which is what its hours-to-days timescale says it should be.

The PBL is a different matter, and the spatial breakdown is what makes it
plain.  At 100 simulated minutes, in the middle of the morning growth phase:

| | share of the 308 land columns | median abs diff | max |
|---|---:|---:|---:|
| PBL height differs by > 100 m | **90.9%** | 826 m | 1243 m |
| sensible heat flux differs by > 10 W m-2 | 0.0% | 2.0 W m-2 | 4.9 W m-2 |
| 2 m temperature differs by > 0.1 K | 0.0% | 0.056 K | 0.094 K |

Domain-mean PBL height at that moment is **888 m at `bldt = 0` against 668 m at
`bldt = 60 s`**, and the per-column median difference of 826 m against a mean
difference of only -220 m says the differences run large in *both* directions:
the growing boundary layer is not merely shallower, it is spatially
reorganised.  By 240 minutes, with the PBL near its afternoon equilibrium, only
13.6% of columns differ by more than 100 m and the median is 47 m -- while the
surface fluxes have become the larger discrepancy, 31.8% of columns differing
by more than 10 W m-2 in HFX.

One honest caveat on the PBL number: PBL height is *diagnosed* by a threshold
search, so it can move discontinuously for a small change in the profile, and a
single column's maximum overstates the physical difference.  But 90.9% of
columns with a median of 826 m during the growth phase is not a threshold
artifact; that is a different boundary layer.

### The verdict, plainly

**The difference is material, and it is material in the place the call
structure said it would be.** If `bldt` gated only the land surface this would
be an easy trade: 36x for hundredths of a kelvin in the soil.  It does not.  It
gates the surface layer and the PBL on the same clock, and a 333 m PBL updated
once per simulated minute is visibly wrong during the morning transition --
which, for a convective-storm case, is precisely the period the forecast exists
to get right.

So the setting is recorded for what it is: the difference between a run that
finishes and a run that does not, bought with a boundary layer that is
noticeably degraded while it is growing.  It is not a physics recommendation,
it is scoped to one nest in one file, and the right way to retire it is to
finish putting Noah-MP's column on the device rather than to widen its use.

For anyone tempted to reach for it anyway: the 2 m fields and the surface
fluxes stay within a few tenths of a kelvin and a few tens of W m-2, so a
verification that looks only at screen-level variables will report that this
change is nearly free.  It is not; the cost is in the PBL, and it is only
visible if PBL height is looked at during the transition.

### Machine hygiene for this measurement

Peak whole-machine VRAM over the run was **5,765 MiB** against the 29,500 MiB
rail, of which about 2,600 MiB is the desktop, so the measurement's own
footprint is roughly 3,200 MiB.  Headroom was checked after acquiring the lock
and before any device work.  The shared GPU lock was held by another lane for
most of this session; this lane queued behind it rather than working around it.
One earlier poller was stopped by the harness's background-command cap *after*
it had acquired the lock, leaving a stale lock directory that this lane
identified as its own and released with the documented
`rm -f gpu.lock/owner && rmdir gpu.lock`; the live poller then acquired
normally two seconds later and released cleanly at the end.  **No process was
killed, stopped or suspended, and no file this lane did not create was
deleted.**

# The conversion route's step 1 was not free, and the record was wrong

The previous section closed by naming the next work: compose the whole column
into one kernel, whose first step is "**the ten proved-identical
`expf`/`logf`/`atanf` device copies are still ten copies, and merging them is
now not merely free**".

Re-derived rather than trusted, that is wrong in both directions.

There are **34 `__device__` symbols defined in more than one Noah-MP kernel**,
and **16 of them have drifted**.  `gpuwm/core/noahmp_kernel_sources.py` states
that glibc's `powf`/`expf`/`logf` are transcribed *once* in this tree, in
`noahmp_leaves.cu`, "because two copies of a 32-entry constant table can drift
and only one of them would be audited".  The reason is exactly right.  The
claim is stale: `leaves` and `fluxprep` hold them as `r_pow`/`r_exp`/`r_log`,
and five other kernels hold their own as `glibc_powf`/`glibc_expf`/`glibc_logf`
in **three mutually different generations**.  The naming convention meant to
prevent the drift is what hid it.

The differences are arithmetic, not cosmetic:

| symbol | groups | why it matters |
|---|---|---|
| `f_min`, `f_max` | `bareflux` vs `radiation`/`snow`/`soilwater`/`water` | `(a < b) ? a : b` against `(b < a) ? b : a`.  Identical on ordinary values; they disagree on **signed zeros and NaN**.  `f_min(+0.0, -0.0)` is `-0.0` under one and `+0.0` under the other. |
| `glibc_powf` | `bareflux`/`radiation` vs `soilwater`/`water` vs `vegeflux` | the `water` copy carries `1**y` and `+0 ** finite-positive` early exits the `bareflux` copy lacks, so `bareflux`'s returns **NaN** for a zero base. |
| `glibc_expf`, `glibc_logf`, `glibc_atanf`, `powf_log2_inline`, `powf_exp2_inline` | `vegeflux` against the rest | an older generation using **literal float constants** where the newer files use `__constant__` tables. |
Of the 16, **8 are renames and 8 are real**, and the split matters more than
the total:

| symbol | groups | verdict |
|---|---|---|
| `combine`, `combo`, `compact`, `divide`, `snowfall`, `snowh2o`, `snowwater` | `snow` vs `water` | **rename.**  Byte-identical except that `water` prefixes every constant with `SN_` to avoid colliding with its own.  Both prefixes resolve through `C_F32` and `C_SN_F32`, and those two tables hold the same 32 bit patterns in the same order -- compared element for element with the comments stripped, because the comments are the only place they differ. |
| `f_abs` | `sflx` vs `soilwater`/`water` | **rename.**  `fabsf(x)` against `__float_as_uint(a) & 0x7FFFFFFF`.  Fortran's `ABS` on `REAL(4)` is a sign-bit clear and so is `fabsf`; they agree on every input including `-0.0`. |

So the merge is more tractable than the raw count suggests: seven of the
sixteen are one `SN_` prefix, and the eight that remain are the libm.  Three of
those deserve their reasoning recorded rather than just their existence.

**The signed-zero one is live in principle.** `-ftz=true` is appended by CuPy
unconditionally and gfortran does not flush, so signed zeros are precisely the
input class this project has already lost three bugs to.  Choosing either
spelling for a merged `f_min` changes the other group's arithmetic until every
call site in that group is shown not to reach one.  That is a per-call-site
obligation, not a one-line edit.

**The `powf` one is not a live defect, and establishing that took WRF rather
than reasoning.** `noahmp_bareflux.cu` calls `glibc_powf(1 - 16*MOZ2, 0.25)`
*inside* a branch guarded on `MOZ`, not on `MOZ2`, which looks exactly like a
reachable zero or negative base.  It is not: WRF's `SFCDIF1`
(`phys/module_sf_noahmplsm.F`) computes

    MOZ  = MIN( (ZLVL-ZPD)/MOL, 1.0)
    MOZ2 = MIN( (2.0 + Z0H)/MOL, 1.0)

from the **same** Monin-Obukhov length with two positive numerators, so `MOZ`
and `MOZ2` always share a sign and `MOZ < 0` implies `MOZ2 < 0`; both bases
exceed one wherever the branch runs.  The `MOZSGN >= 2` reset sets both to zero
together, which does not enter the branch.  A merge still has to resolve the
difference deliberately rather than by whichever copy is pasted first.

**The `vegeflux` generation is the one to retire, not to merge onto.**  ptxas
12.x mis-folds FP32 ties at compile time and `__constant__` is the only barrier
this project has measured to work; that generation uses bare literals.  Its
`expf` overflow threshold is `88.72283935546875f` where the newer files compare
against the double `88.72283172607422` -- **adjacent floats**, so the two
branch differently at exactly one input.  Its `USE_DEVICE_LIBM` escape is
negative-control only (`gpuwm/core/noahmp_vegeflux_gpu.py:139`) and is not
reached on the shipped path.

`tests/test_noahmp_kernel_symbol_duplication.py` pins the entire inventory as a
table, with five negative controls proving the extractor separates drift from
layout: identical bodies collapse to one group, a reordered comparison splits
into two, comments and line breaks do not count, forward declarations are not
definitions, and a symbol defined once is absent.  A merge now shrinks that
table as a visible diff, one symbol at a time, each re-gated against its own
consumer's oracle.  What it cannot do is grow unnoticed.

No kernel source was changed in this section.  The estimate it replaces -- that
the composition's first step was free -- was the reason it looked like the
cheap half of the remaining work.  It is not free, but having separated the
renames from the real divergences it is also not as large as the raw count
first suggested.  The honest sequencing is:

1. **the 18 identical copies and the 8 renames** -- mechanical, no arithmetic
   argument needed, and the seven snow routines collapse the moment `C_F32`
   and `C_SN_F32` become one table;
2. **retire the `vegeflux` generation** onto the `__constant__` one, which is a
   correctness improvement in its own right rather than only a merge step;
3. **adjudicate `f_min`/`f_max` per call site**, which is the only item
   requiring a reachability argument about signed zeros;
4. **then compose**, which is when the 49% ENERGY share becomes reachable.

Steps 1 and 2 need a GPU to re-gate each consumer at `max_ulp 0`, which this
session did not get: the shared card was held by another lane for its whole
length.  The inventory gate is what makes that work resumable without
re-deriving any of this.

# Fifth conversion: the column solver stops being a per-column Python program

*Drew's local RTX 5090, Windows, CUDA 13.0, CuPy 14.1.1.  The absolute
milliseconds on this box move by up to 30% between harnesses and between
hours, so nothing below is compared with a figure from an earlier section;
every comparison is paired, in one process, minutes apart.*

## Result, in the user's unit -- and it has not moved yet

**Noah-MP at d04 is still hours per simulated minute.**  Re-measured today
with `noahmp_lsm_step` itself wrapped -- not the whole RK3 step -- four
repetitions after a warm-up, two independent runs:

| land columns | ms/call | ms/column | s per d04 call | **wall s per simulated minute** |
|---:|---:|---:|---:|---:|
| 48 | 28.3--29.2 | 0.589--0.608 | 212--219 | 7,630--7,881 |
| 160 | 62.7--71.6 | 0.392--0.448 | 141--161 | 5,078--5,801 |
| 352 | 135.4--162.4 | **0.385--0.461** | 138--166 | **4,982--5,977** |

360,000 land columns, `dt = 1.667 s`, `bldt = 0`, 35.99 calls per simulated
minute.  The higher end of each pair was taken while the Noah-MP suite was
running on the same card and is kept in the range rather than discarded,
because that is the honest spread on this box.  The previous section's
1.22--1.31 hours and this 1.38--1.66 hours are the same regime measured by
two harnesses, and this document has already recorded that no single published
millisecond from one of its sections should be compared with another.

This section did not move that number, and says so plainly.  What it did is
establish -- by measurement rather than by argument -- that **no partial
conversion can move it**, build and gate the conversion that can, and leave it
one wiring step from running.  The parts landed are bitwise; the part not
landed is the orchestration that would let a forecast use them.

## The baselines, re-derived rather than trusted

Both six-step carried-state digests were re-derived from this tree before
anything was touched, and again after, over all 45 carried arrays, with the
device and host paths compared array by array as well as by digest:

| domain | six-step SHA-256 | device == host, all 45 |
|---|---|---|
| bare | `b8463904221a6db3ccd29175d83da9685f06a9c743f050ca25681c5ab97d4526` | yes |
| snowpack | `364c088b65a23d9ceeb48ee3370b77a1bdce054e86035b7d3bbdfd65c40bb3ca` | yes |

They agree with the strings the previous section published.  `_CARRIED` holds
45 names, confirmed by assertion and not by counting prose.

## Is Noah-MP wired?  Yes -- but nothing said so

The RUC lane lost a session to device leaves with **zero importers under
`gpuwm/`**, twice in a week, and this document's playbook says to check the
same thing here before trusting any number.  Checked:

`gpuwm/core/noahmp_runtime.py:83-90` imports all eight device batches,
`LEAF_BATCH_EVALUATOR` defaults to `evaluate_leaf_batch_on_device`, and
`gpuwm/core/physics.py:1436` is the one forecast call site.  **A Noah-MP
forecast does take the device path.**  Noah-MP's seam is a module attribute
rather than a `leaves=` argument, which is why it did not rot the way RUC's
did.

What was missing was any test that said so.  `tests/test_noahmp_device_wiring.py`
now replaces every registered CPython leaf answer with a raising tripwire and
steps a snow-covered forecast: if any leaf were answered on the host the step
cannot finish.  Both failing forms were observed firing before it was
committed -- the device evaluator withheld, and `GPUWM_NOAHMP_HOST_LEAVES=1`
set -- so the pass is evidence and not a gate that has never been able to
fail.  A fourth test reads the import edge directly: every entry of
`DEVICE_LEAF_BATCH` must be a function defined in a `gpuwm.core.noahmp_*_gpu`
module, so a batch quietly rebound to a CPython routine goes red here instead
of leaving the whole suite green.

Three things the wiring check found that are **not** in this lane's ownership
and were reported rather than touched:

* `noahmp_driver_gpu.py`, `noahmp_energy_gpu.py`, `noahmp_fluxprep_gpu.py` and
  `noahmp_leaves_gpu.py` have **zero importers under `gpuwm/`** -- they are
  reached only from `tests/`.
* `gpuwm/core/preflight.py:479-484` compiles thirteen Noah-MP translation
  units for `sf_surface_physics = 4`, three of which -- `noahmp_driver`,
  `noahmp_energy`, `noahmp_fluxprep` -- are **never launched** from `gpuwm/`.
  Every Noah-MP forecast pays that compile.
* `noahmp_runtime.py:33-34` and `:209-210` still say "seven leaves are
  batched"; there are eight.

## The decomposition, warm and snow separately

The playbook says decompose before optimising, and decompose the snow case
separately, because RUC's warm and snow calls had different dominant terms.
One 352-column call of each, `cProfile` around `noahmp_lsm_step`:

| | warm, share | snow, share |
|---|---:|---:|
| `_drive_staged_columns` | 92% | 93% |
| `sflx_steps` (the per-column generator frames) | 67% | 69% |
| `energy_steps` -- ENERGY's own composition | 48% | 51% |
| the eight device batches, end to end | 22.9% | 22.0% |
| `noahmp_libm.f32` | 20.3% | 19.6% |
| `sflx_post_steps` | 13.7% | 12.8% |
| `_write_back_batch` | ~1% | ~1% |

**Noah-MP's warm and snow calls cost the same** -- 0.267 s against 0.269 s for
the same 352 columns -- and that is a real difference from RUC rather than a
measurement that failed to separate them.  RUC's snow arm was 4.1x its warm
arm because `_ruc_tanh_array` ran once per snow-covered column *in the
dispatch*.  Noah-MP has no such term: its snow branches are already inside
WATER, PHASECHANGE and TSNOSOI, all of which are device leaves, so the host
residue is the same composition either way.  There is no second dominant term
here to go and find.  There is one, and it is the composition.

## The correction: the floor is not 0.079 ms/column

The previous section computed a ceiling by taking the measured share of "the
eight device batches" -- 22.5% -- and asking what remains when everything else
goes.  It got 0.079 ms/column, about 17 minutes per simulated minute, and
called that "the first genuinely different regime".

**That 22.5% is not the kernels.**  It is a Python list comprehension per
field per column to build each batch's flat rows, and a per-column unpack of
the results afterwards.  Measured directly on one batch: the four SFLX-prefix
kernels over 320 columns, ten timed iterations, paired in one process, cost
**10.43 ms** through the per-column packer and **1.13 ms** through an
array-native one -- and the module docstring for the remaining 1.13 ms names
one guard synchronisation and the bundle construction, not the kernels.

So the same mistake this document has now recorded four times -- naming a
remainder confidently and wrongly -- was in its own ceiling analysis.  The
floor is not 0.079 ms/column.  It is whatever the kernels cost once no Python
runs per column, and the only honest thing to say about it today is that one
batch of four kernels fell 9.2x when its packing was removed.

That also settles the sequencing question the previous section left open.  A
conversion that moves ENERGY's 48% but keeps the per-column seam buys at most
about 2x and lands at hours; the per-column packing it leaves behind grows
with every segment added to the seam.  **The only conversion worth doing is
the one that removes per-column Python entirely**, and every piece of it has
to exist before any of it pays.

## A divergence found before any of the conversion was written

The first thing the new composition needed was glibc's `expf` over an array,
and its very first gate failed at one argument out of 16,390: `expf(-88.0)`.

`__double2float_rn` **flushes a subnormal result to zero on this toolchain**.
Measured on sm_120 with CUDA 13.0, and `--ftz=false` does not change it,
because CuPy appends `-ftz=true` after the caller's options and the compiler
honours the last occurrence.  The conversion instruction has no flush of its
own; under `-ftz=true` the compiler emits an extra multiply after it to
produce one, and without the append the same conversion keeps the subnormal.
glibc 2.39's `expf` and `powf` do return subnormals; gfortran at
`-O0` leaves MXCSR's FTZ and DAZ clear; and `gpuwm.core.noahmp_libm`, verified
against the live glibc over 1,106,247,680 inputs, returns them too.

So `r_exp` and `r_pow` -- the tree's one audited device transcription --
disagreed with the CPython authority across **the whole band where expf
underflows into the subnormals**, 136,357 FP32 arguments in
[-103.616, -87.337), returning `+0.0` for every one.  No fixture reaches that
band, which is why every leaf oracle stayed green.  ENERGY's RHSUR at
`module_sf_noahmplsm.F:2203` evaluates `exp(PSI*GRAV/(RW*TG))` on very dry
soil and can.

`nmp_d2f_rn` recovers the correctly rounded value exactly rather than
approximating it: a binary32 subnormal with mantissa bits *m* is exactly
*m* x 2^-149, scaling by 2^149 is exact in binary64 across this band, and
`rint` rounds to nearest with ties to even, which is the rounding the
conversion owes.  The carry into the smallest normal needs no special case,
because `0x00800000` is already that number.

It went into `noahmp_leaves.cu` **and** `noahmp_fluxprep.cu` in the same edit,
so their byte-identical copies stay one group in the duplication inventory --
fixing one and not the other is exactly how `r_pow` would have acquired a
ninth generation.  The inventory grows by one identical-copy row; the drifted
count is unchanged at 16.  The whole band is now bitwise, and a live control
keeps `__double2float_rn` under observation so the helper retires itself the
day a toolchain stops needing it.

**Seven other kernel files still have the same exposure** -- `bareflux`,
`radiation`, `snow`, `soilwater`, `vegprecip`, `water` and `energy` each carry
their own `glibc_expf`/`glibc_powf` with an unguarded `__double2float_rn`.
Fixing those is the same three-line edit each, but each one is a separate
consumer that has to be re-gated against its own oracle, and doing seven of
them silently in one night is not how this tree has stayed bitwise.  Recorded
here, not fixed.

## What moved

Every piece below is **new code that nothing calls yet**.  It is committed
because it is gated, and because an API drop killed a lane mid-task on this
project; it does not change any forecast until the orchestration lands.

**ENERGY's whole composition, over the column axis.**
`gpuwm/core/noahmp_energy_slab.py` is the same statements as
`gpuwm/core/noahmp_energy.py` with the array spelling of each operation, in
the same order, split into four segments because the batched leaves interrupt
it.  The scalar module remains the authority.

Where the gate is taken is the point.  Everything seg1 and seg2 compute is an
argument of VEGE_FLUX or BARE_FLUX, so the gate records what the scalar
generator actually yields on the nine unmodified-WRF fixture columns and
compares the slab's arguments word for word -- **153 words, max_ulp 0** --
with no second reference implementation and no intermediate that nobody reads.
The argument names come from `noahmp_vegeflux_gpu.CALL_NAMES`, so reordering
that positional list moves the comparison with it instead of silently checking
the wrong slot.  seg3 is gated on all 26 `EnergyState` fields it writes (234
words) and seg4 on 288, with the counts computed from the layer geometry
rather than bounded, so a shrinking comparison fails.

**Four of the eight leaf batches now take arrays instead of per-column call
lists**: WATER, the SFLX prefix, VEGE_FLUX and BARE_FLUX.  No arithmetic was
transcribed for any of them -- the same kernels run in the same order with the
same slot layouts, imported from the existing modules rather than re-typed, so
a slot that moves moves in both packers at once.  Their gates are byte-exact
against the per-column packers, which is a comparison that needs no physics
argument at all: 36 columns and three packed blocks for WATER plus all 40
returned fields; 64 columns and 104 fields, 6,656 comparisons, for the prefix;
all six device arguments and all 43 outputs for the two flux leaves.

**`gpuwm/core/noahmp_slab_libm.py`** states, in one place, which array
operation reproduces which scalar one bitwise, and refuses to guess.  `+ - * /`
and `sqrt` are IEEE-754 correctly rounded, so a CuPy float32 ufunc *is*
`f32(a op b)`; each ufunc is its own launch, so nothing can be contracted into
an FMA either.  `min`/`max` are **not** `cupy.minimum`/`maximum`: the scalar
spelling returns the second argument on a tie and the CuPy builtins do not.
The transcendentals go through one bare elementwise wrapper around the tree's
single audited transcription; the entry points are spelled `slab_powf` and not
`powf` because `test_logf_has_not_forked_between_the_two_libm_modules` treats
a module that defines `powf` as a fork, and it is right to.

## The controls, including three that were wrong first

Every gate above was shown failing before it counted.  Three of the failing
forms are worth recording because they *looked* like controls and were not:

1. **Transposing ELAI and ESAI is unobservable.**  The first spelling of the
   ENERGY transposition control swapped them and passed -- correctly.  ENERGY
   reads that pair only as `ELAI + ESAI`, at :2062 for VAI and at :2140 for
   EMV, and FP32 addition is commutative.  The control now transposes TV and
   TG, which :2211-2223 and :2203 read separately, and the ELAI/ESAI case is
   kept as a **second test requiring it not to move the answer**, so the
   explanation is measured rather than asserted.  This is the same shape as
   RUC's water-column nudge.
2. **The VEGE_FLUX oracle deck is one vegetation type.**  All fourteen `P_*`
   values are constant across every case in `noahmp-vegeflux.csv`, so a
   parameter slab built from the deck alone is uniform and a transposition
   inside it is a silent no-op -- the gate would have looked fine while
   testing nothing.  Three vegetated MODIS classes from
   `noahmp-parameters.csv`, same WRF oracle provenance, are cycled alongside
   it so HVT/VCMX25/MP differ per column.  That is what makes the parameter
   control possible at all.
3. **The strided-pointer trap only bites VEGE_FLUX** of the two flux leaves;
   every BARE_FLUX field is copied into a freshly allocated slab first, so a
   stride cannot leak there.  The test says so rather than implying both
   halves are load-bearing.  Where it *can* bite, it was reproduced
   deliberately: neutering `ascontiguousarray` in the SFLX prefix moves 685
   comparisons with exactly the PAHV/PAHG/PAHB/CANLIQ signature the existing
   module's comment records, and in VEGE_FLUX it fails at 36 of 40 columns on
   TG alone.

Also observed firing: rolling one column's inputs onto its neighbour (in every
gate), rolling every column's BARE_FLUX answer onto its neighbour (the defect
a batched seam actually produces), transposing adjacent packed slots, a
reversed layer axis, a one-layer origin shift, and dropping BTRAN's upper
clamp.

## What remains, in the order it has to happen

1. **THERMOPROP, RADIATION, TSNOSOI and PHASECHANGE in slab form.**  The same
   mechanical byte-exact conversion as the four that landed.  Until all eight
   are done the column cannot be driven without falling back to the
   per-column seam for one leaf, which reinstates the per-column packing for
   every column.
2. **`sflx_post` over the slab.**  About eighty lines of arithmetic --
   SICE, QVAP/QDEW/EDIR, the urban QSFC leg, the snow-epsilon reset, ALBEDO --
   plus ERROR, which already has an oracle-matched kernel (`k_sflx_error`) and
   is therefore a packing job.
3. **The orchestration.**  A module that gathers the land columns once,
   launches seg1, the eight batches and seg2-4 in order over device-resident
   arrays, and writes back with advanced indexing.  There is no data-dependent
   control flow left at the slab level -- every branch is inside a kernel or a
   `where` -- so this is a straight-line launch sequence, not a scheduler.
4. **The driver's marshalling.**  `noahmp_lsm_step` copies its whole field set
   to host NumPy, loops over land columns building per-column keyword dicts,
   and copies back.  The prologue is already vectorised over the slab; what
   remains is to stop indexing it `[j, i]`.
5. **Then, and only then, a paired measurement in the user's unit** -- one
   script, both trees, back to back -- and the six-step digests re-derived on
   both sides.

Steps 1 and 2 are gated the same way the four that landed were, and neither
needs a decision.  Step 3 is where the remaining risk is, and it is the
per-column-values-into-scalar-kernel-arguments class this document has warned
about for four sections.

## What this does not claim

It does not claim a speedup.  Nothing here is on a forecast's path: the eight
per-column batches are still what `noahmp_runtime` calls, the six-step digests
are unchanged for exactly that reason, and the only timing quoted is one
batch's packing measured in isolation.  The registry template,
`GPUWM_NOAHMP_EXPERT_COLUMN_BUDGET` and the maturity grade are untouched, and
the recommendation is unchanged: **expert-only stays and the column budget
stays**, because 4,982--5,977 wall seconds per simulated minute is not a
forecast.

## VRAM and machine hygiene

Whole-machine `nvidia-smi` stayed between 2,396 and 2,552 MiB for the whole
session, of which about 2,400 MiB is the user's desktop; the CuPy pool peaked
at 5.1 MiB in the width harness.  That is under 9% of the 29,500 MiB rail.
The GPU lock was taken once and held for the session.  **No process was
killed, stopped or suspended, and no file this lane did not create was
deleted.**  The four uncommitted files belonging to the other lane in this
worktree were not touched, and every commit used an explicit pathspec.

## Defects found outside this lane's scope, not fixed

* The seven kernel files whose `glibc_expf`/`glibc_powf` still flush the
  subnormals, listed above.
* `noahmp_driver_gpu`, `noahmp_energy_gpu`, `noahmp_fluxprep_gpu` and
  `noahmp_leaves_gpu` have zero importers under `gpuwm/`, and `preflight.py`
  compiles three never-launched translation units on every Noah-MP forecast.
* `noahmp_runtime.py:33-34` and `:209-210` say seven batched leaves; there are
  eight.
* `STATE_SLOT` and `STATE_WIDTH` in `noahmp_water_gpu.py` are two
  hand-maintained dicts with nothing asserting that they tile `NSTATE = 37`
  without overlap or gap.  They do today.
* `noahmp_water_gpu`'s output block interleaves vectors and scalars -- `smc`
  at 0-3, scalars at 4-9, `acc_etrani` at 10-13, scalars at 14-35 -- so a
  reader who assumes "all vectors, then all scalars" transposes six slots.

---

## Addendum: what the converted leaves cost at d04's real width

The section above declines to quote a speedup because nothing is wired.  That
stands.  But the eight leaf batches are ordinary functions, so five of them can
be run **at 360,000 columns** on inputs built by tiling the same fixture
columns their own bitwise gates use -- which is a measurement of the converted
path at the width the question is about, not an extrapolation from 352.

Best of three, then of five, then of three again, three separate runs:

| leaf | 360,000 columns |
|---|---:|
| THERMOPROP | 0.62--0.79 ms |
| TSNOSOI | 0.40--0.47 ms |
| PHASECHANGE | 0.79--0.81 ms |
| RADIATION | 0.72--0.83 ms |
| WATER | 2.31--2.51 ms |
| **five of the eight leaves, one call** | **4.9--5.4 ms** |

**0.18--0.19 wall seconds per simulated minute** for those five leaves at d04,
against 4,982--5,977 for the whole call as it runs today.  Peak CuPy pool
152.4 MiB; whole-machine VRAM never left the 2,372--2,552 MiB band.

And paired, both paths in one process at 4,096 columns -- the largest width
the per-column path finishes quickly -- on the same columns:

| leaf | per-column batch | slab batch | |
|---|---:|---:|---:|
| THERMOPROP | 10.61 ms | 0.31 ms | 34.6x |
| TSNOSOI | 5.59 ms | 0.15 ms | 37.0x |
| PHASECHANGE | 16.60 ms | 0.31 ms | 53.8x |
| RADIATION | 32.04 ms | 0.61 ms | 52.4x |
| **four leaves** | **64.8 ms** | **1.38 ms** | **47x** |

Two things follow, and the second is the one that matters.

**The kernels were never the cost.**  The same CUDA runs on both sides of that
table.  47x is what removing the Python that packs them is worth, and it is
the direct measurement the previous section's ceiling analysis needed and did
not have.

**The survivable regime is real.**  Five of the eight leaves at full d04 width
cost about five milliseconds.  Whatever the remaining three leaves, the
composition and the driver add, they are being added to 5 ms and not to 138
seconds, and the composition is about fifty CuPy launches over the same
arrays.  This does not license a number -- the orchestration is exactly where
this lane's remaining risk is, and a number before it exists would be the
third unreproducible figure this document has had to withdraw.  It does say
that the regime RUC reached at 2.36 s is the one Noah-MP is heading for, and
that the work left is assembly rather than discovery.

## Addendum: a number in this section was wrong, and the rest of the fix

**The subnormal band is 2,133,824 FP32 arguments, not 136,357.**  The section
above says 136,357 twice and so does commit `f5f61b2`.  The whole interval
[-103.616, -87.337) sits inside the binade [64, 128), which holds 2^23 floats,
and 16.279/64 of that is 2.13 million.  The gate always swept the whole band
-- it walks the bit ladder between the two endpoints -- so the code was right
and the figure printed beside it was one this lane asserted instead of reading.
Corrected here rather than edited away, because a report that silently repairs
its own numbers is worth less than one that shows them being repaired.

**And the seven files listed as found-not-fixed are now six files fixed and
one refused.**  `bareflux`, `radiation`, `snow`, `soilwater`, `vegprecip` and
`water` took `nmp_d2f_rn` verbatim at ten conversion sites.  The duplication
inventory shows exactly one row moving -- `nmp_d2f_rn` from two files to eight
-- and it stays a **single group**; `len(DRIFTED)` is untouched at 16, which
is the assertion that says no body split.  `noahmp_energy.cu` needed nothing:
it borrows `r_exp`/`r_pow` from `noahmp_leaves.cu`.  The two `glibc_logf`
sites were left alone because `|log(x)| < 2^-126` requires `|x-1| < 2^-126`
and glibc special-cases `x == 1` before the conversion.

`tests/test_noahmp_kernel_subnormals.py` drives all 2,133,824 expf arguments
and 3,545,500 powf pairs straddling the underflow edge through each unit's own
libm.  **Before the edit every unit differed on 100% of the band**, returning
`+0.0`; after, zero.  Each sweep carries a control that reconstructs the defect
by substituting `__double2float_rn` back into the same source text, so "it was
broken before" is measured on every run rather than remembered.

Two of the six ship no probe that can reach the band -- `bareflux`'s raises to
the 1/4 power, whose result cannot be subnormal, and `vegprecip` exports only
PHENOLOGY and PRECIP_HEAT.  For those the sweep appends an entry point to the
shipped source at compile time, and a test pins that distinction so the file
cannot be read as claiming shipped coverage.

**`noahmp_vegeflux.cu` is refused, not overlooked.**  It has the same defect
spelled as a plain `(float)` cast, measured on the card: `glibc_expf` returns
`+0.0` at -88, -90, -95, -100 and -103.  It is the tree's oldest libm
generation -- literal float constants where the newer files use `__constant__`
tables, and an overflow threshold one ULP from theirs -- and this document's
own duplication argument is that it should be **retired onto** `r_exp`/`r_pow`
rather than gain a tenth copy of the helper.  A live test asserts it still
flushes, so it fails the day somebody fixes it properly.

## Addendum: the composition is complete

`NOAHMP_SFLX`'s second half is converted too, so **every piece of the Noah-MP
column except the orchestration and the driver's host round trip now exists in
slab form and is gated bitwise**: all eight leaf batches, ENERGY's four
composition segments, and `sflx_post`'s two.

`sflx_post` does not transcribe ERROR.  `k_sflx_error` is already the slab
form -- one thread per column, held to `noahmp-sflx-error.csv` at max_ulp 0 on
all sixteen cases -- so it is launched, and its 34-slot packing is parsed off
the `.cu` rather than restated, so a slot the kernel declares and the launch
does not fill raises by name.  32 fields, 228 words, max_ulp 0.  Two of the 32
are `ERRSW` and `ERRENG`, which `SflxResult` does not carry: they are captured
by wrapping `error` for the scalar drive, because a port that computed the two
residuals wrongly and never showed them would pass every whole-column check in
this tree.

Two more vacuity findings, both kept as live measurements:

* **The urban override at :1061-1064 is unreachable from the fixture.**  No
  fixture column is urban, so a slab that dropped it entirely would pass.  A
  test now flips `urban_flag` after the prefix on both sides and requires QSFC
  to move.
* **The dew leg of :983 is never exercised** -- every fixture column has
  FGEV > 0, so QDEW is identically zero and `fmn` is indistinguishable from
  `cupy.minimum` there.  Recorded so a future dewing column makes it fail.

What is left is item 3 and item 4 of the list above: the straight-line launch
sequence, and stopping the driver from indexing its slab `[j, i]`.  There is
no data-dependent control flow remaining at the slab level -- every branch is
inside a kernel or a `where` -- so the orchestration is assembly, and the
risk in it is the one this document has named for five sections: a per-column
value reaching a scalar kernel argument.

---

# Sixth conversion: the orchestration lands, and the forecast takes it

*Drew's local RTX 5090, Windows, CUDA 13.0, CuPy 14.1.1.  As every section
since the second has warned, absolute seconds on this box move by up to 30%
between harnesses and hours; every comparison below is paired, in one
process, minutes apart, and the timing was run twice end to end.*

## Result, in the user's unit

The previous section left the conversion "one wiring step short": every piece
of the column existed in slab form, gated bitwise, and nothing called any of
it.  The wiring now exists.  `gpuwm/core/noahmp_column_slab.py` assembles the
eight leaf batches, ENERGY's four segments and `sflx_post`'s two into one
straight-line launch sequence with no Python per column, and
`noahmp_lsm_step` dispatches to it on every forecast that binds the shipped
device evaluator.

Measured at 360,000 land columns -- a 600x600 all-land nest, tiled from the
8x6 forecast-test domain after one real coupled step, `dt = 5/3 s`,
`bldt = 0`, 36 calls per simulated minute -- **twice, in two separate
processes**:

| | run 1 | run 2 |
|---|---:|---:|
| slab path, s per land-surface call (3 calls each) | 0.217 / 0.222 / 0.227 | 0.202 / 0.202 / 0.203 |
| **wall s per simulated minute** | **7.8 -- 8.2** | **7.3 -- 7.3** |
| staged path, same device leaves, same process, one call | 205.8 s | 166.4 s |
| staged wall s per simulated minute | 7,408 | 5,989 |

**Noah-MP at d04 is now 7.3 -- 8.2 wall seconds per simulated minute**, from
the 4,982 -- 5,977 this document's fifth section published for the same
regime (the paired staged arm lands at 5,989 -- 7,408 on the same box today,
which is that published range plus this machine's documented hour-to-hour
spread).  The paired speedup is 820x -- 950x.  Nothing about `bldt` moved:
this is the every-step cost, at the every-step cadence.

For calibration only, not as a claim: a 15-minute forecast's land-surface
cost goes from about 21 hours to about two minutes.

## Measured ULP

**0**, everywhere it was checked, and "everywhere" now includes the whole
call at the nest width:

* **12 heterogeneous columns against the scalar authority.**  The four
  unmodified-WRF whole-column fixture cases crossed with wind and
  temperature variants -- vegetated and bare, day and night, 45 mm snowpack
  and none, frozen and unfrozen soil in one slab -- driven through `sflx()`
  per column on the host and through `evaluate_sflx_slab` as one call: all
  70 output fields, 1,176 words, zero differing bits, including ERRSW and
  ERRENG captured by wrapping `error()`.
* **A full 65,536-column chunk**, every column bitwise against its source
  column's scalar authority -- the width the runtime actually launches.
* **360,000 columns end to end**: one slab call against one staged call
  with the same device leaves on identical inputs, all 83 written carrier
  fields byte-identical.  Twice, once per timing run.
* **Six steps of the whole forecast**: the slab path against the staged
  host authority reproduces the carried-state digest gate unchanged, and a
  new gate holds slab against staged-with-device-leaves over all 71 written
  fields -- wider than the 45-array digest exactly where the write-back
  branches are.
* `SLAB_COLUMN_CHUNK` at 1, 7 and 65,536 is inert, field by field, and the
  misroute control (every output lane rolled onto its neighbour) is
  observed firing.

## The wiring is proved, not narrated

The RUC lane shipped verified device code that nothing called, twice.  The
booby traps now cover both directions:

* a default forecast runs with `_drive_staged_columns` replaced by a raising
  tripwire and **survives** -- the staged path is provably not what ran;
* the same trap **fires** when `GPUWM_NOAHMP_STAGED_COLUMNS=1` selects the
  staged path by name;
* a third trap replaces `evaluate_sflx_slab` itself and the default forecast
  **fails** -- the orchestration is what answered the columns, not some
  third path;
* the original host-leaf tripwires pass unchanged: with every CPython leaf
  made impossible a forecast still finishes, and withholding the device set
  still trips them.

## What the memory costs, and where it is priced

`preflight.py` gains its first Noah-MP transient terms, each reading the
runtime's own constants so the price and the bound cannot drift apart:

| term | shape | measured | priced |
|---|---|---:|---:|
| `noahmp_lsm/slab_chunk_transients` | `SLAB_COLUMN_CHUNK x 4096 B` | 2,751 B/col at exactly 65,536 columns | 256 MiB |
| `noahmp_lsm/slab_grid_transients` | `nx*ny x 512 B` | 407 B/grid-col at 360,000 | 175.8 MiB |
| `noahmp_lsm/staged_leaf_batches` | `COLUMN_BATCH x 620 B` | the inventory's own derivation | 1.21 MiB |

433 MiB priced against 311.8 MiB measured for the whole d04 call.  The GPU
gate re-measures the chunk term on every run and holds the ceiling to within
2x in both directions.  Whole-machine VRAM peaked at **6,432 MiB** during
the measurement runs, desktop included -- 22% of the 29,500 MiB rail.

`noahmp_libm_slab` also enters `UNMEASURED_KERNEL_MODULES`: it is a fragment
like `noahmp_driver`/`energy`/`thermal`, and `f5f61b2` had left the
local-frame regeneration test red on the card since it landed.

## What this section does not claim

* **No registry string moved and the column budget did not move.**  The
  `land_surface.noah-mp` scaling warning, the expert acknowledgement's
  "7.18 ms per land column", and `GPUWM_NOAHMP_EXPERT_COLUMN_BUDGET`'s
  352-column default now describe a cost that is three orders of magnitude
  stale in the pessimistic direction.  Whether Noah-MP graduates from
  expert-only is a lead decision with its own validation ladder (restart,
  multi-domain, a real case end to end), not a drive-by in a wiring commit.
  What this lane owed was the honest number beside the stale one, and
  `NOAHMP_RUNTIME_RESTRICTIONS["column_solver_location"]` -- which is this
  module's own string -- now carries it.
* The `restart_identity` string still says
  `host-fp32+device-vegeflux-v1`.  It was already one conversion stale and
  the answers are bit-identical, so changing it would invalidate restart
  compatibility over a name.  Recorded, not changed.
* The 7.3 -- 8.2 s figure is the land-surface call alone, measured through
  `noahmp_lsm_step`; it is not a whole-forecast wall clock, which carries
  the dycore and the rest of physics on top.

## 2026-07-27: the restriction surfaces are aligned

The lead authorized retiring the host-era speed warnings the previous
section left standing beside the honest number.  What changed, and what
deliberately did not:

* `gpuwm/physics_compat.py`: `NOAHMP_MEASURED_COLUMN_CEILING` 352 -> 360,000
  (the width the slab path was timed at), `NOAHMP_MEASURED_MS_PER_COLUMN`
  (7.18, host era) replaced by `NOAHMP_MEASURED_SLAB_CALL_SECONDS =
  (0.202, 0.227)`, and the column-budget refusal now quotes the slab
  measurement and a linear projection to the requested width.  The rail
  itself stays: it is still "the largest measured configuration and
  nothing wider", and `GPUWM_NOAHMP_EXPERT_COLUMN_BUDGET` still widens it
  explicitly.
* `gpuwm/physics_registry_v2.json`: the `land_surface.noah-mp` scaling
  warning, the route's expert warnings and the expert template note now
  state the slab regime (0.202--0.227 s per 360,000-column call, 7.3--8.2
  wall s per simulated minute, measured 2026-07-27, twice, one RTX 5090)
  and say the throughput blocker is retired.
* The expert route and template REMAIN expert-only, and
  `expert_acknowledgement_id` keeps its host-era name
  (`noahmp-host-column-throughput-v1`): renaming it would invalidate
  consent records over a label, the same reasoning that left
  `restart_identity` alone.  Graduation to a peer profile still has its
  own validation ladder and is not part of this alignment.
* `tests/test_health_field_census.py`: the census was re-run rather than
  predicted -- 290 measured rows, 670 refusals, peak 601 at root 229 on
  the four-way `lsm4` tie, exactly what the
  `GPUWM_NOAHMP_EXPERT_COLUMN_BUDGET=360000` control had measured -- and
  the pinned constants moved with it.
* `restart_identity` is untouched.
