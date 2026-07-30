# Noah-MP runtime admission — what runs, and what it cost to find out

`sf_surface_physics=4` runs a forecast end to end. This document records the
admission work, the enforced option identity, every published restriction, and
— the part that could not have been derived from any isolated-routine gate —
what running the scheme measured.

Measured against pinned WRF v4.6.1 commit
`d66e442fccc04111067e29274c9f9eaccc3cef28`. Reference hardware: one rented
RTX 5090, CUDA 12.x, CuPy 14.1.1, glibc 2.39.

`docs/mynn_noahmp_ruc_completion_plan.md` still says "Noah-MP: nothing of the
solver is ported". That line was true when it was written and is now three
lanes stale; the column solver, ENERGY, WATER, the snow chain, `NOAHMP_SFLX`
and the driver cold start all landed before this lane started.

## The admission checklist, one gate at a time

Every item is the Noah-MP counterpart of what `d00ee21`/`e65b3ce` did for
MYNN, in the same order:

1. **Dispatch.** `PHYSICS_SLOT_DISPATCH["sf_surface_physics"]` gains
   `4 -> _run_noahmp` (`gpuwm/core/physics.py`). RUC's `3` stays absent: its
   two top-level routines still do not exist, so routing it would run Noah
   under a RUC selector.
2. **The driver, not the column.** `module_surface_driver.F:3127` does not call
   `NOAHMP_SFLX`; it calls `noahmplsm` (`module_sf_noahmpdrv.F:12-1432`), and
   that driver owns the `ITIMESTEP == 1` ocean/sea-ice soil initialisation, the
   two `CYCLE ILOOP` skips, `Q_ML = QV3D/(1+QV3D)`, `P_ML` from the two lowest
   interfaces, `Z_ML = 0.5*DZ8W(1)`, the soil-type water override, the urban
   vegetation remap, the `FICEOLD` reconstruction, the per-column
   `TRANSFER_MP_PARAMETERS`, and the write-back where `TSK = TRAD`,
   `CANWAT = CANLIQ + CANICE`, `Q2MV = Q2MV/(1-Q2MV)` and the SOILENERGY /
   SNOWENERGY integrals live. `gpuwm/core/noahmp_runtime.py` is that driver,
   with the anchors next to the arithmetic. Losing the `Q_ML` conversion is a
   silent O(qv) bias in every flux; losing the open-water skip runs a land
   model on the sea.
3. **Allocation.** `initialize_physics` allocates 45 extra 2-D arrays and four
   snow-stack arrays **only** for this selector, so a Noah run's restart
   inventory, VRAM budget and health-descriptor count are unchanged.
4. **Cold start.** `NOAHMP_INIT` + `SNOW_INIT` run once at driver
   construction, where `module_physics_init.F` runs them. This is not
   optional: zero is the WRF Registry cold state for `TVXY`/`TGXY`, and
   `TV = TG = 0 K` walks straight into a negative saturation vapour pressure.
5. **Preflight.** `physics_field_names_2d` and `physics_array_shapes` count all
   49 additions. Under-counting VRAM is a correctness bar on this hardware.
6. **Output.** `gpuwm/io/wrfout.py` lists every emitted name explicitly, gated
   on the driver's resolved routing rather than on a field's existence, and
   the snow stack gets its own vertical axes — `snow_layers` (3) and
   `snso_layers` (3 + n_soil), WRF's own dimension names from
   `Registry/registry.dimspec:53,55`. They are created lazily, so a run
   without Noah-MP keeps byte-identical file headers.
7. **Restart.** `LAND_SURFACE_PARAMETER_SOURCES` gains
   `4: ("noahmp_params", (mptable, soilparm, genparm))`, and the three table
   paths join `PHYSICS_ASSET_PATHS`. The solar geometry is bound into the same
   header digest — see the restriction table below for why that matters.
8. **Readiness.** `pending_wrf_physics_components` no longer refuses Noah-MP.
   It refuses two narrower things: a soil geometry that is not four layers, and
   Noah-MP paired with the MYNN surface layer.
9. **Registry.** `land_surface.noah-mp` is `implemented: true` at maturity
   `implemented-unverified`, and the 24 `NoahMP_OPTIONS` knobs became real
   `RunConfig` fields with `validate_run_config` refusing anything outside the
   admitted identity before a run starts.

`registry_sha256` is now
`c3abb1b23a11a503f64088f70d3c4020f5556c250d1c4dc5ea407955bd5a56df`;
the admitting lane recorded
`398dd150b232fec029e76c3ac43abe366d7facb139595ae344a72444ac55ade7`
and this lane's rewrite of the scaling warning moved it
(`gpuwm.physics_registry.registry_sha256` over the tracked file). The MYNN
lane's flip recorded
`83c67821b2c7ae393ac80b2e99ab25f685b7ea71df537c2a3c3fe8fd8432cf26`; its later
snow-species commit moved the value again before this one did, so that figure
is two commits stale rather than one.

## The option identity, and what "validated" means for each value

`gpuwm.config.NOAHMP_OPTION_IDENTITY_EVIDENCE` is the authority; it carries the
evidence string in the refusal message, so a user who asks for `opt_run=1` is
told at configuration time *why* it is refused. Three grades appear, and the
distinction is deliberate:

| grade | meaning |
|---|---|
| **fixture** | the four whole-column `noahmp-sflx.csv` cases were generated at this value and the port is bitwise against them |
| **dead-proved** | the value kills a code region, and the kill is proved out of the source in `gpuwm/core/noahmp_sflx.py`'s module docstring rather than assumed |
| **declared only** | the value has no live consumer at all; it is pinned so a plan asking for another value is refused rather than silently ignored |
| **unmeasured** | no other value of this knob has been tried. Not approximated — not measured. |

`dveg=4 opt_crs=1 opt_btr=1 opt_run=3 opt_sfc=1 opt_frz=1 opt_inf=1 opt_rad=3
opt_alb=2 opt_snf=1 opt_tbot=2 opt_stc=1 opt_gla=1 opt_rsf=1 opt_soil=1
opt_pedo=1 opt_crop=0 opt_irr=0 opt_irrm=0 opt_infdv=0 opt_tdrn=0
soiltstep=0.0 noahmp_output=1 noahmp_acc_dt=0.0`, with `num_soil_layers=4` and
`sf_urban_physics=0`.

`opt_gla=1` is the honest awkward one. It is *declared only*, and it is not
evidence that any glacier physics runs: a column whose `VEGTYP` equals
`ISICE_TABLE` **raises**. It exists so a plan asking for `opt_gla=2` is refused
instead of silently ignored.

## Restrictions the identity schema has no field for

MYNN shipped an undisclosed `FLAG_QS` restriction. These are the Noah-MP
equivalents, and all ten are published three times: in
`gpuwm.core.noahmp_runtime.NOAHMP_RUNTIME_RESTRICTIONS` as data, in the
registry option's `warnings` for a user's picker, and here.

| restriction | what gpuwm does | why it differs |
|---|---|---|
| precipitation partition | `PRCPNONC = RAINBL/DT`, `PRCPSNOW = SR*RAINBL/DT`, the other four rates zero | `module_sf_noahmpdrv.F:776-796` picks on `PRESENT(MP_RAINC..MP_HAIL)`; WRF's own surface driver passes all six (`module_surface_driver.F:3180-3181`), so WRF takes the first branch and gpuwm takes the second. gpuwm's LSM seam carries only RAINBL and SR, exactly as its Noah path does. Under `opt_snf=1` FPICE comes from SFCTMP, so the visible consequence is in ATM's convective/large-scale split feeding canopy interception, not in the rain/snow phase. |
| COSZ evaluation time | evaluated at the surface-call time, no offset | WRF hands `noahmplsm` the COSZEN the radiation driver last wrote, which carries radconst's half-radiation-interval offset and is stale between radiation calls. Bounded by `radt/2 + (steps since the last radiation call)*dt`. |
| glacier columns | **raise** | `module_sf_noahmp_glacier.F` (3,080 lines) is not ported and `NOAHMP_SFLX` is not a substitute. |
| sea-ice columns | WRF's skip (`SH2O=1`, `XLAI=0.01`) | This is what WRF does. Listed because it means a Noah-MP run has **no sea-ice surface energy balance**: TSK/HFX/QFX over sea ice keep whatever the surface layer left. |
| `XICE_THRESHOLD` | pinned 0.5 | gpuwm has no configuration field for WRF's namelist value. |
| diagnostic inventory | 26 of ~120 NOAHMP_SFLX outputs stored | The rest are computed and discarded, not zero-filled. WRF's `module_diag_misc.F` accumulators have no gpuwm counterpart, so `noahmp_output`/`noahmp_acc_dt` change nothing. |
| column solver location | **the whole column runs on the device** (updated 2026-07-27; earlier revisions of this row recorded the hybrid leaf-batched and host eras): `noahmp_column_slab`'s orchestration answers every land column with no Python per column, in chunks of `SLAB_COLUMN_CHUNK` (65,536) | The assembled slab is bitwise (max ULP 0) against the scalar column: twelve heterogeneous fixture columns field by field, a full 65,536-column chunk, and 360,000 columns end to end. Measured 2026-07-27, twice, on one RTX 5090: 0.202--0.227 s per 360,000-land-column call through the slab path (7.3--8.2 wall seconds per simulated minute at dt=1.667 s, bldt=0) against 166--206 s through the per-column staged path in the same process. Absolute seconds are a property of the machine. The staged path remains the paired second implementation (`GPUWM_NOAHMP_STAGED_COLUMNS=1`). See `docs/noahmp_device_column_report.md`. |
| soil layers | four only | Every Noah-MP fixture in the tree is four-layer. |
| QSFC units | passed through unconverted | `module_sf_noahmpdrv.F:238` documents QSFC as specific humidity and `:808`/`:1245` pass it straight through, while WRF's surface layer fills the same Registry field with a mixing ratio. The inconsistency is WRF's, it is fully defined, and gpuwm reproduces it rather than inserting a conversion WRF does not have. |
| ACC_* accumulators | zero on entry to every call, not allocated | `soiltstep=0` makes `soil_update_steps=1`, and `:649-660` zeroes all ten before the column loop, so no cross-step accumulator exists. Admitting `soiltstep>0` means allocating them **and** carrying them through restart. |
| surface-layer pairing | MM5 only (1 or 91) | Noah-MP with the MYNN surface layer is refused by both the registry and `physics_compat`. MYNN diagnoses T2/Q2/TH2 itself and supplies UST/HFX/QFX, all of which Noah-MP also writes. Unmeasured. |

## Device `VEGE_FLUX`: exact, faster, and not yet a device column

The first device conversion is now in the runtime. Vegetated columns pause at
ENERGY's physical `VEGE_FLUX` call, all calls execute in one CUDA batch, and
the columns resume with the device results. A collected negative-control test
compiles the same source with CUDA's device libm and observes it disagree with
WRF; the shipped constant-table glibc transcription then matches all 300
physical VEGE_FLUX output values bitwise. The measured max ULP is **0**.

The staging seam is deliberately conservative: a vegetated column is restored
to its pristine entry state and replays the pre-VEGE ENERGY prefix with the
captured device result. That duplicates host work, but prevents a Python
continuation from carrying a partially mutated column across the batch. A host
tripwire proves the resumed execution does not call the CPython VEGE_FLUX.

Two independent width sweeps measured:

| land columns | forced host, ms/column | device VEGE_FLUX, ms/column |
|---:|---:|---:|
| 2 | 15.92--16.30 | 15.10--15.23 |
| 8 | 8.41--8.42 | 7.16--7.18 |
| 48 | 6.53--6.58 | 5.23--5.30 |
| 160 | 6.24--6.25 | 4.94--4.99 |
| 352 | 6.18 | 4.94--4.95 |

That is **1.25x at 352 columns**. The whole-column figure is still flat at
width: work really did reach the device, as the tripwire and batch oracle
show, but ENERGY outside VEGE_FLUX and the rest of SFLX still run once per
column in CPython. Noah-MP therefore remains unsuitable for a production
nest, and its registry template should remain expert-only.

The paired one-hour, 300-step, 130-land-column run measured 254.49 s forced
host versus 207.30 s with the device leaf, or 6.525 versus 5.315 ms per land
column (**1.23x**). All 45 carried arrays plus `u`, `v`, `w`, `thp`, `qv`, and
`mup` were bitwise equal at every 50-step checkpoint. Full hashes and test
provenance are in `docs/noahmp_device_column_report.md`.

## The host-column scaling blocker, profiled before the device leaf

Before the conversion above, there was no device Noah-MP column to call.
`noahmp_lsm_step` copied the surface slab to the host, looped columns through
the bitwise `gpuwm.core.noahmp_sflx.sflx`, and copied back. The measurements
below are retained because they chose VEGE_FLUX as the first leaf and bound
the remaining host work.

**The first version of this section named a cost and named a cause.** The cost
was measured; the cause was inferred. Here is the measurement, on an idle
reference box, of where the cost actually is.

### Is it flat in the column count?

Yes, and that is the whole diagnosis. One whole warmed RK3 step, no water
columns, three repetitions, at HEAD before this lane's rework:

| grid | land columns | step | ms per column |
|---|---:|---:|---:|
| 2x1 | 2 | 34.9 ms | 17.47 |
| 4x2 | 8 | 76.9 ms | 9.61 |
| 8x6 | 48 | 367.5 ms | 7.66 |
| 16x10 | 160 | 1169.9 ms | 7.31 |
| 22x16 | 352 | 2548.3 ms | 7.24 |

A two-point fit over the ends is **21 ms fixed per call plus 7.18 ms per land
column**. The per-column term does not fall as columns are added, so it is not
a launch cost, not a transfer cost and not an under-occupied kernel: it is
per-column host work. That is the same shape MYNN's `phim`/`phih` had. It also
means the three structural items the previous version of this section proposed
— merge the shims, make the leaves `__device__`, replace the flat packings —
are prerequisites for a device column and do not, by themselves, move a
microsecond.

### Inside one call

Instrumenting `sflx` and `cupy.asnumpy`/`asarray` separately, on the
160-land-column call:

| where | per call | share |
|---|---:|---:|
| `gpuwm.core.noahmp_sflx.sflx` | 1143.4 ms | 96.9% |
| both host transfer sets (312 transfers) | 4.1 ms | 0.3% |
| 2-D-to-1-D marshalling, write-back, whole rest of the dycore step | 32.7 ms | 2.8% |

The transfers are 4.1 ms **per call**, not per column: 0.026 ms per column at
160 columns. Fusing them would buy 0.4%.

### What inside `sflx`

`cProfile` over one 48-land-column call, by own time, top entries:

| function | calls | per column | own time |
|---|---:|---:|---:|
| `R4.__new__` | 337,728 | 7,036 | 14.7% |
| `f32` | 282,634 | 5,888 | 11.3% |
| `R4.__mul__` | 67,536 | 1,407 | 7.1% |
| `_struct.pack` | 328,330 | 6,840 | 6.8% |
| `_struct.unpack` | 328,330 | 6,840 | 5.6% |
| `float.__new__` | 183,696 | 3,827 | 3.4% |
| `R4.__add__` | 31,440 | 655 | 3.3% |

**56% of the column's own time is the binary32 arithmetic carrier**, not
physics. By cumulative time `ENERGY` is 84% of the call and `VEGE_FLUX` alone
is 63%. The glibc transcriptions (`logf`, `powf`, `expf`, `atanf` and their
bit-twiddling helpers) are about 11%.

### What that bought, and what it cannot buy

The carrier was flattened as far as CPython allows, in two steps, each held to a
bitwise-identical answer over a six-step run on both a bare and a snowpack
domain, over all four whole-column `noahmp-sflx.csv` fixtures, and over the
one-hour forecast below:

| | ms per land column, 352 columns |
|---|---:|
| HEAD | 7.24 |
| operators round the right operand in place, one `f32` per operation | 6.22 |
| `f32` inlined as `struct` pack/unpack, no call frame | 6.16 |

Five float64-to-binary32 carriers were measured against `struct`; two are
faster and neither is the same function (both saturate an overflow to an
infinity where `f32` raises), and the one bit-identical alternative reads a
shared module-level buffer, so it is wrong under concurrency. The table is in
`gpuwm.core.noahmp_libm.f32`'s docstring. What was taken instead was removing
the call frame, not the rounding.

### The one-hour forecast, before and after

The trajectory gate for the rework is the admitting lane's own run: 16x10x40,
dt 12 s, 300 RK3 steps (one hour of model time), 130 land columns, 30 water,
hashing the whole carried land state plus `u`, `v`, `w`, `theta`, `qv` and `mu`
every 50 steps.

| | wall clock | ms per land column | trajectory |
|---|---:|---:|---|
| HEAD | 299.2 s | 7.671 | `b0ab565b5688...` at step 300 |
| reworked | 250.9 s | 6.432 | `b0ab565b5688...` at step 300 |

**1.193x, and all six 50-step checkpoint digests are identical, not just the
final one.** Measured ULP over the carried state: 0. The four whole-column
`noahmp-sflx.csv` fixtures stay bitwise, and a separate six-step run over a
bare domain and a snowpack domain hashes to
`35badccff04bcbd4bac74adfd126c2e203ff11f799a4d5649657805cdbecfeaf` and
`23de48f07fc6d230ee3069c9935a5a2b8a329deacbf65e17a98dc4ac4e608a88` before and
after.

That was **1.18x on the scaling probe and 1.193x on the forecast**, and it is
where host-side optimisation ended. Even a free carrier would leave about 3.5 ms
per column, because the other 44% is the leaf bytecode itself. A
quarter-million-column nest is still about **27 minutes of wall clock per
land-surface call**. That result established that the remaining lever was
device execution; the first leaf conversion and current measurements are
reported above.
`tests/test_noahmp_runtime.py::test_the_column_cost_is_what_the_registry_says`
brackets the figure at 0.3-30 ms so the published number cannot quietly become
false.

## The nine device libm copies, and why "merge them" is not a refactor

The previous version of this document said composition was blocked by "three
copies of the device glibc shim" and that the fix was to merge them. Both
halves needed checking before anything was merged, so both were measured.

There are **nine** copies, not three. Normalising every `__device__` body for
names and whitespace and hashing it gives five distinct transcriptions of
`expf`, four of `logf`, five of `powf` and three of `atanf`. They differ in
construction, not only in spelling: `noahmp_radiation.cu` and
`noahmp_leaves.cu` both transcribe glibc's `expf`, but one uses `DFMA` against
`__constant__` bit patterns and the other separate `DAD`/`DMU` against
compile-time hex-float literals, and a fused multiply-add is one rounding where
a multiply and an add are two.

So each copy was scored against the pinned glibc 2.39 oracle rows
(`gpuwm/data/noahmp/oracle/glibc-libm-fp32.csv`, 30,000 rows;
`glibc-atanf-fp32.csv`, 4,256 rows) and then swept against the canonical copy
over **all 4,294,967,296 binary32 bit patterns** — every subnormal, both
zeros, every negative, both infinities and every NaN payload:

| role | copies swept | canonical | disagreements over the whole domain | max ULP |
|---|---:|---|---:|---:|
| `expf` | 6 | `noahmp_leaves.cu` `r_exp` | **0** | 0 |
| `logf` | 3 | `noahmp_leaves.cu` `r_log` | **0** | 0 |
| `atanf` | 1 | `noahmp_fluxprep.cu` `r_atan` | **0** | 0 |

Those ten copies are one function under ten names, proved exhaustively rather
than argued. Merging them is a no-op and needs no evidence beyond this table.

`powf` is different, and this is exactly why the merge had to be measured
first. Five of the seven copies deliberately refuse part of glibc's domain:
`noahmp_bareflux.cu`, `noahmp_radiation.cu`, `noahmp_soilwater.cu`,
`noahmp_vegprecip.cu` and `noahmp_water.cu` all return `0x7FC00000` for a
subnormal or negative base, with a comment saying so, while
`noahmp_leaves.cu`'s `r_pow` transcribes the whole thing. Measured, with the
canonical answer beside the guard:

| call | `noahmp_leaves.cu` `r_pow` | the five narrow copies |
|---|---|---|
| `powf(1.4013e-45, 1/3)` | `267fffef` = 8.8817752e-16 | `7fc00000` = NaN |
| `powf(1.4013e-45, 0.5)` | `1a000000` = 2.64697796e-23 | `7fc00000` = NaN |
| `powf(1.4013e-45, -3)` | `7f800000` = +Inf | `7fc00000` = NaN |
| `powf(-5.877e-39, 11)` | `80000000` = -0 | `7fc00000` = NaN |

All seven score 0 mismatches against the 12,000 oracle `powf` rows, because
none of those rows has a subnormal base — a gate that has never been
observed to fail on the case that separates two implementations is not evidence
about that case. The guards are a defensible choice, fail loudly outside the
audited domain rather than return unaudited arithmetic, but it means replacing
the five with `r_pow` **widens five audited domain guards**. That is a change
of behaviour, it needs its own evidence and its own droppable commit, and it is
not de-duplication.

`noahmp_vegeflux.cu` is excluded from the sweep and stays a tenth copy for now:
its libm reads `__constant__` tables the host uploads at module load, so a
probe that does not upload them measures zeros rather than the function. Its
CUDA surface now has a collected physical-argument gate in
`tests/test_noahmp_vegeflux_cuda.py`, including a device-libm negative control.
The standalone oracle tool remains useful for diagnosing the translation.

## What running it measured

None of this was visible from an oracle CSV.

**The census is part of the evidence.** `noahmp_lsm_step` returns how many
columns ran as land, were skipped as water, and were skipped as sea ice,
because "the LSM ran" and "the LSM ran on any land" are different claims and a
test has to be able to tell them apart.

**`TAUSSXY` is inert on a snow-free column, and that is WRF.** The first
version of the runtime test asserted that every carried array moves on land.
It failed on the snow age. `SNOW_AGE` sets `TAGE = 0` whenever `SNEQV <= 0` and
then `TAUSS = TAGE`, so a snow-free column's snow age is pinned at its cold
zero for the whole run. The assertion moved to the snowpack test, where a
snowpack exists for it to age — it is not a hole in the gate, it is a property
of the scheme.

**Four Registry restart carriers do not reach the next step.** Measured on the
runtime configuration, after a real restore:

| perturbation | fields that moved (of 20 watched) |
|---|---:|
| `TSLB(1)` + 1 ULP | 8 |
| `TVXY` + 1 ULP | 7 |
| `SH2O(1)` + 1 ULP | 2 |
| `TGXY` + 1 ULP | 2 (LH, QFX only) |
| `EAHXY` + 50 Pa | 13 |
| `CANLIQXY` + 0.5 mm | 15 |
| `TAHXY` + 5 K | **0** |
| `CHXY` x 1.5 | **0** |
| `CMXY` x 1.5 | **0** |
| `FWETXY` + 0.3 | **0** |
| `ALBOLDXY` + 0.1 | 1 (itself) |

`FWET` is unconditionally reassigned by `PRECIP_HEAT` before ENERGY sees it;
`TAH`, `CH` and `CM` are re-derived inside `VEGE_FLUX` and `SFCDIF1` by
iterations whose entry values are initial guesses. So a restart test that
perturbed only those four would prove nothing, and a reader who assumed "it is
Registry restart state, so it must matter" would draw the wrong conclusion
about what a checkpoint must preserve exactly.

This also chose the perturbation for the falsification test. A one-ULP `TGXY`
nudge propagates only into LH and QFX in one step, which is why the first
attempt at "prove the restart check can fail" proved nothing: it watched
TGXY/TVXY/HFX/TSLB/TSK, none of which moved. `TSLB` is the target now, and the
measurement is pinned in
`test_four_registry_carriers_do_not_reach_the_next_step` so it cannot drift
silently.

**The test's own thermal setup was the first "defect".** An early run asserted
a positive sensible heat flux over a 297 K ground and got -30 W/m2. The port
was right: the first model level sits near 300 K, so the flux is correctly
downward. The test now uses a 303 K ground under a 300 K boundary layer and
asserts something stronger than a sign — that `sign(HFX) == sign(TG - T1)` on
every land column, which ties the flux the driver wrote back to the state the
column solved for.

## The flaky memory-pool assertion, and what replaced it

`tests/test_noahmp_water_cuda.py` asserted
`cp.get_default_memory_pool().used_bytes() < 8 MiB` under the docstring "a
whole-column kernel that spilled to global memory would be a defect". Both
halves were wrong.

*Flaky*: `used_bytes()` is process-global and counts every live block in the
pool, so any array another test still holds is included. Measured: with a
single unrelated 16 MiB array alive, the value is 16,777,216 both before and
after the kernel runs — above the old bound. The delta attributable to the
kernel is 0, because the pool recycles the packing buffers.

*Wrong quantity*: spills do not go to the memory pool. They go to local memory,
reported per function as `local_size_bytes`. Measured for `k_water`:
`local_size_bytes` 224, `num_regs` 123, `shared_size_bytes` 0. The replacement
bounds the frame at 2 KiB and has a companion that recompiles the identical
source under `-maxrregcount=24` — which raises the frame to 656 bytes — so the
gate's sensitivity is demonstrated rather than asserted.

## What is still open

1. **No gpuwm/WRF forecast comparison exists for this scheme.** That is what
   `validation-candidate` would require and it is why the registry row is
   `implemented-unverified`. The whole-column fixture proves one call; a
   trajectory proves the coupling.
2. **Finish the device column.** `VEGE_FLUX` is complete: it is physically
   batched on the device, oracle-matched at max ULP 0, trajectory-matched, and
   worth 1.25x at width. The whole-column curve is still flat, so the remaining
   order is:
   1. Remove the conservative pre-VEGE replay with an explicit ENERGY
      continuation, then move `BARE_FLUX` (about 7% of the original call).
   2. Merge the ten proved-identical `expf`/`logf`/`atanf` copies into one
      header. Free, and the table above is the evidence.
   3. Settle `powf`: either preserve the five narrow domain guards in the
      merged copy, or widen them deliberately in a droppable commit with its
      own subnormal-base evidence.
   4. Give the leaves physical argument lists. The flat per-lane packings are
      fixture scaffolding, and they are what makes step 1 expensive to write.
   `TSNOSOI` and `PHASECHANGE` are already `__device__` cores with `__global__`
   fixture-packing wrappers over them, so that item is done.
3. **The six-way precipitation partition**, which needs the LSM seam to carry
   the per-interval convective/snow/graupel/hail split rather than RAINBL+SR.
4. **Glacier columns**, which currently raise.
5. **Sea-ice surface energy balance**, which does not exist under any LSM here.
6. **`soiltstep > 0`**, which needs the ten ACC_* carriers allocated and
   restart-bound.
7. **The health-descriptor ceiling.** `MAX_HEALTH_FIELDS = 1024` and
   `collect_state_fields` auto-walks `driver.fields`, so Noah-MP adds 49
   descriptors per domain. A four-domain Noah-MP nest is therefore about 196
   descriptors above whatever the same nest costs under Noah; the plan
   document's open item 4 is still open and now has a number.
