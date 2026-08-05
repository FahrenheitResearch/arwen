# Grell-Freitas (`cu_physics = 3`) — what a release integrator needs to know

Lane: `lane/gf-port`. Target: `integration/release-1.6`. This is the short
form. The conformance argument lives in the registry entry
(`gpuwm/physics_registry_v2.json`,
`components.cumulus.options.grell-freitas`) and in the parity suites; this
file is only the set of things that will bite someone folding the lane in.

## 1. The label is `implemented-unverified`, and a run does not change that

`scientific_evidence: none`. Grell-Freitas is bitwise against WRF v4.6.1
at the whole-driver boundary — 216 oracle columns, 208 of them word for
word on both the float32 CPU authority and the CUDA translation unit, the
remaining 8 bounded by the driver's own `module_gfs_physcons` mixed
precision at max 34 ULP — and that is **conformance evidence, not
scientific validation**. No scored gpuwm/WRF forecast comparison exists
for this scheme.

A first real-case trajectory (below) establishes that the scheme
integrates a real domain without diverging, and that a dual run of it is
byte-reproducible. It does **not** establish forecast skill, and it is not
grounds to move the label. Moving to `supported` needs a scored
comparison against obs (MRMS/ASOS), on the project's own
obs-skill-is-the-referee terms.

## 2. Selection surface

* `cu_physics = 3`, per domain, on the tree route only
  (`reachability: component-override` — the Shin-Hong/SASE posture).
* `cudt_minutes = 0` is **enforced, not chosen**:
  `validate_run_config` refuses a nonzero `cudt` for `cu_physics = 3` by
  name. GF runs on the model step (WRF's usual GF configuration,
  `STEPCU = 1`) and carries no NCA hold, so the KF cadence knob has no
  meaning here.
* `clos_choice = 0` (the 16-member ensemble closure) is the only admitted
  arm — it is the only one the oracle covers. `ishallow` is 0 or 1, both
  covered, 0 is the WRF Registry default.
* Both Grell-family keys are refused wherever no Grell scheme is selected,
  so unrelated configs cannot carry them.
* **ArWen structural seam:** 390 (`pbl = 0`, `cu = 3`) cells of the WRF
  v4.6.1 compatibility matrix are admitted by WRF and refused by ArWen.
  WRF reads `KPBL = 0` there and indexes below the column base; ArWen
  refuses rather than reproduces. This is the second named structural
  seam and it is deliberate.

## 3. The corrected-k22 default, and the flag that is not a config key

WRF's shallow k22 trigger is a `MAXLOC` over the array section
`heo_cup(2:kbmax)` whose result `module_cu_gf_sh.F` uses as an absolute
level index without adding the section offset. **The shipped kernel uses
the corrected indexing** (owner ruling, no inherited WRF bugs).

The WRF-faithful off-by-one exists only behind a launch argument
(`k22_wrf_faithful`) that the parity suites set directly. It has **no
`RunConfig` spelling** — deliberately, and there is a test that asserts
so. An integrator should not add one.

The restart identity binds the choice:
`grell-freitas-wrf461-gfdrv-corrected-k22-v1`. A checkpoint written under
the shipped identity cannot resume under a WRF-faithful build.

Measured ledger for the difference: k22 moves on 3 of 18 fixture cases,
all three rejected under both modes with identical `ierr`, and **zero**
output words differ at either boundary.

## 4. Engine-seam deviations (the plumb-list for any label upgrade)

The kernel is bitwise; these are about what the engine can hand it today.

1. **Forcing tendencies.** GFDRV sums advective (`RTHFTEN`/`RQVFTEN`),
   radiative (`RTHRATEN`) and boundary-layer (`RTHBLTEN`/`RQVBLTEN`)
   forcing into its forced state. The adapter feeds the **radiative** term
   (read off the driver through `bind_driver`) and **zeros** for the other
   two: the dycore exports no advective theta/qv forcing pair, and the PBL
   stack couples its rates before the driver retains them. Degraded: the
   rate-of-change closure family's forcing, and with `ishallow = 1` the
   shallow `blqe` member (`dhdt = 0`). The instability closures see the
   current state and the omega/moisture-convergence closures are fully fed.
2. **Convective momentum tendencies.** The kernel computes GF's
   `dudt`/`dvdt`; `CumulusResult` carries no momentum slots, so they are
   not coupled. WRF couples them.
3. **w on mass levels** is the KF-precedent average `0.5*(w[k]+w[k+1])`.

## 5. VRAM: GF costs +3.20 GiB of non-pool backing store

This is the one number that changes what a host can run, and it is not in
the CuPy pool where anyone would look for it.

One GF thread owns one whole GFDRV column, so the per-thread local frame
is the deep+shallow local-array stack: **22,416 B measured** (registered
at `gpuwm/core/preflight.py::KERNEL_MAX_LOCAL_SIZE_BYTES["gf"]`, nz=40
tier). By the reservation law in `docs/kernel_local_memory_bounds.md`,

```
(22,416 - 1,024) * 1,536 * 170  =  5.20 GiB          [RTX 5090]
```

the driver reserves that for the device's whole resident-thread capacity
on the kernel's first launch and holds it for the life of the process.

The base stack's largest frame is KF's at nz=49 (9,216 B → 1.99 GiB), so
selecting GF **replaces** that reservation rather than adding to it:
net **+3.20 GiB**. `gpuwm check` prices it under `NON-POOL`.

## 6. WORK ITEM: the four-domain GF arm does not fit, and it is a code fix

`configs/real74_4dom_gf.toml` is the ratified target and **is not runnable
on the Windows host**. Measured by `gpuwm check`:

| config | device-footprint projection | forecast peak envelope |
|---|---|---|
| `real74_4dom.toml` (KF base) | 24.12 GiB | — |
| `real74_4dom_gf.toml` | **27.79 GiB** | 44.87 GiB |
| `real74_2dom_gf.toml` (what runs) | 16.67 GiB | 25.98 GiB |

The card is 32,607 MiB with no ECC, the four-domain base stack has been
measured peaking at 30,879–31,352 MiB machine-wide, and this project has
documented 32.2 GB producing **corrupted d03/d04 output rather than a
clean refusal**. There is no room for another 3.2 GiB and the failure mode
is silent. `gpuwm check` also reports that the RRTMGP `column_chunk` lever
cannot close the four-domain gap: the grid itself would have to come down.

Disk is an independent limit: four domains at these cadences is ~244 GB of
wrfout per arm, ~488 GB for a dual run, against 242 GB free on the largest
volume. Not even one arm fits, and nothing may be deleted.

**The route is a code change, not a config one:** specialize the `gf.cu`
per-thread local frame the way `KF_KMAX`/`REFL_KMAX` were specialized
(`docs/kernel_local_memory_bounds.md`), so the reservation is sized by the
arrays a column actually needs rather than by the compiled ceiling.
Acceptance: `gpuwm check configs/real74_4dom_gf.toml` back under the
four-domain base's footprint, GF parity suites still bitwise (the frame is
allocation size, not a value any loop reads — the same argument that made
the KF/REFL specialization safe).

**This is registered here as its own work item and is NOT a blocker for
folding the lane in.** The two-domain arm carries the GF evidence: GF is
selected on d01 and d02 and nowhere else, `feedback = 0` makes the nesting
one-way, so d01/d02 integrate the same trajectory they would inside the
four-domain tree.

## 7. Restart: `cu_physics = 3` could not checkpoint until a4c06235

The GF selectable-wiring commit (`ea0cada3`) taught the restart manifest
GF's algorithm identity and its coefficient-table absence, but not GF's
**callable shape**. The adapter stores the `PhysicsDriver` that
`bind_driver` hands it — that is how it reads the held radiative rates —
and `_callable_state_check` walks one level into object attributes, so the
driver's own arrays appeared under the cumulus callable unclassified.
Every checkpoint of a GF run raised `RestartManifestError`. The first
real-case GF trajectory integrated 179 outer steps and died at its first
restart.

Fixed in `a4c06235`: `_driver` is classified rebuild-on-load in
`CUMULUS_CALLABLE_CONTAINERS`, on the KF `_history_state` precedent. Both
are back-references, not state. **Nothing of GF's is serialized as new
restart state, because GF holds none** — `cu_physics = 3` is a stateless
Task-1 callable with no NCA persistence, no trigger history (no `W0AVG`),
and no closure memory. Every GF quantity with memory between calls already
lived on the driver and was already serialized there: held rates through
`cu_rates`' `cu_rthcuten`/`cu_rqvcuten`/`cu_rqccuten`/`cu_rqicuten`
scratch slots, precipitation accumulators through
`cu_rainc`/`cu_raincv`/`cu_pratec`.

Gated by `tests/test_restart.py::
test_short_grell_freitas_restart_is_bit_identical` — 20 steps + restart +
20 steps == 40 steps, FP32-bit-exact, on a state where GF actually
convects — and by the CPU-tier
`test_every_cumulus_adapter_attribute_is_classified`, which walks every
selectable cumulus adapter live.

**Integrator note:** a GF adapter that ever grows real state must be
serialized, not added to the container allowlist. The classification
comment in `gpuwm/io/restart.py` says so at the point of use.

## 8. Test battery

```
tests/test_gf_deep_cuda.py      tests/test_gf_deep_parity.py
tests/test_gf_driver_parity.py  tests/test_gf_engine_smoke.py
tests/test_gf_gfdrv_cuda.py     tests/test_gf_shallow_cuda.py
tests/test_gf_shallow_parity.py tests/test_gf_wrf461_parity.py
                                -> 765 passed
tests/test_restart.py           -> 96 passed, 1 skipped
```

Cross-lane touchpoints that also carry GF assertions and must stay green:
`test_namelist_import.py` (native import of namelist `cu_physics = 3`),
`test_wrf461_compatibility.py` (the cumulus axis: 3,840 -> 5,760 matrix
cells, and the `pbl = 0` structural seam), `test_config_freeze.py`,
`test_experiment.py`, `test_physics_vertical_preflight.py`.
