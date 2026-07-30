# NSSL2 runtime-adapter validation

## Scope and baseline

- Baseline commit: `7e11ce5da677a96d466c6e13a54d752cc49b8c10`
- Worktree: `/workspace/gpuwm-nssl2-runtime-adapter-work`
- Runtime source SHA-256: `bee2db0c8b8d5f4d962dec26b466799b6cad29737beb1bb9368616d4785abcca`
- Preflight source SHA-256: `a24e532526fda32538874bf5d168f25cbaae6def027af56ec90dfc820b439992`
- Test source SHA-256: `acf03fc88370ca689b5b65c3c651177da4a8b211839f5d03c5fca1590b3f10de`

The adapter is exposed as
`gpuwm.core.nssl2_runtime.apply_nssl2_production`. It is deliberately not
imported by `gpuwm.core.microphysics.apply`; the global `mp_physics=18`
selector remains fail-closed until a real fused GS implementation is admitted.
The adapter requires an explicit `NSSL2RuntimeHooks` bundle, and neither the
fused GS stage nor either allowed NUCOND/QVEXCESS form has a default.

## DomainState boundary

Each call rebuilds the WRF microphysics preparation from the current completed
RK state:

```text
theta = thb + thp
rho   = 1 / alt
pii   = (p / P0) ** (Rd / Cp)
z8w   = (phb + php) / g
dz    = z8w[k+1] - z8w[k]
w     = state.w
```

`theta`, `rho`, `pii`, `dz`, and `z8w` use existing restart-rebuilt `mp_*`
scratch names; pressure and staggered vertical velocity remain direct state
views. Preflight now admits those buffers plus post-process `refl_t` and the
single carrying `refl_10cm` handoff for MP18.

The adapter binds all 16 canonical Registry fields directly, in this order:

```text
qv qc qr qi qs qg qh
qndrop qnr qni qns qng qnh qnn
qvolg qvolh
```

The nine persistent precipitation fields must be identity aliases of their
canonical DomainState scratch slots:

```text
mp_rainnc mp_rainncv
mp_snownc mp_snowncv
mp_graupelnc mp_graupelncv
mp_hailnc mp_hailncv mp_sr
```

Copied or renamed aliases are rejected before state mutation.

`driver.microphysics_updates == 0` is the only first-call authority. When KF
is enabled, `qrcuten`, `qscuten`, `qicuten`, and `qccuten` are the persistent
raw rates in `driver.cu_rates`; each must identity-alias its canonical
`cu_rq*cuten` scratch slot. Coupled RK tendencies are never substituted.

## Ordered diagnostics and finish

The adapter delegates one admitted call to the production coordinator:

```text
gather/init/sediment -> precip reducer -> explicit fused GS
-> explicit NUCOND -> explicit QVEXCESS
-> direct concentration-space radar/radii -> one scatter -> finish
```

For a combined NUCOND/QVEXCESS implementation, its one callback occupies the
two adjacent stages. A wrapper publishes radar temperature only after the
final condensation/QVEXCESS callback returns:

```text
temperature_k = post_process_theta * prepared_pii
```

Thus `radardd02` and effective-radius diagnostics consume the same durable
concentration workspace before its only scatter. Radius kernels preserve their
official metre output boundary; the finish callback converts `state.effc`,
`state.effi`, and `state.effs` once in place to the existing radiation-facing
micron convention, checks native bounds, then invokes the real
`moist_physics_finish`. An output-due `REFL_10CM` buffer is stashed exactly
once and only after the entire coordinator call succeeds. A non-output call
does not allocate, touch, or stash either REFL buffer.

An exception in fused GS, NUCOND, or QVEXCESS cannot reach diagnostics,
scatter, radius conversion, moist finish, or the REFL stash. Tests cover both
fused-GS and final-QVEXCESS failures, including an unchanged radar-temperature
sentinel when QVEXCESS raises.

The returned `MicrophysicsDiagnostics` aliases the nine persistent fields and
must be accepted once by the existing outer
`PhysicsDriver.accept_microphysics` boundary. That existing boundary alone
advances `microphysics_updates` and accumulates the pending Noah rain handoff.

## Required future selector cadence

The public `radiation_due` flag is retained for isolated cadence/no-op tests.
An eventual MP18 selector must explicitly call the adapter with
`radiation_due=True` on every microphysics invocation. The effective radii are
the completed scheme's persistent state for the next radiation call, not a
diagnostic gated by the radiation driver's less frequent schedule. This
matches the current WSM6 and Morrison adapters, whose scheme launchers update
their effective-radius state on every microphysics call. No narrower WRF gate
has been admitted here.

`output_due`, by contrast, remains the history-output gate. Non-output calls
pass no radar temperature or output buffer into the coordinator and perform
no REFL stash.

## Validation record

Executed on the CUDA validation node with `/venv/main` activated:

```text
ruff check gpuwm/core/nssl2_runtime.py tests/test_nssl2_runtime.py
All checks passed!

ruff check --ignore F841,E731 gpuwm/core/preflight.py
All checks passed!

python -m pytest -q tests/test_nssl2_runtime.py
11 passed in 0.65s

python -m pytest -q \
  tests/test_nssl2_runtime.py \
  tests/test_nssl2_contract.py \
  tests/test_nssl2_production_coordinator.py \
  tests/test_nssl2_driver_support.py \
  tests/test_nssl2_nucond_diagnostics.py
49 passed in 1.02s

python -m pytest -q tests/test_preflight.py \
  -k 'scratch_registry_feature_matrix or \
      scratch_registry_classifiable_by_restart_manifest or \
      physics_lifetime_audit_is_exact_name_closed_world'
3 passed, 42 deselected in 0.12s

python -m compileall -q gpuwm/core/nssl2_runtime.py
(no output)

git diff --check
(no output)
```

The full preflight scratch subset was also attempted. Four independent tests
passed; two real74 fixture tests could not start because this Linux worktree's
checked-in config names a local Windows ERA5 path that is not present on the
validation node. The failure occurs in `load_experiment_case` before the
scratch assertions and is unrelated to this slice.

`gpuwm/core/preflight.py` has four existing Ruff findings at the baseline
(two F841 and two E731). The current file reports the identical four findings,
shifted only by this slice's ten inserted lines; the new/changed logic adds no
finding.

Only the runtime adapter, preflight admission, its tests, and this evidence
note are changed. The microphysics selector, dycore, production coordinator,
fused/process kernels, QVEXCESS/process28 work, and restart implementation are
untouched.
