# MP18 selector integration checklist

## Scope and release decision

This is the integration plan for wiring the already admitted NSSL option-18
runtime into `gpuwm.core.microphysics.apply`.  It is intentionally a planning
artifact: **the selector must remain locked** until every mandatory gate below
passes on the production GPU path.

The selector itself is a small branch.  The release-critical work is the
persistent runtime binding, scratch/restart ownership, output cadence, removal
of steady-state allocations and host synchronizations, and end-to-end GPU
verification.

## Hard blockers before the selector can be enabled

1. **Avoid the import cycle.** `nssl2_runtime.py` imports
   `MicrophysicsDiagnostics`, `save_pre_mp_theta`, and
   `moist_physics_finish` from `microphysics.py`; `nssl2_default_hooks.py`
   imports `nssl2_runtime.py`.  A top-level NSSL import in
   `microphysics.py` can therefore observe a partially initialized module.
   Import the NSSL adapter only inside the `mp_physics == 18` branch.  Build
   the persistent binding through a lazy import inside `PhysicsDriver`
   initialization, not at module import time.

2. **Eliminate every per-call allocation.** The current
   `gather_initialize_and_sediment` allocates a `16 * nz * ny * nx` state
   slab, a `5 * ny * nx` category-export slab, an ignored `ny * nx`
   accumulator, and sometimes a full-volume zero-rate array on every MP call.
   This is not an admissible production path.

3. **Complete scratch and restart ownership.** The fused GS callback adds
   `nssl2_fused_temperature` and `nssl2_primary_ice_target`; the reusable
   driver workspace adds three more buffers.  All must be named preflight
   scratch, classified write-before-read and restart-rebuilt, prewarmed by
   `--alloc`, and identity-checked before the first mutation.

4. **Fix every REFL allowlist.** Option 18 must be included in the four
   scheduling/consumption sites listed below.  If production schedules the
   native NSSL radar but a consumer omits 18, the one-frame stash remains live
   and the next output-due call correctly fails as an overwrite.

5. **Remove production value-reduction syncs.** `validate_values=False` must
   be explicit in the selector-owned binding and runtime call.  In addition,
   `PhysicsDriver.accept_microphysics` currently performs nine synchronous
   `cp.isfinite(...).all()` reductions through `_validated_array`, then two
   more `cp.any` reductions for the SR range.  A trusted native-MP18 path must
   retain exact identity/shape/dtype/C-contiguity checks but skip all eleven
   value reductions.  Scheduled global health checks remain the value
   authority.

## Required runtime architecture

### Persistent per-domain binding

Create one restart-rebuilt binding when `PhysicsDriver` is initialized for
`mp_physics == 18`.  The binding should own:

- the real `NSSL2FusedGS` callback;
- the concrete default runtime hooks, created with `validate_values=False`;
- one reusable `NSSL2DriverWorkspace`;
- the ignored sediment accumulator;
- the canonical float32 step used by the fused GS, NUCOND, and outer runtime.

The binding is domain-specific.  Before any theta parking, gather, or
precipitation update, validate that its state and every scratch pointer belong
to the current `DomainState`, its workspace shape is exact, and
`np.float32(binding.dt_s) == np.float32(dt_s)`.  The last check is mandatory:
the fused callback and NUCOND closure capture `dt_s`, while the outer runtime
also receives a step argument; permitting those values to diverge is a hidden
correctness failure.

Recommended named buffers are:

| Scratch name | Shape | Per-call lifecycle |
| --- | --- | --- |
| `nssl2_driver_state` | `(16, nz, ny, nx)` | gather fully overwrites |
| `nssl2_driver_surface_export` | `(5, ny, nx)` | five sediment launches overwrite their planes |
| `nssl2_driver_ignored_accumulator` | `(ny, nx)` | explicitly zero before sediment RMW |
| `nssl2_fused_temperature` | `(nz, ny, nx)` | fused prepass fully overwrites |
| `nssl2_primary_ice_target` | `(nz, ny, nx)` | fused prepass fully overwrites |
| `nssl2_nucond_ss` | `(nz, ny, nx)` | NUCOND fully overwrites before read |

The new resident request beyond the already registered NUCOND buffer is
`72 * nz * ny * nx + 24 * ny * nx` bytes in FP32.  Including NUCOND, these
six buffers request `76 * nz * ny * nx + 24 * ny * nx` bytes.  Do not claim
cross-slot alias savings: the volume buffers are simultaneously live inside
one coordinator call.

When `cu_used=False`, the CUDA gather branch does not read the KF rate
pointers.  Pass a valid existing volume pointer as the unused argument rather
than allocate a zero volume.  When `cu_used=True`, production already requires
the four canonical KF raw-rate aliases.  Any allocation fallback retained for
isolated direct tests must be impossible to reach from the selector.

### Selector contract

Only after the gates pass, replace the exact `NotImplementedError` branch in
`gpuwm/core/microphysics.py` with a branch-local import and one call through
the persistent binding:

- require `state.physics` and the real MP18 binding;
- map `refl_10cm_due` exactly to `output_due`;
- pass `radiation_due=True` on **every** MP18 call;
- pass `validate_values=False` explicitly;
- return the runtime's `MicrophysicsDiagnostics` unchanged;
- let the existing dycore call `accept_microphysics` exactly once; do not
  increment `microphysics_updates` in the selector or runtime.

No `PhysicsDriver.compute` or dycore cadence change is required.  Physics
radiation runs pre-RK and therefore consumes effective radii produced by the
previous post-RK microphysics call.  On the first step it consumes the valid
initial radius defaults from `DomainState`.  This is why `radiation_due` must
not be tied to the less frequent radiation scheduler.

Noah similarly consumes the previously accepted SR and pending precipitation
on its next due pre-RK surface call.  The existing post-RK acceptance order is
the required order.

### Native radar handoff

Add 18 to all four native-radar scheduling/consumption allowlists:

1. `gpuwm/runtime.py`, `write_case_output`;
2. `gpuwm/runtime.py`, the single-domain `refl_due` predicate;
3. `gpuwm/runtime.py`, `_submit_tree_history_frame`;
4. `gpuwm/verify/cases/nest_ideal_common.py`,
   `consume_history_reflectivity`.

Do **not** add 18 to `gpuwm/core/refl.py::compute_refl_10cm`.  That dispatcher
is the generic Kessler/WSM6/Morrison implementation.  MP18 reflectivity is the
native concentration-space NSSL radar stage inside the production
coordinator, after the final condensation callback and before scatter/finish.

The due path must remain transactional: calculate into `refl_10cm`, stash only
after the full coordinator succeeds, consume exactly once at history output,
and drain output before a boundary checkpoint.  A non-output call must not
create or alter the stash.

## Exact code changes

- `gpuwm/core/microphysics.py`
  - Keep the selector closed until release.
  - At release, add only the lazy MP18 branch described above.

- `gpuwm/core/nssl2_default_hooks.py`
  - Add a production binding factory that acquires the named scratch arrays,
    constructs the real fused callback and reusable workspace, captures the
    canonical step, and creates hooks with validation disabled.
  - Keep direct/oracle hook construction value-validating by default.

- `gpuwm/core/nssl2_driver_support.py`
  - Extend `NSSL2DriverWorkspace` to own/reference every reusable driver
    buffer.
  - Thread a supplied workspace through gather/sediment and prove full
    overwrite/required zeroing.
  - Remove the selector-reachable `cp.empty`, `cp.zeros`, and `zeros_like`
    paths.

- `gpuwm/core/nssl2_production_coordinator.py` and
  `gpuwm/core/nssl2_runtime.py`
  - Thread the supplied workspace/binding through the whole call; do not
    create another workspace.
  - Validate workspace identity, domain identity, and captured-step equality
    before `save_pre_mp_theta` or any kernel launch.
  - Preserve the sole gather, sole scatter, stage order, and post-success REFL
    stash.

- `gpuwm/core/physics.py`
  - Lazily construct `nssl2_binding` once for MP18 and set it to `None` for all
    other schemes so the driver attribute set is closed-world.
  - Add a trusted-MP18 acceptance path: exact canonical aliases and structural
    invariants are mandatory; per-field finite and SR reductions are skipped.
    The default/untrusted acceptance behavior for all other callers remains
    unchanged.

- `gpuwm/core/preflight.py`
  - Register the five new named buffers with the exact shapes above (NUCOND is
    already registered).
  - Add every buffer to the write-before-read lifetime table and resident
    estimate; make `run_alloc_preflight` materialize them.
  - Preserve per-slot max-over-domains shared-arena accounting; do not infer
    arbitrary aliasing between simultaneously live slots.

- `gpuwm/io/restart.py`
  - Add all five new scratch names to exact `REBUILT_SCRATCH_SLOTS` (the
    current rebuilt prefixes do not cover `nssl2_*`).
  - Add `nssl2_binding` to `DRIVER_REBUILT_ATTRS` so the driver closed-world
    audit accepts the domain-owned callbacks/arrays.
  - Rebuild the binding during normal physics initialization, then restore
    `microphysics_updates` and serialized precipitation scratch in place.
    Never serialize the workspace contents or ephemeral REFL handoff.

- `gpuwm/runtime.py` and
  `gpuwm/verify/cases/nest_ideal_common.py`
  - Update the four REFL allowlists above and no generic radar dispatcher.

## Validation policy

- Direct runtime, hook, oracle, and malformed-input tests use
  `validate_values=True`.
- The production selector hard-codes `validate_values=False`; do not make this
  an environment-variable or non-restart-stable runtime toggle.
- The initialized/restored health gate and scheduled global health cadence
  (every fourth internal step plus output, restart, and final mandatory
  instants; every step in `health_debug`) are the production value authority.
- Tests must prove true/false validation modes produce identical finite
  outputs for the same valid inputs.

## Required tests before unlock

### CPU/contract tests

Add `tests/test_nssl2_selector.py` and update the existing focused suites to
cover:

- branch-local import succeeds without a circular/partial initialization;
- missing/wrong-domain binding and float32-step mismatch fail before any
  mutation;
- `refl_10cm_due -> output_due`, `radiation_due is True`, and production
  validation is false on every selector call;
- direct adapter and selector return the same diagnostics and accept exactly
  once;
- two calls reuse the exact workspace/scratch identities and do not grow the
  CuPy memory pool after warmup;
- poisoned driver-state/export/fused/NUCOND scratch is fully overwritten, and
  the ignored accumulator is explicitly reset;
- `cu_used=False` makes no zero-rate allocation; `cu_used=True` uses the four
  canonical KF aliases;
- all four MP18 REFL schedule/consume paths work, non-due calls leave no
  stash, and a due stash is consumed exactly once;
- generic `compute_refl_10cm` still rejects 18;
- radiation at step N sees radii from MP step N-1, and every MP call refreshes
  the three persistent radii;
- Noah sees accepted SR/pending precipitation on its next due call;
- trusted MP18 acceptance enforces identity/shape/dtype/C-contiguity and has
  no value-reduction calls; untrusted acceptance retains existing checks;
- preflight shape, lifetime, arena-byte, resident-estimate, and `--alloc`
  materialization contracts include every new buffer;
- restart binding is rebuilt against fresh scratch, workspace is absent from
  the checkpoint, the counter/`first_step` transition is exact, and output
  boundary REFL policy remains single-consume.

Retire the selector fail-closed assertions in `tests/test_nssl2_runtime.py` and
`tests/test_nssl2_gpu.py` only in the final release commit, after all other
gates pass.

### Mandatory RTX 5090 gates

All of the following are release blockers:

1. Focused lint/compile/unit suite, including the actual selector path.
2. Actual-GPU first-call and second-call selector tests, both output-due and
   non-output-due, using the real fused GS and NUCOND hooks.
3. One-step and multi-step direct-runtime versus selector identity.
4. Real-hook uninterrupted versus split/restart bitwise identity, including
   `microphysics_updates`, precipitation accumulators, state moments, radii,
   and the first post-restore call.
5. Full d01-d04 `--alloc` preflight for the production 500 m configuration,
   with measured allocation no greater than the estimate.
6. Compute Sanitizer memcheck and racecheck on a reduced real-hook case.
7. A warmed performance run (exclude NVRTC/first-use compilation):
   - CUDA-event selector+accept overhead no more than 2% over the direct
     runtime+accept path;
   - CuPy pool `total_bytes()` stable after warmup;
   - no `cudaMalloc`, device-to-host copy, or device synchronization inside a
     steady-state MP18 call/accept outside scheduled health/output;
   - stable p95 step time and no secular resident-memory growth.
8. A 15-minute four-domain nested smoke with every expected d01-d04 history
   frame, native REFL field, clean health gates, and restart continuation.
9. The requested full 12/15-hour 1974 d01-d04 verification with d04 at 500 m,
   complete wrfout inventory, restart continuity, health clean, and CPU/WRF
   comparison artifacts retained.

The selector is releasable only when these gates are recorded with exact
commit, configuration fingerprint, GPU identity, command lines, and output
artifact locations.
