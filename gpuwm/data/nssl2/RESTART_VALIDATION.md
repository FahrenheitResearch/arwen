# NSSL2 restart validation

## Scope and baseline

- Baseline commit: `504dd3d8606c7916086dfc4056386f3ed61589c8`
- Outer archive format: existing restart v5 (unchanged)
- MP18/NSSL2 restart contract: nested schema version 1
- Implementation: `gpuwm/io/restart.py`
- Tests: `tests/test_restart.py`
- Source SHA-256: `1ffc8a54d5e04bc266d84dbd29eb3142d2aa6db7875059fdfe13f13669446eaf`
- Test SHA-256: `f0778df1bf0aa266ea6cabc8a12a1880f0f3242557ebc2660dda3708232a8476`

The existing v5 raw-NPY payload, self-manifest, atomic temporary-file rename,
configuration/setup fingerprints, and validate-before-mutate restore path are
reused. MP18 adds a fail-closed, independently versioned identity inside
`physics_setup.microphysics.restart_contract`; no global format bump is needed.

## Durable state contract

The exact canonical three-dimensional Registry inventory is:

```text
qv qc qr qi qs qg qh
qndrop qnr qni qns qng qnh qnn
qvolg qvolh
h_diabatic
```

The exact persistent two-dimensional NSSL precipitation inventory is:

```text
mp_rainnc mp_rainncv
mp_snownc mp_snowncv
mp_graupelnc mp_graupelncv
mp_hailnc mp_hailncv
mp_sr
```

`driver.microphysics_updates` is the first-call authority: zero means the next
MP18 invocation is the first call, and a positive value means first-call
initialization has already occurred. `header.elapsed_seconds` is the clock
authority. The existing driver manifest also carries pending/held driver state.

There is no additional NSSL coordinator workspace to checkpoint. Its local
workspace is assembled for one invocation, and the coordinator performs one
final scatter into the canonical Registry arrays above before returning. A
restart write therefore occurs at a state boundary at which those arrays,
precipitation slots, driver state, counter, and clock are authoritative.

## Refusal behavior

MP18 writes fail unless all canonical arrays and precipitation slots exist with
the prepared shapes and `float32` dtype, an MP18 `PhysicsDriver` is attached,
the first-call counter is a non-negative integer, and elapsed time is finite
and non-negative.

MP18 restores reject, before mutating live state:

- absent, stale, malformed, or extended nested contract versions;
- missing or extra state/scratch members;
- shape or dtype disagreement;
- generic microphysics aliases such as `nc`, `nr`, `ni`, `ns`, and `ng`;
- historical NSSL slab aliases such as `ccw`, `crw`, `cci`, `csw`, `chw`,
  `chl`, `cn`, `vhw`, and `vhl`;
- legacy driver precipitation aliases;
- invalid first-call counters or elapsed time;
- unsupported outer restart versions.

## Split-run identity

The restart test advances a deterministic MP18 state for eight steps and
compares it with a three-step run, v5 checkpoint/restore into a fresh prepared
state, and five-step continuation. It checks raw bytes (no tolerance) at the
checkpoint boundary and final boundary for every serialized state, scratch,
and driver array, then explicitly checks all 16 NSSL prognostics,
`h_diabatic`, all nine precipitation slots, `microphysics_updates`, and
`elapsed_seconds`. The fixture includes one-time first-call behavior so an
incorrectly restored counter provably diverges.

## Validation record

Executed on the CUDA validation node in `/workspace/gpuwm-nssl2-restart-work`
with `/venv/main` activated:

```text
ruff check gpuwm/io/restart.py tests/test_restart.py
All checks passed!

python -m pytest -q tests/test_restart.py -k nssl2
20 passed, 52 deselected in 0.60s

python -m pytest -q tests/test_restart.py
71 passed, 1 skipped in 6.60s

python -m pytest -q \
  tests/test_nssl2_contract.py tests/test_nssl2_production_coordinator.py
23 passed in 0.22s

git diff --check
(no output)
```

The only changed paths are this evidence note, `gpuwm/io/restart.py`, and
`tests/test_restart.py`. Frozen NSSL kernels, the production coordinator, and
the MP18 selector were not edited.
