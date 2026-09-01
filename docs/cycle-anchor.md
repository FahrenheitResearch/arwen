# The cycle anchor, the ingestion gate and the consistency instrument

This is the state-handoff half of the cycling spine (`gpuwm cycle`). It
replaces "the restart format is a pickle of live Python objects" with a
versioned on-disk directory, and it adds the one instrument that can tell
"the DA cycle assimilated radar" apart from "the DA cycle silently dropped
its analysis and produced a very expensive free forecast".

Modules: `gpuwm/cycle/anchor.py`, `gpuwm/cycle/consistency.py`,
`gpuwm/cycle/ingestion.py`. Leg runner: `tools/cycle_mpas_leg.py`.
Shared seam (owned jointly with the clock and placement lanes):
`gpuwm/cycle/contracts.py`.

## Why files and not a library

The MPAS GPU port is a separate codebase. It pins gpuwm by SHA, re-verifies
the checkout before and after every run, and freezes its own execution
sources. The spine therefore cannot be a library the port imports and
cannot live inside the pinned gpuwm surface. It has to be a separate
process that talks to both sides through files with versioned schemas.

That constraint pays for itself three times: the cycle becomes
crash-recoverable, a Rust DA engine or the renderer can read the same
artifacts, and the pickle-as-restart-format problem disappears.

## Directory layout

```
<root>/anchors/anchor_<cycle:03d>_<YYYYmmdd_HHMMSS>/
  anchor.json              manifest, schema gpuwm-cycle.anchor/v1
  parent_prognostic.nc     rho, rho_theta, rho_u, rho_w, scalars, time_seconds
  parent_derived.nc        exner and the rest of the carried diagnostics
  parent_seam.nc           physics-seam mapping (optional)
  analysis_increment.nc    the DA increment in the anchor's own space (optional)
  children/<grid_id>/state.nc
  COMMIT                   phase-two marker: every file with its sha256
```

`.nc` when `netCDF4` imports, `.npz` otherwise. **The manifest records
which** under `array_format` (`"netcdf4"` or `"npz"`), so a consumer never
guesses, and `read_anchor` refuses an anchor written in a format this
process cannot read rather than half-reading it.

## Manifest keys (`anchor.json`)

| key | meaning |
|---|---|
| `schema` | `gpuwm-cycle.anchor/v1` |
| `cycle_index` | integer cycle number; also the directory prefix |
| `anchor_ticks` | model time at this boundary, in ticks (1 tick = 1 ms) |
| `tick_hz` | 1000, carried so a reader never assumes it |
| `valid_time` | ISO-8601 instant |
| `array_format` | `"netcdf4"` or `"npz"` |
| `written_at` | wall clock of publication, provenance only |
| `parent.kind` | one of `mpas-cuda`, `arwen`, `replay` |
| `parent.mesh_id` | the mesh this state belongs to |
| `parent.n_cells`, `n_edges`, `n_levels` | dimensions read off `rho` / `rho_u` |
| `parent.prognostic_sha256` | canonical state hash (recipe below) |
| `parent.path` | relative path of the prognostic file |
| `derived.derived_from_sha256` | **the prognostic sha the diagnostics were built from** |
| `derived.path`, `derived.fields` | where they are and what they are |
| `seam.path` | physics-seam mapping, or `null` |
| `children[]` | `grid_id`, `placement`, `state` (one of `PLANNED/LIVE/RETIRED/REFUSED/DIVERGED`), `state_sha256`, `path` |
| `analysis.state` | `APPLIED`, `SKIPPED_NO_OBS`, `REJECTED`, `NULL_ARM` |
| `analysis.increment_sha256`, `analysis.path` | the increment carried here |
| `analysis.ingestion` | the three-hash receipt block (below) |
| `diagnostics_rebuilt` | `true` / `false` / `null` (not asked) |
| `consistency` | the residual block written by the leg runner |
| `resumed_from` | the anchor this one was advanced from |

`children[].arrays` and `analysis.arrays` are **write-time inputs**, not
manifest keys: `write_anchor` takes the arrays, writes them, and records
their paths and shas.

## The canonical state hash

So an independent implementation (Rust, Fortran, another Python) can
reproduce the number:

1. walk the field names in `sorted()` order;
2. for each, feed the sha256 `name.encode()`, then
   `str(dtype).encode()`, then `repr(tuple(shape)).encode()`, then
   `np.ascontiguousarray(array).tobytes()`;
3. nothing else enters -- no file bytes, no compression, no dict order.

It is byte-exact and therefore FP-representation-sensitive **by design**.
The same values as float32 hash differently from float64, and a one-ULP
change is a different state. That is what makes it usable as the ingestion
gate's evidence: the question the gate asks is "is this literally the same
array", not "is it close".

One trap worth naming, because it bit this implementation: numpy's
`ascontiguousarray` promotes a 0-d array to shape `(1,)`. `time_seconds` is
0-d. The hash and the writer both go through `_contiguous`, which restores
the original shape, so a scalar round-trips as a scalar.

## Three-phase commit

1. **Phase one** -- every array file, the manifest, and the child states are
   written into `anchor_<...>.tmp/` and fsynced.
2. **Phase two** -- a `COMMIT` marker naming every file with its sha256 is
   written and fsynced.
3. **Phase three** -- the directory is renamed into place with `os.replace`.

A crash leaves a `.tmp` directory, which `latest_anchor` never returns and
`read_anchor` refuses by name. A directory edited after publication is
caught on load: every member's sha is checked against the marker, so a
truncated `parent_derived.nc` is a refusal, not a silent shape error.

**Reading a directory listing to discover anchors is out of contract.** Use
`latest_anchor(root)` and `anchor_for_cycle(root, i)`; they only ever return
committed anchors.

## Write-time refusals

Each names what was observed and offers a remedy:

- a prognostic mapping missing any of `rho, rho_theta, rho_u, rho_w,
  scalars` -- names the missing list, the required list and what was present;
- a non-finite value in any prognostic array -- names the field, the count
  and the first flat index;
- a `time_seconds` that disagrees with `anchor_ticks` by more than half a
  tick -- names both, because the clock authority is the cycle clock and not
  the state file;
- an unknown `parent_kind` or child `state`;
- an anchor directory that already exists -- the spine is forward-only and
  never overwrites a published boundary.

## The ingestion gate (`gpuwm-cycle.analysis-ingestion/v1`)

Three hashes, taken in three places by two processes:

1. `background_sha256` -- the parent state the **spine** wrote into the
   anchor, hashed by the spine.
2. the increment -- `increment_sha256`, `increment_nonzero_cells`, and a
   per-field `increment_l2`, so a refusal can say what was thrown away.
3. `analysis_sha256` -- the state after application, hashed by the
   **forecast process** off its own rehydrated state. Taken anywhere else
   it proves nothing: the interesting failure is precisely that the
   increment never reached the device.

`verify_ingestion` gates **both directions**, because an exact-zero delta
means the experiment never ran.

**Direction one -- the analysis was dropped.** Nonzero increment, identical
shas:

```
analysis was not ingested | observed: fields=['qr', 'u', 'v'],
increment_l2={'qr': 0.0003, 'u': 1.5811388300841898, 'v': 2.0},
increment_nonzero_cells=4,
increment_sha256='a5e555bda2c32fd469d1f8195492a8c6a6e08675ec3b4705355335daf7661aba',
label='cycle=3 valid=2026-08-14T02:00:00Z',
remedy='check that the increment reached the device state before the first
step, not the host copy that was discarded at rehydration',
shared_sha256='4f2c000000000000000000000000000000000000000000000000000000000000'
```

**Direction two -- the null arm moved.** Zero increment, different shas:

```
null-increment arm is not bit-stable | observed:
analysis_sha256='bbbb...bbbb', background_sha256='aaaa...aaaa',
fields=['qr', 'u', 'v'], increment_nonzero_cells=0,
label='cycle=3 null arm',
remedy='rehydration itself perturbed the state; compare the restored
prognostics field by field against the anchor before assimilating anything'
```

Otherwise the receipt block comes back with keys exactly
`background_sha256`, `increment_sha256`, `increment_nonzero_cells`,
`increment_l2`, `analysis_sha256`, `state` (plus `schema` and `fields`), and
`state` is `APPLIED` or `NULL_ARM`. That block is what the supervisor writes
into the transition receipt as `analysis["ingestion"]`.

## The consistency instrument (`gpuwm-cycle.state-consistency/v1`)

Every pre-existing gate on the restart path is an **identity** gate, and a DA
increment must break identity by construction, so none of them can see the
failure this instrument exists for: prognostics rewritten by an analysis,
diagnostics carried across the boundary unchanged, a model that resumes on
an `exner` describing the atmosphere it had before the radar spoke. It does
not crash. It is wrong.

`hydrostatic_residual(prognostic, derived)` recomputes
`exner = (rd * rho_theta / p0) ** (rd / (cp - rd))` and reports the maximum
and mean relative departure of the carried exner from the recomputation,
the `argmax_index`, `n_points`, and -- always -- `resolution_floor`.

**Measured floor: 2.220e-16** (`tests/test_cycle_consistency.py` prints it
on every run). The floor is measured, not asserted: exner is recomputed
twice from the same input and the disagreement between the two
recomputations is the smallest signal the instrument can resolve. On this
float64 path that difference is exactly 0, so the reported floor is clamped
to machine epsilon -- the honest limit of the representation rather than a
flattering zero.

**Threshold 1e-6, justified against that floor rather than a guess.** The
clean-state residual is 0.0 (floor 2.2e-16); a 1 % increment to a single
`rho_theta` cell drives the max residual to ~3e-3 and the mean over a
4x10 state to 9.1e-5. 1e-6 sits ten orders of magnitude above the floor and
three orders below the smallest increment anybody would call an analysis, so
it cannot fire on arithmetic and cannot miss a real analysis. Both
directions are asserted in the tests.

`require_consistent(...)` returns the residual block or refuses naming the
label, the observed max and mean, the threshold, the argmax index and the
floor.

## The stale-derived refusal, and why it exists

`derived_is_stale(manifest)` is the cheap structural test:
`derived.derived_from_sha256 != parent.prognostic_sha256`. The leg runner
takes it against the state **as it will be integrated**, not as it was
stored -- applying an increment is exactly what makes a carried derived
block describe the wrong atmosphere.

Policy, non-negotiable: if the derived block is stale and the backend cannot
rebuild diagnostics, the leg **refuses to advance**:

```
REFUSED: cannot resume on stale derived diagnostics | observed:
backend='replay', label='cycle=0 valid=... anchor=anchor_000_...',
rebuildable_fields=['exner'],
remedy='run this leg under --backend mpas-cuda, whose
_rebuild_saved_diagnostics owns these fields; the replay stub must not
invent them', unrebuildable_fields=['normal_velocity']
```

`diagnostics_rebuilt: true|false` is recorded in the next anchor either way.

This is fail-closed at precisely the place recon named as unguarded, and it
is default-on: there is no flag that turns it off. The two tests
`test_stale_derived_refuses_when_rebuild_unavailable` and
`test_rebuildable_stale_derived_advances_and_receipts_ingestion` hold both
directions -- refusing everything is not passing.

## The leg runner

```
python -m tools.cycle_mpas_leg --root RUN --backend replay \
    --history "RUN/history/frame_*.npz" --cycle-seconds 120
```

Order of operations: read anchor N (committed only) -> apply
`analysis_increment.nc` -> rebuild-or-refuse on stale derived -> the
positive residual check -> hash the rehydrated state in the forecast process
-> the three-hash ingestion gate -> advance -> publish anchor N+1 carrying
the ingestion receipt, the residual and `diagnostics_rebuilt`.

Backends:

- `--backend replay --history GLOB` is a **loud stub**. It stamps
  `parent_kind: "replay"` into the anchor and prints
  `REPLAY BACKEND: this leg did not integrate a dycore` on every run. Its
  rebuild capability is deliberately narrow: it knows the equation of state,
  so it can rebuild `exner`; it knows nothing about the port's discrete
  curl, so handed a stale `normal_velocity` it refuses rather than inventing
  one.
- `--backend mpas-cuda --port-root PATH` imports the port's
  `_construct_device_stack` / `_run_steps` / `_rebuild_saved_diagnostics` by
  path. The import is guarded; a failure names every module it wanted and
  every place it looked, and exits 2.

Exit codes: `0` published, `2` backend unavailable, `3` refused.
