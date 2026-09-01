# Level-2 regional spectral numerics

Optional scale-selective spectral operators for a running forecast:
exact exponential hyperdiffusion of chosen scalar fields, divergent-mode
damping of the C-grid wind, applied (or only proposed) once per completed
slow large step.  Default **off**, and off is bitwise inert — a config
that never mentions `[spectral_numerics]` runs the bytes it always ran.

The subsystem is research instrumentation with a hard audit trail, not an
admitted physics change.  Nothing here is meteorologically endorsed until
the applied A/B campaign runs and passes its gates; until then `shadow`
is the mode that answers questions.

## Modes

| mode     | reads state | writes state | receipts |
|----------|-------------|--------------|----------|
| `off`    | no          | no           | none     |
| `shadow` | yes         | **no** (bitwise inert) | every cadence step |
| `apply`  | yes         | yes, only after every field and budget passed | every cadence step |

The hook fires once per domain per model time step, immediately after the
RK slow-mode state commits — never inside acoustic substeps, before
output and nest feedback.

## Configuration

```toml
[spectral_numerics]
mode = "shadow"              # off | shadow | apply
cadence_steps = 1            # fire every Nth large step
boundary = "tapered"         # tapered | reflect | periodic
edge_taper_cells = 12
# receipt_directory = "out/spectral-receipts"   # optional JSON per step
# maximum_scalar_rms_increment = 0.05           # budgets: refuse before
# maximum_wind_rms_increment = 0.05             # any state mutation

[[spectral_numerics.scalar]]
field = "thp"
space = "linear"             # or "log" with a positive floor
[spectral_numerics.scalar.diffusion]
order = 3                    # 1 = Laplacian, 3 = sixth order
reference_wavelength_m = 18000.0
e_fold_time_s = 450.0
protect_wavelength_m = 72000.0

[spectral_numerics.wind]
enabled = true
staggering = "cgrid"
[spectral_numerics.wind.divergent]
order = 2
reference_wavelength_m = 12000.0
e_fold_time_s = 300.0
protect_wavelength_m = 48000.0
```

Unknown keys refuse at load.  `boundary = "periodic"` additionally
requires `periodic_domain = true`, and the declaration must be true: a
domain with open, specified or nested lateral boundaries refuses the
wrap at run start.  `tapered` leaves every outer-edge cell exactly
unchanged; C-grid outer faces of a nonperiodic domain receive zero wind
increment, preserving externally supplied boundary values.

## Where it runs, and where it refuses

Honored on the routes that integrate through the shared executor:
`gpuwm run` with a domain tree, and both prepared runners.  Refused
(loudly, at start) with an active mode on the frozen single-domain loop
and the ensemble member leg, and on any tile-streamed domain — a
streamed domain's state object is the attach-time snapshot, so the hook
cannot see the forecast planes.  Set `mode = "off"` or run resident.

## Identity, receipts, capsule

- A present `[spectral_numerics]` table binds the restart/config
  identity value for value; an absent table stays absent, so existing
  fingerprints and checkpoints are untouched.
- Every cadence step yields a hash-bound receipt embedding the operator
  pin hash and the resolved config hash.  With `receipt_directory` set
  they are also written to disk atomically.
- The run's certification capsule carries a `spectral_numerics` receipts
  block (per-domain step and receipt counts plus a receipt hash chain).
  A completed `apply` run with missing receipts refuses a clean capsule.

## CLI

```
gpuwm spectral-op pins        # immutable arithmetic identity + hash
gpuwm spectral-op response --reference-wavelength-m 18000 \
    --e-fold-time-s 450 --dt-s 60          # exact per-call response table
gpuwm spectral-op benchmark --backend numpy   # analytic controls + timing
gpuwm spectral-op check RECEIPT.json          # validate a step receipt
gpuwm spectral-op calibrate --input bands.json --output proposal.json \
    --dt-s 60   # damping-only proposal from Level-1 band power ratios
```

All of it is CPU-reachable on an install without CuPy.  The calibrate
output is a proposal, never a configuration change: it damps observed
power excess only, amplifies nothing, and names the shadow-then-A/B gate
it still has to pass.
