# Level-2 regional spectral numerics — integration handoff record

Lane `lane/spectral-level2`, integrated 2026-08-17 off integration tip
`0f45dcfec`.  Post-cut material: this lane is not part of the 2.5.0 merge
wave; the coordinator merges it after the release.

## What the delivery actually contained

Package `arwen-level2-regional-spectral-numerics` (Downloads), SHA256SUMS
verified, 42/42 files OK.  Two findings the next reader needs:

1. **The combined patch was empty.**
   `patches/arwen-level2-combined-from-current-head.patch` is zero bytes,
   and the package's own `ARTIFACT-MANIFEST.json` records it that way
   (the sha256 is the empty-string digest).  Everything the prompt said
   the patch carried — `docs/public/LEVEL2_SPECTRAL_NUMERICS.md`, the
   two handoff documents, the delivered seam survey, the survey tool,
   `tests/test_spectral_operators.py`, `tests/test_spectral_operator_cli.py`
   — was therefore **not delivered**.  The operator sources survived in
   `repo-overlay/gpuwm/spectral_ops/` (17 `.py` files, plus `__pycache__`
   which was not adopted) and the four `demo/` artifacts.
2. **The delivery's own container never ran its suites.**  Its
   TEST-RESULTS.md shows the CLI suite failing at collection
   (`cannot import name 'cli' from 'gpuwm'`).  The suites in this repo
   are the first real runs of the operators.

The missing surfaces were reconstructed here: the two test suites were
written fresh against the delivered non-negotiable contracts, the survey
tool is `tools/spectral_seam_survey.py`, the survey record is
`docs/handoffs/CURRENT-CORE-SPECTRAL-SEAM-SURVEY.json`, and the public
doc is `docs/public/LEVEL2_SPECTRAL_NUMERICS.md`.

## The seam, as it exists today

The delivered survey was lost with the patch, so the seam was re-derived
from the live tree (the tip had moved through the 2.5 reset; nothing to
diff against, so this record IS the baseline):

- **Authoritative slow-large-step commit point:**
  `gpuwm.core.model.execute_experiment`, the STEP op's `on_step` closure.
  The stepper (`dycore.step` itself, or a `StreamedDomain` with the same
  signature) returns with the RK slow-mode state committed; the very next
  statement is `refresh_model_time(node.state, clock, after_step=True)`.
  The `SpectralLargeStepHook` call sits immediately after that refresh,
  inside the same domain turn — once per domain per model time step,
  never in an acoustic substep (those live entirely inside `dycore.step`),
  before health validation, output, nest feedback and the next large step.
  Step numbering matches `step_observer`: the committed step is
  `clock.step_count + 1`.
- **Wiring:** `gpuwm/spectral_seam.py` (`SpectralSeam`, `attach_seam`,
  `seam_capsule_receipts`).  Attached to the model object so spawn- and
  restart-leg walks that re-enter `execute_experiment` keep one receipt
  ledger.  Hooks are built lazily per grid (spawn-born nests get theirs
  at their first committed step) with `dx/dy/dt` from that domain's
  `RunConfig`.
- **Routes that honor the config** (they integrate through
  `execute_experiment`): `runtime.run_experiment:domain-tree`,
  `prepared_single_domain_forecast`, `prepared_domain_tree_forecast`.
- **Routes that refuse an active config** (they commit slow steps in the
  frozen single-domain loop, outside the wired seam):
  `runtime.run_experiment:single-domain`, the ensemble member leg.
  Refusal: `gpuwm.experiment.refuse_unrouted_spectral_numerics`, the
  `[perturbation]` honored-or-refused governance applied unchanged.
  `mode = "off"` passes everywhere — off is the absence of the operator.
- **Supervisor note:** `supervisor:success` re-emits a capsule for the
  run its worker completed; the worker's own capsule carries the spectral
  receipts.  Threading them through the supervisor's re-emission is left
  to the pin-owner if wanted.

Drift control: `python -m tools.spectral_seam_survey --check` compares
the live tree's structural anchors against the committed survey record,
and `tests/test_spectral_seam.py` pins the source order (stepper call →
after-step refresh → hook → poison) inside `on_step`.

## Contract placement

- Arithmetic contracts (off reads no state; shadow computes receipts but
  is state-bitwise inert; apply mutates only after every field and budget
  passed; periodic requires the declaration; C-grid nonperiodic outer
  faces get zero increment) live in the vendored operator package and are
  held by `tests/test_spectral_operators.py` (46 tests).
- Seam contracts: streamed-domain refusal and the false-periodic-
  declaration refusal at attach; the restart/config identity binds the
  resolved operator config when the table is present and stays absent
  when absent (pre-feature fingerprints byte-identical); per-domain step
  receipts ledger into every emitting route's certification capsule
  (`receipts.spectral_numerics`, with a receipt hash chain per domain);
  a completed **apply** run with missing receipts refuses a clean
  completion capsule.  Held by `tests/test_spectral_seam.py` (24 tests).
- Front door: `gpuwm spectral-op` (pins / benchmark / response / check /
  calibrate), CPU-reachable without CuPy, held by
  `tests/test_spectral_operator_cli.py` (7 tests, including validation
  of the construction checkout's own shadow-step receipt).

The committed operator pin hash is
`549502b5f1b66fff4dda949ba5a16cfb9ed71bb52877c2ef2f395d36c031c2ad`
(pinned literally in the operator suite).  The frozen Level-1 spectral
pins (`gpuwm/verify/spectral.py` / `tests/test_spectral.py`) are
untouched and their suite is green.

## Measured limits of the delivered arithmetic (recorded, not redesigned)

- The realized Helmholtz split keeps ~6% divergence-RMS leakage on
  white-noise input: the coefficient-space projector is exact (1e-17),
  but realizing the parts through `irfft2` loses the anti-Hermitian `ky`
  content of the `kx = {0, Nyquist}` columns.  `damp_divergence`'s
  receipt recomputes realized divergence, so receipts stay honest.
  Pin-owner item if tighter separation is ever needed.
- A tapered nudge's outer-edge increment is the full relaxed raw
  mismatch (the taper preserves its operand at the edge); the receipt
  reports it as `boundary_max_abs` rather than claiming zero.

## Validation state (ladder steps 1-7 of the delivered prompt)

1. CPU operator + CLI suites: **written here and green** (48 + 7).
2. Frozen Level-1 spectral/chaos suites: **green, pins unmoved**.
3. Touched config/restart/capsule/CLI/stage-1 suites: **green**
   (`test_experiment`, `test_clock`, `test_config_freeze`,
   `test_certification_capsule`, `test_restart`,
   `test_checkpoint_route_contract`, `test_stage1_manifest`,
   `test_cli`, `test_docs_extras_agree_with_code`, case-token and EOL
   gates).
4. No-CuPy venv: `gpuwm spectral-op pins/check/calibrate` exit 0.
5. **CuPy parity and benchmark on a real card: PASS.**
   `tools/spectral_gpu_parity.py` on the local RTX 3080 (sm_86, CuPy
   14.0.1 / CUDA 13.0) at the battery's 49 x 400 x 480 3 km shape.  Every
   operator -- hyperdiffuse across tapered/reflect/periodic and log
   space, `damp_divergence`, `damp_c_grid_divergence`,
   `solve_helmholtz` -- lands at 0.2%-5.2% of its round-off bound in
   float32 and 0.4%-4.6% in float64.  Receipt scalars agree to 1e-15
   relative.  Parity is NOT bitwise and is not claimed to be: the pins
   say only that transform arithmetic follows the input/backend FFT
   dtype, and pocketfft and cuFFT are different implementations.  Bound
   held: `eps(dtype) * sqrt(ny*nx) * rms(field)`.
   Receipt: `evidence/spectral-level2/gpu-parity-3km-rtx3080.json`.
6. **One-domain shadow forecast: bit-identity PROVED.**  Real HRRR
   pressure-level bytes, cycle 2026-08-17 17Z, d01 142 x 114 x 49 at
   12 km, 60 committed steps, through `gpuwm sim` ->
   `prepared_single_domain_forecast`.  Three arms -- feature ABSENT, mode
   OFF, mode SHADOW -- prepared separately (the single-domain proof binds
   the experiment config, so each arm gets its own preparation; the three
   prepared trees are byte-identical in all 119 data arrays and differ
   only in the three receipts that record the config digest).  All three
   land on ONE canonical state digest over 138 arrays
   (`f1abd030dcfb2a57...`) and byte-identical wrfout frames.  Shadow wrote
   60/60 receipts, `complete: true`.
7. **Multi-domain shadow forecast: bit-identity PROVED across the
   chain.**  The wizard's 12-3 km nested ladder (d01 142 x 114 at 12 km
   radt 1.0 -> d02 128 x 112 at 3 km radt 1.0 INHERITED, which is the
   shape the radt-floor fix produced), through
   `prepared_domain_tree_forecast`.  Eight runs -- absent x3, off,
   shadow x3, shadow-without-on-disk-receipts -- all land on ONE pair of
   per-domain trajectory digests, and all seven wrfout frames are
   byte-identical.  Receipt ledger d01 60/60, d02 240/240,
   `complete: true`, 300 receipt files on disk, all 360 receipts across
   both proofs validated by `gpuwm spectral-op check`, every one
   `backend: cupy`, `applied: false`, and NONE with an exact-zero
   increment.
8. Applied A/B campaign: **not run here, deliberately.**  It is the
   post-cut science program's opening act.

## Two defects the live runs found, and their fixes

1. **The seam never fired on either prepared route.**  `attach_seam`
   read the experiment out of `model._activation_context`, which only
   `build_experiment_state` sets; both prepared runners construct
   `ExperimentState()` directly.  The first shadow tree run wrote ZERO
   receipts, bound no `receipts.spectral_numerics`, and still emitted a
   clean PASS capsule -- an operator that was neither honored nor
   refused.  `execute_experiment` now takes the resolved experiment as a
   keyword-only argument and treats it as the authority; all four
   forecast call sites pass it, and an `ast` gate over every call site in
   the package fails closed for a route added later.
2. **A nested chain's receipt files overwrote each other.**  Every domain
   commits step N and the file was named by the step alone, so 300
   computed receipts left 240 files (59 of them d01's).  The domain now
   leads the file name; a call with no source identity keeps the bare
   spelling.  The operator pin hash is untouched -- PINS describes
   arithmetic, and a file name is not arithmetic.

## Measured cost of the hook (RTX 3080)

Once per domain per slow step, cadence 1, one scalar target (`thp`) plus
C-grid wind, tapered boundary, 8 taper cells.  Median slow-step wall time
from the runners' own `Timing for main:` lines, median of three runs per
arm:

| domain | absent | off | shadow | hook | % of a measured slow step |
| --- | --- | --- | --- | --- | --- |
| d01 142 x 114 x 49, 12 km | 104.10 ms | 102.01 ms | 111.15 ms | 7.05 ms | **6.8%** |
| d02 128 x 112 x 49, 3 km | 39.90 ms | 40.35 ms | 51.09 ms | 11.19 ms | **28.0%** |

Whole two-domain forecast hour: 25.0 s absent, 27.4 s shadow (+9.7%).
`off` is inside run-to-run noise (-2.0% and +1.1%), which is the OFF
contract measured rather than asserted: with `mode = "off"` the seam is
never built, so the STEP op pays one `is not None` test.

An in-process benchmark of the same hook at the same shapes agrees to the
same order: 11.18 ms median compute at d01's shape, 8.18 ms at d02's, plus
0.27 ms and 1.54 ms respectively when the step also writes its receipt
file.

At the battery's much larger 49 x 400 x 480 shape with two scalar targets
plus wind the hook costs 64.3 ms on this card against 3.13 s on numpy --
a 48.7x backend speedup -- so the fraction above is a per-shape number,
not a constant.

## Still open for the pin owner

- The applied A/B campaign (step 8) and every meteorological claim.
- The realized Helmholtz split's ~6% divergence-RMS leakage and the
  tapered nudge's outer-edge increment, both recorded above.
- `supervisor:success` re-emits a capsule for its worker's run; threading
  the spectral receipts through that re-emission is unfinished.
- The HRRR **native** single-domain route (`--source hrrr`) generates its
  own `experiment.toml` from the domain spec and namelist and binds it in
  the portable source manifest, so a user cannot put `[spectral_numerics]`
  in it at all.  The operator has no front door on that route; the
  pressure-level route (`--source hrrr-prs`) and every other
  `--experiment-config` route do.
- Nothing in Level 2 is meteorologically admitted by any of this.  The
  state is: arithmetic implemented and tested, runtime seam wired AND
  proved to fire on the routes that promise it, shadow bit-identity
  proved on one domain and across a nested chain, GPU parity and cost
  measured, applied scientific gates NOT run.
