# Fable handoff: additive spectral verification for Arwen

## Base and scope

Target repository: `FahrenheitResearch/arwen`

Audited base commit:

```text
891a07ea248e6afe3fa6e180cc2ac58a8992a936
ArWen 2.4.1
```

This handoff is an **additive verification lane**. It must not change model
integration, physics, nesting, restart bytes, existing gate thresholds, or the
frozen `gpuwm.verify.spectral` v1 arithmetic.

The existing tree already has:

- `gpuwm/verify/spectral.py` with a committed pin hash;
- `tests/test_spectral.py` with failure controls and the literal v1 hash;
- `gpuwm/verify/chaos_envelope.py` wiring v1 spectral distance into the chaos
  envelope; and
- `tools/matched_wrfout_envelope.py` for registration/build/identity checks.

Do not create a second replacement for that lane. Land this package beside it.

## What the patch adds

```text
gpuwm/verify/spectral_compare.py
    Float64 physical-band scalar and vector arithmetic.
    Adds cross-spectrum, signed correlation, coherence, amplitude/phase error,
    and planar Helmholtz rotational/divergent KE.

gpuwm/verify/spectral_io.py
    Lazy NetCDF + NPY/NPZ readers, fail-closed missing-value policy,
    C-grid destaggering, vertical reductions, full-file SHA-256 identity.

gpuwm/verify/spectral_receipt.py
    TOML preregistration, canonical parameter hash, scoring, self-hashed
    receipts, explicit gate policy, input rehash audit.

gpuwm/verify/spectral_plot.py
    Receipt-only PNGs and SHA-256 plot manifest.

gpuwm/verify/spectral_cli.py
    CPU-only `gpuwm spectral register|score|run|check|plot|pins` front door.

tools/spectral_gate_calibrate.py
    Nearest-rank gate-fragment generator from predeclared known-good receipts.

configs/verify/spectral_compare_example.toml
    No invented gates; real WRF variable/stagger examples.

docs/public/SPECTRAL_VERIFICATION.md
    Public contract, interpretation, workflow, and limitations.

tests/test_spectral_compare_v2.py
    Arithmetic and Helmholtz failure controls.

tests/test_spectral_receipt_v2.py
    Registration/hash/gate/input-identity failure controls.
```

The package also includes a minimal `gpuwm/cli.py` integration patch and a
stage-1 manifest amendment.

## Non-negotiable invariants

1. The current v1 literal remains exactly:

   ```text
   f3d1d17f80742c7114be8984cdfb55c10db117ecaa6e926b3ccc27414834062e
   ```

2. The delivered v2 arithmetic pin remains exactly:

   ```text
   51517e31336c7b415efe4cc969f62f740b003d95bc2ad14ea88f54f7849b0f5c
   ```

   An arithmetic change requires an explicit new pin/schema decision and
   remeasurement; do not merely update the test literal.

3. `gpuwm/verify/spectral.py` and `tests/test_spectral.py` are not edited.
4. `gpuwm spectral` remains CPU-only. Do not add it to
   `capabilities.COMMAND_REQUIREMENTS`.
5. Registration must be durably written before any model-output file is
   opened. The one-shot `run` path must preserve that order.
6. Missing/nonfinite/masked input is a refusal.
7. An unresolved or low-reference-power gate is `incomplete`, not pass.
8. No empirical threshold is introduced without a receipt population and a
   source-hash-bound calibration record.
9. No internal regridding. Both sides must already share a grid.
10. Every numerical statement in docs or a PR description comes from a test,
   receipt, or an explicitly marked estimate.

## Apply order

From a clean checkout at the audited commit:

```bash
git checkout 891a07ea248e6afe3fa6e180cc2ac58a8992a936
git switch -c feature/spectral-verification-v2

git apply patches/0001-spectral-verification-v2.patch
git apply patches/0002-spectral-stage1.patch
```

If main has advanced, rebase first and inspect these seams manually:

- imports in `gpuwm/cli.py`;
- `build_parser()` registration order;
- `_LONG_RUNNING_COMMANDS`;
- `_dispatch()`'s function-dispatched command tuple; and
- the end of `tools/battery/stage1_files.txt`.

Do not resolve a conflict by changing v1 pins.

## Required test sequence

### Fast new controls

```bash
pytest -q \
  tests/test_spectral_compare_v2.py \
  tests/test_spectral_receipt_v2.py
```

Expected from the delivered base package:

```text
22 passed
```

### Frozen-lane regression

```bash
pytest -q \
  tests/test_spectral.py \
  tests/test_chaos_envelope.py
```

This proves the additive lane did not redefine existing receipts.

### CLI and capability regression

Run in an environment with no CuPy extra installed:

```bash
gpuwm spectral pins
gpuwm spectral --help
pytest -q tests/test_cli.py tests/test_cli_capability_refusals.py
```

`gpuwm spectral pins` must succeed without asking for a GPU runtime.

### Stage-1 manifest hygiene

```bash
pytest -q tests/test_stage1_manifest.py
```

### Full CPU lane appropriate to the merge

Use the repository's current stage-1/battery command, not a copied list in this
handoff. The manifest is the authority.

## Synthetic end-to-end proof

The delivered artifact contains `demo/` with:

- `demo.toml`;
- `arwen.npz` and `wrf.npz` fixtures;
- `registration.json`;
- `receipt.json`;
- 16 PNGs; and
- `plots/manifest.json`.

Rebuild it with the patch on `PYTHONPATH`:

```bash
python demo/build_demo.py
gpuwm spectral check demo/receipt.json --rehash-inputs
```

The receipt is intentionally `informational`; the demo does not smuggle in a
made-up pass interval.

## First real campaign

Use the already matched Arwen/WRF output lineage, not unrelated runs. Start
with one domain and one lead.

Recommended initial fields:

```text
T at matched levels
U/V at matched levels, x/y destaggered
W z-destaggered and either fixed-level or column_max_abs
one pressure/geopotential plane
```

Initial workflow:

1. Copy `configs/verify/spectral_compare_example.toml` into the campaign
   evidence directory.
2. Pin exact output paths, valid time, grid spacing, crop, fields, reductions,
   and wavelength bands.
3. Commit or otherwise timestamp/hash that TOML before reading scores.
4. Run `gpuwm spectral register`.
5. Run a known-good tiny-perturbation population through `score`.
6. Use `tools/spectral_gate_calibrate.py` only on those known-good receipts.
7. Review the fragment and register the candidate campaign with those gates.
8. Score the candidate.
9. Keep RMSE/FSS/object/conservation receipts beside the spectral receipt.

Do not begin with reflectivity gates. dBZ spectra are morphology diagnostics,
not kinetic-energy spectra.

## Review checklist by file

### `spectral_compare.py`

- Parseval closes at the pinned tolerance.
- Identical fields have zero error and unit agreement.
- Scaling by two gives power ratio four and amplitude ratio two.
- Sign reversal keeps power ratio one but signed correlation minus one.
- Added high-wavenumber structure lands in the short band.
- Total vector mode energy closes to rotational plus divergent energy.
- A windowed pure mode strongly, but not impossibly perfectly, separates into
  the correct Helmholtz component.
- Bands reject overlap and duplicate names.

### `spectral_io.py`

- NetCDF import remains lazy.
- `.npy` with a requested variable refuses.
- multi-array `.npz` without a key refuses.
- masked/nonfinite values refuse.
- time selection only consumes a dimension explicitly named as time.
- auto destagger refuses more than one staggered dimension.
- field rank after time/destagger is exactly 2-D or 3-D.

### `spectral_receipt.py`

- registration does not `stat()` output;
- resolved paths are bound without requiring existence;
- parameter hash covers implementation pins, v1 pin identity, bands, fields,
  pairs, and gates;
- scoring hashes each unique source once;
- receipt self-hash covers every comparison and gate row;
- input rehash finds changed bytes;
- no gates means informational;
- missing/unresolved target means incomplete;
- violated gate means fail.

### `spectral_cli.py`

- `run` writes registration before calling score;
- command is added to the normal `.func` dispatch tuple;
- command is not GPU-gated;
- exit 1 is used for `fail` or `incomplete`, not for informational evidence.

## Last-mile items for the Fable agent

The delivered patch intentionally does not guess campaign integration details
that only the live tree/run inventory can prove. Complete these in-repo:

1. Add a link to `docs/public/SPECTRAL_VERIFICATION.md` from the current
   verification index if the index has changed since the audited commit.
2. Confirm setuptools package discovery includes all new `gpuwm.verify`
   modules in both wheel and sdist.
3. Run the no-CuPy publish-job environment, not merely
   `GPUWM_NO_LOCAL_GPU=1` on a machine where CuPy still resolves.
4. Decide whether the real campaign stores registrations/receipts under the
   existing public receipt tree or a run-private evidence directory.
5. Bind the real campaign to the current evaluator Git commit in its enclosing
   evidence document. The generic receipt binds arithmetic and inputs; the
   campaign record should additionally bind the repository commit.
6. Measure wall time and peak memory on d01/d03 frames before claiming cost.
7. Only after CPU results are frozen, consider a separate optional CuPy batch
   backend with a CPU-vs-GPU tolerance/identity receipt. Do not make it a
   prerequisite for this merge.

## Future work, not part of this patch

- pressure-level/equal-area preprocessing receipts;
- lead-time × wavelength matrices;
- direct scheduled integration with the chaos envelope;
- global scalar/vector spherical harmonics for a global parent;
- shadow-mode scale-selective nudging diagnostics; and
- GPU FFT batching.

Each requires a new pin/schema identity. None should mutate v1 or reinterpret
v2 receipts.
