# Scale-resolved spectral verification

Arwen has two deliberately separate spectral metric classes.

1. `gpuwm.verify.spectral` is the existing frozen `gpuwm.spectral-pins/v1`
   chaos-envelope metric. It compares radially averaged power for the pinned
   `W` and `REFL_10CM` column reductions. Its pin hash and existing receipts
   must remain unchanged.
2. `gpuwm.verify.spectral_compare` is the additive model-to-reference class.
   It compares arbitrary preregistered scalar fields and horizontal wind in
   physical wavelength bands, including amplitude, signed phase/location
   agreement, error power, and rotational/divergent kinetic energy.

The second class does **not** replace pointwise RMSE, FSS, object verification,
conservation receipts, or the v1 chaos envelope. It answers a different
question:

> At which spatial scales does Arwen carry the right amount and kind of
> structure, and at which scales is the disagreement amplitude, displacement,
> or numerical noise?

## Why this is useful

A gridpoint metric can grade a translated but otherwise realistic storm very
harshly. A power-only spectrum can show that the storm has the right scale and
amplitude, but cannot tell whether the structure is in the same place. The v2
comparison therefore carries both power and a complex cross-spectrum.

For every preregistered wavelength band it reports:

- candidate and reference power;
- power ratio and amplitude ratio;
- signed spectral correlation;
- coherence squared;
- weighted mean absolute per-mode phase error;
- total error power normalized by reference power;
- the part of error attributable to amplitude mismatch; and
- the remaining phase/location error.

For a horizontal-wind field it repeats those quantities for:

- total kinetic energy;
- rotational kinetic energy; and
- divergent kinetic energy.

It also records the divergent-energy fraction on both sides. This is useful for
finding high-wavenumber acoustic/divergent noise that can be invisible in a
single wind RMSE.

## The arithmetic contract

The regional comparison is intentionally not a global spherical-harmonic
transform. Current Arwen domains are limited-area projected grids; treating the
unmodeled remainder of Earth as zero would create a boundary discontinuity and
spectral leakage.

The v2 regional contract is:

1. Read both fields on the same horizontal grid.
2. Apply the preregistered equal-width crop to remove the lateral-boundary or
   relaxation frame.
3. Remove a least-squares plane `a + b*x + c*y`.
4. Apply a separable symmetric Hann window normalized to unit mean square.
5. Compute a float64/complex128 two-dimensional FFT.
6. Keep only `k > 0` through the smaller physical axis Nyquist. This circular
   support disk has complete directional coverage; diagonal-only corner modes
   are not scored.
7. Sum mode contributions in half-open physical wavelength bands.

For a scalar transform `F`, mode power is

```text
|F|² / N²
```

where `N = ny*nx`. Summing all modes therefore closes to the spatial mean square
by Parseval.

For two transforms `A` and `R`, a band carries

```text
P_A = Σ |A|² / N²
P_R = Σ |R|² / N²
C   = Σ A conj(R) / N²
E   = Σ |A - R|² / N²
```

and

```text
power_ratio          = P_A / P_R
spectral_correlation = Re(C) / sqrt(P_A P_R)
coherence_squared    = |C|² / (P_A P_R)
weighted_phase_error = Σ |A conj(R)| |arg(A conj(R))| / Σ |A conj(R)|
```

The signed correlation distinguishes equal-power phase reversal from genuine
agreement. Coherence remains high for a consistent phase offset. The weighted
absolute phase metric is evaluated per Fourier mode, so the conjugate pair of
a real field cannot cancel its phase to zero. Use these diagnostics together,
not one alone.

The minimum possible error if the two band amplitudes were perfectly aligned
is

```text
amplitude_mismatch_power = (sqrt(P_A) - sqrt(P_R))²
```

and the remaining error is recorded as `phase_location_error_power`.

## Quick start

Start from the example:

```text
configs/verify/spectral_compare_example.toml
```

The safe two-command workflow is:

```bash
# 1. This reads only policy. It does not open either wrfout.
gpuwm spectral register \
  configs/verify/spectral_compare_example.toml \
  --output out/spectral-registration.json

# 2. This validates the registration, hashes each input, scores, and receipts.
gpuwm spectral score \
  out/spectral-registration.json \
  --output out/spectral-receipt.json \
  --plot-dir out/spectral-plots

# Audit receipt bytes and, optionally, rehash all source files.
gpuwm spectral check out/spectral-receipt.json --rehash-inputs
```

A convenience command preserves the same publication order:

```bash
gpuwm spectral run configs/verify/spectral_compare_example.toml \
  --registration out/spectral-registration.json \
  --receipt out/spectral-receipt.json \
  --plot-dir out/spectral-plots
```

`run` durably writes the registration before the scoring function opens the
first output byte.

The command is CPU-only. It is intentionally not placed behind the CuPy/GPU
capability gate.

## Source formats

The reader supports:

- WRF-shaped NetCDF files (`.nc`, `.nc4`, `.cdf`);
- `.npz` fixtures with named arrays; and
- one-array `.npy` fixtures.

NetCDF time selection is applied only when the leading dimension explicitly
identifies time. Masked, missing, empty, or nonfinite values are refusals.

### Reductions

A scalar or vector component can use:

- `plane`: the source is already two-dimensional;
- `level`: select the preregistered zero-based vertical index;
- `surface`: select vertical index zero;
- `column_max`;
- `column_max_abs`; or
- `column_mean`.

### Staggering

The reader can remove one C-grid stagger before the vertical reduction:

- `x`: average adjacent west-east points;
- `y`: average adjacent south-north points;
- `z`: average adjacent vertical points;
- `auto`: use a single NetCDF dimension ending in `_stag`; or
- `none`.

For a standard WRF horizontal wind, explicitly set U to `x` and V to `y`.
For W, use `z` if the campaign intends mass-level vertical velocity. The
choice is registration policy and must not be changed after seeing output.

## Recommended first Arwen campaign

Use matched Arwen/WRF history frames on the same grid and score these fields
first:

1. `T` at fixed model levels or pressure-interpolated common levels;
2. horizontal U/V at matched levels, with C-grid destaggering;
3. W at a matched mass level or a preregistered column reduction;
4. pressure/geopotential fields; and
5. optionally vorticity/divergence planes derived by one shared authority.

Reflectivity is useful as a morphology diagnostic, but dBZ power is not
physical atmospheric energy. Keep CSI, FSS, object timing, and neighborhood
verification as the primary reflectivity claims.

For multi-domain campaigns, each `[[pairs]]` record carries its own `dx_m` and
`dy_m`. A physical band that a domain cannot resolve is marked `unresolved`.
Any gate targeting it becomes `INCOMPLETE`, never `PASS`.

## Reading the result

Typical interpretations:

| Power ratio | Signed correlation | Meaning |
|---:|---:|---|
| near 1 | near 1 | correct amount of structure and aligned phase/location |
| near 1 | low | correct power, but displaced/rephased structure |
| below 1 | high | aligned but too weak or over-diffused |
| above 1 | low | excess or misplaced structure; possible noise |
| above 1 in divergent KE only | variable | possible acoustic/split-explicit or boundary noise |

A small reference power can produce an enormous ratio from a physically small
candidate signal. Every row retains absolute powers, and gates may declare a
`minimum_reference_power`. Such a gate becomes `INCOMPLETE` below its floor;
it does not silently skip to green.

## Gates and calibration

There are no built-in empirical thresholds. A source TOML with no `[[gates]]`
records produces an `informational` receipt, not a pass claim.

Use a predeclared known-good population to build a chaos/equivalence envelope,
then derive a candidate gate fragment:

```bash
python tools/spectral_gate_calibrate.py \
  out/member01-receipt.json \
  out/member02-receipt.json \
  out/member03-receipt.json \
  --percentile 95 \
  --output out/calibrated-spectral-gates.toml
```

The calibrator can derive:

- a symmetric log-space envelope around a power or amplitude ratio of one;
- lower-tail signed-correlation and coherence limits;
- an upper weighted mean absolute per-mode phase-error limit;
- an upper nearest-rank normalized-error limit; and
- an upper nearest-rank absolute divergent-fraction difference.

Every source receipt SHA-256 is written into the fragment header. Review and
paste the fragment into the campaign TOML **before** scoring the candidate.
Never include the candidate being judged in the calibration population.

Gate outcomes are fail closed:

- any violated gate: `fail`;
- any missing, unresolved, or low-signal gate target: `incomplete`;
- every evaluated gate inside its preregistered interval: `pass`; and
- no gates: `informational`.

## Receipt identity

A registration binds:

- source TOML SHA-256;
- candidate/reference labels;
- crop;
- physical wavelength bands;
- pair paths, geometry, time indices, domain, and valid time;
- field variables, reductions, levels, and staggering;
- gates;
- v2 arithmetic pins and hash; and
- the existing v1 spectral pin hash.

A score receipt additionally binds:

- full-file SHA-256 and size for every source;
- resolved source dimensions and reductions;
- all scalar/vector comparison rows;
- gate rows and verdict; and
- a self-hash over the complete receipt.

`gpuwm spectral check --rehash-inputs` detects a source file changed after the
receipt was produced.

## Plot contract

Plots are generated only from the validated receipt. They never reopen model
output or recompute a metric. Each PNG receives a SHA-256 in
`plots/manifest.json`.

The plotter emits separate charts for:

- candidate/reference band power;
- power ratio;
- signed spectral correlation; and
- normalized error power.

Vector fields receive one set for total, rotational, and divergent energy.

## Limits and cautions

### A taper changes a pure Helmholtz mode

A nonperiodic regional field needs a taper. Multiplication by the Hann window
convolves neighboring Fourier modes, so a mathematically pure divergent or
rotational test wave acquires a small component in the other partition. The
synthetic controls require strong separation, not impossible post-window
purity.

### This version does not regrid

Both sides must already be on one common grid. Regridding inside the scorer
would hide a second numerical operator under the verification metric. For
science-mode statements across different projections, generate a separately
receipted common-grid product first.

### Model levels are not always common physical levels

Comparing `level_index = 20` is valid only when both outputs share the same
vertical coordinate. For different model configurations, interpolate both
sides to preregistered pressure/height surfaces using one shared authority.

### Regional FFT is not global SH

The linked Spherical Harmonic Exponentials work inspired the broader
scale-space direction, but no graphics reflectance code is copied here. A
future global Arwen/MPAS parent can add a true spherical-harmonic backend under
a separate pin schema. It must not reinterpret these regional receipts.

### Power agreement is not forecast skill by itself

A forecast can have an excellent spectrum and put every storm in the wrong
county. Keep deterministic, neighborhood, object, observational, conservation,
and distributional verification beside this class.

## Future extensions

Additions should use new schemas rather than modifying v1/v2 receipt meanings:

- pressure-level preprocessing with one pinned interpolation authority;
- equal-area science-grid preparation;
- lead-time × wavelength error-growth matrices;
- direct connection to the existing chaos-envelope builder;
- optional CuPy FFT batching after CPU/GPU bit/tolerance controls exist;
- global scalar/vector spherical harmonics for a global parent; and
- offline shadow-mode large-scale nudging diagnostics.
