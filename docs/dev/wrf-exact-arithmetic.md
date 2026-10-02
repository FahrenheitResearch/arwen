# Strict arithmetic verification

Set `GPUWM_WRF_EXACT=1` before starting a process to select WRF verification
arithmetic. It is off by default. Default runs retain their existing compiler
options and compiled-image cache keys.

The verification compiler disables FMA contraction and fast math, preserves
subnormals, and selects precise division and square root. These choices follow
the strict WRF reference build's `-fno-fast-math -ffp-contract=off` arithmetic
constraints. They do not by themselves prove routine or forecast identity.

The override is applied before CuPy computes the cache key and at the final
NVRTC program boundary. The latter replaces CuPy's appended FTZ option and also
covers direct NVRTC loads and generated array kernels. Exact images use a
distinct cache key. Start a new process to change verification mode.

Set `GPUWM_WRF_EXACT_COMPILE_RECEIPT` to an output JSON path to retain the
compiler receipt. Cache requests and actual NVRTC compilations are listed
separately with source hashes and effective options.

The native WRF-file door preserves the file's `T = theta - 300 K` words in
strict mode. The matching reconstruction offset is 300 K; dry base pressure,
inverse density and geopotential retain their imported carriers. WRF history
writes the stored perturbation directly. Other initialization and restart
doors have not yet been qualified with this verification representation.

The acoustic path follows WRF's operation order: rounded half-level
geopotential, separate wind tendencies, total-flux continuity before reference
subtraction, perturbation-theta transport, rounded reciprocals, and heating
removal before adding the reference numerator. Specified-frame updates are a
separate driver operation, as in WRF. Damping uses the scalar library sine
implementation. All nine acoustic routine groups match the compiled WRF
oracle across the retained fixture corpus. Whole-forecast identity remains
a separate measurement.

Additional experiment switches select separately measured controls:
`GPUWM_WRF_EXACT_BIGSTEP=1`, `GPUWM_WRF_EXACT_ADVECTION=1`,
`GPUWM_WRF_EXACT_DIFFUSION=1` and
`GPUWM_WRF_EXACT_DIAGNOSTICS=1`. They require `GPUWM_WRF_EXACT=1` and must be
set before import. Their presence is not a claim of routine identity.

The big-step control restores pressure-force interpolation, Coriolis and
curvature sums and tendency application, native PH/PHB geopotential differences,
and stored north-south map reciprocals. It also preserves canonical theta
through lateral-boundary coupling. The pressure-gradient fixture comparison
uses pressure normalized through full storage until the diagnostic control
retains the original perturbation pressure. That representation seam is
reported separately from exact operator comparisons on identical supplied words.

The diffusion control restores WRF's rounded geometry, reciprocal density,
strain and mixing-coefficient products, and separately staged stress/flux
divergences. It also selects WRF's tensor donor coordinates and stability
boundary mask. These last two controls cover independently reproduced boundary
defects, so the measured diffusion stage includes their effect as well as
arithmetic order. Defaults retain the frozen comparison behavior.

All seven deformation outputs, stability, km_opt=4 coefficient outputs and
horizontal wind/theta/vapor mixing outputs match the retained compiled WRF
arrays across 14 fixtures, using geometry computed by the production kernels.
This does not qualify the deliberate constant-coefficient or km_opt=2 vertical
operator choices, which are inactive in the configured twin.

The advection control follows WRF's separately rounded centered and dissipative
stencils, directional map spacing and tendency writes, hybrid face masses,
Courant operations and limiter recombination. All five implemented routine
totals and the defined optional positive-definite horizontal/vertical outputs
match compiled WRF across seven periodic or specified-boundary fixtures.
Radiative-open discrepancies and the absent monotonic option remain outside
the qualified native twin configuration.

The diagnostic control restores WRF's hypsometric options 1 and 2, moisture
and scalar-library EOS order, and original perturbation-pressure storage.
Physics receives full pressure reconstructed from that stored value; the
pressure force and history receive the stored perturbation. It matches all
P, full-P, AL and ALT words from unchanged compiled WRF in eight real/perturbed,
wet/dry diagnostic cases. The retained pressure carrier requires one additional
mass-grid float32 array in verification mode.

The default keeps the documented cancellation-resistant layer geometry and
log-pressure thickness. Selecting the reference formula here is an experiment,
not a change to the default conditioning improvement. Native pressure storage
loses bits in the near-zero operator probe; the realistic initialization
fixtures show no such pressure round-trip loss. That probe alone does not
attribute the forecast gap to pressure representation.
