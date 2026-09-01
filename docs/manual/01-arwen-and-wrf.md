# 1. What ArWen is and how it relates to WRF-ARW

## 1.1 The relationship in one paragraph

ArWen integrates a WRF-ARW-class compressible nonhydrostatic core (RK3,
split-explicit acoustics, one-way static nesting) in FP32 on CUDA
[README.md:33-34]. Every physics scheme is a transcription of WRF v4.6.1 source
(commit `d66e442f`), and every option carries a machine-readable maturity label in
the physics registry (`gpuwm/physics_registry_v2.json`); the registry, not any prose
page, is the authority, and `tests/test_registry_reachability.py` keeps the two from
drifting [docs/public/PHYSICS.md:3-8]. The model state is FP32, like WRF's default
REAL, and no end-to-end bit-identity with WRF is claimed anywhere
[README.md:480-483]. Where ArWen deliberately diverges from WRF it documents the
divergence in a numbered ledger (section 1.5), and the standard for judging a
divergence is observational skill, not similarity to WRF output.

The WRF reference build used for matched comparisons is WRF v4.6.1 at the pinned
commit, built with GNU gfortran 15.2.0 (WRF `configure` option 34, dmpar; Intel
oneAPI supplies the MPI layer only), run on 48 MPI ranks against a single RTX 5090
GPU run [docs/public/VERIFICATION.md:65-69].

## 1.2 What is kept from WRF-ARW

The core uses a WRF-style hybrid terrain-following, dry-mass vertical coordinate.
Prognostic state includes staggered horizontal wind, vertical velocity, perturbation
potential temperature, dry mass, geopotential, water vapor, and hydrometeors. A
three-stage Runge-Kutta outer integrator wraps split-explicit acoustic steps:
forward-backward horizontal acoustic updates, implicit vertical treatment, and
recoupling to the large step. Parent timesteps divide exactly down the nest tree
[docs/gpuwm-project-history.md:65].

The damping and stability stack is deliberately WRF-shaped rather than a generic
Laplacian safety net: `epssm` off-centering, external-mode divergence damping,
`smdiv`, sixth-order diffusion, Smagorinsky mixing, positive-definite scalar
transport, optional vertical-velocity damping, and the `damp_opt=3` upper sponge
applied to `w` rather than to every state. Lateral boundaries use
specified/relaxation zones with Davies-style weighting. Slow physics tendencies are
held through RK stages in WRF order; microphysics runs after the final stage
[docs/gpuwm-project-history.md:67]. Two solver details that WRF users know as
correctness traps are tracked as verification subjects in their own right: the
dry-mass flux accumulator `mudf` lifecycle across RK stages, and WRF's open-top and
moist pressure-coupling factors `cqu`/`cqv`/`cqw` [docs/gpuwm-project-history.md:71].

WRF's namelist is a first-class input: a WRF namelist imports through the
preprocessing front door, and options ArWen does not implement are refused by name
rather than silently substituted (chapter 5).

## 1.3 The maturity ladder

The registry's raw maturity strings are `wrf-matched-run`,
`wrf-matched-run-candidate`, `supported`, `experimental-runtime`,
`implemented-unverified`, and `planned`; the public physics page renders the first
two as **model-validated** and **validation-candidate** [docs/public/PHYSICS.md,
registry vocabulary note]. This manual uses the page vocabulary. Definitions
[docs/public/PHYSICS.md:14-21]:

- **model-validated**: a matched multi-hour ArWen-vs-WRF forecast of the reference
  case has been run with this option and its decay tables are published.
- **validation-candidate**: executable and gated, with a ratified reference
  comparison, but deliberately not the default.
- **supported**: production option from the longest-certified slice:
  WRF-transcribed, standing unit and runtime gates, exercised by the certified
  reference configurations.
- **experimental-runtime**: executable, carrying a documented runtime restriction or
  an unratified composition; selecting it warns and does not block.
- **implemented-unverified**: runs on the GPU and is column-oracle-measured against
  unmodified WRF Fortran, but no dedicated ArWen/WRF forecast-trajectory comparison
  exists for it yet. The registry records its measured ULP (units in the last
  place) distances and open divergences verbatim.
- **planned / port-in-progress**: not selectable; nothing can resolve to it.

"Certified" in these definitions, and wherever this manual applies it to
physics, is the `gpuwm certify` contract: a run's capsule checked fail-closed against
published bands from a matched WRF comparison of a pinned reference
configuration [docs/public/CERTIFICATION.md]. The *certified slice* is the set
of options those reference configurations exercise; chapter 3's tables name it
scheme by scheme.

The composition rule is machine-readable in the registry (clause C2): a template's
maturity rank does not exceed the lowest maturity rank among the component options it
selects; a composed suite is only as conformant as its weakest member
[gpuwm/physics_registry_v2.json]. `implemented-unverified` is carried by 23 of the
registry's 40 component options [docs/public/PHYSICS.md:32-38], so most of the
physics inventory is at the oracle-measured tier, not the matched-run tier. Chapter 3
gives the per-scheme evidence.

## 1.4 The obs-skill standard for judging divergence

ArWen's governing ruling is model quality over optics: where the physics argument
favors departing from WRF, ArWen departs, and the referee is skill against
observations, not agreement with WRF. Wide-domain runs are graded against MRMS
(`noaa-mrms-pds`, full files) as the verification truth for reflectivity-class
fields, with surface observations (ASOS class) for near-surface state; ArWen's own
multi-radar composite is a feed-space diagnostic, never the grader. Agreement with
WRF remains a verification instrument (the matched-run protocol, chapter 7), because
a transcription that cannot reproduce its source is wrong for uninteresting reasons;
it is not the definition of correct.

## 1.5 The divergence ledger

Deliberate runtime deviations from WRF v4.6.1 are numbered D1-D12 in
`PROVENANCE.md:250`. The standing rule, stated in the physics page:
implement the defined behaviour and document the divergence rather than reproduce an
undefined read, refusing with the Fortran line named where reproduction is refused
[docs/public/PHYSICS.md:367-371].

| id | subject |
|---|---|
| D1 | retired compatibility-mode microphysics / `h_diabatic` cadence |
| D2 | `REFL_10CM` computed on output-due microphysics steps only, where WRF's `nwp_diagnostics == 1` sets `diag_flag` every step |
| D3 | experiment-schema fail-loud rejections |
| D4 | integer-tick clock: WRF-recurrent `dtbc` / running seconds, exact-calendar scope |
| D5 | nest-interpolation (SINT) geometry precomputed FP64-on-host, stored FP32; bitwise-identical to WRF's per-op REAL construction for refinement ratios 1-4, proven in `tests/test_nest_interp.py`; diverges by exactly 1 ULP at ratio 5. Only geometry is precomputed; flux and limiter arithmetic is evaluated per field at force time |
| D6 | `adjust_tempqv` evaluates FP64 on device, stores FP32; the temperature chain cancels ~275 K of magnitude and the Magnus exponential amplifies the residue, so a REAL-internal kernel is irreducibly hundreds of ULPs from any FP64 mirror in qv |
| D7 | Noah sea-ice runtime thermodynamics not implemented; the four init behaviors are mirrored, WRF's separate `seaice_noah` call is not ported |
| D8 | the vertical diffusion of vertical velocity (`vertical_diffusion_w_2`) takes the vertical momentum exchange coefficient where WRF hands it the horizontal one; taking WRF's is unstable at dx 250 m against dz 17 m, and the WRF oracle lane measured the two coefficients equal at all 589,824 points on the idealized case where it can matter (section 2.7) |
| D9 | Thompson aerosol-aware (mp=28) admission, aerosol ingest, lateral boundaries, PBL mixing |
| D10 | LES-nest inflow perturbation, a designed ArWen-over-WRF extension |
| D11 | RUC LSM returns the defined one-layer answer where WRF reads an uninitialised `ilnb` on thin snow, a real WRF defect in which the value depends on grid traversal order |
| D12 | configured initial-state theta bubbles (`[perturbation]`), an ArWen-over-WRF extension |

[PROVENANCE.md:250-1345]

Where WRF's own arithmetic is undefined, each case is decided on its consumers and
published rather than hidden: P3's first-step 0/0 supersaturation is floored to
exactly -1 (fully subsaturated, the intended meaning; default-on in all three arms,
inert from step 2 onward, the step-1 delta declared)
[docs/public/PHYSICS.md:256-264], while Shin-Hong reproduces WRF's own
`prfac2 = 0/0` NaN — every consumer tolerates it — and deliberately does not
perform WRF's `q2xk(kpbl+1)` out-of-bounds read [docs/public/PHYSICS.md:876-878].

Two schema extensions leave WRF-expressible territory on purpose and are named as
such: per-domain `isfflx` (a scalar in the WRF Registry, per-domain in ArWen's TOML,
so such a configuration cannot round-trip back to a namelist)
[docs/public/LES.md:357-362], and per-domain radiation-cadence inheritance, which is
WRF's own namelist guidance made the default rather than a per-nest derivation
(section 3.9).

## 1.6 The arbitrary acceptance test

A design law shapes the input system: adding a future model must be metadata and
table work, not a new code path; a per-model adapter file fails the test. The 2.5.0
initialization engine (chapter 5) is built to that law: the domain wizard reads
cadence, horizon, and coverage off a registry row and names no model in code; a
synthetic registry row drives the real CLI in `tests/test_wizard_sources.py` with no
code added anywhere. Where a per-model fact is genuinely a correctness fact (an
upstream bucket publishing a different product under identical object keys, chapter
5), it is carried as table data, not as a code branch.
