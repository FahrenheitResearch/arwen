# Compiled WRF big-step coupling reference

WRF v4.7.1, commit `f52c197ed39d12e087d02c50f412d90d418f6186`.
The complete `module_big_step_utilities_em.F` SHA256 is
`bd177b6b5ba7949cf9e694d7ad654fd9ae2f07d39d85802f0716c5318889a815`.
The complete `module_model_constants.F` SHA256 is
`5b80377fecdc18a5f0ad38d3b6c15cfc86ad5d76701adbbbb08a08698d0f7062`.
The builder refuses either source if its bytes change. Each extracted routine
body is unchanged and separately hashed in `coupling-receipt.json`.

The harness calls `calc_cq`, `calc_mu_uv`, `calc_ww_cp`, `calc_php`, `w_damp`
and `rk_rayleigh_damp` with their complete native argument lists. It imports
WRF's generated configuration type and Registry scalar index from the real
compiled module interfaces. The complete model-constants source is compiled
with WRF's EM and REAL32 defines. Logging and display-name services are the
only substitutes. Numerical code, dimensions and boundary decisions are WRF's.
The reference uses gfortran 15.2, `-O0`, `-ffp-contract=off` and bounds checks.
The receipt rejects vector libm symbols.

Nine cases use the unchanged 49-level real-state crop in `state-real.npz`:
the original state with periodic halos, clamped boundary halos, map factors
from 0.5 to 2, a terrain displacement reaching 2500 m, zero values, normal
near-zero values, a dry-mass perturbation, and two CFL threshold families.
The threshold cases retain adjacent IEEE words around Courant numbers 1 and
2 and both signs of vertical velocity. No random states are used.

`coupling.npz` contains the Fortran output words. `coupling-receipt.json`
pins every input/output stream and every packaged output array. Array halos
outside the routine's defined output domain are retained in the raw stream
receipt; packaged arrays contain all defined output rows. Raw `cqw` is saved
alongside `cqwr`, the reciprocal representation consumed by the engine.
The Fortran `calc_php` output is also saved. `php_ru` and `php_rv` are clearly
separate, supplementary consuming responses formed from that output for an
isolated actual `advance_uv` launch with a 1 Pa acoustic mass perturbation.

`coupling-baseline.json` records every port output hash and its exact ULP,
changed-word and absolute-distance measurements against Fortran. These are
measured fixture bounds, not mathematical bounds for arbitrary states and
not adjustable tolerances. The tests assert the complete output hashes and
the exact measured tables. `w_damp` instead asserts bit identity directly.

The observed differences are:

* `calc_mu_uv`: 1 ULP in the dry-mass perturbation case. The engine adds
  each column's base and perturbation mass first, while WRF adds the two
  perturbations before the two base values. This representation is declared
  in `gpuwm/core/ieva.py`.
* `calc_cq`: all consumed interior factors are bit identical. Its clamped
  boundary case differs only on unused horizontal boundary-normal scratch
  faces, which the engine deliberately fills with periodic duplicate values.
  `gpuwm/core/kernels/acoustic.cu` declares that choice. The raw Fortran
  vertical mixing-ratio temporary is represented as its consumed reciprocal.
* `calc_ww_cp`: maximum 2271232 ULP in a near-zero cancellation, with an
  absolute difference of `1.7048861883636507e-26`. WRF multiplies the v flux
  by the stored inverse map factor; the engine divides by the map factor.
  The dry-mass addition grouping also differs. Supplying WRF's float32 flux
  words to the real engine column scan makes every Omega word bit identical
  on all nine cases. This isolates the difference to coupling arithmetic.
* `calc_php`: maximum 8388608 ULP in the isolated consuming x response;
  maximum absolute response difference `6.738584488630295e-6` across x/y.
  WRF stores and rounds full half-level geopotential before taking a face
  difference. The fused engine consumer takes separate perturbation/base
  face differences and then adds them. The ordering is documented in
  `acoustic.cu`; an exact operator trace reproduces all engine response words.
  A standalone engine half-level output array does not exist.
* `w_damp`: bit identical for its full tendency array and both returned CFL
  scalars after the default-on rounding correction. A fused hybrid mass
  crossed the strict activation threshold before the correction; the CFL
  diagnostic had the same arithmetic. The earlier contraction allowance in
  `tests/test_w_crit_cfl.py` is retired and its four compiled cases now assert
  bit identity too.
* `rk_rayleigh_damp`: DIFFERENT, an explicitly absent `damp_opt=2` tendency
  path. `gpuwm/core/diffusion.py` documents that its existing implicit
  `damp_opt=3` utility is not this WRF routine. All four Fortran tendency
  outputs are saved and the absent path's zero-output effect is measured;
  this is not a numerical parity claim.

Rebuild with `coupling_build.py WRF_SOURCE WRF_BUILD BUILD`, then run
`coupling_fixture.py FIXTURE_DIRECTORY BUILD` with the engine source on
`PYTHONPATH`. Run `pytest -q tests/test_bigstep_coupling_wrf471_parity.py`
through the required GPU ownership protocol. The separate measurement tool
uses the engine's own CUDA source loader and production launch functions.
