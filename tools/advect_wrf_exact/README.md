# Exact advection verification

The opt-in arithmetic follows the native WRF directional order. Ordinary
transport applies the meridional contribution, the zonal contribution, then
the vertical contribution. Each horizontal spacing coefficient rounds before
it multiplies the flux difference. Vertical wind includes WRF's top-row
extrapolation and one-sided vertical flux.

Positive-definite transport retains each face's hybrid mass averaging,
physical spacing, and Courant-number operations. Its western second-order
face divides velocity by face mass before multiplying the timestep. The
limiter uses the separate base and perturbation masses and applies vertical,
zonal, then meridional tendencies. Low and correction fluxes retain their
separate subtraction order.

Run `compare.py` with `GPUWM_WRF_EXACT=1` and
`GPUWM_WRF_EXACT_ADVECTION=1`. The references are stored words returned by
compiled, unmodified WRF v4.7.1 Fortran, with pinned input and reference
archive hashes. Comparison covers every defined word, including signed
zeros and halo storage. Observing the two optional limiter channels first
requires unchanged ordinary GPU tendency words.

Seven nonradiative fixture cases match all five implemented routines and
both defined optional limiter outputs on the RTX 5090. The radiative-open
momentum and positive-definite routines still differ. Monotonic transport
has no CUDA implementation. Those paths are absent from the specified-domain
twin and are reported separately. No observation-based forecast-skill claim
follows from these routine comparisons.

`limiter_workspace.py` is a causal witness. It adds only final output stores
to the compiled WRF routine and checks that all original total-tendency
words remain unchanged. It also requires a CUDA register observation to
preserve every computed native limiter-scale word.
