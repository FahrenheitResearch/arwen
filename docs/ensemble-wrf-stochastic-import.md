# WRF stochastic namelist import

Public ensemble and native-input doors admit connected SPPT, SKEBS and the
GF (`spp_conv`), MYNN surface/PBL (`spp_pbl`) and RUC (`spp_lsm`) consumers.
An active run prints a loud warning and records it in source-selection,
runtime-authority, stochastic checkpoint and ensemble receipts:
`uncalibrated spread, WRF default amplitudes`. Explicit parameter overrides
remain recorded in the process configuration. This warning describes spread
whose forecast-error calibration has not been established; it does not prevent
execution. No observation-based skill or calibrated probability claim follows
from a successful run.
All-off controls retain the ordinary path, imported TOML bytes and state words.
WRF `nens` is a random-stream label and never sets the ensemble member count.

For example, an experiment can declare:

```toml
[ensemble]
members = 2
base_seed = 29
[ensemble.stochastic]
sppt = true
skebs = true
[ensemble.stochastic.spp]
conv = 1
pbl = 1
lsm = 1
```

SPP still requires the corresponding GF, WRF MYNN and RUC physics selectors.
Unsupported combinations refuse with the missing consumer's name. Native
`rand_perturb`, `multi_perturb`, boundary and `pert_*` selectors have no runtime
consumer for their requested increments. They refuse because their fields
would stay unperturbed while the receipt claimed the selector. Turning the
selector off is the remedy; changing an amplitude cannot supply its consumer.

The shipped uniform-wind initial-condition kernel is also available through
`[ensemble.perturbation]`, `kind = "uniform-wind"`, `amplitude = 0.5` (m/s).
It draws one seeded uniform increment per wind axis and member, applies it
after physics initialization and before the first step, and records the seed,
amplitude and kernel identity. This is an illustrative engine amplitude,
separately labelled in the receipt; it does not implement WRF `rand_perturb`.
Reconstruction and restart retain advanced states rather than applying it twice.

Ordinary concurrent members may have different physics configurations on one
card. Each keeps its own original model, clock, stream and allocator. Admission
charges the largest complete member envelope for each live member. Native
component packs retain their separate compatibility checks.

`tests/test_ensemble_random_admission_gpu.py` checks two members on a
16x12x40 idealized case for three 12-second steps: SPPT, SKEBS, each connected
SPP consumer and uniform-wind initialization. It checks seed replay, finite
state, committed production receipts and an all-off byte control. Its concurrent
WSM6/six-soil-layer and Thompson mp8/nine-soil-layer members each match their
own sequential forecast words. The fixture supplies initialization directly;
it does not qualify real-data preparation, rendered products or forecast skill.

The mapping follows WRF v4.6.1's [Registry](https://github.com/wrf-model/WRF/blob/v4.6.1/Registry/registry.stoch)
and [stochastic configuration checks](https://github.com/wrf-model/WRF/blob/v4.6.1/share/module_check_a_mundo.F).
SKEBS clipping follows its original [T/U/V update calls](https://github.com/wrf-model/WRF/blob/v4.6.1/dyn_em/module_first_rk_step_part2.F#L157-L214).

| WRF controls | Engine controls |
| --- | --- |
| `sppt`; `gridpt_stddev_sppt`, `stddev_cutoff_sppt`, `lengthscale_sppt`, `timescale_sppt` | SPPT enable, standard deviation, cutoff, length in metres, time in seconds |
| `skebs`, obsolete `stoch_force_opt`; `tot_backscat_psi/t`, `ztau_psi/t`, `rexponent_psi/t` | Both SKEBS processes with their original backscatter, time and exponent |
| `gridpt_stddev_sppt`, `stddev_cutoff_sppt` with SKEBS enabled | Also set SKEBS clipping, even when SPPT is off, following WRF's original T/U/V update calls |
| `kmin/maxforc`, `lmin/maxforc`, corresponding `forct` controls | Shared k/l bounds per SKEBS process; unequal bounds refuse |
| `spp_conv/pbl/lsm` and their standard deviation, cutoff, length and time controls | GF, MYNN and RUC parameter patterns; the selected original physics must contain the consumer |
| `spp = 1` on any domain | Enables all three SPP consumer arrays, following WRF's configuration check |
| `nens`, `iseed_sppt`, `iseed_skebs`, `iseed_spp_conv/pbl/lsm`, `iseed_rand_pert` | Complete signed integer seed-label authority in provider receipts and checkpoints |

Per-domain arrays retain Registry defaults for omitted elements. SPPT/SKEBS
forcing controls must resolve uniformly across domains in the current provider.
SPP switches stay per domain; each selected consumer uses the common process
parameters, and parameters on off domains are inert. Vertical structure must be
0. Unimplemented random-field,
multi-perturbation and boundary consumers refuse by name. SPPT/SPP retain WRF's
restriction on temperature wavenumber controls. Spectral restart reseeding
(`hrrr_cycling = .false.`) refuses because it would replace a complete spectral
history. The registered `zsigma2_eps/eta` coefficients have no WRF consumer;
WRF computes that noise variance from the accepted time step.

The packaged WRF v4.7.1 vocabulary has the same types, scopes and resolved defaults
for all 94 `&stoch` entries declared in WRF v4.6.1's `registry.stoch`.

The engine uses keyed Philox and cuFFT. It does not reproduce WRF's
date-dependent compiler RNG or WRF realization bytes. Explicit WRF labels use
`gpuwm-wrf-process-seed-labels.v1`: the leading 64 SHA256 bits of the derivation
label, original global member seed, process name, `nens`, and that process's
`iseed_*`, separated by colons. The two SKEBS components keep separate process
names. Labels do not alter the original member ID, recipe or seed. Their complete
authority and derivation version are checked before restart mutation.

Omitting `wrf_seed_labels` preserves existing process keys and checkpoint schema.
All-off namelists preserve their original imported TOML bytes and allocate no
stochastic owner. CPU import, constructor, seed and restart-contract gates pass.
The new labelled path still requires an actual forecast and restart GPU gate.
