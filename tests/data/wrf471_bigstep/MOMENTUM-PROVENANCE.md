Compiled WRF v4.7.1 momentum comparisons

The reference calls unchanged horizontal_pressure_gradient, coriolis and
curvature bodies with their complete WRF argument lists. Source SHA256 is
bd177b6b5ba7949cf9e694d7ad654fd9ae2f07d39d85802f0716c5318889a815;
constants SHA256 is
5b80377fecdc18a5f0ad38d3b6c15cfc86ad5d76701adbbbb08a08698d0f7062.
WRF commit is f52c197ed39d12e087d02c50f412d90d418f6186. The actual
generated grid_config_rec_type is imported. No numerical module is stubbed.
Compilation uses gfortran 15.2.0, O0, FMA contraction disabled and bounds
checks. momentum-receipt.json records extraction, compiler and fixture pins.

Eight cases use the real state, specified or periodic boundaries, southern
rotation, a steep terrain perturbation, map-factor extremes, vertical motion,
zero and near-zero motion. All 49 layers, staggered faces and native memory
halos are retained. Periodic duplicate faces are made consistent before the
native halo fill. The pressure comparison supplies both implementations the
same recovered FP32 perturbation P=(P_total-PB). Original stored-P output is
also retained, and the representation loss is counted separately.

Measured maxima over these cases are 15651 ULP for pressure gradient, 81 ULP
for Coriolis and 3061 ULP for curvature. Absolute maxima are 0.02978515625,
0.0001220703125 and 0.001953125 respectively. Complete actual and reference
output hashes are pinned alongside every-array word comparisons.

Pressure gradient uses different FP32 interpolation, mass weighting and
geopotential subtraction grouping. Its explicit operator-tree attribution
control reproduces every production output word. For Coriolis and curvature,
a transient diagnostic changes only the four-term addition order and disables
FMA; every resulting output word is identical to compiled WRF. Those controls
explain arithmetic differences and are not substituted for the native oracle.

Scope is nonhydrostatic pressure gradient, isotropic map factors, map_proj=1
curvature and whole-domain kts=1. Polar/map_proj=6 curvature is explicitly not
implemented in coriolis_map.cu and is not claimed. The combined momentum
launcher is also pinned, with a 355 ULP corpus maximum. Forecast kernels are
unchanged by this comparison family.

Rebuild with momentum_build.py and momentum_fixture.py, measure with
momentum_measure.py and run tests/test_bigstep_momentum_wrf471_parity.py.
The case maxima are measurements, not general error budgets; any change to
the pinned output words requires a new explanation and measurement.

An sm_89 / NVRTC 12.9.86 replay has distinct production V tendency words
in the vertical-motion, steep-terrain and map-extreme curvature and combined
probes. All remaining production outputs retain the original pins. The
same WRF-order controls match every reference word in all eight cases.
This identifies float32 evaluation and contraction as the cause; it does
not isolate compiler build from architecture without a same-card comparison.
momentum-measurements-sm89-nvrtc12.9.86.json retains every exact output pin,
and momentum-diagnostics-sm89-nvrtc12.9.86.json retains the causal controls.
momentum-device-sm89-nvrtc12.9.86.json pins the compiler build, loaded NVRTC
library hash, runtime, device architecture and both measurement files.
The tests select this measured compile-platform table and continue using
the original exact table for every other platform, without skipping a GPU.
