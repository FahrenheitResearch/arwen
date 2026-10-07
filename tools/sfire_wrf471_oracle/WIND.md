# Atmosphere-to-fire wind oracle

`extract_wind.py` pins the complete WRF v4.7.1 fire driver by SHA-256 and
extracts `interpolate_atm2fire` without changing its body. A small module
wrapper supplies the compiled WRF utility module and gravity constant.
`build_wind.sh WRF_SOURCE_ROOT NATIVE_ORACLE_BUILD WIND_BUILD` links that
wrapper against the native utility, constants and service objects built by
`build.sh`. Scalar libm, float32 arithmetic, disabled contraction and runtime
bounds checks are required by the compiler receipt.

The corpus contains 42 cases: 12 full-domain cases, the same 12 using four
paired atmosphere/fire tiles, nine compact atmospheric routine controls,
and nine compact controls using the actual coupled driver's inclusive
mass-cell bounds. Both cards pass 60 checks and grade 25,980 words with zero
differences. It covers anisotropic integer refinement, nonuniform terrain,
heterogeneous atmospheric and fire roughness, roughness above the target,
targets below the first mass level, interpolation across higher levels,
targets above the model top and both fine-roughness coupling modes. The
compact cases cover three-dimensional and one-dimensional base
geopotential inputs and calm columns with signed zero. Refined U/V and all written atmospheric UAH/VAH
diagnostic nodes compare output words, including the upper continuation
halo. Fire allocation halos are unwritten by the original routine and are
excluded from grading.

`gpuwm.core.sfire_wind.interpolate_atm2fire` accepts original shared
staggered allocations. `interpolate_native_atm2fire` accepts compact
atmospheric U/V arrays and pads them with CUDA copy kernels. All heights,
face roughness averages, log interpolation, native boundary continuation,
bilinear refinement and optional roughness scaling run on the GPU.

The coupled driver subtracts one from WRF's outer atmospheric node bounds
before calling the wind routine, while retaining every physical refined
fire cell. The compact adapter therefore defaults to mass-cell bounds
`1..nx` and `1..ny`. It computes interior staggered faces and linearly
continues the terminal faces. The explicit
`domain_includes_terminal_face=True` option retains the smaller-domain
routine-control convention. Native X/Y diagnostic shapes are retained in
both conventions.

The original routine accepts frame corrections and terrain but never uses
them in this interpolation. The port retains that behavior. The optional
roughness scaling evaluates real exponent `2.0` through the scalar libm
power routine, matching the native `-O0` oracle. Replacing it with a square
produced one 1-ULP wind difference in each of the paired high-target cases;
the GPU expression removes that difference.
