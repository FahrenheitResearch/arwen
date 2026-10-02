# Native LW extension fixture

`native_profile_50hpa.npz` contains original little-endian float32 driver
inputs and the PLEV, PLAY, TLEV, TLAY and O3VMR arrays passed by an admitted
WRF 4.7.1 wrapper to its longwave solver. Expected arrays are captured
native outputs. They were not computed by ArWen or its tests.

The column has 49 model layers, 62 radiation layers and a 50 hPa model top.
The exact extension interfaces are 50, 46, 42, 38, 34, 30, 26, 22, 18,
14, 10, 6, 2, 0 hPa. Midpoints are 48, 44, 40, 36, 32, 28, 24, 20,
16, 12, 8, 4, 1 hPa. WRF overwrites the final interface with zero.

The accepted executable SHA-256 is
7680ed5b3483c7ab3d90589707a15fb2bb524cecd7de78948619b4ce794c2de8.
`provenance.json` records original source, object/executable, input,
history and raw-array hashes, the observational source hooks, layout and
the complete seven-history identity receipt. Logical inputs retain their
native int32 words. Gravity is the accepted module constant 9.81 m s-2.

The capture uses one original column at the first physics call. Copies
of that column exercise partial batch widths; separately modified top
interfaces exercise the existing variable-top fallback. Those modified
columns are test constructions, and are not claimed native observations.

The five asserted profiles use copies and basic float32 add, subtract,
multiply and divide operations. Temperature interpolation is linear in
pressure, and ozone is a pressure-thickness average. Captured input
temperatures are not recomputed from potential temperature. These expected
words do not depend on platform exp, log, pow or trigonometric functions.
A regression refuses host math and NumPy transcendental calls while
preparing the native column. The separate trace-gas EXP path uses the
repository's deterministic `noahmp_libm` transcription and does not feed
these five profiles. Exact comparisons therefore remain appropriate on
glibc and Windows UCRT; no temperature or ozone tolerance is introduced.
