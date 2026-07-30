# Official WRF v4.6.1 fused-GS oracle

This harness calls the exact `nssl_2mom_gs` routine from official NCAR WRF
v4.6.1 commit `d66e442fccc04111067e29274c9f9eaccc3cef28`.  The source file is pinned by
its canonical LF SHA-256,
`1eb1b138b75ff3b0cfe33c23779f4ec9b72e57a5455a53ef11c9e55ae0f42722`.
`visibility.patch` changes accessibility only; the two acceptance builds do
not patch an executable WRF statement.

The fixture is a direct full-process oracle for the option-18 GS call.  Inputs
use WRF concentration-space units: mass mixing ratios are kg/kg, number and
predicted CCN are #/m3, and predicted graupel/hail volume is m3/m3.  It records
all 16 prognostics plus native post-GS temperature and potential temperature.
The vertical-velocity input is cell-slab shaped.  GS receives the native
center value `0.5*(w(k)+w(min(k+1,nz)))`, including the top-level clamp.

The 30 four-level cases span two identical repetitions, dt=0.1/1/10/60/300 s,
warm/cold/mixed-phase columns, exact temperature gates, variable graupel/hail
density, zero and moment-threshold cleanup, all-active and shared-donor
competition, the two shared frozen-vapor limiters, and explicit option-18
rain-freezing boundaries.  Rain-freezing coverage includes strict -5 C and
8-mm Bigg gates, the default `ibiggsnow=3` 0.3-mm split, strict -30 C heat-cap
override, qxmin and minimum donor transfers, malformed q/N moment donors,
negative-to-positive `fwet1`, simultaneous Bigg plus qiacr, and long-dt caps.

Driver sedimentation is deliberately excluded.  The later driver calls to
`NUCOND` and the separate `QVEXCESS` branch are also excluded.  Native
post-GS warm ice melting remains included because it executes inside GS.
Temperature and potential temperature are both authoritative outputs: WRF
does not recompute its saved temperature after every late GS state update.

`instrumentation.patch` is applied only to a third, separately linked
diagnostic executable.  Its canonical state output must compare byte-for-byte
with the uninstrumented acceptance fixture.  The raw trace exposes Bigg tail
routing, the rain-freezing heat limiter before/after the -30 C override,
shared frozen deposition/sublimation factors, process rates, donor caps, and
post-limiter aggregate tendencies.  `normalize_diagnostics.py` converts that
trace to named long-form records and rejects unknown or non-finite fields.

Reproduce from a fresh official checkout:

```text
tools/nssl2_wrf461_fused_gs_oracle/build.sh \
  /path/to/WRF-v4.6.1 /new/empty-build-path
```

`fetch_official_wrf.sh` is an optional helper that creates the required fresh
checkout and verifies both the commit and source hash.  The build refuses a
dirty or reused source/build directory, compiles two independent acceptance
executables, requires identical bytes, compiles the diagnostic executable,
requires its state CSV to match acceptance, normalizes/validates diagnostics,
and emits SHA-256 evidence files.
