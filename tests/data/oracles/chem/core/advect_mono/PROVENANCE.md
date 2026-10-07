WRF v4.7.1 monotonic scalar advection oracle

Reference commit: f52c197ed39d12e087d02c50f412d90d418f6186.
Source: dyn_em/module_advect_em.F:9495-10560 (monotonic) and
:6069-7885 (PD counterexample). WRF source is public domain.
Source SHA256: 58e9a9f29315181250e1f8cf1dccb3b577d66b261892efe371d53e508fd891f9.
The harness checks this against tools/chem_wrf471_oracle/SOURCES.sha256.
No extra source was fetched. Both routines enter the wrapper byte for byte;
mono-prelude.inc supplies configuration storage and the error buffer only.

Compiler: GNU Fortran (Ubuntu 15.2.0-16ubuntu1) 15.2.0.
glibc: ldd (Ubuntu GLIBC 2.43-2ubuntu2.4) 2.43.
Build flags: -O0 -cpp -DEM_CORE=1 -DNMM_CORE=0 -DRWORDSIZE=4 -DIWORDSIZE=4
-DDWORDSIZE=8 -DLWORDSIZE=4 -DWRF_CHEM=1 -ffree-form
-ffree-line-length-none -fallow-argument-mismatch.
No fast math, no FMA. The libmvec guard sees no vector symbols in the oracle;
its positive control imports _ZGVbN4v_expf. See libmvec-report.txt.
Compiler and complete source/tool/wrapper hashes are in compiler.txt and
oracle-sha256sums.txt. Fixture hashes are in fixture-sha256sums.txt.

Build command, CPU only:
bash tools/chem_wrf471_oracle/build.sh WRF_SOURCE_ROOT BUILD_DIR run_advect_mono

Cases: 72 combinations, 12 x 10 x 8 mass cells, horizontal order 5,
vertical order 3 or 5, boundary b0 periodic, b1 specified, b2 open,
map0 unit maps, map1 separate varying x/y map factors,
front0 smooth sinusoidal field, front1 sharp front with exact zeros,
c1 nominal horizontal Courant 0.1, c2 0.4, c3 0.8.
c2 cases include nonzero implicit vertical flux. Coupled ru/rv/ww have
both signs; hybrid masses and stretched eta weights vary with height.
The stage field differs from the old field and the initial tendency is
nonzero, so the additive accumulator and the two time levels are exercised.
These are identity cases, not physically balanced model trajectories.

Two bounds_v3/bounds_v5 cases use divergence-free periodic x transport,
mass 65536, spacing 1024, dt 16, Courant 0.25, initial plateau 0.25..0.75.
They also publish pinned WRF PD tendencies as independent counterexamples.
Horizontal order 3 is a fatal WRF case, not an admitted monotonic stencil;
WRF stops with "module_advect: advect_scalar_mono, h_order not known 3",
which is why the port refuses it rather than substituting another stencil.

Array order is WRF (i,k,j), raw little-endian float32. The h_tendency dump
excludes degraded boundary rings because WRF does not define its complete
horizontal diagnostic there. z_tendency and tendency cover all mass cells.
There is no weather-map rendering or data ingest in this fixture producer.
