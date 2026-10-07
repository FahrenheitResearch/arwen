Source conversion fixture

Reference: NOAA-EMC/UPP commit
1296eebb295251d0fe1cc697f47058c62d887977.
https://github.com/NOAA-EMC/UPP/blob/1296eebb295251d0fe1cc697f47058c62d887977/sorc/ncep_post.fd/MDLFLD.f#L2442
https://github.com/NOAA-EMC/UPP/blob/1296eebb295251d0fe1cc697f47058c62d887977/sorc/ncep_post.fd/params.F#L57

MDLFLD.f sha256:
a73e5a105711b9257209ecdca96d20c3ccb3e401a2264a547489827ee617d38c
params.F sha256:
90d9632639efdf0920faa25f84c96ae291eb2a8a1e8a7252709d6d8a315afb72

GNU Fortran (Ubuntu 15.2.0-16ubuntu1) 15.2.0.
glibc: Ubuntu GLIBC 2.43-2ubuntu2.4.
Flags: -O0 -cpp -DRWORDSIZE=4 -ffp-contract=off.
Writer: unchanged oracle_io.F90, little endian REAL(4).
The lane script checks both source pins and the exact copied UPP statement.
Its -O0 objects contain no _ZGV imports; the -Ofast positive control
contains _ZGVbN4v_expf. No WRF model constants or service stub is compiled.

This is an isolated UPP arithmetic statement oracle, not a full UPP run.
The eight cells span zero, small and large smoke, pressures 10000..100000 Pa
and temperatures 220..310 K. The inverse is the port's explicitly defined
algebra, evaluated in Fortran alongside the byte-copied forward statement.
Poisoning GRID1, density and inverse with huge previous values before every
cell produces identical fixture bytes. All expression inputs are assigned;
the port has no persistent state or undefined reads. This experiment does
not qualify undefined reads elsewhere in UPP.

Measured max ULP / nonzero / count against the fixture:
effective density, CPU reference: 0 / 0 / 8.
forward MASSDEN, CPU reference: 0 / 0 / 8.
inverse, CPU reference: 0 / 0 / 8.
inverse, Rust debug and release: 0 / 0 / 8.
Inverse against the original mixing ratio, including the forward rounding:
1 / 2 / 8. No CUDA kernel was added or run.
Independent GRIB packing of P, T and MASSDEN adds an unmeasured residual
relative to the original unencoded model tracer. The native model arrays
are not present, so the real-file residual cannot be measured from GRIB alone.

Fixture file hashes are in oracle-sha256sums.txt. Rebuild command, from the
export root on Linux:
bash tools/chem_wrf471_oracle/build_source_conversion.sh ../oracle-build
