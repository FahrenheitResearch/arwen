# Kain--Fritsch lookup table provenance

`kf_lutab.npz` is a deterministic FP32 data rendering of the lookup-table
construction in the local WRF v4.6.1 authority. The upstream source is
<https://github.com/wrf-model/WRF.git>, tag `v4.6.1`, commit
`d66e442fccc04111067e29274c9f9eaccc3cef28`. The canonical source range is
`phys/module_cu_kfeta.F:3174-3301`; a machine-local staging path is not part
of the authority.

`tools/generate_kf_lutab.py` transcribes the 250 saturation-equivalent-
potential-temperature points, 220 pressure points from 5000 to 110000 Pa,
the parcel-temperature and saturation-mixing-ratio tables, and the 200-point
logarithm table. It reproduces WRF default-`REAL` arithmetic, including the
repeated pressure/log increments and final in-loop `QS` store at source lines
3191-3299. Runtime interpolation is a direct transcription of the unclipped
index, fraction, and cross-term expressions in `TPMIX2DD` at source lines
3009-3039.

The local authority's `phys/module_cu_kfeta.F` has SHA-256, over the file as
the upstream repository stores it (**LF** line endings, which is what a Linux
checkout and the oracle build below see):

    e6376c2d85c45470f49d545b25d513b5ec111bf36b87beebc740bf42825c6e5f  module_cu_kfeta.F

The LF digest is the pin. A Windows checkout with `core.autocrlf=true` renders
the same bytes with CRLF line endings and hashes to

    b2ee225b2148d54afa464f941d967cb197a8a73e23d6fd3450086c6f3a705895  module_cu_kfeta.F (CRLF)

which is the value this file recorded from 2026-07-16 to 2026-07-25 as if it
were the authority. It is the same content, but it does not verify where the
oracle is built, so anyone checking the pin on Linux would have concluded the
source had moved. Both are recorded here so neither host can be surprised;
`sha256sum module_cu_kfeta.F` on the pinned WSL/Linux tree must print the LF
digest.

`LICENSE.txt` is copied verbatim from the repository root at the same tag.

## Precision, and which libm the table is built against

`KF_LUTAB` declares every table variable as default `REAL`, and WRF v4.6.1
does not promote it: `arch/preamble:63` ships `NATIVE_RWORDSIZE = 4` and every
`PROMOTION` line in `arch/configure.defaults` leaves `-fdefault-real-8`
commented out. The table WRF builds is therefore **binary32 throughout** —
matching it means matching *single-precision* elementary functions, not
rounding a double-precision result.

The routine calls exactly three transcendentals: `exp` (twice per parcel
state), `**` with a real exponent, and `alog`. On the reference toolchain
(gfortran 13.3.0, glibc 2.39, x86-64) those lower to the scalar glibc entry
points `expf`, `powf` and `logf` — confirmed by `nm -u` on the oracle binary,
which shows `expf@GLIBC_2.27`, `logf@GLIBC_2.27`, `powf@GLIBC_2.27` and no
libmvec symbol. The Newton iteration is serially dependent, so the vectoriser
never replaces them: `-O0 -ffp-contract=off` and `-O2 -ftree-vectorize`
produce bit-identical tables.

The generator therefore calls the glibc 2.39 float32 transcriptions in
`gpuwm/core/noahmp_libm.py` (`expf`, `powf`, `logf`), not NumPy's float32
transcendentals. NumPy is used only for storage and for correctly-rounded
`+ - * /`, both of which are stable across releases.

## Oracle

`tools/kf_lutab_oracle.F90` is `module_cu_kfeta.F:3174-3301` copied verbatim
with the declarations it needs, the `SVP*` constants `kf_eta_init` passes
(`share/module_model_constants.F:76-79`), and a raw dump wrapped around it.
Built and run as documented in its header, it emits little-endian binary32
column-major streams with SHA-256:

    04c8e3cd4138e440d486b5899575063fde81b4beba5d82c8ffe6a2d766fd5ce0  ttab.bin
    39ae8b543154f035cc47512929335c15c7880c30f33f41cd8a4518c3e8975d7a  qstab.bin
    a6e207a3f0f9b3631aea74ce89b070b57eed6c298fb40f2488dca4627391874d  the0k.bin
    6836d80708f02642d1dc53b547817dbab623545cfee778671290d7d825eeaf4d  alu.bin
    206e7022af0e9ef62d5203083a83efac175607e8263c873b19b7708d6bbe352c  scalars.bin

`generate()` reproduces all **110,420** table cells from that oracle
bit-for-bit, and does so identically under NumPy 2.2.6 / CPython 3.13.7
(Windows) and NumPy 2.5.1 / CPython 3.12.13 (Linux).

Reproducibility is a claim about the *arrays*, not the container: NumPy's
`savez_compressed` writes a DEFLATE stream whose bytes depend on the host zlib,
so the `.npz` digest below identifies the shipped artifact and is not
reproduced by regenerating on a different Python. Regeneration is verified by
array equality, which is exact.

SHA-256 (regenerated 2026-07-25):

    3248ad89f73084d615a55e319ee3a0151a3b6a07402143d745fe1afb11751f73  kf_lutab.npz

## Finding: the 2026-07-15 table did not match WRF (superseded)

The table shipped from 2026-07-15 to 2026-07-25 (SHA-256
`91d383e7...29cc11fb`) was built on `np.exp`, `np.power` and `np.log` at
float32. Those are neither WRF's functions nor stable across NumPy releases,
and the table was wrong on both counts. Measured against the gfortran/glibc
oracle above:

| member        | cells wrong    |   %  | max ULP | max abs err | max rel err |
|---------------|----------------|-----:|--------:|------------:|------------:|
| `temperature` | 17,074 / 55,000| 31.0 |      65 |  9.92e-4 K  |    4.15e-6  |
| `qsat`        | 27,741 / 55,000| 50.4 |    1480 |  2.10e-7    |    9.78e-5  |
| `log_ratio`   |     21 /   200 | 10.5 |       1 |  2.38e-7    |    1.19e-7  |
| `thetae_base` |      0 /   220 |  0.0 |       — |           — |           — |

`thetae_base` survived because it is evaluated at `TMIN = 150 K`, where the
`expf` argument is far enough from a rounding boundary that NumPy and glibc
agree on all 220 points. The errors concentrate in the Newton-iterated
members: worst `temperature` cells sit near `it = 87-92` at 86-110 kPa
(~239 K, the cold end of the boundary layer), and the worst `qsat` cells at
the same indices, where a 1-ULP `expf` difference is amplified by the
`0.622*es/(p-es)` quotient and then by the secant iteration.

The regeneration moved 44,836 cells. **Every moved cell lands exactly on the
value WRF produces** — the new table is the oracle, bit-for-bit, so the change
is a strict fidelity improvement with no judgement call in it.

The NumPy-version instability that surfaced this (25 `temperature` cells
differing between NumPy 2.2.6 and 2.5.1, max abs 4.58e-5) was a symptom, not
the defect: fixing only the version-dependence would have pinned a table that
was still wrong in 31% of its `temperature` cells. This is the same class of
defect `22b63fb` fixed for the MYNN surface tables.

### Erratum: how far the two real74 KF pins actually moved

The regeneration moved two pins in
`tests/test_kf.py::test_real74_12z_extracted_column_gates_and_provenance`.
Commit `4239185` and the comment above those pins reported both moves as
"~1e-6 relative" / "4.5e-6 relative". Only the `cape_before` figure was right.
Recomputed from the pinned values themselves:

| pin | before | after | absolute | **relative** |
|---|---|---|---:|---:|
| `cape_before` | 432.29333945446024 | 432.29276931463795 | 5.701e-4 | **1.32e-6** |
| closure `FABE` | 0.09302343911952345 | 0.0930192928993745 | 4.146e-6 | **4.46e-5** |

The `FABE` error was reporting the *absolute* difference (4.15e-6) as if it
were relative; the relative move is 4.46e-5, thirty-four times larger than
`cape_before`'s and an order of magnitude larger than what was published. That
is still far inside the unchanged `rtol=2.0e-13` exact pins, so nothing was
concealed by it and no tolerance was widened — but the number quoted as the
size of the change was wrong, and the amplification is the point: `FABE` is the
converged ratio `cape_after/cape_before` out of a secant iteration, so it
magnifies a table move rather than tracking it. Neither `4239185`'s message nor
any other commit message can be corrected in place; this table is the
correction of record, and the test comment now carries these figures.

## The runtime port: no deviations from WRF v4.6.1 on any defined path

Until 2026-10-07 the CUDA column (`gpuwm/core/kernels/kf.cu`) was a
re-derivation of the scheme with its own guards and its own density, graded
only against a float64 mirror of itself. It was replaced that day by a
line-for-line port of `KF_eta_PARA`, graded at 0 ULP on every output of
`KF_eta_CPS` against the byte-unmodified Fortran by the column oracle in
`tools/kf_wrf461_oracle` (`tests/test_kf_wrf461_oracle.py`). The guards this
section used to list are gone with the old kernel: no `0.99*p` cap on the
saturation vapor pressure, no bounding of the dewpoint/`ALU` lookup, no
top-two-levels LCL refusal, and the layer masses use WRF's own RHO input
(`DP(K)=rhoe(k)*g*DZQ(k)`, module_cu_kfeta.F:779) instead of an in-scheme
`p/(R*Tv)`.

What remains differs from the Fortran only where the Fortran's own behaviour
is undefined or is a model abort, and each is documented in `kf.cu` beside
the code:

- A source layer whose LCL is the first mass level (`KLCL = 1`) makes WRF
  read `Z0(0)`, `T0(0)` and `W0AVG1D(0)`, outside its arrays. The port does
  not evaluate that candidate (and returns no convection if the closure's
  recomputed LCL lands there).
- `wrf_error_fatal` paths -- negative `QG` at the surface (lines 2028-2034)
  and the cloud-top mass-balance stop (2060-2067, 2497-2499) -- stop the
  whole WRF model. The port returns no convection for the column; the oracle
  never counts such a column as a match (none occurred in the graded sets).
- A table index outside the whole `TTAB`/`QSTAB`/`THE0K`/`ALU` array (WRF
  reads a neighbouring module variable) is clamped to the array so the
  kernel stays inside its allocation. Every in-bounds index, including the
  edge cells WRF's own `OUT OF BOUNDS` message flags, reads exactly the
  Fortran's word: the tables are uploaded in Fortran column-major order.
- A zero `TMB` in the negative-vapor borrowing divides by zero in WRF and in
  the port alike; the resulting non-finite tendency is caught by the
  driver's validation (`validate_kf_outputs`) and stops the run with a named
  error instead of propagating as it would in WRF.
