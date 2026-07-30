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

## Deliberate runtime deviations from WRF v4.6.1

The column port retains a small set of bounds/error-handling guards that are
not present in `module_cu_kfeta.F`. They do not activate for the packaged
real74 columns or the trigger/shallow parity fixtures:

- Saturation vapor pressure is capped at `0.99*p` before forming `qes`. WRF
  evaluates the denominator directly; the cap keeps arbitrary validation
  inputs from producing a non-positive denominator.
- The source-layer dewpoint/`ALU` lookup bounds `EMIX`/`A1`, the lookup index,
  and interpolation fraction. WRF extrapolates its private table expression;
  the port bounds fixed CUDA table storage for near-dry or out-of-range test
  inputs.
- An LCL at/below the first mass level or within the top two mass levels is a
  no-convection result. WRF checks an LCL above the model top, but its later
  neighbor and plume accesses are not defined for these degenerate columns.
- If negative-vapor borrowing encounters zero `TMA` or `TMB`, the port rejects
  the column before division. WRF divides without a zero guard at lines
  2039-2045.
- WRF aborts the domain for negative surface `QG` and eventually for its
  cloud-top mass-balance diagnostic (lines 2028-2067, 2497-2499). The CUDA
  kernel cannot issue a per-thread host fatal error, so those exceptional
  columns return zero outputs; the float64 mirror returns zero for the former
  and raises `RuntimeError` for the latter. Ordinary WRF no-cloud/numerical
  `RETURN` paths remain ordinary zero-output columns in both ports.
- Environmental density for the layer-mass integrals is computed in-scheme as
  `rho = p/(R*Tv)` with `Tv = T*(1+0.608*qenv)` (`kf.cu:291-293`; the float64
  mirror's `np_kf_column` uses the identical form) — the expression WRF itself
  carries commented out at `module_cu_kfeta.F:777`
  (`!        RHOE(K)=P0(K)/(R*TV0(K))`). WRF instead consumes the
  driver-supplied moist density `rho = 1./alt*(1.+qv)` from `phy_prep`
  (`module_big_step_utilities_em.F:4856`) as `RHOE` in
  `DP(K)=rhoe(k)*g*DZQ(k)` (kfeta.F:779). O(0.1%) layer-mass difference in
  humid columns; recorded 2026-07-16 per the Task 6b review (adopting WRF's
  form would add a rho input to the frozen kernel surface).

Two previously added guards were removed for literal parity: candidate-source
search now ends only 300 hPa above the surface (not also at `0.5*psfc`), and
the precipitation-efficiency shear depth is the un-clamped
`z(cloud_top)-z(LCL)` expression from line 1611. The cloud-top guards guarantee
that denominator is positive. Source depth also follows WRF's strict
`DPTHMX > 5000 Pa`; equality is insufficient.
