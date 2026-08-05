# The batched symmetric eigensolver, and why the DA path no longer needs cuSOLVER

`gpuwm/core/kernels/jacobi_eigh.cu` + `gpuwm/core/jacobi_eigh.py`

## What this replaced

`gpuwm.da.letkf.analyze` factors one small symmetric matrix per analysis
gridpoint -- `(R-1)I/rho + C Yb`, `R` the ensemble size, one per local patch.
Until this kernel that single line,

```python
evals, evecs = xp.linalg.eigh(amat)          # (G, R, R)
```

was the **only** call in the whole project that reached a linear-algebra
library.  The dycore, the physics, the microphysics and every diagnostic are
hand-written CUDA and elementwise CuPy; nothing else needs BLAS or LAPACK.
On the device that one call pulled in cuSOLVER, and through it cuBLAS and
cuSPARSE.

That dependency failed twice in the field, both times wearing the same
disguise:

* a Windows box where the `nvidia-*-cu12` wheels were installed but invisible
  to a compiled extension, because since Python 3.8 an extension's dependent
  DLLs resolve through `os.add_dll_directory` and never through `PATH`
  (`evidence/da-demo/live-fire-2/cusolver-fix-receipt.json`);
* a rented Ada node whose CUDA install simply shipped `cusolver: false`.

Both present as *elementwise CuPy works, the factorisation does not* -- so
every cheap piece of evidence says the GPU is fine, because for everything
except the factorisation it is.

### It is worse than "sometimes missing": it is order-dependent

Measured on a rented Ada node on 2026-08-05, cupy 14.1.1, receipt in
`evidence/da-demo/jacobi-eigensolver-cusolver-fault.json`.  The wheel's
`libcusolver.so.11` was installed, `ldconfig` did not know it, and the
shipped `LD_LIBRARY_PATH` did not contain it.  In that one environment:

```
$ python -c "import cupy; cupy.linalg.eigh(A)"
ImportError: libcusolver.so.11: cannot open shared object file

$ python -c "import cupy; cupy.linalg.inv(A); cupy.linalg.eigh(A)"
ok
```

Same box, same interpreter, same environment; the only difference is what
the process did first.  `inv` reaches cuSOLVER through
`cupy_backends.cuda.libs.cusolver`, whose link resolution finds the wheel;
batched `eigh` reaches it through `cupyx.cusolver`, whose does not -- and
once the first call has pulled the library into the process, the second one
finds it already loaded.

So on such a box the answer to "is cuSOLVER available?" depends on the call
order inside the process asking.  That is the mechanism behind "it worked
yesterday", and it is why the fix is a **loader path** rather than an
install: `pip install` would not have repaired that node.

Elementwise arithmetic, cuBLAS `matmul`, `inv` and `svd` were all fine
throughout.  This project's own kernel ran on the unfixed box with a
reconstruction residual of `3.7e-15` and gave byte-identical results across
processes, which is the whole point: **the DA path no longer has the
dependency to get wrong.**

## Why a purpose-built kernel wins here

A batch of hundreds of thousands of `k <= 64` matrices is the case a
general-purpose library is worst at.  Householder tridiagonalisation plus a
shifted QL iteration -- what LAPACK does, and the right answer for one large
matrix on one core -- is a serial dependence down the diagonal with a
data-dependent shift per step.  There is nothing in it for a thread block.

A Jacobi sweep splits into `k/2` rotations on **disjoint index pairs** that
apply simultaneously, so one block can own one matrix from load to store with
the whole `k x k` problem resident in shared memory.  Two further properties
decided it over the alternatives:

* **Demmel and Veselic (1992)**, *Jacobi's method is more accurate than QR*
  (SIMAX 13, 1204-1245): on a symmetric **positive definite** matrix, Jacobi
  attains a relative accuracy governed by the condition number of the
  *scaled* matrix rather than of `A`.  QR-based methods promise only absolute
  accuracy proportional to `||A||`.  `(R-1)I/rho + C Yb` is positive definite
  by construction -- `C Yb` is positive semi-definite and the shift is
  `(R-1)/rho > 0` -- so the stronger bound is the one in force.
* **An already-diagonal matrix is an exact fixed point.**  Every rotation is
  skipped by the threshold test, `U` comes back the literal identity and `w`
  the literal diagonal, bit for bit.  No library eigensolver promises this,
  and `gpuwm/da/letkf.py`'s inactive-gridpoint argument is written around the
  absence of the promise.

Alternatives considered and rejected: Cholesky (gives `A^-1` cheaply but not
the **symmetric** square root Hunt et al. step 6 requires -- `L L^T` is not
`A^1/2`); Newton-Schulz / Denman-Beavers iteration for `A^-1/2` (quadratic
but conditionally convergent, and the convergence condition is exactly what
this matrix does not guarantee).

## The algorithm as built

One block per matrix; `k` padded up to even `M`; `A` and the accumulated
rotations `V` in dynamic shared memory.  Per round: compute `M/2` rotations,
apply them all on the left, then all on the right (and to `V`), then write the
exact zeros the rotations were aiming for.  `M-1` rounds make a sweep; sweeps
run until no pair in a whole sweep clears the threshold.

* **Ordering**: round-robin ("chess tournament") -- index 0 fixed, the rest
  rotating -- which emits all `M(M-1)/2` pairs in `M-1` rounds of `M/2`
  disjoint pairs.  Fixed, compiled in, part of the contract.
* **Rotation**: Golub and Van Loan Algorithm 8.4.1, the root of smaller
  magnitude so `|theta| <= pi/4`.
* **Threshold**: skip when `|a_pq| <= eps sqrt(|a_pp| |a_qq|)` -- the
  Demmel-Veselic criterion, measured against the geometric mean of the two
  diagonals rather than against `||A||`.  This is what buys the relative
  accuracy *and* what makes a diagonal input a fixed point.
* **Symmetry**: the two triangles are averaged once per sweep.  A congruence
  by an orthogonal matrix is symmetric analytically, but the left and right
  passes see different intermediates and drift by an ulp or so per round.
* **Odd `k`**: padded to `M = k+1` with a unit diagonal and exact zeros off
  it.  The padding line is never touched -- an exact zero never clears the
  threshold, and neither a left nor a right rotation between two real indices
  writes into it -- so it stays exactly `e_M` and is dropped on the way out.

### Tiers

| `M` | threads/matrix | matrices/block | barrier |
|---|---|---|---|
| `<= 32` | 32 | up to 8 | `__syncwarp()` |
| `33..48` | 128 | 1 | `__syncthreads()` |
| `49..64` | 256 | 1 | `__syncthreads()` |

One warp per matrix for small `k` means the barrier is a warp barrier and a
warp that converges early cannot strand a sibling.  The wider tiers put one
matrix on the whole block, so every thread runs the same number of sweeps and
the block barrier is equally safe.

`MAX_K = 64` is a shared-memory ceiling, not an algorithmic one: two `k x k`
float64 working copies is `16 k^2` bytes, 64 KiB at `k = 64`, which is one
block per multiprocessor even with the opt-in limit raised.

## Canonical form, and what it does not fix

An eigendecomposition is not unique.  The kernel pins what can be pinned:

* eigenvalues **ascending**, ties broken by source column index (a total
  order, not merely a sort);
* each eigenvector scaled so its **largest-magnitude component is positive**,
  ties in magnitude broken by the lowest row index.

What no convention can fix is the basis chosen inside a **degenerate
eigenspace**, and that is not a corner case here: `C Yb` is routinely
rank-deficient -- a gridpoint whose localisation lens caught one usable
observation hands the solver an `R-1`-fold degenerate eigenvalue -- so two
correct solvers legitimately return different `U`.

## What is actually asserted

`gpuwm.da.letkf` takes exactly two things out of the eigendecomposition:

```python
pa = U diag(1/w) U^T                    # Pa~ = A^-1
wa = U diag(sqrt((R-1)/w)) U^T          # Wa  = sqrt(R-1) A^-1/2
```

Both are **matrix functions** of `A`, and a matrix function of a symmetric
matrix is unique -- independent of sign, of ordering, and of the basis chosen
inside a degenerate eigenspace.  So the headline invariant is agreement of
`Pa~` and `Wa`, not of eigenvectors.  In order of what they are worth
(`tests/test_jacobi_eigh_gpu.py`):

1. `||Pa_mine - Pa_cusolver||` and `||Wa_mine - Wa_cusolver||`, relative,
   over the whole battery.
2. Eigenvalues -- also unique -- elementwise once both are sorted.
3. Backward stability `U diag(w) U^T = A` and orthogonality `U^T U = I`,
   which need no reference solver.
4. Eigenvectors, **only** where the spectrum is well separated, both sides
   put in the canonical form above, tolerance scaled by the smallest
   eigenvalue gap because Davis-Kahan says that is what bounds the
   achievable agreement.

The battery covers, at `k` in `{2, 3, 10, 11, 20, 21, 36, 64}`: the realistic
`letkf` shape; a patch with exactly **one** observation (rank-1 `C Yb`, hence
`k-1`-fold degeneracy); a patch with **none** (a scaled identity, perfectly
degenerate); **near-degenerate** eigenvalues a few ulp apart; a genuinely
**singular** matrix; a spectrum spanning `1e10`; and everything scaled to
`1e-150`.

## Determinism

**Bitwise reproducible run to run on the same hardware, at fixed `k`.**  There
are no atomics, no cross-block communication, and no reduction whose order can
move: the sweep ordering is compiled in, the sort is a sorting network, and
the sign convention is a first-maximum scan.  A matrix also solves to the same
bytes wherever it sits in the batch and whatever the batch size, so
`LetkfConfig.chunk_points` stays a memory knob at the solver.

**Not** bitwise stable across sweep orderings: a different pairing order
diagonalises the same matrix to a different rounding.  The ordering is
therefore part of the kernel's contract rather than a tuning parameter, and
changing it is a re-pin, not a refactor.

One honest caveat that is **upstream of this kernel**: the *analysis* as a
whole is not bitwise invariant to `LetkfConfig.chunk_points` on the device,
because CuPy selects reduction and batched-GEMM kernels by array shape, so
`s.mean(axis=0)` and `cmat @ yb` sum in a different order when the chunk
changes.  Measured at `3.6e-16` relative, and **identical under
`eigensolver='library'`** -- it predates this work and is not caused by it.
`tests/test_letkf_eigensolver.py` pins that the kernel adds none of its own.

## Performance

Float64, `(R-1)I + C Yb` with `C Yb` of realistic observational rank, min of
five launches after a warm-up, RTX 5090 (sm_120, 170 SMs).

| `k` | `G` | `cupy.linalg.eigh` | this kernel | ratio |
|---|---|---|---|---|
| 10 | 220,000 | 164.2 ms | 96.1 ms | **1.7x** |
| 10 | 170,000 | 145.6 ms | 76.2 ms | **1.9x** |
| 20 | 200,000 | 700.7 ms | 341.1 ms | **2.0x** |
| 36 | 200,000 | **69,727 ms** | 1,106 ms | **63x** |

`k = 64` was not measured; the run did not reach it.

**Read these as lower bounds on the kernel and treat the absolute numbers as
soft.**  The card was shared throughout with several other lanes -- an
ensemble sweep, a live nowcast daemon, a cycling forecast and a test session
-- sitting at 94% utilisation and 31 of 32 GiB.  Both solvers took the same
contention on the same inputs in the same process, so the ratios are the
robust part; the milliseconds are inflated for both.

### The `k = 36` result is UNEXPLAINED and must not be relied on

The 70-second cuSOLVER figure above is a real measurement, and I do not know
what it means.  My first explanation -- that `cusolverDn<t>syevjBatched` caps
at `n = 32`, so `k = 36` falls back to one call per matrix -- is **refuted**.
Measured on a rented Ada node (sm_89, cupy 14.1.1, `G = 10,000`, float64,
light load), sweeping straight through the supposed boundary:

| `k` | 16 | 24 | 30 | 31 | 32 | 33 | 34 | 36 | 40 |
|---|---|---|---|---|---|---|---|---|---|
| cuSOLVER (ms) | 36 | 149 | 194 | 211 | 264 | 218 | 214 | 222 | 294 |
| this kernel (ms) | 18 | 31 | 89 | 154 | 144 | 90 | 157 | 151 | 194 |

There is no discontinuity at 32/33 at all.  Scaling the Ada `k = 36` cuSOLVER
number to 200,000 patches gives roughly 4.4 s, not 70 s.

So the two measurements disagree by about 16x on cuSOLVER while agreeing on
this kernel, and contention does not explain that asymmetry -- a busy card
slows both sides.  Candidates not yet separated: the cupy version (14.0.1
locally against 14.1.1 on the node), a workspace or memory threshold crossed
somewhere between `G = 10,000` and `G = 200,000`, or an sm_120-specific
cuSOLVER path.  **Re-measure on a quiet sm_120 card before quoting the 70-second
number or planning around it.**

What survives both measurements: `cupy.linalg.eigh` on a stacked array routes
to `cupyx.cusolver.syevj`, cuSOLVER's own cyclic Jacobi, so this is a
like-for-like ALGORITHM comparison throughout, and this kernel is ahead at
every size tried on both architectures -- by 1.5-2.0x on the Ada node across
`k = 16..40`, and by 1.7-2.0x locally at `k = 10` and `k = 20`.  The
dependency-removal case does not rest on the outlier.

### Still outstanding

The KDMX live-fire-3 cycle has NOT been re-run both ways, so there is no
end-to-end analysis comparison and no FSS(30 dBZ, 27 km) number in this
document.  The harness is written and the exact arguments are reconstructed
from `evidence/da-demo/live-fire-3/cycle-report.json`; it needs a quiet card.

A provenance note for whoever runs it, because the `N = 10` baseline is
quoted at more frames than the repository actually carries.  Committed, in
`evidence/da-demo/live-fire-3/verification-addendum.json`: legs 6 and 7 only,
**FSS(30 dBZ, 27 km) = 0.7274 at +15 min and 0.7557 at +30 min**.  The rolling
verifier did measure all six free-forecast frames -- including the `+90 min`
figure that gets quoted -- but those live in an untracked gallery file and
have not been committed, so they are not yet a citable baseline.  Use the two
committed frames, or cite the gallery explicitly as uncommitted.  Committing
the full ladder is tracked elsewhere as a blocking item; this document should
be updated to the six-frame baseline once it lands.

## Configuration

`LetkfConfig.eigensolver` / `RadarAssimilationConfig.eigensolver`:

* `"auto"` (default) -- this kernel on the device when `R` is in range,
  `xp.linalg.eigh` otherwise.  On numpy that is in-process LAPACK and no CUDA
  library at all, which keeps `--solve-device host` a genuine fallback.
* `"jacobi"` -- require this kernel; refuse rather than fall back.
* `"library"` -- `xp.linalg.eigh`, i.e. cuSOLVER on the device.  The escape
  hatch and the A/B reference.

`LetkfDiagnostics.eigensolver` records which ran, and
`LetkfDiagnostics.max_jacobi_sweeps` the worst sweep count, because the two
solvers agree to rounding rather than bitwise and an increment array does not
say on its face which produced it.

## Refusals

Everything is fail-closed, matching the rest of `gpuwm.da.letkf`:

| condition | behaviour |
|---|---|
| `k` outside `[2, 64]`, or a dtype other than float32/float64 | `JacobiEighError` naming the range; `"auto"` falls back to the library instead |
| non-finite entry on input | `JacobiEighError` naming how many matrices; nothing returned |
| still rotating at the 40-sweep cap | `JacobiEighError`; a partially diagonalised matrix is not an answer |
| `eigensolver="jacobi"` on numpy | `LetkfError` saying the kernel is CUDA |

The sweep cap is generous on purpose -- the worst case in the whole battery is
19 sweeps, on a `64 x 64` with condition number `1e12`, far outside anything
the LETKF matrix can be.  It exists to make non-convergence loud, not to bound
the work.

## `gpuwm doctor`

`gpuwm doctor` now carries a `radar-DA eigensolver` line immediately after the
cupy line.  It **solves a small batch both ways and checks the answer**, which
is the only probe that distinguishes the fault: an import test, a
`show_config()` read, or an elementwise operation all come back green on a box
where the factorisation cannot run.

* both work -> `ok`, no remedy;
* kernel works, no cuSOLVER -> `ok`, with the absence reported and a remedy
  that checks for **installed-but-unreachable first**, because that is the
  commoner fault and installing again does not fix it;
* kernel fails, cuSOLVER works -> `present`, pointing at
  `eigensolver='library'`;
* neither -> `MISSING`, naming both failures and stating that forecasts are
  unaffected.

None of these is blocking: an install that never assimilates radar touches
neither solver, and one that does has the bundled kernel.

Two properties of the probe are load-bearing and are commented as such in
`gpuwm/doctor.py`, with the measurement above as the reason:

1. it runs in a **fresh process**, and
2. it calls **nothing else from `cupy.linalg` before `eigh`**.

A probe that shares a process with other work, or that warms up with a
different factorisation, reports green on a box where the analysis will fail.
This was verified against the real fault, not a simulated one: the payload
the probe returned on the unfixed node is in the receipt, and rendering the
check from it produces the verdict quoted there.
