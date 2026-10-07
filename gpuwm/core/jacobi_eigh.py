"""Batched symmetric eigendecomposition on the device, without cuSOLVER.

One thread block per matrix, two-sided cyclic Jacobi in shared memory up to
k = 64 and in a launcher-owned global scratch slab above it.  The
algorithm, the ordering, the padding of odd sizes and the canonical output
form are all documented in ``gpuwm/core/kernels/jacobi_eigh.cu``; this module
is the launcher and the refusal surface.

Why this exists
---------------
``gpuwm.da.letkf`` factors one small symmetric matrix per analysis gridpoint
-- k x k with k the ensemble size, a few hundred thousand of them per cycle.
Until this module, that single call was the ONLY thing in gpuwm that needed a
linear-algebra library: the dycore, the physics and the microphysics are
hand-written CUDA and elementwise CuPy throughout, and the one ``eigh`` pulled
in cuSOLVER (and, through it, cuBLAS and cuSPARSE) for the whole DA path.
That dependency cost real time twice -- once where the NVIDIA wheels were
installed but invisible to a compiled extension under Python's post-3.8 DLL
resolution, and once on a rented node whose CUDA install simply shipped
without it.  Both faults present as the same masquerade: elementwise CuPy
works, so the GPU is obviously fine, and only the factorisation fails.

A batch of small matrices is also the case a general-purpose library is
worst at and a purpose-built kernel is best at, so removing the dependency and
going faster are the same piece of work rather than a trade.

Supported range
---------------
``MIN_K <= k <= MAX_K``, float32 or float64, on CuPy.  Outside that range this
module REFUSES rather than degrading: :func:`supported` answers the question
without raising and :func:`batched_eigh` raises :class:`JacobiEighError`.  A
caller that wants a library fallback should ask :func:`supported` first --
``gpuwm.da.letkf`` does exactly that.

Two tiers, one algorithm.  Up to :data:`SHARED_MAX_K` each matrix holds two
k x k working arrays in shared, so a float64 problem needs about ``16 k^2``
bytes, and at k = 64 that is 64 KiB -- one block per multiprocessor even with
the opt-in limit raised.  k = 64 therefore WORKS but is bandwidth-starved; see
the measurements in ``docs/da_jacobi_eigensolver.md``.  Above it the kernel
is compiled with ``JACOBI_GLOBAL_WORK`` and the two working arrays live in a
global scratch slab this launcher allocates (bounded by
:data:`GLOBAL_SCRATCH_BYTES`, the batch launched in slices that fit it); the
arithmetic, the pair ordering and the canonical form are the same source
lines, so the answer is still fixed by the input alone.  Before the global
tier an ensemble above 64 members fell back to cuSOLVER, which is not
bit-reproducible across cards.

:data:`MAX_K` is the largest size the test battery measures converging
inside :data:`SWEEP_CAP`; above it this module refuses rather than guess.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

__all__ = [
    "JacobiEighError",
    "MIN_K",
    "MAX_K",
    "SHARED_MAX_K",
    "GLOBAL_SCRATCH_BYTES",
    "SUPPORTED_DTYPES",
    "SWEEP_CAP",
    "Tier",
    "plan",
    "supported",
    "batched_eigh",
]

#: Kernel translation unit and entry point, named once so tests and the
#: kernel-manifest sweep do not restate them.
KERNEL_MODULE = "jacobi_eigh"
KERNEL_SYMBOL = "jacobi_eigh_batched"


class JacobiEighError(RuntimeError):
    """Any refusal by this solver.  Never raised for a merely hard matrix."""


#: Smallest problem the round-robin ordering is defined for.  A 1x1 matrix is
#: its own eigendecomposition and does not belong in a batched solver.
MIN_K = 2

#: Largest problem that fits two working copies in the opt-in shared-memory
#: limit of every architecture this project targets: the shared tiers.
SHARED_MAX_K = 64

#: Largest problem either tier takes.  Above :data:`SHARED_MAX_K` the global
#: tier runs; 256 is the largest size ``tests/test_jacobi_eigh_gpu.py``
#: measures converging inside :data:`SWEEP_CAP` and byte-identical run to
#: run.  Above it convergence inside the cap is unmeasured, and that is the
#: breakage the refusal prevents: a matrix still rotating at the cap is
#: refused only after the whole localised gather has been paid for.
MAX_K = 256

#: Target size of the global tier's scratch slab, in bytes.  Each matrix in
#: flight needs ``2 * m * m`` reals; a batch whose slab would exceed this is
#: launched in slices that fit, so the slab is a fixed overhead (inside the
#: share of the card ``gpuwm.da.letkf``'s auto-sizer leaves unpromised)
#: rather than a per-gridpoint cost.  Matrices are independent, so slicing
#: never moves a byte of the answer (``tests/test_jacobi_eigh_gpu.py``).
#: Sized to stay resident in L2: on an RTX 4090 (72 MB L2) a 2048-matrix
#: k = 128 batch took 0.95 s at 32 MiB and 3.8 s at 256 MiB, the same bytes.
GLOBAL_SCRATCH_BYTES = 32 * 1024 * 1024

#: Fewest matrices one global-tier launch carries, whatever the slab target,
#: so a large k still puts a block on most multiprocessors (k = 256 took
#: 1.4 s for 256 matrices with 64 in flight, 2.6 s with 32).
GLOBAL_MIN_IN_FLIGHT = 64

SUPPORTED_DTYPES = (np.dtype(np.float32), np.dtype(np.float64))

#: Sweeps after which a matrix is declared a failure rather than an answer.
#: Cyclic Jacobi converges quadratically once the off-diagonal is small; the
#: worst case measured across the test battery -- a 64x64 with a condition
#: number of 1e12, which is far outside anything the LETKF matrix can be --
#: took 19.  The cap is generous on purpose: it exists to make non-convergence
#: LOUD, not to bound the work.
SWEEP_CAP = 40

#: Threads per matrix by worked size.  At 32 the block barrier is a warp
#: barrier and several matrices share a block, which is what keeps a k = 10
#: problem from spending its life in ``__syncthreads``.  Above that one matrix
#: owns the block and the wider tiers buy back the per-thread work that a
#: single warp would otherwise serialise.
_THREAD_TIERS = ((32, 32), (48, 128), (64, 256), (128, 256), (256, 512))

#: Cap on matrices per block, so one tier cannot monopolise the shared memory
#: of a multiprocessor.
_MAX_MATRICES_PER_BLOCK = 8

_DEFAULT_SHARED_LIMIT = 48 * 1024


@dataclass(frozen=True)
class Tier:
    """The compiled specialisation for one ``(k, dtype)``."""

    k: int
    m: int
    threads_per_matrix: int
    matrices_per_block: int
    sweep_cap: int
    real_bytes: int
    shared_bytes_per_matrix: int
    shared_bytes: int
    #: ``True`` for the global-work tier (k above :data:`SHARED_MAX_K`).
    global_work: bool = False

    @property
    def global_bytes_per_matrix(self) -> int:
        """Scratch-slab bytes one matrix in flight needs (0 for shared tiers)."""
        return 2 * self.m * self.m * self.real_bytes if self.global_work else 0

    @property
    def defines(self) -> tuple[tuple[str, int], ...]:
        """Exactly what ``get_kernel_int_defines`` compiles with.

        ``JACOBI_M`` is deliberately absent: the kernel derives it from
        ``JACOBI_K`` in the preprocessor, so host and device cannot disagree
        about the padding.
        """
        defines = (
            ("JACOBI_K", self.k),
            ("JACOBI_TPB", self.threads_per_matrix),
            ("JACOBI_MPB", self.matrices_per_block),
            ("JACOBI_SWEEPS", self.sweep_cap),
            ("JACOBI_REAL_BYTES", self.real_bytes),
        )
        # Appended only for the global tier, so every shared tier compiles
        # from exactly the define tuple (and cache key) it always had.
        if self.global_work:
            defines += (("JACOBI_GLOBAL_WORK", 1),)
        return defines

    @property
    def block(self) -> tuple[int, int, int]:
        if self.threads_per_matrix == 32:
            return (32, self.matrices_per_block, 1)
        return (self.threads_per_matrix, 1, 1)


def _shared_per_matrix(m: int, real_bytes: int,
                       global_work: bool = False) -> int:
    # work[m*m] + vecs[m*m] + cos[m/2] + sin[m/2] + diag[m] + sign[m]
    # then perm[m] + flag[1] as int32.  The global tier keeps work and vecs
    # in the scratch slab instead.
    reals = 3 * m if global_work else 2 * m * m + 3 * m
    return reals * real_bytes + (m + 1) * 4


@lru_cache(maxsize=None)
def plan(k: int, dtype_str: str) -> Tier:
    """The specialisation for ``k``, or :class:`JacobiEighError` explaining why not.

    ``dtype_str`` rather than a dtype so the result memoises; callers pass
    ``np.dtype(x).str``.
    """
    dtype = np.dtype(dtype_str)
    if dtype not in SUPPORTED_DTYPES:
        raise JacobiEighError(
            f"jacobi_eigh solves in float32 or float64, not {dtype.name}."
        )
    k = int(k)
    if k < MIN_K or k > MAX_K:
        raise JacobiEighError(
            f"jacobi_eigh supports {MIN_K} <= k <= {MAX_K}, got k={k}."
            "  The two k x k working copies are held in shared memory up to"
            f" k = {SHARED_MAX_K} and in a global scratch slab above it;"
            f" {MAX_K} is the largest size measured converging inside the"
            f" {SWEEP_CAP}-sweep cap.  Use a library eigensolver for larger"
            " problems."
        )
    m = k + (k & 1)
    global_work = k > SHARED_MAX_K
    threads = next(t for bound, t in _THREAD_TIERS if m <= bound)
    per_matrix = _shared_per_matrix(m, dtype.itemsize, global_work)
    if threads == 32:
        per_block = max(
            1, min(_MAX_MATRICES_PER_BLOCK, _DEFAULT_SHARED_LIMIT // per_matrix))
    else:
        per_block = 1
    return Tier(
        k=k,
        m=m,
        threads_per_matrix=threads,
        matrices_per_block=per_block,
        sweep_cap=SWEEP_CAP,
        real_bytes=dtype.itemsize,
        shared_bytes_per_matrix=per_matrix,
        shared_bytes=per_matrix * per_block,
        global_work=global_work,
    )


def supported(k: int, dtype) -> bool:
    """Can this solver take ``(k, dtype)``?  Never raises, never touches a device."""
    try:
        plan(int(k), np.dtype(dtype).str)
    except (JacobiEighError, TypeError, ValueError):
        return False
    return True


@lru_cache(maxsize=None)
def _kernel(defines: tuple[tuple[str, int], ...], shared_bytes: int):
    from gpuwm.core.kernels import get_kernel_int_defines

    fn = get_kernel_int_defines(KERNEL_MODULE, KERNEL_SYMBOL, defines)
    if shared_bytes > _DEFAULT_SHARED_LIMIT:
        # Opt in, once, for the tiers whose two working copies do not fit the
        # 48 KiB default.  Raising this is a per-function attribute, so it has
        # to happen before the first launch of THIS specialisation.
        fn.max_dynamic_shared_size_bytes = shared_bytes
    return fn


def _launch_global(fn, tier: Tier, a, w, v, status, n: int) -> None:
    """The global tier: one matrix per block, the batch in slab-sized slices.

    The slab holds the working arrays of one slice and is reused slice to
    slice.  Each matrix reads and writes only its own stretch, indexed by its
    position in the slice, so the slicing is invisible in the answer.
    """
    import cupy as cp

    per_launch = max(1, min(n, max(GLOBAL_MIN_IN_FLIGHT,
                                   GLOBAL_SCRATCH_BYTES
                                   // tier.global_bytes_per_matrix)))
    slab = cp.empty((per_launch * 2 * tier.m * tier.m,), dtype=a.dtype)
    for lo in range(0, n, per_launch):
        hi = min(n, lo + per_launch)
        fn((hi - lo,), tier.block,
           (a[lo:hi], w[lo:hi], v[lo:hi], status[lo:hi],
            np.int64(hi - lo), slab),
           shared_mem=tier.shared_bytes)
    del slab


def batched_eigh(a, *, return_sweeps: bool = False):
    """``(w, v)`` for a stack of symmetric matrices, ascending, U in columns.

    Parameters
    ----------
    a
        CuPy array ``(n, k, k)``.  Only the values actually present are read;
        the matrix is NOT symmetrised here, because the caller that needs a
        symmetric input is the caller that knows which triangle is
        authoritative.  ``gpuwm.da.letkf`` averages the two triangles before
        it calls, exactly as it did for ``cupy.linalg.eigh``.
    return_sweeps
        Also return the largest sweep count any matrix in the batch needed.
        Free -- the launch already reports it per matrix, and the refusal
        check already reduces over that array -- and it is the only cheap
        early warning that a caller's matrices are worse conditioned than it
        believes.

    Returns
    -------
    ``(w, v)`` matching ``cupy.linalg.eigh``'s contract -- ``w`` ascending
    ``(n, k)``, ``v[i, :, j]`` the unit eigenvector for ``w[i, j]`` -- with
    the additional sign convention documented in the kernel: the
    largest-magnitude component of every eigenvector is positive, ties broken
    by the lowest row index.

    Raises
    ------
    JacobiEighError
        On an unsupported ``k`` or dtype, on a non-finite input, or on a
        matrix still rotating at the sweep cap.  Nothing is returned in those
        cases: a partially diagonalised matrix is not an answer and this
        module will not let one be mistaken for one.
    """
    import cupy as cp

    if a.ndim != 3 or a.shape[1] != a.shape[2]:
        raise JacobiEighError(
            f"jacobi_eigh wants a stack of square matrices (n, k, k), got"
            f" shape {tuple(a.shape)}."
        )
    n, k, _ = (int(x) for x in a.shape)
    tier = plan(k, np.dtype(a.dtype).str)
    a = cp.ascontiguousarray(a)

    w = cp.empty((n, k), dtype=a.dtype)
    v = cp.empty((n, k, k), dtype=a.dtype)
    status = cp.empty((n,), dtype=cp.int32)
    if n == 0:
        return (w, v, 0) if return_sweeps else (w, v)

    fn = _kernel(tier.defines, tier.shared_bytes)
    if tier.global_work:
        _launch_global(fn, tier, a, w, v, status, n)
    else:
        blocks = (n + tier.matrices_per_block - 1) // tier.matrices_per_block
        fn((blocks,), tier.block,
           (a, w, v, status, np.int64(n)),
           shared_mem=tier.shared_bytes)

    worst = int(status.min())
    if worst < 0:
        if worst == -2:
            bad = int((status == -2).sum())
            raise JacobiEighError(
                f"jacobi_eigh was handed a non-finite entry in {bad} of {n}"
                " matrices.  No eigendecomposition was produced for them; the"
                " caller's own finiteness guard is the place to fix this."
            )
        bad = int((status == -1).sum())
        raise JacobiEighError(
            f"jacobi_eigh reached its {tier.sweep_cap}-sweep cap on {bad} of"
            f" {n} matrices of size {k} without the off-diagonal falling"
            " below threshold.  These are not answers and are not returned."
            "  A symmetric matrix that resists this many cyclic Jacobi sweeps"
            " is either not symmetric or is conditioned far outside anything"
            " the caller believes it is producing."
        )
    if return_sweeps:
        return w, v, int(status.max())
    return w, v
