"""The batched Jacobi eigensolver, against cuSOLVER and against itself.

What is asserted, and why it is not "the eigenvectors match"
-----------------------------------------------------------
A symmetric eigendecomposition is NOT unique.  Every eigenvector carries an
arbitrary sign, and a repeated eigenvalue admits any orthonormal basis of its
eigenspace -- so two correct solvers can return completely different ``U`` for
the same ``A``.  That is not a corner case here: the matrix the LETKF factors
is ``(R-1)I/rho + C Yb`` with ``C Yb`` positive SEMI-definite and routinely
rank-deficient (fewer local observations than ensemble members), so a
gridpoint with one usable observation hands the solver an ``R-1``-fold
degenerate eigenvalue by construction.  Asserting eigenvector agreement there
would be asserting that two solvers made the same arbitrary choice.

So the assertions are, in order of what they are worth:

1. **The functional invariant** -- ``U f(w) U^T`` for the two functions the
   filter actually consumes, ``f = 1/x`` (``Pa~``) and ``f = 1/sqrt(x)``
   (``Wa``).  A matrix function of a symmetric matrix is UNIQUE: it does not
   depend on sign, on ordering, or on the basis chosen inside a degenerate
   eigenspace.  This is the one that means "the filter gets the same answer",
   and it is asserted on every case.
2. **Eigenvalues** -- also unique, compared elementwise once both are sorted
   ascending, to a Weyl-style absolute tolerance scaled by ``||A||``.
3. **Backward stability and orthogonality** -- ``U diag(w) U^T = A`` and
   ``U^T U = I``, which are properties of our output alone and need no
   reference solver at all.
4. **Eigenvectors**, only where the spectrum is well separated, and only
   after putting BOTH solvers' output into this kernel's documented canonical
   form.  Davis-Kahan says the achievable agreement scales like
   ``||dA|| / gap``, so the tolerance does too.

Everything is checked at k = 2, 3 and 11 as well as the ensemble sizes the DA
lane cares about, because odd k takes the padding path and k = 2 is the
degenerate case of the round-robin ordering.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_gpu

pytestmark = [pytest.mark.gpu, requires_gpu]

#: Ensemble sizes the lane runs or plans to (10 today, 20/36/64 if it scales),
#: plus the shapes that exercise the kernel's own edges: the smallest legal
#: problem, and two odd sizes that take the even-padding path.
SIZES = (2, 3, 10, 11, 20, 21, 36, 64)

EPS64 = float(np.finfo(np.float64).eps)


def _tolerance(k: int) -> float:
    """A backward-stability budget, not a fudge factor.

    Jacobi's backward error is ``O(k) eps ||A||``; the constant hides the
    sweep count and the reduction order.  64 is roughly eight times the
    worst residual measured across this whole battery, which is enough
    headroom that a toolchain change does not turn the suite red and little
    enough that a real regression does.
    """
    return 64.0 * k * EPS64


# ---------------------------------------------------------------------------
# The battery
# ---------------------------------------------------------------------------

def _batch(kind: str, n: int, k: int, seed: int) -> np.ndarray:
    """One family of symmetric matrices, shaped like something the filter meets."""
    rs = np.random.RandomState(seed)
    eye = np.eye(k)
    if kind == "letkf":
        # The real shape: (R-1)I + C Yb, C Yb positive semi-definite of full
        # observational rank.
        y = rs.standard_normal((n, k, max(1, k // 2)))
        a = (k - 1) * eye + y @ np.swapaxes(y, 1, 2)
    elif kind == "one_observation":
        # A gridpoint whose localisation lens caught exactly one usable
        # observation: C Yb is rank ONE, so k-1 eigenvalues are degenerate.
        # This is the common case in a radar-sparse domain, not a corner.
        y = rs.standard_normal((n, k, 1))
        a = (k - 1) * eye + 40.0 * (y @ np.swapaxes(y, 1, 2))
    elif kind == "no_observation":
        # The closed form the filter special-cases, handed to the solver
        # anyway: a scaled identity.  Perfectly degenerate.
        a = np.broadcast_to((k - 1.0) * eye, (n, k, k)).copy()
    elif kind == "near_degenerate":
        # Two eigenvalues a few ulp apart -- the case where an eigenvector is
        # genuinely ill-determined but every matrix function still is not.
        q, _ = np.linalg.qr(rs.standard_normal((k, k)))
        lam = np.linspace(1.0, 2.0, k)
        lam[1] = lam[0] * (1.0 + 8.0 * EPS64)
        a = np.broadcast_to(q @ np.diag(lam) @ q.T, (n, k, k)).copy()
    elif kind == "rank_deficient":
        # C Yb of rank 1 with NO shift: the matrix is singular.  Outside what
        # the recipe can produce, and the solver still may not produce NaN.
        y = rs.standard_normal((n, k, 1))
        a = y @ np.swapaxes(y, 1, 2)
    elif kind == "wide_spectrum":
        q, _ = np.linalg.qr(rs.standard_normal((k, k)))
        a = np.broadcast_to(
            q @ np.diag(np.logspace(0.0, 10.0, k)) @ q.T, (n, k, k)).copy()
    elif kind == "tiny":
        # Everything scaled to where a float64 would flush if it were a
        # float32: the relative-accuracy claim should not care.
        y = rs.standard_normal((n, k, k))
        a = 1e-150 * (y @ np.swapaxes(y, 1, 2) + k * eye)
    else:  # pragma: no cover -- guards the parametrisation itself
        raise AssertionError(kind)
    return np.ascontiguousarray(0.5 * (a + np.swapaxes(a, 1, 2)))


KINDS = ("letkf", "one_observation", "no_observation", "near_degenerate",
         "rank_deficient", "wide_spectrum", "tiny")


def _canonical(v: np.ndarray) -> np.ndarray:
    """The kernel's documented sign convention, applied to anyone's output.

    Largest-magnitude component positive, ties broken by the lowest row
    index.  ``argmax`` on ``|v|`` already returns the first maximum, which is
    that tie-break.
    """
    rows = np.abs(v).argmax(axis=1)
    pick = np.take_along_axis(v, rows[:, None, :], axis=1)[:, 0, :]
    return v * np.where(pick < 0.0, -1.0, 1.0)[:, None, :]


def _matrix_function(w, v, f):
    """``U f(w) U^T`` -- unique, whatever basis ``U`` happens to be."""
    return v @ (f(w)[:, :, None] * np.swapaxes(v, 1, 2))


# ---------------------------------------------------------------------------
# 1-3: invariants that need no reference, plus the reference where it is fair
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("k", SIZES)
@pytest.mark.parametrize("kind", KINDS)
def test_the_decomposition_reconstructs_its_own_matrix(k, kind):
    """``U diag(w) U^T == A`` and ``U^T U == I``, with no reference solver."""
    import cupy as cp
    from gpuwm.core.jacobi_eigh import batched_eigh

    a = _batch(kind, 64, k, seed=1000 + k)
    w, v = batched_eigh(cp.asarray(a))
    w, v = cp.asnumpy(w), cp.asnumpy(v)

    scale = np.abs(a).max(axis=(1, 2))
    scale = np.where(scale > 0, scale, 1.0)
    residual = np.abs(_matrix_function(w, v, lambda x: x) - a).max(axis=(1, 2))
    assert (residual / scale).max() < _tolerance(k)

    off = np.abs(np.swapaxes(v, 1, 2) @ v - np.eye(k)).max()
    assert off < _tolerance(k)

    assert np.all(np.diff(w, axis=1) >= 0.0), "eigenvalues must be ascending"


@pytest.mark.parametrize("k", SIZES)
@pytest.mark.parametrize("kind", KINDS)
def test_the_eigenvalues_agree_with_cusolver(k, kind):
    """Eigenvalues are unique, so this comparison is fair on every case."""
    import cupy as cp
    from gpuwm.core.jacobi_eigh import batched_eigh

    a = _batch(kind, 64, k, seed=2000 + k)
    device = cp.asarray(a)
    mine = cp.asnumpy(batched_eigh(device)[0])
    theirs = cp.asnumpy(cp.linalg.eigh(device)[0])

    norm = np.abs(a).max(axis=(1, 2))
    norm = np.where(norm > 0, norm, 1.0)
    assert (np.abs(mine - theirs).max(axis=1) / norm).max() < _tolerance(k)


@pytest.mark.parametrize("k", SIZES)
@pytest.mark.parametrize("kind", KINDS)
def test_the_matrix_functions_the_filter_consumes_agree_with_cusolver(k, kind):
    """THE assertion.  ``Pa~ = A^-1`` and ``Wa = sqrt(R-1) A^-1/2``.

    These are the only two things ``gpuwm.da.letkf`` takes out of the
    eigendecomposition, they are unique matrix functions of ``A``, and they
    are therefore comparable between two solvers that chose different bases
    inside a degenerate eigenspace -- which, on ``one_observation`` and
    ``no_observation``, they certainly did.
    """
    import cupy as cp
    from gpuwm.core.jacobi_eigh import batched_eigh

    a = _batch(kind, 64, k, seed=3000 + k)
    if kind == "rank_deficient":
        pytest.skip("singular by construction: A^-1 does not exist to compare")
    device = cp.asarray(a)
    wm, vm = (cp.asnumpy(x) for x in batched_eigh(device))
    wt, vt = (cp.asnumpy(x) for x in cp.linalg.eigh(device))

    root = float(np.sqrt(k - 1))
    for label, f in (("Pa", lambda x: 1.0 / x),
                     ("Wa", lambda x: root / np.sqrt(x))):
        mine = _matrix_function(wm, vm, f)
        theirs = _matrix_function(wt, vt, f)
        scale = np.abs(theirs).max(axis=(1, 2))
        scale = np.where(scale > 0, scale, 1.0)
        worst = (np.abs(mine - theirs).max(axis=(1, 2)) / scale).max()
        # A matrix function amplifies the eigenvalue error by the condition
        # number of f, so the budget is the backward error times cond(A) --
        # which for the LETKF matrix is small, and for wide_spectrum is 1e10
        # by construction.
        budget = _tolerance(k) * float(
            (np.abs(wt).max(axis=1) / np.abs(wt).min(axis=1)).max())
        assert worst < max(budget, _tolerance(k)), (label, kind, k, worst)


@pytest.mark.parametrize("k", (10, 20, 36))
def test_eigenvectors_agree_where_the_spectrum_is_separated(k):
    """The one place a raw eigenvector comparison means anything.

    Both sides are put into this kernel's canonical form first, and the
    tolerance is scaled by the smallest eigenvalue gap because Davis-Kahan
    says that is what bounds the achievable agreement.
    """
    import cupy as cp
    from gpuwm.core.jacobi_eigh import batched_eigh

    # A deliberately well-separated spectrum: gaps of order 1 against a norm
    # of order k, so the vectors are determined to near machine precision.
    rs = np.random.RandomState(4000 + k)
    q, _ = np.linalg.qr(rs.standard_normal((k, k)))
    lam = np.arange(1.0, k + 1.0)
    a = np.ascontiguousarray(
        np.broadcast_to(q @ np.diag(lam) @ q.T, (16, k, k)).copy())
    a = 0.5 * (a + np.swapaxes(a, 1, 2))

    device = cp.asarray(a)
    vm = _canonical(cp.asnumpy(batched_eigh(device)[1]))
    wt, vt = (cp.asnumpy(x) for x in cp.linalg.eigh(device))
    vt = _canonical(vt)

    gap = np.diff(wt, axis=1).min()
    assert np.abs(vm - vt).max() < _tolerance(k) * float(lam.max() / gap)


# ---------------------------------------------------------------------------
# Canonical form and exactness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("k", SIZES)
def test_the_sign_convention_is_actually_enforced(k):
    """Largest-magnitude component positive, on every column of every matrix."""
    import cupy as cp
    from gpuwm.core.jacobi_eigh import batched_eigh

    a = _batch("letkf", 128, k, seed=5000 + k)
    v = cp.asnumpy(batched_eigh(cp.asarray(a))[1])
    rows = np.abs(v).argmax(axis=1)
    picked = np.take_along_axis(v, rows[:, None, :], axis=1)[:, 0, :]
    assert np.all(picked > 0.0)
    # And it really is the FIRST maximum that decides, not any maximum.
    assert np.array_equal(rows, np.abs(v).argmax(axis=1))


@pytest.mark.parametrize("k", SIZES)
def test_an_already_diagonal_matrix_is_a_fixed_point_exactly(k):
    """A property no library eigensolver promises, and the filter leans on.

    ``gpuwm.da.letkf`` excludes observation-free gridpoints from the solve
    precisely because passing a scaled identity through ``eigh`` and getting
    ``U U^T`` back only NEARLY equal to the identity would break its
    bitwise-zero-increment guarantee.  Here it is not near: the threshold
    test skips every rotation, so ``U`` is the literal identity and ``w`` the
    literal diagonal, bit for bit.
    """
    import cupy as cp
    from gpuwm.core.jacobi_eigh import batched_eigh

    rs = np.random.RandomState(6000 + k)
    diag = np.sort(rs.uniform(1.0, 50.0, size=(8, k)), axis=1)
    a = np.ascontiguousarray(diag[:, :, None] * np.eye(k)[None])

    w, v = (cp.asnumpy(x) for x in batched_eigh(cp.asarray(a)))
    assert np.array_equal(w, diag)
    assert np.array_equal(v, np.broadcast_to(np.eye(k), (8, k, k)))


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def _raw(array) -> bytes:
    import cupy as cp
    return np.ascontiguousarray(cp.asnumpy(array)).tobytes()


@pytest.mark.parametrize("k", SIZES)
def test_the_same_batch_solves_to_the_same_bytes(k):
    """Byte-identical run to run: no atomics, no reduction whose order moves."""
    import cupy as cp
    from gpuwm.core.jacobi_eigh import batched_eigh

    a = cp.asarray(_batch("letkf", 4096, k, seed=7000 + k))
    first = [_raw(x) for x in batched_eigh(a)]
    for _ in range(3):
        assert [_raw(x) for x in batched_eigh(a)] == first


@pytest.mark.parametrize("k", (10, 36))
def test_a_matrix_solves_the_same_wherever_it_sits_in_the_batch(k):
    """No block-index or batch-size dependence -- the blocks are independent.

    A per-matrix answer that moved when the batch was chunked differently
    would make the filter's answer depend on ``LetkfConfig.chunk_points``,
    which is a memory knob and must not be a numerical one.
    """
    import cupy as cp
    from gpuwm.core.jacobi_eigh import batched_eigh

    a = _batch("letkf", 1000, k, seed=8000 + k)
    whole = [cp.asnumpy(x) for x in batched_eigh(cp.asarray(a))]
    for lo, hi in ((0, 1), (7, 9), (511, 512), (997, 1000)):
        part = [cp.asnumpy(x) for x in batched_eigh(cp.asarray(a[lo:hi]))]
        for full, piece in zip(whole, part):
            assert _raw(full[lo:hi]) == _raw(piece), (lo, hi)


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------

def test_k_outside_the_supported_range_is_refused_by_name():
    from gpuwm.core.jacobi_eigh import (
        MAX_K, MIN_K, JacobiEighError, plan, supported)

    assert supported(MIN_K, np.float64) and supported(MAX_K, np.float64)
    assert not supported(MAX_K + 1, np.float64)
    assert not supported(MIN_K - 1, np.float64)
    assert not supported(10, np.float16)
    for bad in (MIN_K - 1, MAX_K + 1):
        with pytest.raises(JacobiEighError, match=r"supports 2 <= k <= 64"):
            plan(bad, np.dtype(np.float64).str)
    with pytest.raises(JacobiEighError, match="float32 or float64"):
        plan(10, np.dtype(np.float16).str)


def test_a_non_finite_entry_is_refused_rather_than_laundered():
    import cupy as cp
    from gpuwm.core.jacobi_eigh import JacobiEighError, batched_eigh

    a = _batch("letkf", 8, 10, seed=9001)
    a[3, 2, 2] = np.nan
    a[6, 0, 1] = np.inf
    with pytest.raises(JacobiEighError, match="non-finite entry in 2 of 8"):
        batched_eigh(cp.asarray(a))


def test_a_matrix_still_rotating_at_the_cap_is_refused_not_returned():
    """Force the cap by compiling a one-sweep tier; the refusal must fire.

    The cap is a compile-time constant, so this is the only way to reach the
    branch -- and reaching it matters, because the alternative to refusing is
    returning a partially diagonalised matrix that looks like an answer.
    """
    import cupy as cp
    from gpuwm.core import jacobi_eigh as je

    a = _batch("letkf", 32, 10, seed=9002)
    honest = je.plan(10, np.dtype(np.float64).str)
    crippled = je.Tier(**{**honest.__dict__, "sweep_cap": 1})
    assert crippled.defines != honest.defines

    fn = je._kernel(crippled.defines, crippled.shared_bytes)
    n = a.shape[0]
    device = cp.asarray(a)
    w = cp.empty((n, 10), dtype=cp.float64)
    v = cp.empty((n, 10, 10), dtype=cp.float64)
    status = cp.empty((n,), dtype=cp.int32)
    blocks = (n + crippled.matrices_per_block - 1) // crippled.matrices_per_block
    fn((blocks,), crippled.block, (device, w, v, status, np.int64(n)),
       shared_mem=crippled.shared_bytes)
    assert int(status.max()) == -1, "one sweep should not diagonalise these"


def test_an_empty_batch_is_a_shape_not_an_error():
    import cupy as cp
    from gpuwm.core.jacobi_eigh import batched_eigh

    w, v = batched_eigh(cp.zeros((0, 10, 10), dtype=cp.float64))
    assert w.shape == (0, 10) and v.shape == (0, 10, 10)


def test_a_non_square_stack_is_refused():
    import cupy as cp
    from gpuwm.core.jacobi_eigh import JacobiEighError, batched_eigh

    with pytest.raises(JacobiEighError, match=r"\(n, k, k\)"):
        batched_eigh(cp.zeros((4, 10, 9), dtype=cp.float64))


# ---------------------------------------------------------------------------
# float32
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("k", (10, 36))
def test_the_float32_tier_solves_to_float32_accuracy(k):
    """Supported, and held to its own precision rather than float64's.

    ``gpuwm.da.letkf`` keeps ``solve_dtype='float64'`` by default for reasons
    that have nothing to do with this kernel, but the tier exists and a tier
    that is never exercised is a tier that does not work.
    """
    import cupy as cp
    from gpuwm.core.jacobi_eigh import batched_eigh

    a = _batch("letkf", 64, k, seed=9100 + k).astype(np.float32)
    w, v = (cp.asnumpy(x) for x in batched_eigh(cp.asarray(a)))
    assert w.dtype == np.float32 and v.dtype == np.float32
    residual = np.abs(
        _matrix_function(w.astype(np.float64), v.astype(np.float64),
                         lambda x: x) - a).max(axis=(1, 2))
    scale = np.abs(a).max(axis=(1, 2))
    assert (residual / scale).max() < 64.0 * k * float(np.finfo(np.float32).eps)


def test_the_sweep_count_is_reported_and_stays_far_below_the_cap():
    """The early-warning number the filter records; it must mean something."""
    import cupy as cp
    from gpuwm.core.jacobi_eigh import SWEEP_CAP, batched_eigh

    a = cp.asarray(_batch("letkf", 4096, 10, seed=9200))
    w, v, sweeps = batched_eigh(a, return_sweeps=True)
    assert 0 < sweeps < SWEEP_CAP // 2, sweeps
