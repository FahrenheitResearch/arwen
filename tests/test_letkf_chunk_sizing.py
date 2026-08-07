"""The LETKF chunk sizing model and its out-of-memory degradation path.

Pinned at the geometry that failed in the field.  On 2026-08-05 the
Level-II A/B sweep's three single-radar arms all died at the same place:
``letkf.py``'s eigenvalue check, with ``cudaErrorMemoryAllocation``, on a
card with roughly thirty gigabytes free.  The retired heuristic priced
only the member-slot arrays, called its budget "not a hard limit" in its
own docstring, and from ``--memory-budget-mib 6144`` chose 7248-point
chunks against an 1845-slot single-radar stencil -- while the three-radar
arms (3 x 1845 = 5535 slots), sized by the same formula to 2421 points,
solved twice the observation load on the same card without incident.

What this file pins:

* the sizing arithmetic at exactly that shape -- the budget is a ceiling
  the chosen chunk provably stays under, and the 7248-point choice cannot
  recur from the same inputs;
* the ceiling the card imposes, which outranks the budget when the two
  disagree, and the refusal that names whichever of the two actually
  bound;
* the degradation path -- a device allocation failure mid-analysis halves
  the chunk and re-solves the same span instead of failing the analysis,
  bit-identically here on the host and to a few ulp on a card, for the
  reason the relevant test states;
* the refusal -- raised by name, with the remedy and the required figure
  in the message, only when even a single gridpoint cannot fit.

No cupy import anywhere in this module, deliberately: the sizing
arithmetic is pure, the degradation driver is exercised through numpy
with synthetic allocation failures, and ``tests/conftest.py`` auto-marks
any cupy-importing module as ``gpu``.  The on-device counterpart lives in
``tests/test_letkf_chunk_sizing_gpu.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

import gpuwm.da.letkf as letkf_mod
from gpuwm.da.letkf import (
    _DEVICE_FREE_FRACTION,
    GriddedObs,
    GridGeometry,
    LetkfConfig,
    LetkfDiagnostics,
    LetkfError,
    Localization,
    _is_device_memory_error,
    analyze,
    chunk_points_for_budget,
    solve_bytes_per_point,
)

# The failing configuration of record: A1-fixed900-1radar-chained cycle02
# (and B1/B2 cycle05), Level-II A/B sweep, 2026-08-05.  Numbers from the
# cycle reports and the failure triage, not invented for the test.
FAILING_SLOTS = 1845
FAILING_MEMBERS = 10
FAILING_NPTS = 853776
FAILING_ITEMSIZE = 8                       # float64 solve
BUDGET_BYTES = 6144 * (1 << 20)            # --memory-budget-mib 6144
OLD_CHUNK = 7248                           # what the retired heuristic chose
MULTIRADAR_SLOTS = 3 * FAILING_SLOTS       # the arms that survived
PASSING_CHUNK = 2421                       # ... at this chunk


def _tiny_case(members=6, nz=3, ny=6, nx=6, seed=11, fields=("theta", "u")):
    """A domain small enough that every path here is instant."""
    rng = np.random.default_rng(seed)
    shape = (nz, ny, nx)
    grid = GridGeometry(
        dx_m=1000.0, dy_m=1000.0,
        heights_m=np.array([250.0, 800.0, 1600.0][:nz]),
    )
    prior = {
        f: rng.standard_normal((members,) + shape) + 5.0 * i
        for i, f in enumerate(fields)
    }
    mask = np.zeros(shape, dtype=bool)
    mask.reshape(-1)[rng.choice(nz * ny * nx, size=10, replace=False)] = True
    sim = prior[fields[0]] * 0.8 + 0.1
    values = np.where(
        mask, sim.mean(axis=0) + rng.standard_normal(shape) * 0.4, np.nan)
    obs = GriddedObs(name="probe", values=values, errors=0.4, simulated=sim,
                     mask=mask)
    return grid, prior, obs, fields


# ---------------------------------------------------------------------------
# The sizing arithmetic, at the shape that failed
# ---------------------------------------------------------------------------

def test_failing_shape_now_sizes_under_the_budget():
    """6144 MiB at 1845 slots x R10 f64 must not choose 7248 again."""
    per_point = solve_bytes_per_point(
        FAILING_SLOTS, FAILING_MEMBERS, FAILING_ITEMSIZE)
    chunk = chunk_points_for_budget(
        FAILING_SLOTS, FAILING_MEMBERS, FAILING_ITEMSIZE,
        BUDGET_BYTES, FAILING_NPTS)
    assert chunk >= 1
    # The property the word "budget" promises: chunk x cost <= budget.
    assert chunk * per_point <= BUDGET_BYTES
    # The field failure cannot recur from the same inputs.
    assert chunk < OLD_CHUNK
    # And the fix is not a throughput lobotomy: the solve saturates near
    # 2048 points (module docstring); staying within a factor of a few of
    # that keeps the batched win.
    assert chunk >= 512


def test_multiradar_shape_still_saturates():
    """The three-radar stencil keeps a card-filling chunk under the budget."""
    per_point = solve_bytes_per_point(
        MULTIRADAR_SLOTS, FAILING_MEMBERS, FAILING_ITEMSIZE)
    chunk = chunk_points_for_budget(
        MULTIRADAR_SLOTS, FAILING_MEMBERS, FAILING_ITEMSIZE,
        BUDGET_BYTES, FAILING_NPTS)
    assert chunk * per_point <= BUDGET_BYTES
    assert chunk >= 512


@pytest.mark.parametrize("slots,members,itemsize", [
    (FAILING_SLOTS, FAILING_MEMBERS, 8),
    (MULTIRADAR_SLOTS, FAILING_MEMBERS, 8),
    (405, 30, 8),          # the docstring's 8 km / 2 km / R30 example
    (69, 8, 4),            # small stencil, float32 solve
    (1, 2, 4),             # degenerate floor
])
@pytest.mark.parametrize("budget_mib", [0.25, 64, 512, 6144])
def test_budget_is_a_ceiling_everywhere(slots, members, itemsize, budget_mib):
    """For every shape: the chunk fits, or it is 0 and the caller refuses."""
    budget = int(budget_mib * (1 << 20))
    per_point = solve_bytes_per_point(slots, members, itemsize)
    chunk = chunk_points_for_budget(slots, members, itemsize, budget, 10 ** 9)
    if per_point > budget:
        assert chunk == 0
    else:
        assert chunk >= 1
        assert chunk * per_point <= budget


def test_the_model_prices_more_than_the_member_slot_block():
    """Slot-only scratch is real memory; the model must not omit it again.

    The retired heuristic priced ``6 * slots * members * itemsize`` and
    nothing per-slot, which is exactly how it under-read the single-radar
    geometry.  The replacement must price the index, weight and mask
    arrays too: strictly more than any pure member-slot term, and growing
    when only the slot-only terms grow.
    """
    member_slot_only = 6 * FAILING_SLOTS * FAILING_MEMBERS * FAILING_ITEMSIZE
    assert solve_bytes_per_point(
        FAILING_SLOTS, FAILING_MEMBERS, FAILING_ITEMSIZE) > member_slot_only
    # float32 solve halves the member-slot block but not the float64
    # distance scratch or the int64 gather indices: the price must not
    # halve with it.
    f64 = solve_bytes_per_point(FAILING_SLOTS, FAILING_MEMBERS, 8)
    f32 = solve_bytes_per_point(FAILING_SLOTS, FAILING_MEMBERS, 4)
    assert f32 > f64 // 2


# ---------------------------------------------------------------------------
# The refusal, by name
# ---------------------------------------------------------------------------

def test_refusal_names_the_knob_and_the_figure():
    """When one gridpoint cannot fit, say which knob and how much."""
    grid, prior, obs, fields = _tiny_case()
    config = LetkfConfig(
        localization=Localization(horizontal_m=3000.0, vertical_m=1500.0),
        analysis_fields=fields, rtps_alpha=0.0,
        memory_budget_mib=0.01)
    with pytest.raises(LetkfError,
                       match=r"Raise memory_budget_mib to at least \d+"):
        analyze(prior, [obs], grid, config)


def test_explicit_chunk_points_bypasses_the_budget():
    """An explicit chunk is the operator's override: no budget refusal."""
    grid, prior, obs, fields = _tiny_case()
    d = LetkfDiagnostics()
    inc = analyze(prior, [obs], grid, LetkfConfig(
        localization=Localization(horizontal_m=3000.0, vertical_m=1500.0),
        analysis_fields=fields, rtps_alpha=0.0,
        chunk_points=13, memory_budget_mib=0.01), diagnostics=d)
    assert d.chunk_points_initial == 13
    assert d.chunk_oom_shrinks == 0
    for f in fields:
        assert np.all(np.isfinite(inc[f]))


# ---------------------------------------------------------------------------
# The card outranks the budget
# ---------------------------------------------------------------------------
# The budget is what the caller PROMISED itself; the free-memory reading
# is what the card will actually honour, and the chained cycling layout is
# where they disagree.  A fresh process gets a clean card; a chained one
# inherits whatever the forecast legs before it left resident, and the
# budget knows nothing about that.  Both tests below drive the real
# analyze() with the reading stubbed, because a test that allocated its
# way down to a chosen free figure would be a test of whatever else is on
# the box that day.


def test_the_card_outranks_the_budget(monkeypatch):
    """A generous budget does not survive a card that has no room."""
    grid, prior, obs, fields = _tiny_case()
    loc = Localization(horizontal_m=3000.0, vertical_m=1500.0)
    unconstrained = LetkfDiagnostics()
    baseline = analyze(prior, [obs], grid, LetkfConfig(
        localization=loc, analysis_fields=fields, rtps_alpha=0.0,
        memory_budget_mib=4096.0), diagnostics=unconstrained)
    per_point = unconstrained.solve_bytes_per_point
    assert per_point > 0

    # A card reporting four gridpoints' worth free, against a budget that
    # would happily promise the whole domain.  Four times the fraction is
    # 3.2, and the sizer floors, so the arithmetic says three -- stated as
    # a multiple of the price rather than solved backwards from the answer,
    # because dividing by the fraction lands a hair under a whole point
    # and turns this into a test of rounding.
    free = 4 * per_point
    monkeypatch.setattr(letkf_mod, "_device_free_bytes", lambda xp: free)
    constrained = LetkfDiagnostics()
    inc = analyze(prior, [obs], grid, LetkfConfig(
        localization=loc, analysis_fields=fields, rtps_alpha=0.0,
        memory_budget_mib=4096.0), diagnostics=constrained)

    assert constrained.chunk_points == 3
    assert constrained.chunk_points < unconstrained.chunk_points
    assert constrained.chunk_oom_shrinks == 0
    # Smaller batches, same answer -- on the host, to the byte.
    for f in fields:
        assert np.array_equal(inc[f], baseline[f]), f


def test_the_refusal_names_the_card_when_the_card_is_the_limit(monkeypatch):
    """Two ceilings, two remedies: say which one actually bound.

    "Raise memory_budget_mib" is the wrong advice when the budget was
    never the constraint, and an operator who follows it learns nothing.
    """
    grid, prior, obs, fields = _tiny_case()
    monkeypatch.setattr(letkf_mod, "_device_free_bytes", lambda xp: 4096)
    with pytest.raises(LetkfError, match=r"set by the card \(0 MiB free"):
        analyze(prior, [obs], grid, LetkfConfig(
            localization=Localization(horizontal_m=3000.0, vertical_m=1500.0),
            analysis_fields=fields, rtps_alpha=0.0,
            memory_budget_mib=4096.0))


def test_an_unanswerable_card_falls_back_to_the_budget(monkeypatch):
    """A card that will not say how much is free is not a refusal."""
    grid, prior, obs, fields = _tiny_case()
    loc = Localization(horizontal_m=3000.0, vertical_m=1500.0)
    monkeypatch.setattr(letkf_mod, "_device_free_bytes", lambda xp: None)
    d = LetkfDiagnostics()
    inc = analyze(prior, [obs], grid, LetkfConfig(
        localization=loc, analysis_fields=fields, rtps_alpha=0.0,
        memory_budget_mib=4096.0), diagnostics=d)
    assert d.chunk_points >= 1
    assert d.chunk_points * d.solve_bytes_per_point <= 4096 * (1 << 20)
    for f in fields:
        assert np.all(np.isfinite(inc[f]))


# ---------------------------------------------------------------------------
# The degradation path
# ---------------------------------------------------------------------------

def _flaky_eigendecompose(failures, real, exc_factory):
    """The real eigensolver seam, refusing its first ``failures`` calls.

    ``_eigendecompose`` is called exactly once per chunk that carries an
    active gridpoint, inside the driver's try, which makes it the precise
    place a device allocation failure surfaced in the field (the batched
    eigensolve is the deepest allocation of phase 2).
    """
    calls = {"n": 0}

    def wrapper(xp, amat, which):
        calls["n"] += 1
        if calls["n"] <= failures:
            raise exc_factory()
        return real(xp, amat, which)

    return wrapper


def test_allocation_failure_halves_the_chunk_and_changes_nothing(monkeypatch):
    """OOM mid-analysis degrades the chunk; the increments are bitwise same.

    Bitwise HERE, and the qualifier is load-bearing.  Each gridpoint's
    transform is mathematically independent of how gridpoints are
    batched, and numpy honours that to the byte, because its batched
    eigensolve and its matmuls are per-matrix loops whose summation order
    does not know the batch extent.  The DEVICE does not: cuBLAS and the
    batched eigensolver pick work partitionings from the batch extent, so
    the same gridpoint's sums land in a different order and the result
    moves by a few ulp.  Measured on an RTX 3090 (sm_86, cupy 14.1.1,
    float64): at most 3.1e-15 absolute across chunks of 16, 8 and 1
    against a 32-point reference.  ``test_letkf_chunk_sizing_gpu.py``
    pins that as a tolerance; this file pins the exactness that survives
    on the host, so a regression that made the degradation path an
    ACCURACY path -- dropped points, double-counted increments, a
    different localisation -- fails here rather than hiding under a
    tolerance chosen to accommodate it.
    """
    grid, prior, obs, fields = _tiny_case()
    loc = Localization(horizontal_m=3000.0, vertical_m=1500.0)

    baseline = analyze(prior, [obs], grid, LetkfConfig(
        localization=loc, analysis_fields=fields, rtps_alpha=0.0,
        chunk_points=16))

    flaky = _flaky_eigendecompose(
        1, letkf_mod._eigendecompose,
        lambda: MemoryError("synthetic allocation failure"))
    monkeypatch.setattr(letkf_mod, "_eigendecompose", flaky)
    d = LetkfDiagnostics()
    degraded = analyze(prior, [obs], grid, LetkfConfig(
        localization=loc, analysis_fields=fields, rtps_alpha=0.0,
        chunk_points=16), diagnostics=d)

    assert d.chunk_points_initial == 16
    assert d.chunk_oom_shrinks == 1
    assert d.chunk_points == 8
    for f in fields:
        assert np.array_equal(baseline[f], degraded[f]), f


def test_repeated_failures_keep_halving_down_to_one(monkeypatch):
    """Three refusals: 16 -> 8 -> 4 -> 2, and the analysis still lands."""
    grid, prior, obs, fields = _tiny_case()
    loc = Localization(horizontal_m=3000.0, vertical_m=1500.0)

    baseline = analyze(prior, [obs], grid, LetkfConfig(
        localization=loc, analysis_fields=fields, rtps_alpha=0.0,
        chunk_points=16))

    flaky = _flaky_eigendecompose(
        3, letkf_mod._eigendecompose,
        lambda: MemoryError("synthetic allocation failure"))
    monkeypatch.setattr(letkf_mod, "_eigendecompose", flaky)
    d = LetkfDiagnostics()
    degraded = analyze(prior, [obs], grid, LetkfConfig(
        localization=loc, analysis_fields=fields, rtps_alpha=0.0,
        chunk_points=16), diagnostics=d)

    assert d.chunk_oom_shrinks == 3
    assert d.chunk_points == 2
    for f in fields:
        assert np.array_equal(baseline[f], degraded[f]), f


def test_a_wall_at_chunk_one_refuses_with_the_remedy(monkeypatch):
    """When no smaller solve exists, the refusal is honest and named."""
    grid, prior, obs, fields = _tiny_case()
    flaky = _flaky_eigendecompose(
        10 ** 6, letkf_mod._eigendecompose,
        lambda: MemoryError("synthetic allocation failure"))
    monkeypatch.setattr(letkf_mod, "_eigendecompose", flaky)
    with pytest.raises(LetkfError, match="already at one gridpoint"):
        analyze(prior, [obs], grid, LetkfConfig(
            localization=Localization(horizontal_m=3000.0, vertical_m=1500.0),
            analysis_fields=fields, rtps_alpha=0.0, chunk_points=4))


def test_wrong_answers_are_not_retried(monkeypatch):
    """Only allocation failures degrade; a wrong answer propagates."""
    grid, prior, obs, fields = _tiny_case()
    flaky = _flaky_eigendecompose(
        1, letkf_mod._eigendecompose,
        lambda: LetkfError("a non-positive eigenvalue, say"))
    monkeypatch.setattr(letkf_mod, "_eigendecompose", flaky)
    d = LetkfDiagnostics()
    with pytest.raises(LetkfError, match="non-positive eigenvalue"):
        analyze(prior, [obs], grid, LetkfConfig(
            localization=Localization(horizontal_m=3000.0, vertical_m=1500.0),
            analysis_fields=fields, rtps_alpha=0.0, chunk_points=16),
            diagnostics=d)
    assert d.chunk_oom_shrinks == 0


# ---------------------------------------------------------------------------
# The predicate that separates the two
# ---------------------------------------------------------------------------

def _cupy_shaped(name, module, message):
    """An exception class wearing the device stack's name and address."""
    cls = type(name, (Exception,), {})
    cls.__module__ = module
    return cls(message)


def test_predicate_recognises_the_device_allocation_failures():
    runtime_err = _cupy_shaped(
        "CUDARuntimeError", "cupy_backends.cuda.api.runtime",
        "cudaErrorMemoryAllocation: out of memory")
    assert _is_device_memory_error(runtime_err)
    assert _is_device_memory_error(_cupy_shaped(
        "OutOfMemoryError", "cupy.cuda.memory",
        "Out of memory allocating 1,069,804,800 bytes"))
    assert _is_device_memory_error(_cupy_shaped(
        "CUSOLVERError", "cupy_backends.cuda.libs.cusolver",
        "CUSOLVER_STATUS_ALLOC_FAILED"))
    assert _is_device_memory_error(MemoryError())
    # The chain is walked: a wrapped allocation failure is still one.
    wrapped = LetkfError("the batched eigensolver failed")
    wrapped.__cause__ = runtime_err
    assert _is_device_memory_error(wrapped)


def test_predicate_rejects_everything_else():
    # A LetkfError that merely MENTIONS memory is a verdict, not an
    # allocation failure: only the device stack's own classes are
    # text-matched.
    assert not _is_device_memory_error(
        LetkfError("raise memory_budget_mib; the card is out of memory"))
    assert not _is_device_memory_error(
        ValueError("cudaErrorMemoryAllocation"))
    assert not _is_device_memory_error(_cupy_shaped(
        "CUDARuntimeError", "cupy_backends.cuda.api.runtime",
        "cudaErrorInvalidValue: invalid argument"))
    assert not _is_device_memory_error(np.linalg.LinAlgError("did not converge"))
