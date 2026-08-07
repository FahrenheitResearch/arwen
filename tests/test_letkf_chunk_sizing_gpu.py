"""The chunk sizing and degradation path, on the device it exists for.

The CPU-side arithmetic and driver are pinned in
``tests/test_letkf_chunk_sizing.py``; this module asserts the parts that
only mean something with a card present: the free-memory reading, the
ceiling holding on a real analysis, and the degradation path fired by the
device stack's own exception classes rather than by look-alikes.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_gpu

pytestmark = [pytest.mark.gpu, requires_gpu]


def _tiny_device_case(members=6, nz=3, ny=8, nx=8, seed=23,
                      fields=("theta", "u")):
    import cupy as cp

    from gpuwm.da.letkf import GriddedObs, GridGeometry

    rng = np.random.default_rng(seed)
    shape = (nz, ny, nx)
    grid = GridGeometry(
        dx_m=1000.0, dy_m=1000.0,
        heights_m=np.array([250.0, 800.0, 1600.0][:nz]),
    )
    prior = {
        f: cp.asarray(rng.standard_normal((members,) + shape) + 5.0 * i)
        for i, f in enumerate(fields)
    }
    mask = np.zeros(shape, dtype=bool)
    mask.reshape(-1)[rng.choice(nz * ny * nx, size=12, replace=False)] = True
    sim = rng.standard_normal((members,) + shape) * 0.8 + 0.1
    values = np.where(
        mask, sim.mean(axis=0) + rng.standard_normal(shape) * 0.4, np.nan)
    obs = GriddedObs(name="probe", values=cp.asarray(values), errors=0.4,
                     simulated=cp.asarray(sim), mask=cp.asarray(mask))
    return grid, prior, obs, fields


def test_device_free_bytes_answers_on_a_real_card():
    import cupy as cp

    from gpuwm.da.letkf import _device_free_bytes

    free = _device_free_bytes(cp)
    assert free is not None
    assert free > 0


def test_budget_ceiling_holds_on_the_device():
    from gpuwm.da.letkf import LetkfConfig, LetkfDiagnostics, Localization
    from gpuwm.da.letkf import analyze

    grid, prior, obs, fields = _tiny_device_case()
    d = LetkfDiagnostics()
    inc = analyze(prior, [obs], grid, LetkfConfig(
        localization=Localization(horizontal_m=3000.0, vertical_m=1500.0),
        analysis_fields=fields, rtps_alpha=0.0,
        memory_budget_mib=64.0), diagnostics=d)
    assert d.chunk_points >= 1
    assert d.chunk_points * d.solve_bytes_per_point <= 64 * (1 << 20)
    assert d.chunk_oom_shrinks == 0
    for f in fields:
        assert bool(np.all(np.isfinite(inc[f].get())))


def test_the_real_device_exception_classes_degrade(monkeypatch):
    """cupy's own OOM classes, not stand-ins, drive the halving.

    And the degraded answer is checked to ROUNDING, not to the byte.
    The host-side twin of this test asserts bitwise equality and is right
    to: numpy's batched eigensolve and matmuls are per-matrix loops.  The
    device's are not -- cuBLAS and the batched eigensolver choose work
    partitionings from the batch extent -- so re-solving the same span at
    a smaller chunk reorders each gridpoint's sums.  Measured here on an
    RTX 3090 (sm_86, cupy 14.1.1, float64): 2.2e-16 absolute at chunk 16,
    rising to 3.1e-15 at chunk 1, on increments of order one.  The
    tolerance below is 1e-12, four orders above the worst measurement and
    still nine orders below any increment this filter produces, so it
    admits reordering and nothing else.  If this ever needs loosening,
    the degradation path has become an accuracy path and the loosening is
    the bug report.
    """
    import cupy as cp
    from cupy_backends.cuda.api.runtime import CUDARuntimeError

    import gpuwm.da.letkf as letkf_mod
    from gpuwm.da.letkf import LetkfConfig, LetkfDiagnostics, Localization
    from gpuwm.da.letkf import analyze

    grid, prior, obs, fields = _tiny_device_case()
    loc = Localization(horizontal_m=3000.0, vertical_m=1500.0)
    baseline = analyze(prior, [obs], grid, LetkfConfig(
        localization=loc, analysis_fields=fields, rtps_alpha=0.0,
        chunk_points=32))

    real = letkf_mod.gaspari_cohn
    calls = {"n": 0}

    def flaky(distance, cutoff):
        # WHICH call site: on the DEVICE array, not on a shape.  Stencil
        # construction also passes a 2-D array -- the (dj, di) offset grid
        # it thresholds to find the support radius -- but it does so on
        # the host, before the chunk loop and outside its try, so a shape
        # test fires there and never reaches the path under test.  The
        # chunk loop is the only caller handing this function device
        # memory, which makes the namespace the honest discriminator.
        if isinstance(distance, cp.ndarray):
            calls["n"] += 1
            if calls["n"] == 1:
                # Status 2 is cudaErrorMemoryAllocation; raising the class
                # the runtime itself raises is the point of this test.
                raise CUDARuntimeError(2)
        return real(distance, cutoff)

    monkeypatch.setattr(letkf_mod, "gaspari_cohn", flaky)
    d = LetkfDiagnostics()
    degraded = analyze(prior, [obs], grid, LetkfConfig(
        localization=loc, analysis_fields=fields, rtps_alpha=0.0,
        chunk_points=32), diagnostics=d)

    assert d.chunk_points_initial == 32
    assert d.chunk_oom_shrinks == 1
    assert d.chunk_points == 16
    for f in fields:
        moved = float(cp.abs(baseline[f] - degraded[f]).max())
        assert moved <= 1e-12, (f, moved)
        assert bool(cp.all(cp.isfinite(degraded[f])))
    # The tolerance is a comparison against the baseline at every
    # gridpoint, so a retry that skipped the span it failed on cannot
    # pass it -- those points would differ by the whole increment, not by
    # a rounding.  This asserts the baseline was not itself all zero,
    # which is the one way the comparison could be vacuous.
    assert any(bool(cp.any(baseline[f] != 0)) for f in fields)
