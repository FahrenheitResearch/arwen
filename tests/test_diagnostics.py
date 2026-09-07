# tests/test_diagnostics.py
import numpy as np
import pytest
from conftest import requires_gpu
from gpuwm.config import RunConfig
from gpuwm.core.grid import make_vertical_coord, make_base_state

pytestmark = pytest.mark.gpu

def _setup(nx=32, nz=16, ztop=6400.0):
    cfg = RunConfig(nx=nx, ny=1, nz=nz, dx=100.0, dy=100.0, ztop=ztop,
                    dt=0.5, run_seconds=1.0)
    vc = make_vertical_coord(nz)
    b = make_base_state(vc, lambda z: 300.0 + 0.003 * np.asarray(z, float),
                        p_surf=cfg.p_surf, ztop=cfg.ztop)
    return cfg, vc, b

@requires_gpu
def test_eos_matches_numpy_reference():
    import cupy as cp
    from gpuwm.core.state import init_theta_perturbation
    from gpuwm.core.diagnostics import update_diagnostics
    from gpuwm.verify.npref import np_calc_p_alpha
    cfg, vc, b = _setup()
    rng = np.random.default_rng(0)
    def thp_f(x, z):
        return rng.normal(0.0, 1.0, (cfg.nz, cfg.ny, cfg.nx))
    s = init_theta_perturbation(cfg, vc, b, thp_f)
    update_diagnostics(s)
    p_ref, al_ref, alt_ref = np_calc_p_alpha(
        cp.asnumpy(s.thp).astype(np.float64),
        cp.asnumpy(s.php).astype(np.float64),
        cp.asnumpy(s.mup).astype(np.float64), b, vc)
    np.testing.assert_allclose(cp.asnumpy(s.p), p_ref, rtol=1e-6)
    np.testing.assert_allclose(cp.asnumpy(s.alt), alt_ref, rtol=1e-6)
    np.testing.assert_allclose(cp.asnumpy(s.al), al_ref, rtol=1e-6, atol=1e-6)

@requires_gpu
@pytest.mark.parametrize("nz,ztop,bound", [(16, 6400.0, 1e-5),
                                           (160, 2400.0, 1e-5)])
def test_rest_state_pressure_equals_base(nz, ztop, bound):
    import cupy as cp
    from gpuwm.core.state import init_at_rest
    from gpuwm.core.diagnostics import update_diagnostics
    cfg, vc, b = _setup(nz=nz, ztop=ztop)
    s = init_at_rest(cfg, vc, b)
    update_diagnostics(s)
    p = cp.asnumpy(s.p)[:, 0, 0]
    # Slack is discrete-vs-analytic base-state error plus FP32 rounding.
    # It must NOT grow with nz: at rest the EOS reproduces pb exactly in
    # exact arithmetic, so any nz dependence is the diagnostic's own
    # cancellation.
    np.testing.assert_allclose(p, b.pb, rtol=bound)


def _eos_error(nz, ztop, hypso, nx=32, seed=0):
    """max |dp/p| and max |d(al)| of the device EOS against the FP64 mirror."""
    import cupy as cp
    from gpuwm.core.state import init_theta_perturbation
    from gpuwm.core.diagnostics import update_diagnostics
    from gpuwm.verify.npref import np_calc_p_alpha
    cfg, vc, b = _setup(nx=nx, nz=nz, ztop=ztop)
    rng = np.random.default_rng(seed)
    s = init_theta_perturbation(
        cfg, vc, b, lambda x, z: rng.normal(0.0, 1.0, (nz, cfg.ny, nx)))
    update_diagnostics(s, hypso)
    p_ref, al_ref, _ = np_calc_p_alpha(
        cp.asnumpy(s.thp).astype(np.float64),
        cp.asnumpy(s.php).astype(np.float64),
        cp.asnumpy(s.mup).astype(np.float64), b, vc, hypsometric_opt=hypso)
    rel_p = float(np.abs((cp.asnumpy(s.p).astype(np.float64) - p_ref)
                         / p_ref).max())
    abs_al = float(np.abs(cp.asnumpy(s.al).astype(np.float64) - al_ref).max())
    return rel_p, abs_al


@requires_gpu
@pytest.mark.parametrize("hypso", [1, 2])
def test_eos_error_does_not_scale_with_vertical_resolution(hypso):
    """The EOS's own error must not track 1/dz.

    ``calc_p_alpha`` recovers the layer geopotential thickness, ~368
    J/kg at nz=64 over a 2400 m LES column, by differencing two total
    geopotentials of ~2.4e4 J/kg.  Storing those totals in FP32 puts
    ulp(2.4e4) into a difference that shrinks as 1/nz, so the diagnosed
    alpha -- and through it p, the pressure-gradient force and the
    acoustic sound speed -- degrade linearly with vertical resolution.
    This test measures the structure rather than a per-GPU ulp count:
    the error at nz=160 may not exceed three times the error at nz=16.
    """
    rel_lo, al_lo = _eos_error(16, 6400.0, hypso)
    rel_hi, al_hi = _eos_error(160, 2400.0, hypso)
    assert rel_hi <= 3.0 * rel_lo, (
        f"opt {hypso}: rel p error {rel_hi:.4e} at nz=160 is "
        f"{rel_hi / rel_lo:.1f}x the {rel_lo:.4e} at nz=16")
    assert al_hi <= 3.0 * al_lo, (
        f"opt {hypso}: abs al error {al_hi:.4e} at nz=160 is "
        f"{al_hi / al_lo:.1f}x the {al_lo:.4e} at nz=16")


@requires_gpu
@pytest.mark.parametrize("hypso", [1, 2])
def test_directly_written_phb_still_diagnoses_that_phb(hypso):
    """A phb written around the setter must still diagnose ITS OWN column.

    ``state.dphb_resid`` is derived from ``phb``, and three in-tree
    callers assign ``state.phb[...]`` directly (nest spawn, relocation,
    the real74 verify case).  The residual spelling is chosen so that a
    stale residual is a bounded <=1-ulp perturbation of the shipped
    answer rather than a wrong ``alt``: the kernel still forms the base
    layer thickness from the phb it was handed and only ADDS the stored
    correction.  Had the kernel instead read a stored absolute thickness,
    a bypassed write would silently diagnose the PREVIOUS geopotential --
    a plausible, wrong alt, and so a wrong p, pressure-gradient force and
    acoustic sound speed on exactly the paths hardest to eyeball.
    """
    import cupy as cp
    import dataclasses
    from gpuwm.core.state import init_at_rest
    from gpuwm.core.diagnostics import update_diagnostics
    from gpuwm.verify.npref import np_calc_p_alpha
    cfg, vc, b = _setup()
    s = init_at_rest(cfg, vc, b)
    update_diagnostics(s, hypso)
    # Bypass every setter, and move the LAYER THICKNESSES, not just the
    # column's offset: a 2 % vertical stretch is the shape a respawned or
    # relocated child's own terrain produces.  A constant shift would not
    # discriminate -- it leaves every dphb untouched.
    shifted = np.asarray(b.phb, dtype=np.float64) * 1.02
    s.phb[...] = cp.asarray(shifted, dtype=cp.float32)
    update_diagnostics(s, hypso)
    moved = dataclasses.replace(b, phb=shifted)
    p_ref, _, alt_ref = np_calc_p_alpha(
        cp.asnumpy(s.thp).astype(np.float64),
        cp.asnumpy(s.php).astype(np.float64),
        cp.asnumpy(s.mup).astype(np.float64), moved, vc,
        hypsometric_opt=hypso)
    # The shipped tolerance, not the tightened one: a stale correction
    # costs precision, never correctness.
    np.testing.assert_allclose(cp.asnumpy(s.p), p_ref, rtol=2e-5)
    np.testing.assert_allclose(cp.asnumpy(s.alt), alt_ref, rtol=2e-5)
