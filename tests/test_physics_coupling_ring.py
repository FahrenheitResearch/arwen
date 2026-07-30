# tests/test_physics_coupling_ring.py
"""Physics-tendency coupling: WRF's one-cell boundary exclusion fires for
``specified .OR. nested`` (SEAM A, spec-zone census).

Oracle: WRF v4.6.1 phys/module_physics_addtendc.F -- every coupling
helper narrows its loops by one physical cell on all four sides when
``config_flags%specified .or. config_flags%nested``:

  add_a2a     :2312-2319  A-grid mass: i in [ids+1, ide-2], j likewise
  add_a2a_ph  :2359-2366
  add_a2c_u   :2412-2419  u faces: i in [ids+1, ide-1], j in [jds+1, jde-2]
  add_a2c_v   :2466-2473  v faces: j in [jds+1, jde-1], i in [ids+1, ide-2]
  add_c2c_u/v :2521-2528 / :2573-2580

(``periodic_x`` lifts the i clip -- gpuwm has no channel mode: explicit
non-goal, as for the microphysics ring.)

gpuwm's couplers carried the mask under ``cfg.specified`` ONLY, so nested
children (production d02+ run nested=true, specified=false) received ring
physics tendencies WRF never applies there.  Dry theta/u/v ring
tendencies are dynamically dead (the child LBC replacement overwrites
them), but the MOIST ring sources are live: they fold into ``q0_eff``
and become the first interior cell's PD upwind donor/limiter state
(gpuwm/core/moist.py), where WRF's ring sc_tend is zero before its
full-field fold (dyn_em/module_em.F:1889-1912).

These pins were written RED at 6eb26c00 (nested arms + the
specified==nested equivalence) and must stay green forever after.
"""
import numpy as np
import pytest

from conftest import requires_gpu
from test_mp_spec_zone_ring import _ring_mask, _square_moist_state


def _flags(kind):
    return ({"specified": True} if kind == "specified"
            else {"nested": True} if kind == "nested" else {})


def _ones(shape):
    import cupy as cp
    return cp.ones(shape, dtype=cp.float32)


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("kind", ["specified", "nested"])
def test_column_coupling_masks_the_ring(kind):
    """couple_column_tendencies (radiation/cumulus scalar products,
    WRF add_a2a bounds): ring exactly zero, interior coupled, for BOTH
    boundary-forced flags -- the nested arm is the SEAM A pin."""
    import cupy as cp
    from gpuwm.core.physics import couple_column_tendencies

    s, cfg = _square_moist_state(mp=1, **_flags(kind))
    rate = _ones(s.p.shape)
    out = couple_column_tendencies(
        s, cfg, rtheta=rate, rqv=rate, rqc=rate, rqr=rate, rqi=rate,
        rqs=rate)
    ring = _ring_mask(cfg.ny, cfg.nx, cfg.spec_zone)
    for name in ("rtheta", "rqv", "rqc", "rqr", "rqi", "rqs"):
        field = cp.asnumpy(getattr(out, name))
        assert (field[:, ring] == 0.0).all(), (
            f"{name} ring coupled under {kind} (WRF add_a2a excludes it, "
            "module_physics_addtendc.F:2312-2319)")
        assert np.abs(field[:, ~ring]).min() > 0.0, f"{name} interior dead"


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("kind", ["specified", "nested"])
def test_ysu_coupling_masks_the_ring(kind):
    """couple_ysu_tendencies: A-grid scalars per add_a2a; u/v faces per
    add_a2c_u/v (first/last normal face and outer parallel row/col
    excluded).  The nested arm is the SEAM A pin."""
    import cupy as cp
    from gpuwm.core.physics import couple_ysu_tendencies

    s, cfg = _square_moist_state(mp=1, **_flags(kind))
    nz, ny, nx = s.p.shape
    ysu = {name: _ones((nz, ny, nx))
           for name in ("du", "dv", "dtheta", "dqv", "dqc", "dqi")}
    out = couple_ysu_tendencies(s, cfg, ysu)
    ring = _ring_mask(ny, nx, cfg.spec_zone)
    for name in ("rtheta", "rqv", "rqc", "rqi"):
        field = cp.asnumpy(getattr(out, name))
        assert (field[:, ring] == 0.0).all(), f"{name} ring coupled ({kind})"
        assert np.abs(field[:, ~ring]).min() > 0.0
    ru = cp.asnumpy(out.ru)          # (nz, ny, nx+1)
    rv = cp.asnumpy(out.rv)          # (nz, ny+1, nx)
    # add_a2c_u: faces ids and ide excluded, outer j rows excluded
    assert (ru[:, :, 0] == 0.0).all() and (ru[:, :, -1] == 0.0).all(), kind
    assert (ru[:, 0, :] == 0.0).all() and (ru[:, -1, :] == 0.0).all(), kind
    assert np.abs(ru[:, 1:-1, 1:-1]).min() > 0.0
    # add_a2c_v: faces jds and jde excluded, outer i cols excluded
    assert (rv[:, 0, :] == 0.0).all() and (rv[:, -1, :] == 0.0).all(), kind
    assert (rv[:, :, 0] == 0.0).all() and (rv[:, :, -1] == 0.0).all(), kind
    assert np.abs(rv[:, 1:-1, 1:-1]).min() > 0.0


@pytest.mark.gpu
@requires_gpu
def test_specified_and_nested_coupling_are_bitwise_identical():
    """WRF's OR makes the two flags indistinguishable inside the coupling
    helpers: identical states and rates must produce bitwise-identical
    coupled tendencies under specified=T and nested=T."""
    import cupy as cp
    from gpuwm.core.physics import (couple_column_tendencies,
                                    couple_ysu_tendencies)

    outs = {}
    for kind in ("specified", "nested"):
        s, cfg = _square_moist_state(mp=1, **_flags(kind))
        nz, ny, nx = s.p.shape
        rate = _ones((nz, ny, nx))
        col = couple_column_tendencies(s, cfg, rtheta=rate, rqv=rate,
                                       rqc=rate)
        ysu = {name: _ones((nz, ny, nx))
               for name in ("du", "dv", "dtheta", "dqv", "dqc")}
        pbl = couple_ysu_tendencies(s, cfg, ysu)
        outs[kind] = (col, pbl)
    for got, want in zip(outs["nested"], outs["specified"]):
        for name in ("ru", "rv", "rtheta", "rqv", "rqc"):
            np.testing.assert_array_equal(
                cp.asnumpy(getattr(got, name)),
                cp.asnumpy(getattr(want, name)), err_msg=name)


@pytest.mark.gpu
@requires_gpu
def test_periodic_coupling_stays_unmasked():
    """Neither flag set (WRF: neither specified nor nested -> full-tile
    loops): the ring IS coupled -- guards against over-masking the frozen
    periodic/idealized paths."""
    import cupy as cp
    from gpuwm.core.physics import couple_column_tendencies

    s, cfg = _square_moist_state(mp=1)
    rate = _ones(s.p.shape)
    out = couple_column_tendencies(s, cfg, rqv=rate)
    ring = _ring_mask(cfg.ny, cfg.nx, cfg.spec_zone)
    assert np.abs(cp.asnumpy(out.rqv)[:, ring]).min() > 0.0
