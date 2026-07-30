# tests/test_mp_spec_zone_ring.py
"""Specified-zone ring exclusion for microphysics (WRF tile clipping).

Oracle: WRF v4.6.1 (reference bundle) --
  sz        solve_em.F:3618-3622: ``sz = spec_zone`` when ``specified .or.
            nested`` (the driver's own ``specified`` argument is that OR,
            solve_em.F:3692), else 0;
  prep      solve_em.F:3629-3639: moist_physics_prep tiles clip to
            its = max(i_start, ids+sz) .. ite = min(i_end, ide-1-sz),
            jts = max(j_start, jds+sz) .. jte = min(j_end, jde-1-sz)
            (the i clip is skipped only for periodic_x channels);
  driver    module_microphysics_driver.F:802-806/:870-879: the scheme tile
            loop applies the identical clip;
  finish    solve_em.F:4040-4048: moist_physics_finish_em (theta update +
            h_diabatic capture) applies the identical clip; ``grid%sr = 0.``
            is whole-field each call (solve_em.F:3691).

Therefore the outermost ``spec_zone`` mass ring is NEVER touched by
microphysics in a specified/nested WRF domain: ring RAINNC stays exactly
0.0 (measured bit-exact on the CPU reference wrfouts, every domain, every
lead), ring theta/moisture keep their boundary-installed values, and ring
h_diabatic stays 0 (allocator zero init, frame/module_domain.F:770-777 /
tools/gen_allocs.c:411-415; set_physical_bc3d writes halo indices only,
module_bc.F:867-885).  Qualifications: WRF's optional ``mp_zero_out``
path uses whole-field mass bounds (solve_em.F:4002-4038) -- gpuwm has no
mp_zero_out; and specified+periodic_x channels clip j only -- gpuwm has
no channel mode.  Both are explicit non-goals.

These pins were written RED against 4d2ce99 (gpuwm ran every scheme
whole-field, after the end-of-step boundary overwrite) and must stay green
forever after.  gpuwm's mass grid is 0-based, so WRF's clipped tile
``ids+sz .. ide-1-sz`` maps to columns ``sz .. nx-1-sz`` and the excluded
ring is ``i < sz or i > nx-1-sz or j < sz or j > ny-1-sz``.
"""
import numpy as np
import pytest

from conftest import requires_gpu
from gpuwm.config import RunConfig
from gpuwm.core import constants as c


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _ring_mask(ny, nx, sz=1):
    """Boolean (ny, nx) mask of the WRF-excluded specified-zone ring."""
    m = np.ones((ny, nx), dtype=bool)
    m[sz:ny - sz, sz:nx - sz] = False
    return m


def _square_moist_state(ny=8, nx=9, nz=40, mp=1, dt=30.0, **flags):
    """Balanced moist state, saturated cloudy lowest 3 km, ny > 1.

    Same construction as test_kessler._moist_state but with a real 2-D
    horizontal footprint so the domain HAS a ring, plus arbitrary boundary
    flags (specified/nested) forwarded to RunConfig.
    """
    import cupy as cp
    from gpuwm.core.diagnostics import update_diagnostics
    from gpuwm.core.grid import make_base_state, make_vertical_coord
    from gpuwm.core.moist import init_moist_balanced
    from gpuwm.core.state import DTYPE
    from test_kessler import _teten_qvs

    cfg = RunConfig(nx=nx, ny=ny, nz=nz, dx=1000.0, dy=1000.0, ztop=10000.0,
                    dt=dt, run_seconds=0.0, moist=True, mp_physics=mp,
                    **flags)
    vc = make_vertical_coord(cfg.nz)
    b = make_base_state(vc, lambda z: 300.0 + 0.003 * np.asarray(z, float),
                        p_surf=cfg.p_surf, ztop=cfg.ztop)
    s = init_moist_balanced(cfg, vc, b, lambda z: 0.0 * np.asarray(z, float))
    z = s.height_half()
    zc = z[:, None, None]
    qc = np.where((zc > 500.0) & (zc < 3000.0), 2.0e-3, 0.0)
    s.qc[...] = cp.asarray(np.broadcast_to(qc, s.p.shape), dtype=DTYPE)

    def state_qvs():
        th = b.thb[:, None, None] + cp.asnumpy(s.thp).astype(np.float64)
        pii = (cp.asnumpy(s.p).astype(np.float64) / c.P0) ** c.RCP
        return _teten_qvs(th, pii)

    for _ in range(6):
        qv = np.where(zc < 3000.0, 1.02 * state_qvs(), 0.3 * state_qvs())
        s.qv[...] = cp.asarray(np.broadcast_to(qv, s.p.shape), dtype=DTYPE)
        update_diagnostics(s)
    rh = cp.asnumpy(s.qv).astype(np.float64) / state_qvs()
    assert rh[np.broadcast_to(zc < 2500.0, rh.shape)].min() > 1.015
    return s, cfg


def _mp_state_fields(cfg):
    """State attributes the configured scheme (plus finish) mutates."""
    fields = ["thp", "h_diabatic", "qv", "qc", "qr"]
    if cfg.mp_physics in (6, 10):
        fields += ["qi", "qs", "qg", "effc", "effi", "effs"]
    if cfg.mp_physics == 10:
        fields += ["nc", "nr", "ni", "ns", "ng", "effr"]
    return fields


# ---------------------------------------------------------------------------
# ring geometry (CPU)
# ---------------------------------------------------------------------------

def test_ring_slices_tile_the_wrf_excluded_ring():
    """The production slice set covers WRF's excluded ring exactly once
    (no gaps, no double-restore overlap) and never reaches the clipped
    tile interior ``sz..n-1-sz``, including degenerate tiny domains where
    WRF's clip leaves an empty tile."""
    from gpuwm.core.microphysics import spec_zone_ring_slices

    for ny, nx, sz in ((8, 9, 1), (5, 5, 2), (2, 7, 1), (1, 4, 1),
                       (4, 1, 1), (3, 3, 2), (200, 250, 1)):
        count = np.zeros((3, ny, nx), dtype=np.int32)  # Ellipsis-led slices
        for slc in spec_zone_ring_slices(ny, nx, sz):
            count[slc] += 1
        interior = ~_ring_mask(ny, nx, sz)
        assert (count[0][interior] == 0).all(), (ny, nx, sz)
        assert (count[0][~interior] == 1).all(), (ny, nx, sz)
        assert (count == count[0][None]).all()          # Ellipsis broadcast


# ---------------------------------------------------------------------------
# ring non-mutation pins (GPU) -- RED at 4d2ce99
# ---------------------------------------------------------------------------

@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("mp,ncalls", [(1, 40), (6, 10), (10, 10)])
def test_specified_run_never_touches_the_spec_zone_ring(mp, ncalls):
    """Core pin: on a specified domain, repeated microphysics leaves every
    ring column bitwise untouched (theta', every hydrometeor/moment, the
    effective radii), accumulates ring RAINNC == exactly 0.0, keeps ring
    h_diabatic == 0.0, and still runs normally in the interior."""
    import cupy as cp
    from gpuwm.core import microphysics

    s, cfg = _square_moist_state(mp=mp, specified=True)
    ring = _ring_mask(cfg.ny, cfg.nx, cfg.spec_zone)
    fields = _mp_state_fields(cfg)
    before = {k: cp.asnumpy(getattr(s, k)).copy() for k in fields}
    result = None
    hd_first_interior_max = None
    for i in range(ncalls):
        result = microphysics.apply(s, cfg, cfg.dt)
        if i == 0:
            # Sample the heating rate while the scheme is actively
            # condensing; late calls sit at an FP32 saturation fixed
            # point where the increment is legitimately exactly zero.
            hd_first = cp.asnumpy(s.h_diabatic)
            hd_first_interior_max = float(np.abs(hd_first[:, ~ring]).max())

    for k in fields:
        after = cp.asnumpy(getattr(s, k))
        np.testing.assert_array_equal(
            after[:, ring], before[k][:, ring],
            err_msg=f"microphysics mutated spec-zone ring {k} (mp={mp})")

    rainnc = cp.asnumpy(result.rainnc)
    assert (rainnc[ring] == 0.0).all(), (
        f"ring RAINNC accumulated (max {np.abs(rainnc[ring]).max()} mm); "
        "WRF's is exactly 0.0 at every lead")
    # per-call surface diagnostics: WRF never writes them in the ring
    # (rainncv zero-init, never touched; sr whole-field-zeroed each call)
    assert (cp.asnumpy(result.rainncv)[ring] == 0.0).all()
    assert (cp.asnumpy(result.sr)[ring] == 0.0).all()
    hd = cp.asnumpy(s.h_diabatic)
    assert (hd[:, ring] == 0.0).all(), "ring h_diabatic must stay 0 (WRF " \
        "start_em zero init + clipped finish tiles)"

    # non-vacuous: the interior really condensed/heated
    interior = ~ring
    dthp = np.abs(cp.asnumpy(s.thp) - before["thp"])
    assert dthp[:, interior].max() > 1.0e-3
    if mp == 1:
        assert rainnc[interior].min() > 0.05     # rain at every interior col
        assert hd_first_interior_max > 1.0e-4    # heating registered


@pytest.mark.gpu
@requires_gpu
def test_nested_domain_ring_is_masked_like_specified():
    """WRF applies the same sz clip on nested domains (specified .OR.
    nested, solve_em.F:3618/:3692) -- the child-domain flag alone must
    trigger the exclusion."""
    import cupy as cp
    from gpuwm.core import microphysics

    s, cfg = _square_moist_state(mp=1, nested=True)
    before = {k: cp.asnumpy(getattr(s, k)).copy()
              for k in ("thp", "qv", "qc", "qr")}
    result = None
    for _ in range(5):
        result = microphysics.apply(s, cfg, cfg.dt)
    ring = _ring_mask(cfg.ny, cfg.nx, cfg.spec_zone)
    for k, want in before.items():
        np.testing.assert_array_equal(
            cp.asnumpy(getattr(s, k))[:, ring], want[:, ring],
            err_msg=f"nested-domain microphysics mutated ring {k}")
    assert (cp.asnumpy(result.rainnc)[ring] == 0.0).all()
    # non-vacuous
    assert np.abs(cp.asnumpy(s.qv) - before["qv"])[:, ~ring].max() > 1.0e-5


@pytest.mark.gpu
@requires_gpu
def test_masking_leaves_interior_bitwise_identical():
    """The exclusion is EXACTLY a ring operation: per-column schemes give
    bitwise-identical interior trajectories with and without the guard
    (this is the equivalence argument for implementing WRF's clipped tiles
    as a ring capture/restore around whole-field column kernels).  This
    pin proves the INTERIOR half only; exact ring restoration is proven
    by test_specified_run_never_touches_the_spec_zone_ring."""
    import cupy as cp
    from gpuwm.core import microphysics

    s_spec, cfg_spec = _square_moist_state(mp=1, specified=True)
    s_per, cfg_per = _square_moist_state(mp=1)      # periodic control
    for _ in range(5):
        microphysics.apply(s_spec, cfg_spec, cfg_spec.dt)
        microphysics.apply(s_per, cfg_per, cfg_per.dt)
    ring = _ring_mask(cfg_spec.ny, cfg_spec.nx, cfg_spec.spec_zone)
    for k in ("thp", "qv", "qc", "qr", "h_diabatic"):
        a = cp.asnumpy(getattr(s_spec, k))
        b = cp.asnumpy(getattr(s_per, k))
        np.testing.assert_array_equal(
            a[:, ~ring], b[:, ~ring],
            err_msg=f"ring guard changed interior {k}")
    # the control's ring DID move (periodic default keeps whole-field MP,
    # every frozen idealized fixture is bitwise unaffected by the guard)
    per_rainnc = cp.asnumpy(s_per.scratch((cfg_per.ny, cfg_per.nx),
                                          "mp_rainnc"))
    assert per_rainnc[ring].min() > 0.0


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("mp", [1, 6, 10])
def test_ring_save_slots_registry_matches_live_guard(mp):
    """Every mp_ring_save_* buffer a live guarded call creates must be
    enumerated -- with its exact shape -- by spec_zone_ring_save_slots,
    the preflight-registry source of truth; otherwise the allocation gate
    under-budgets the guard (unbudgeted cp.zeros after preflight).  The
    due-reflectivity call forces the refl_10cm stash into existence so
    its ring saves are exercised for every scheme, Kessler's rain-only
    fallback included."""
    from types import SimpleNamespace
    from gpuwm.core import microphysics
    from gpuwm.core.microphysics import spec_zone_ring_save_slots

    s, cfg = _square_moist_state(mp=mp, specified=True)
    s.physics = SimpleNamespace(refl_10cm=None)   # minimal stash owner
    microphysics.apply(s, cfg, cfg.dt, refl_10cm_due=True)
    s.physics.refl_10cm = None                    # consume the handoff
    microphysics.apply(s, cfg, cfg.dt)            # captures refl_10cm slot
    registry = spec_zone_ring_save_slots(cfg)
    live = {name: buf.shape for name, buf in s._scratch.items()
            if name.startswith("mp_ring_save_")}
    assert any(name.startswith("mp_ring_save_refl_10cm_") for name in live)
    unregistered = {name: shape for name, shape in live.items()
                    if registry.get(name) != shape}
    assert not unregistered, unregistered


@pytest.mark.gpu
@requires_gpu
def test_ring_restored_even_when_dispatch_raises():
    """The restoration invariant holds on EVERY exit path: a due
    reflectivity call with no physics driver raises AFTER the scheme has
    mutated the field (refl.py stash refusal), and the ring must still
    come back bit-identical -- WRF never dispatches those columns at
    all."""
    import cupy as cp
    from gpuwm.core import microphysics

    s, cfg = _square_moist_state(mp=1, specified=True)
    assert getattr(s, "physics", None) is None
    ring = _ring_mask(cfg.ny, cfg.nx, cfg.spec_zone)
    before = {k: cp.asnumpy(getattr(s, k)).copy()
              for k in ("thp", "qv", "qc", "qr")}
    with pytest.raises(RuntimeError, match="physics driver"):
        microphysics.apply(s, cfg, cfg.dt, refl_10cm_due=True)
    for k, want in before.items():
        np.testing.assert_array_equal(
            cp.asnumpy(getattr(s, k))[:, ring], want[:, ring],
            err_msg=f"ring {k} not restored after raising dispatch")
    assert (cp.asnumpy(s.h_diabatic)[:, ring] == 0.0).all()
    rainnc = cp.asnumpy(s.scratch((cfg.ny, cfg.nx), "mp_rainnc"))
    assert (rainnc[ring] == 0.0).all()
    # the scheme DID run before raising: interior moisture moved
    assert np.abs(cp.asnumpy(s.qv) - before["qv"])[:, ~ring].max() > 1e-6


# ---------------------------------------------------------------------------
# single-domain PBL cadence override guard (CPU)
# ---------------------------------------------------------------------------

def test_single_domain_pbl_cadence_override_only_at_bldt_zero():
    """The single-domain integration loop's bldt=0 convenience override
    (bldt_seconds = dt, stepbl = 1) must not clobber a configured positive
    bldt: PhysicsDriver already computed WRF's STEPBL calendar from it
    (physics.py: max(nint(bldt*60/dt), 1))."""
    from types import SimpleNamespace
    from gpuwm.runtime import apply_single_domain_pbl_cadence

    cfg0 = RunConfig(nx=8, ny=8, nz=8, dx=1e3, dy=1e3, ztop=1e4, dt=60.0,
                     run_seconds=0.0, bldt=0.0)
    phys = SimpleNamespace(bldt_seconds=999.0, stepbl=7)
    apply_single_domain_pbl_cadence(phys, cfg0)
    assert phys.bldt_seconds == 60.0 and phys.stepbl == 1

    cfg30 = RunConfig(nx=8, ny=8, nz=8, dx=1e3, dy=1e3, ztop=1e4, dt=60.0,
                      run_seconds=0.0, bldt=30.0)
    phys = SimpleNamespace(bldt_seconds=1800.0, stepbl=30)
    apply_single_domain_pbl_cadence(phys, cfg30)
    assert phys.bldt_seconds == 1800.0 and phys.stepbl == 30
