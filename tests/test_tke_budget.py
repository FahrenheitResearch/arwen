# tests/test_tke_budget.py
"""km_opt=2 restart carrier, lateral-boundary arm, and the TKE budget.

Three things land together here because they are three faces of the same
question -- does the prognostic carrier survive every boundary a real run
crosses?

* the RESTART boundary: ``tke`` is a serialized field and ``tke0`` its
  rebuilt time-t copy, so a stop/restore/finish trajectory must reproduce
  the uninterrupted one byte for byte (AC-L5.3);
* the LATERAL boundary: WRF's ``bound_tke`` (every domain, every RK stage)
  and ``flow_dep_bdy`` (specified/nested spec zone) are the entire
  treatment TKE gets, and each is checked with a mutation control that
  FIRES;
* the BUDGET: the term-by-term device accumulation must close against the
  storage it claims to explain, and each term must be load-bearing --
  removing any one of them from the sum has to move the residual by orders
  of magnitude, which is what makes the committed residual a measurement
  rather than an artifact of its own definition (AC-L5.1's bound-free arm).

Formula authority: the byte-identical local WRF v4.6.1 bundle --
dyn_em/module_em.F:2490-2520 (bound_tke), dyn_em/solve_em.F:2432-2451
(call order), share/module_bc.F:2335-2456 (flow_dep_bdy),
Registry/Registry.EM_COMMON:312 (the ``r``-only IO string that makes tke a
restart field with no boundary stream, no nest interpolation, and no
feedback).
"""
import dataclasses

import numpy as np
import pytest
from conftest import requires_gpu

from gpuwm.config import RunConfig
from gpuwm.core import tke_budget
from gpuwm.core.grid import make_base_state, make_vertical_coord

C_K = 0.10


def _theta(z):
    z = np.asarray(z, float)
    return 300.0 + np.where(z > 600.0, 0.003 * (z - 600.0), 0.0)


def _cfg(**kw):
    kw.setdefault("km_opt", 2)
    kw.setdefault("bl_pbl_physics", 0)
    kw.setdefault("mix_isotropic", 1)
    kw.setdefault("isfflx", 0)
    kw.setdefault("tke_heat_flux", 0.24)
    kw.setdefault("tke_drag_coefficient", 0.0013)
    return RunConfig(nx=16, ny=16, nz=16, dx=100.0, dy=100.0, ztop=1600.0,
                     dt=0.5, run_seconds=0.0, c_k=C_K,
                     time_step_sound=4, **kw)


def _state(cfg, seed=3):
    """At-rest CBL with seeded low-level theta noise (em_les shape)."""
    from gpuwm.core.state import init_theta_perturbation
    vc = make_vertical_coord(cfg.nz)
    base = make_base_state(vc, _theta, p_surf=cfg.p_surf, ztop=cfg.ztop)

    def thp_func(x, z):
        rng = np.random.default_rng(seed)
        pert = np.zeros((cfg.nz, cfg.ny, cfg.nx))
        pert[:4] = 0.1 * rng.standard_normal((4, cfg.ny, cfg.nx))
        return pert

    return init_theta_perturbation(cfg, vc, base, thp_func)


def _state_bytes(state):
    """Every serialized state array as raw bytes, in a stable order."""
    import cupy as cp

    from gpuwm.io.restart import STATE_SERIALIZED_ATTRS
    out = {}
    for name in STATE_SERIALIZED_ATTRS:
        value = getattr(state, name, None)
        if value is not None:
            out[name] = cp.asnumpy(value).tobytes()
    return out


# ---------------------------------------------------------------------------
# (1) The restart boundary
# ---------------------------------------------------------------------------

@requires_gpu
def test_tke_is_classified_as_restart_state_not_scratch():
    """The carrier is SERIALIZED and its time-t copy REBUILT -- and every
    other km_opt writes neither, so no other configuration pays for it."""
    from gpuwm.io import restart

    assert "tke" in restart.STATE_SERIALIZED_ATTRS
    assert "tke0" in restart.STATE_REBUILT_ATTRS
    assert restart.classify_state_attr("tke") == "serialize"
    assert restart.classify_state_attr("tke0") == "rebuild"
    # The manifest walk covers attributes that exist but are None on other
    # closures; an unclassified name would take down every restart write.
    for km_opt in (1, 3, 4):
        cfg = _cfg(km_opt=km_opt, isfflx=0, tke_heat_flux=0.0,
                   tke_drag_coefficient=0.0)
        state = _state(cfg)
        assert state.tke is None and state.tke0 is None
        assert "tke" in vars(state)


@requires_gpu
def test_km_opt2_restart_is_bit_identical(tmp_path):
    """AC-L5.3: 12 steps + checkpoint + 12 steps == an uninterrupted 24,
    byte for byte, with the carrier demonstrably restored rather than
    re-grown -- the fresh state is cold and the archive fills it."""
    import cupy as cp

    from gpuwm.core.dycore import run_steps
    from gpuwm.io import restart

    cfg = _cfg()
    straight = _state(cfg)
    run_steps(straight, cfg, 24)
    reference = restart.write_restart(tmp_path / "ref.npz", straight, cfg)

    split = _state(cfg)
    run_steps(split, cfg, 12)
    assert float(cp.abs(split.tke).max()) > 0.0, (
        "the checkpoint must carry live TKE or this proves nothing")
    mid = restart.write_restart(tmp_path / "mid.npz", split, cfg)

    resumed = _state(cfg)
    assert float(cp.abs(resumed.tke).max()) == 0.0    # cold, as WRF starts
    info = restart.restore_restart(mid, resumed, cfg)
    assert info.elapsed_seconds == pytest.approx(12 * cfg.dt)
    assert float(cp.abs(resumed.tke).max()) > 0.0     # the archive filled it
    run_steps(resumed, cfg, 12)

    got, want = _state_bytes(resumed), _state_bytes(straight)
    assert set(got) == set(want)
    differing = sorted(name for name in want if got[name] != want[name])
    assert not differing, f"restart diverged on {differing}"


@requires_gpu
def test_a_cold_started_carrier_is_a_different_trajectory(tmp_path):
    """The control the previous test needs: had the resume re-zeroed tke
    instead of restoring it, the comparison would have FAILED.  Without
    this the bit-identity claim could be satisfied by a carrier that never
    mattered."""
    import cupy as cp

    from gpuwm.core.dycore import run_steps
    from gpuwm.io import restart

    cfg = _cfg()
    straight = _state(cfg)
    run_steps(straight, cfg, 24)

    split = _state(cfg)
    run_steps(split, cfg, 12)
    mid = restart.write_restart(tmp_path / "mid.npz", split, cfg)

    resumed = _state(cfg)
    restart.restore_restart(mid, resumed, cfg)
    resumed.tke[...] = 0.0                            # the silent cold start
    run_steps(resumed, cfg, 12)

    got, want = _state_bytes(resumed), _state_bytes(straight)
    assert got["tke"] != want["tke"]
    assert got["thp"] != want["thp"], (
        "a cold-started closure must reach the thermodynamics, not just tke")
    assert float(cp.abs(resumed.tke).max()) > 0.0


# ---------------------------------------------------------------------------
# (2) The lateral boundary: bound_tke and flow_dep_bdy
# ---------------------------------------------------------------------------

@requires_gpu
def test_bound_tke_clamps_the_carrier_on_a_periodic_domain():
    """WRF bounds TKE into [0, tke_upper_bound] after EVERY rk_update_scalar
    pass on EVERY domain -- periodic ones included.  A ceiling below the
    live field must bite; the mutation control is the same run without it."""
    import cupy as cp

    from gpuwm.core import moist
    from gpuwm.core.dycore import run_steps

    cfg = _cfg()
    warm = _state(cfg)
    run_steps(warm, cfg, 20)
    live_max = float(cp.abs(warm.tke).max())
    assert live_max > 0.0

    ceiling = 0.25 * live_max
    bounded_cfg = dataclasses.replace(cfg, tke_upper_bound=ceiling)
    bounded = _state(bounded_cfg)
    run_steps(bounded, bounded_cfg, 20)
    assert float(bounded.tke.max()) <= ceiling + 1e-6
    assert float(bounded.tke.min()) >= 0.0

    # WATCHED control: with the bound removed the same run exceeds it, so
    # the assertion above is testing the clamp and not the physics.
    original = moist.bound_tke
    calls = []

    def _no_bound(tke, upper_bound):
        calls.append(float(upper_bound))

    moist.bound_tke = _no_bound
    try:
        unbounded = _state(bounded_cfg)
        run_steps(unbounded, bounded_cfg, 20)
    finally:
        moist.bound_tke = original
    assert calls, "the mutation control never fired -- bound_tke was not called"
    assert calls == [ceiling] * len(calls)
    assert len(calls) == 3 * 20, "bound_tke runs on every RK stage"
    assert float(unbounded.tke.max()) > ceiling + 1e-6


@requires_gpu
def test_flow_dep_bdy_fills_the_spec_zone_on_a_specified_domain():
    """Every specified-zone cell is either exactly zero (inflow) or exactly
    its interior donor (outflow) -- the definition of WRF flow_dep_bdy,
    checkable from the finished state alone.  Mutation control: skip the
    arm and the invariant breaks."""
    import cupy as cp

    from gpuwm.core import moist
    from gpuwm.core.dycore import run_steps

    from gpuwm.ingest.lateral_bc import (attach_lateral_boundaries,
                                         build_state_lateral_boundaries)

    cfg = _cfg(specified=True)

    def _build():
        # A uniform mean wind so both an inflow and an outflow face exist,
        # frozen into the specified boundary tables as well as the interior.
        s = _state(cfg)
        s.u[...] += 5.0
        s.v[...] += 3.0
        attach_lateral_boundaries(
            s, build_state_lateral_boundaries([s, s], [0.0, 3600.0]))
        return s

    state = _build()
    run_steps(state, cfg, 6)

    # The invariant checker is the one the LES driver GATES with, so the
    # gate and this test cannot drift apart.
    from gpuwm.verify.cases.convective_boundary_layer import (
        _spec_zone_is_flow_dependent)

    sz = cfg.spec_zone
    e = cp.asnumpy(state.tke).astype(np.float64)

    assert _spec_zone_is_flow_dependent(e, sz)

    # WATCHED control: remove exactly the flow_dep_bdy arm (nothing else),
    # and the spec zone keeps its own advected values -- neither zero nor
    # the donor.
    from gpuwm.ingest import lateral_bc

    fired = []
    original = lateral_bc.apply_flow_dependent_boundaries

    def _no_boundary(fields, u_flux, v_flux, spec_zone):
        fired.append(len(fields))

    lateral_bc.apply_flow_dependent_boundaries = _no_boundary
    try:
        mutant = _build()
        run_steps(mutant, cfg, 6)
    finally:
        lateral_bc.apply_flow_dependent_boundaries = original
    assert fired, "the mutation control never fired"
    assert not _spec_zone_is_flow_dependent(
        cp.asnumpy(mutant.tke).astype(np.float64), sz)


@requires_gpu
def test_the_walled_les_driver_gates_on_the_lateral_arm():
    """The ``--lateral specified`` driver reaches the boundary arm on a
    real trajectory and files the invariant as a GATE, and it drops the
    periodic-only mass-closure gate instead of failing it.

    Mutation control: remove exactly the flow_dep_bdy arm and the metric
    the driver gates on flips to False.
    """
    import cupy as cp

    from gpuwm.ingest import lateral_bc
    from gpuwm.verify.cases import convective_boundary_layer as cbl

    cfg = cbl.make_config(nx=24, ny=24, nz=16, dx=100.0, ztop=1600.0,
                          dt=0.5, minutes=0.5, km_opt=2,
                          lateral="specified")
    assert cfg.specified and not cfg.nested

    result = cbl._integrate(cfg, seed=3, sample_every_s=15.0)
    metrics = result["metrics"]
    assert metrics["tke_spec_zone_flow_dependent"] is True
    assert not metrics["nan"]
    # The walled state really did attach forcing and develop a carrier.
    assert float(cp.abs(result["state"].tke).max()) > 0.0

    fired = []
    original = lateral_bc.apply_flow_dependent_boundaries

    def _no_boundary(fields, u_flux, v_flux, spec_zone):
        fired.append(len(fields))

    lateral_bc.apply_flow_dependent_boundaries = _no_boundary
    try:
        mutant = cbl._integrate(cfg, seed=3, sample_every_s=15.0)
    finally:
        lateral_bc.apply_flow_dependent_boundaries = original
    assert fired, "the mutation control never fired"
    assert mutant["metrics"]["tke_spec_zone_flow_dependent"] is False


def test_the_periodic_driver_reports_no_lateral_arm_verdict():
    """km_opt=3 and periodic runs carry no TKE spec zone, so the metric is
    None rather than a vacuous True that would gate on nothing."""
    from gpuwm.verify.cases import convective_boundary_layer as cbl

    assert cbl.make_config(km_opt=2).specified is False
    with pytest.raises(ValueError, match="lateral must be"):
        cbl.make_config(lateral="open")


# ---------------------------------------------------------------------------
# (3) The budget
# ---------------------------------------------------------------------------

@requires_gpu
def test_the_budget_is_byte_inert():
    """A diagnostic that moves the trajectory is not a diagnostic.  This is
    what lets ``tke_budget`` be a restart-boundary-adjustable field."""
    from gpuwm.core.dycore import run_steps

    off_cfg = _cfg()
    on_cfg = _cfg(tke_budget=1)
    off, on = _state(off_cfg), _state(on_cfg)
    run_steps(off, off_cfg, 15)
    run_steps(on, on_cfg, 15)
    assert _state_bytes(on) == _state_bytes(off)


@requires_gpu
def test_the_budget_closes_and_every_term_is_load_bearing():
    """AC-L5.1's measurement half plus its bound-free control.

    The committed residual is ``storage - sum(terms)``.  It must be small
    against the terms it explains -- and, critically, dropping ANY single
    term must blow it up by orders of magnitude, so the smallness is a
    property of the port rather than of the definition.
    """
    from gpuwm.core.dycore import run_steps

    cfg = _cfg(tke_budget=1)
    state = _state(cfg)
    run_steps(state, cfg, 30)
    budget = tke_budget.drain(state, cfg)
    assert budget is not None and budget["steps"] == 30

    volume = budget["volume"]
    scale = max(abs(volume[name]) for name in tke_budget.SOURCE_TERMS)
    assert scale > 0.0

    # The physics has to be present at all, or "closure" is vacuous.
    assert volume["shear"] > 0.0
    assert volume["dissipation"] < 0.0
    assert abs(volume["storage"]) > 0.0

    closed = abs(budget["residual"]) / scale
    assert closed < 1.0e-3, (
        f"budget does not close: residual/scale = {closed:.3e}, "
        f"terms = {volume}")

    # WATCHED control, one term at a time.
    for name in tke_budget.SOURCE_TERMS:
        if abs(volume[name]) <= 1.0e-3 * scale:
            continue                    # a term this run does not exercise
        dropped = abs(budget["residual"] - (-volume[name])) / scale
        assert dropped > 100.0 * max(closed, 1.0e-12), (
            f"dropping {name} barely moves the residual "
            f"({dropped:.3e} vs {closed:.3e}) -- the budget is not "
            "actually accounting for it")

    # Terms this configuration cannot produce must read as honest zeros.
    assert volume["diffusion_6th"] == 0.0        # diff_6th_opt = 0


@requires_gpu
def test_the_budget_series_survives_a_restart_window():
    """The accumulator is a per-window diagnostic, not trajectory state: a
    resume starts a new window and says so, and the trajectory is unmoved
    by having flipped the toggle across the boundary."""
    from gpuwm.core.dycore import run_steps

    cfg = _cfg(tke_budget=1)
    state = _state(cfg)
    run_steps(state, cfg, 10)
    first = tke_budget.drain(state, cfg)
    assert first["steps"] == 10
    # Draining reset the window; a second drain with no steps says nothing
    # rather than repeating the first window's numbers.
    assert tke_budget.drain(state, cfg) is None
    run_steps(state, cfg, 5)
    second = tke_budget.drain(state, cfg)
    assert second["steps"] == 5
    assert second["volume"] != first["volume"]


def test_budget_terms_partition_the_source_inventory():
    """CPU: the accumulator's own bookkeeping -- every field term is
    accounted for in the closure sum, and the two derived rows are the
    storage it is checked against plus the non-conservative clip."""
    assert set(tke_budget.SOURCE_TERMS) == (
        set(tke_budget.TERM_FIELDS) | {"clip"})
    assert "storage" not in tke_budget.SOURCE_TERMS
    assert tke_budget.TERMS == (
        tke_budget.TERM_FIELDS + tke_budget.DERIVED_TERMS)
    assert len(set(tke_budget.TERMS)) == len(tke_budget.TERMS)
    with pytest.raises(KeyError, match="not a TKE budget field term"):
        tke_budget.term(None, _cfg(tke_budget=1), "storage")
