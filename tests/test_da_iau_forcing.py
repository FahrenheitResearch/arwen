"""DA lane 7: the increment as an RK-stage tendency, its hydrostatic php,
and the MASPT recorder.  CPU arithmetic here; the dycore tests are in
tests/test_da_iau_forcing_gpu.py."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.da import iau
from gpuwm.da.maspt import MasptRecorder


def test_box_density_integrates_to_one_over_any_step_partition():
    knots = iau.box_density(100.0, 1800.0)
    rng = np.random.default_rng(1)
    edges = np.concatenate([[0.0], np.sort(rng.uniform(0, 2500, 57)),
                            [2500.0]])
    weights = [iau.integrate_density(knots, a, b)
               for a, b in zip(edges[:-1], edges[1:])]
    assert abs(sum(weights) - 1.0) < 1e-12
    assert iau.integrate_density(knots, 0.0, 100.0) == 0.0
    assert abs(iau.integrate_density(knots, 100.0, 1000.0) - 0.5) < 1e-12


def test_interpolation_densities_are_a_partition_of_one():
    dens = iau.interpolation_densities([-1800.0, -900.0, 0.0],
                                       -1800.0, 1800.0)
    shares = [iau.integrate_density(k, -np.inf, np.inf) for k in dens]
    assert abs(sum(shares) - 1.0) < 1e-12
    # the last slot holds beyond its time: it owns the whole second half
    assert shares[-1] > 0.5
    for t in np.linspace(-1800, 1800, 41)[:-1]:
        total = sum(iau.integrate_density(k, t, t + 90.0) for k in dens)
        assert abs(total - 90.0 / 3600.0) < 1e-12


def test_interpolation_densities_refuse_slots_off_the_window():
    with pytest.raises(iau.IauError):
        iau.interpolation_densities([0.0, 4000.0], 0.0, 3600.0)
    with pytest.raises(iau.IauError):
        iau.interpolation_densities([10.0, 5.0], 0.0, 3600.0)


def test_resolve_iau_mode():
    assert iau.resolve_iau_mode(None, None) == "oneshot"
    assert iau.resolve_iau_mode(None, 900.0) == "split"
    assert iau.resolve_iau_mode("3d", 1800.0) == "3d"
    with pytest.raises(iau.IauError):
        iau.resolve_iau_mode("3d", None)
    with pytest.raises(iau.IauError):
        iau.resolve_iau_mode("oneshot", 900.0)


def _stub_state(nz=3, ny=4, nx=5):
    z = lambda *s: np.zeros(s, dtype=np.float32)  # noqa: E731
    return SimpleNamespace(
        thp=z(nz, ny, nx), u=z(nz, ny, nx + 1), v=z(nz, ny + 1, nx),
        w=z(nz + 1, ny, nx), php=z(nz + 1, ny, nx), qv=None,
        mup=np.full((ny, nx), 100.0, np.float32),
        mub2d=np.full((ny, nx), 9.0e4, np.float32),
        c1h=np.ones(nz, np.float32), c2h=np.zeros(nz, np.float32),
        c1f=np.ones(nz + 1, np.float32), c2f=np.zeros(nz + 1, np.float32),
        has_msf=False, elapsed_seconds=0.0,
        rth_t=z(nz, ny, nx), ru_t=z(nz, ny, nx + 1), rv_t=z(nz, ny + 1, nx),
        rph_t=z(nz + 1, ny, nx), rw_t=z(nz + 1, ny, nx))


def test_fold_couples_by_column_mass_and_sums_to_the_increment():
    state = _stub_state()
    dth = np.ones_like(state.thp)
    forcing = iau.IauForcing(
        state, [iau.IauNode({"thp": dth}, iau.box_density(0.0, 300.0))])
    iau.attach(state, forcing)
    cfg = SimpleNamespace(dt=45.0, specified=False, nested=False)
    added = np.zeros_like(state.thp, dtype=np.float64)
    t = 0.0
    while t < 400.0:
        state.elapsed_seconds = t
        state.rth_t[...] = 0.0
        tend = iau.fold_step_tendencies(state, cfg, None)
        if tend is not None:
            tend.add_to_slow(state)
            # uncouple and integrate as the RK final stage does
            added += cfg.dt * state.rth_t / (9.0e4 + 100.0)
        t += cfg.dt
    assert abs(forcing.applied_fraction - 1.0) < 1e-12
    np.testing.assert_allclose(added, 1.0, rtol=2e-6)
    receipt = forcing.receipt()
    assert receipt["steps_forced"] == 7       # 0..315 overlaps [0, 300]


def test_fold_outside_the_window_returns_the_physics_object_itself():
    state = _stub_state()
    forcing = iau.IauForcing(
        state, [iau.IauNode({"thp": np.ones_like(state.thp)},
                            iau.box_density(600.0, 300.0))])
    sentinel = object()
    state.elapsed_seconds = 0.0
    assert forcing.fold(state, SimpleNamespace(dt=60.0), sentinel) is sentinel


def test_specified_ring_is_excluded_like_the_physics_couplers():
    state = _stub_state()
    forcing = iau.IauForcing(state, [iau.IauNode(
        {"thp": np.ones_like(state.thp), "u": np.ones_like(state.u)},
        iau.box_density(0.0, 60.0))])
    tend = forcing.fold(state, SimpleNamespace(dt=60.0, specified=True,
                                               nested=False), None)
    rth = tend.slow["rth_t"]
    assert (rth[:, 0, :] == 0).all() and (rth[:, :, -1] == 0).all()
    assert (rth[:, 1:-1, 1:-1] > 0).all()
    ru = tend.slow["ru_t"]
    assert (ru[:, :, 0] == 0).all() and (ru[:, :, -1] == 0).all()


def test_shares_must_sum_to_one():
    state = _stub_state()
    half = ((0.0, 0.5 / 60.0), (60.0, 0.5 / 60.0))
    with pytest.raises(iau.IauError):
        iau.IauForcing(state, [iau.IauNode({"thp": np.ones_like(state.thp)},
                                           half)])


def test_a_field_the_step_never_reaches_is_refused():
    state = _stub_state()
    state.mystery = np.zeros_like(state.thp)
    with pytest.raises(iau.IauError):
        iau.IauForcing(state, [iau.IauNode(
            {"mystery": np.ones_like(state.thp)},
            iau.box_density(0.0, 60.0))])


def test_step_tendencies_sum_scalars_with_the_physics_without_writing_it():
    base_qv = np.full((2, 2, 2), 3.0, np.float32)
    base = SimpleNamespace(scalar_for=lambda n: base_qv if n == "qv" else None,
                           add_to_slow=lambda s: None)
    own = np.full((2, 2, 2), 2.0, np.float32)
    tend = iau.IauStepTendencies(base, {}, {"qv": own, "qr": own})
    np.testing.assert_array_equal(tend.scalar_for("qv"), 5.0)
    np.testing.assert_array_equal(base_qv, 3.0)
    assert tend.scalar_for("qr") is own
    assert tend.scalar_for("qc") is None


def test_maspt_recorder_measures_the_lowest_level_tendency():
    nz, ny, nx = 2, 12, 12
    state = SimpleNamespace(p=np.full((nz, ny, nx), 1.0e5),
                            mup=np.zeros((ny, nx)),
                            thp=np.zeros((nz, ny, nx)),
                            thb=np.full(nz, 300.0), elapsed_seconds=0.0)
    rec = MasptRecorder(start_seconds=0.0, horizon_seconds=600.0, margin=2)

    def step(s, run, **kw):
        s.p[0] += 100.0          # 1 hPa a step
        s.p[0, 0, 0] += 1e6      # boundary noise, outside the interior
        s.elapsed_seconds += run.dt

    stepped = rec.stepper(step)
    run = SimpleNamespace(dt=60.0)
    for _ in range(12):
        stepped(state, run)
    receipt = rec.receipt()
    assert len(receipt["series"]["maspt"]) == 10   # horizon is 600 s
    assert abs(receipt["maspt_first_30min"] - 60.0) < 1e-9   # hPa/h
    assert receipt["insertion_jump"] is None


def test_maspt_recorder_measures_the_insertion_jump():
    nz, ny, nx = 2, 8, 8
    state = SimpleNamespace(p=np.full((nz, ny, nx), 1.0e5),
                            mup=np.zeros((ny, nx)),
                            thp=np.zeros((nz, ny, nx)),
                            thb=np.full(nz, 300.0), elapsed_seconds=0.0)
    rec = MasptRecorder(start_seconds=0.0, horizon_seconds=600.0)
    rec.before_insertion(state)
    state.thp[0] += 2.0                       # the analysis warms 2 K
    state.p[0] += 50.0

    def step(s, run, **kw):
        s.elapsed_seconds += run.dt

    rec.stepper(step)(state, SimpleNamespace(dt=60.0))
    jump = rec.receipt()["insertion_jump"]
    from gpuwm.core import constants as c
    expected = 302.0 * (100050.0 / c.P0) ** c.RCP - 300.0
    assert abs(jump["lowest_level_t_max_abs_k"] - expected) < 1e-9
    assert abs(jump["lowest_level_p_mean_abs_hpa"] - 0.5) < 1e-12


def _leg_context(tmp_path, *, leg, t_start, t_end, restart=None, pending=None,
                 analysis_due=True):
    from tools.da_member_leg import MemberLegContext

    return MemberLegContext(
        args=None, inputs=None, identity=None, cfg_perturb=None,
        hot_cfg=None, leg=leg, absolute_leg_number=leg, t_start=t_start,
        t_end=t_end, stage_root=tmp_path / "stage", out=tmp_path / "out",
        restart=restart, pending=pending, analysis_due=analysis_due)


def test_4d_runner_splits_the_leg_before_an_analysis_and_rewinds_after(
        tmp_path):
    from tools.da_cycle_prepared import Iau4dLegRunner

    calls = []

    def fake(context, name):
        calls.append(context)
        root = (context.stage_root / "restart"
                / f"leg{context.leg:03d}" / str(name))
        root.mkdir(parents=True, exist_ok=True)
        restart = root / "root.npz"
        restart.write_bytes(b"x")
        return SimpleNamespace(restart=restart,
                               record={"elapsed_seconds": context.t_end})

    runner = Iau4dLegRunner(fake, window_seconds=3600.0)
    # leg 0: no analysis pending, one due at its end: split at 1800 s
    r0 = runner(_leg_context(tmp_path, leg=0, t_start=0.0, t_end=3600.0), 3)
    assert [(c.t_start, c.t_end) for c in calls] == [(0.0, 1800.0),
                                                    (1800.0, 3600.0)]
    assert calls[0].stage_root.name == "iau-mid"
    assert calls[1].restart == runner.mid[3] and calls[1].pending is None
    assert calls[1].resumed
    mid0 = runner.mid[3]
    # leg 1: the analysis at 3600 s is pending: rewind to 1800 s, force
    # there, and split again before the next analysis at 7200 s
    calls.clear()
    r1 = runner(_leg_context(tmp_path, leg=1, t_start=3600.0, t_end=7200.0,
                             restart=r0.restart, pending={"thp": 1}), 3)
    assert [(c.t_start, c.t_end) for c in calls] == [(1800.0, 5400.0),
                                                    (5400.0, 7200.0)]
    assert calls[0].restart == mid0 and calls[0].pending == {"thp": 1}
    assert calls[0].iau_rewound and not calls[1].iau_rewound
    assert not mid0.exists()               # restored once, then removed
    assert r1.record["iau4d"]["rewound_from_seconds"] == 1800.0
    # the free leg after the last analysis: rewound, not split
    calls.clear()
    runner(_leg_context(tmp_path, leg=2, t_start=7200.0, t_end=10800.0,
                        restart=r1.restart, pending={"thp": 1},
                        analysis_due=False), 3)
    assert [(c.t_start, c.t_end) for c in calls] == [(5400.0, 10800.0)]
    assert runner.mid == {}


def test_4d_runner_never_splits_or_rewinds_the_control(tmp_path):
    from tools.da_cycle_prepared import Iau4dLegRunner

    calls = []

    def fake(context, name):
        calls.append(context)
        return SimpleNamespace(restart=tmp_path / "r", record={})

    runner = Iau4dLegRunner(fake, window_seconds=3600.0)
    runner(_leg_context(tmp_path, leg=0, t_start=0.0, t_end=3600.0),
           "control")
    assert [(c.t_start, c.t_end) for c in calls] == [(0.0, 3600.0)]


def test_4d_runner_refuses_a_pending_analysis_without_its_mid_restart(
        tmp_path):
    from tools.da_cycle_prepared import Iau4dLegRunner

    runner = Iau4dLegRunner(lambda c, n: None, window_seconds=3600.0)
    with pytest.raises(RuntimeError):
        runner(_leg_context(tmp_path, leg=1, t_start=3600.0, t_end=7200.0,
                            restart=tmp_path / "r", pending={"thp": 1}), 2)


def test_node_groups_share_one_each():
    state = _stub_state()
    one = np.ones_like(state.thp)
    forcing = iau.IauForcing(state, [
        iau.IauNode({"thp": one}, iau.box_density(0.0, 600.0)),
        iau.IauNode({"thp": one}, iau.box_density(300.0, 60.0),
                    group="hydrometeors")])
    assert set(forcing.groups) == {"analysis", "hydrometeors"}
