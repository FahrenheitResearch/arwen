"""Adaptive history alarms are deadlines, independent of the nominal step."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta
from fractions import Fraction
from types import SimpleNamespace
import tomllib
import sys

import numpy as np
import pytest

from gpuwm.core.adaptive_clock import AdaptiveClockDriver
from gpuwm.core.adaptive_timestep import real_time_fp32
from gpuwm.core.clock import build_schedule, execute_schedule, resolve_clock
from gpuwm.experiment import build_experiment
from gpuwm.namelist_import import import_namelists
from tests.test_namelist_import import INPUT_TEXT, _pair
from test_clock import _chain_experiment


def _experiment(*, adaptive=True, windows=False, ratios=(1, 3)):
    exp = _chain_experiment(ratios, run_seconds=1800.0)
    domains = []
    for dc in exp.domains:
        interval = 17.0 if dc.parent_id == 0 else 25.0
        run = replace(dc.run, use_adaptive_time_step=adaptive,
                      output_interval_s=interval, cu_physics=0,
                      ra_physics=0, ra_lw_physics=0, ra_sw_physics=0,
                      bl_pbl_physics=0, sf_sfclay_physics=0,
                      sf_surface_physics=0)
        domains.append(replace(
            dc, run=run, history_interval_s=interval,
            history_begin_s=(5.0 if dc.parent_id == 0 else 7.0)
                if windows else 0.0,
            history_end_s=(70.0 if dc.parent_id == 0 else 77.0)
                if windows else None))
    return replace(exp, domains=tuple(domains), restart_interval_s=60.0)


def _model(exp, clock):
    nodes = {}
    for dc in exp.domains:
        node = SimpleNamespace(
            cfg=dc, clock=clock.domain_clock(dc.grid_id), children=[],
            parent=nodes.get(dc.parent_id), state=SimpleNamespace(physics=None))
        nodes[dc.grid_id] = node
        if node.parent is not None:
            node.parent.children.append(node)
    return SimpleNamespace(root=nodes[exp.root.grid_id], node=nodes.__getitem__)


def test_imports_an_adaptive_history_interval_off_the_nominal_step_grid(tmp_path):
    inp = INPUT_TEXT.replace(" history_interval = 60, 15,",
                             " history_interval = 0, 0,\n"
                             " history_interval_s = 17, 25,")
    inp = inp.replace("&domains\n", "&domains\n"
                      " use_adaptive_time_step = .true.,\n"
                      " step_to_output_time = .true.,\n")
    text, _ = import_namelists(*_pair(tmp_path, inp=inp))
    exp = build_experiment(tomllib.loads(text), source="synthetic namelist")
    assert exp.root.run.use_adaptive_time_step
    assert [dc.history_interval_s for dc in exp.domains] == [17.0, 25.0]
    clock = resolve_clock(exp)
    assert [spec.history_ticks for spec in clock.domains] == [1700, 2500]


def test_imports_adaptive_radiation_periods_off_the_nominal_step_grid(tmp_path):
    inp = INPUT_TEXT.replace(" radt = 12, 3,", " radt = 0.13, 0.17,")
    inp = inp.replace("&domains\n", "&domains\n"
                      " use_adaptive_time_step = .true.,\n")
    text, _ = import_namelists(*_pair(tmp_path, inp=inp))
    exp = build_experiment(tomllib.loads(text), source="synthetic namelist")
    clock = resolve_clock(exp)
    assert [spec.radt_ticks for spec in clock.domains] == [780, 1020]
    assert [dc.run.radt for dc in exp.domains] == [0.13, 0.17]


def test_the_same_fixed_step_history_interval_still_names_its_misalignment(tmp_path):
    inp = INPUT_TEXT.replace(" history_interval = 60, 15,",
                             " history_interval = 0, 0,\n"
                             " history_interval_s = 17, 25,")
    with pytest.raises(ValueError, match="history_interval_s.*whole number"):
        import_namelists(*_pair(tmp_path, inp=inp))
    with pytest.raises(ValueError, match="history_interval_s.*whole number"):
        resolve_clock(_experiment(adaptive=False))


@pytest.mark.parametrize("ratios", [(1,), (1, 3), (1, 3, 3)])
@pytest.mark.parametrize("windows", [False, True])
def test_the_live_driver_lands_on_every_history_and_keeps_tree_clocks_exact(
        ratios, windows):
    exp = _experiment(windows=windows, ratios=ratios)
    clock = resolve_clock(exp, lbc_interval_s=300.0)
    schedule = build_schedule(exp, clock)
    model = _model(exp, clock)
    driver = AdaptiveClockDriver(model, cfl_source=lambda gid: (0.0, 0.0),
                                 tick_den=clock.tick_den)
    history = {dc.grid_id: [] for dc in exp.domains}
    events = []
    steps = []

    def on_step(gid, dom):
        # The shortened step reaches the kernel config and WRF REAL dt
        # mirror together, including the child step that divides it.
        run = model.node(gid).cfg.run
        exact = Fraction(dom.step_ticks, clock.tick_den)
        assert run.dt == float(exact)
        assert np.float32(dom.dt_fp32).tobytes() == real_time_fp32(exact).tobytes()
        driver.before_step(gid)
        steps.append((gid, dom.ticks, dom.step_ticks))

    def on_history(gid, ticks):
        history[gid].append(ticks)
        events.append(("history", gid, ticks))

    report = execute_schedule(
        schedule, clocks={gid: model.node(gid).clock for gid in history},
        on_period_steps=driver, on_step=on_step,
        on_history=on_history,
        on_restart=lambda ticks: events.append(("restart", 1, ticks)),
        on_lbc_reset=lambda ticks: events.append(("lbc", 1, ticks)))
    for dc in exp.domains:
        begin = int(dc.history_begin_s)
        end = int(dc.history_end_s if dc.history_end_s is not None
                  else exp.run_seconds)
        expected = list(range(begin * clock.tick_den,
                              end * clock.tick_den + 1,
                              int(dc.history_interval_s) * clock.tick_den))
        assert history[dc.grid_id] == expected
        assert report.clocks[dc.grid_id].ticks == clock.run_ticks
    assert any(ticks < clock.root.step_ticks for gid, now, ticks in steps
               if gid == exp.root.grid_id)
    assert [ticks for kind, gid, ticks in events if kind == "restart"] == list(
        range(60 * clock.tick_den, clock.run_ticks + 1, 60 * clock.tick_den))
    assert [ticks for kind, gid, ticks in events if kind == "lbc"] == list(
        range(0, clock.run_ticks, 300 * clock.tick_den))
    # When history and restart coincide, durable history is published first.
    for index, (kind, gid, ticks) in enumerate(events):
        if kind == "restart":
            matching_history = [i for i, event in enumerate(events)
                                if event[0] == "history" and event[2] == ticks]
            assert all(i < index for i in matching_history)


def test_adaptive_history_begin_is_exact_and_fixed_begin_keeps_its_rounding():
    exp = _experiment(windows=True)
    adaptive = resolve_clock(exp)
    assert [spec.history_begin_ticks for spec in adaptive.domains] == [500, 700]
    fixed_domains = tuple(replace(dc, history_interval_s=60.0,
                                  run=replace(dc.run, use_adaptive_time_step=False,
                                              output_interval_s=60.0))
                          for dc in exp.domains)
    fixed = resolve_clock(replace(exp, domains=fixed_domains))
    assert [spec.history_begin_ticks for spec in fixed.domains] == [60, 20]


def test_subsecond_history_still_refuses_filename_aliasing(tmp_path):
    from test_experiment import BASE

    text = BASE.format(experiment="restart_interval_s = 0.0",
                       shared="use_adaptive_time_step = true", d01="", d02="")
    text = text.replace("history_interval_s = 900.0", "history_interval_s = 0.5")
    with pytest.raises(ValueError, match="whole number of seconds"):
        build_experiment(tomllib.loads(text), source="synthetic experiment")


def test_adaptive_physics_uses_requested_time_periods_through_history_shortening():
    exp = _experiment(ratios=(1, 3))
    domains = tuple(replace(
        dc, run=replace(dc.run, ra_physics=90, ra_lw_physics=-1,
                        ra_sw_physics=-1, radt=0.13,
                        cu_physics=1, cudt_minutes=0.17,
                        sf_sfclay_physics=1, sf_surface_physics=2,
                        bl_pbl_physics=1, bldt=0.19)) for dc in exp.domains)
    exp = replace(exp, run_seconds=180.0, domains=domains)
    clock = resolve_clock(exp, lbc_interval_s=60.0)
    model = _model(exp, clock)
    calls = {gid: {label: [] for label in ("radt", "cudt", "bldt")}
             for gid in (1, 2)}
    for gid in (1, 2):
        model.node(gid).state.physics = SimpleNamespace(
            stepra=1, stepcu=1, stepbl=1,
            radt_minutes=0.13, cudt_minutes=0.17,
            radt_seconds=60.0, cudt_seconds=60.0, bldt_seconds=60.0)
    driver = AdaptiveClockDriver(model, cfl_source=lambda gid: (0.0, 0.0),
                                 tick_den=clock.tick_den)

    # An old observed gap must not change the solar/PBL consumers'
    # configured period when the exact clock calendar is authoritative.
    driver._radiation_actual.update({1: 900.0, 2: 900.0})

    def on_step(gid, dom):
        driver.before_step(gid)
        physics = model.node(gid).state.physics
        for label, attr in (("radt", "radiation_due_override"),
                            ("cudt", "cumulus_due_override"),
                            ("bldt", "surface_pbl_due_override")):
            if getattr(physics, attr):
                calls[gid][label].append(dom.ticks)
        assert physics.bldt_seconds == pytest.approx(11.4)
        assert physics.radt_seconds == 7.8
        assert physics.cudt_seconds == 10.2

    execute_schedule(
        build_schedule(exp, clock),
        clocks={gid: model.node(gid).clock for gid in (1, 2)},
        on_period_steps=driver, on_step=on_step)
    for gid in (1, 2):
        for label in ("radt", "cudt", "bldt"):
            interval = getattr(clock.spec(gid), label + "_ticks")
            assert calls[gid][label] == list(range(0, clock.run_ticks, interval))


def test_surface_pbl_override_preserves_fixed_predicate_and_selects_due_calls():
    from gpuwm.core.physics import _surface_pbl_step_due

    assert [step for step in range(1, 17)
            if _surface_pbl_step_due(step, 5, 5.0)] == [1, 5, 10, 15]
    assert not _surface_pbl_step_due(1, 5, 5.0, False)
    assert _surface_pbl_step_due(2, 5, 5.0, True)


@pytest.mark.parametrize("physics_active", [False, True])
def test_offgrid_alarms_land_on_delayed_child_activation(physics_active):
    exp = _experiment()
    domains = tuple(replace(
        dc, start_time=exp.start_time + timedelta(seconds=60)
            if dc.parent_id else None,
        run=replace(dc.run, ra_physics=90 if physics_active else 0,
                    ra_lw_physics=-1, ra_sw_physics=-1, radt=0.13))
        for dc in exp.domains)
    exp = replace(exp, domains=domains, run_seconds=180.0,
                  restart_interval_s=0.0)
    clock = resolve_clock(exp)
    model = _model(exp, clock)
    driver = AdaptiveClockDriver(model, cfl_source=lambda gid: (0.0, 0.0),
                                 tick_den=clock.tick_den)
    starts, child_steps, histories = [], [], []
    execute_schedule(
        build_schedule(exp, clock),
        clocks={gid: model.node(gid).clock for gid in (1, 2)},
        on_period_steps=driver,
        on_domain_start=lambda gid, dom: starts.append((gid, dom.ticks)),
        on_step=lambda gid, dom: child_steps.append(dom.ticks) if gid == 2 else None,
        on_history=lambda gid, ticks: histories.append(ticks) if gid == 2 else None)
    assert starts == [(2, 60 * clock.tick_den)]
    assert child_steps[0] == 60 * clock.tick_den
    assert histories == [seconds * clock.tick_den
                         for seconds in (60, 85, 110, 135, 160)]


@pytest.mark.parametrize("adaptive,expected", [(True, 7.8), (False, 60.0)])
def test_actual_dudhia_solar_receiver_uses_adaptive_requested_period(
        monkeypatch, adaptive, expected):
    from gpuwm.core import dudhia

    monkeypatch.setitem(sys.modules, "cupy", np)
    seen = []
    monkeypatch.setattr(dudhia, "wrf_solar_geometry", lambda *args, **kw:
                        (seen.append(kw["hour_offset_seconds"]) or
                         np.ones((1, 1)), 1360.0))
    monkeypatch.setattr(dudhia, "dudhia_shortwave_columns", lambda *args, **kw:
                        (np.zeros((1, 2)), np.ones(1), np.ones(1)))
    adapter = object.__new__(dudhia.DudhiaShortwaveRadiation)
    adapter.start_time = datetime(2000, 1, 1)
    adapter.latitude_deg = np.zeros((1, 1))
    adapter.longitude_deg = np.zeros((1, 1))
    adapter.icloud, adapter.swrad_scat, adapter.update_count = 0, 1.0, 0
    cfg = replace(_experiment(ratios=(1,)).root.run,
                  use_adaptive_time_step=adaptive, radt=0.13)
    atmosphere = {name: np.ones((2, 1, 1)) for name in (
        "temperature", "qc", "qi", "pressure", "qv", "dz", "exner")}
    adapter(atmosphere=atmosphere,
            fields={"albedo": np.zeros((1, 1)), "glw": np.ones((1, 1))},
            state=SimpleNamespace(elapsed_seconds=0.0), cfg=cfg)
    assert seen == [0.5 * expected]


@pytest.mark.parametrize("adaptive,expected", [(True, 10.2), (False, 10.8)])
def test_actual_kf_launcher_receives_requested_adaptive_cudt(
        monkeypatch, adaptive, expected):
    from gpuwm.core import kf

    monkeypatch.setitem(sys.modules, "cupy", np)
    seen = []

    def launch(*args, **kwargs):
        seen.append(kwargs)
        result = {name: np.zeros((2, 1, 1), np.float32) for name in (
            "rthcuten", "rqvcuten", "rqccuten", "rqicuten", "rqrcuten", "rqscuten")}
        result.update(nca_seconds=np.full((1, 1), 20.0, np.float32),
                      rainc=np.full((1, 1), 2.0, np.float32))
        return result

    monkeypatch.setattr(kf, "launch_kf", launch)
    cfg = replace(_experiment(ratios=(1,)).root.run,
                  use_adaptive_time_step=adaptive, dt=1.2,
                  cudt_minutes=0.17, mp_physics=10)
    state = SimpleNamespace(elapsed_seconds=10.2)
    adapter = kf.KainFritsch()
    adapter.w0avg = np.zeros((2, 1, 1))
    adapter._history_state, adapter._history_time = state, 10.2
    atmosphere = {name: np.ones((2, 1, 1)) for name in (
        "u", "v", "temperature", "qv", "qc", "pressure", "exner", "dz")}
    result = adapter(atmosphere=atmosphere, fields={}, state=state, cfg=cfg)
    assert seen[0]["cudt"] == pytest.approx(expected)
    assert seen[0]["dt"] == 1.2
    np.testing.assert_array_equal(
        result.pratec, np.float32(2.0) / np.float32(expected))
