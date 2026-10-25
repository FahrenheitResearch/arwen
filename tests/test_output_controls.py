"""Operational output alarms and GPU-resident accumulation semantics."""
from dataclasses import replace
from datetime import datetime
from pathlib import Path
import tomllib

import pytest

from gpuwm.experiment import load_experiment
from gpuwm.core.clock import resolve_clock
from gpuwm.prepared_single_domain_forecast import _history_output_schedule
from gpuwm.runtime import history_output_due

ROOT = Path(__file__).resolve().parents[1]


def test_history_switch_matches_clock_fixed_runner_and_inventory(tmp_path):
    # NOAA HRRR's two-cadence history (15 min to f18, then hourly), written
    # into a copy of the recipe: 2.9.4's recipe is hourly, the switch stays.
    recipe = (ROOT / "gpuwm/data/recipes/hrrr_v4_gsd41.toml").read_text(encoding="utf-8")
    # Line-anchored: the [output] table's full_history_interval_s ends the same way.
    assert recipe.count("\nhistory_interval_s = 3600.0\n") == 1
    copy = tmp_path / "hrrr_v4_gsd41.toml"
    copy.write_text(recipe.replace("\nhistory_interval_s = 3600.0\n",
        "\nhistory_interval_s = 900.0\nhistory_interval2_s = 3600.0\n"
        "history_interval_change_s = 64800.0\n"), encoding="utf-8")
    exp = load_experiment(copy)
    exp = replace(exp, run_seconds=48*3600)
    clock = resolve_clock(exp)
    spec = clock.root
    expected = list(range(0, 64801, 900)) + list(range(68400, 172801, 3600))
    assert [s for s in range(0, 172801, 20) if spec.history_offset_due(s * clock.tick_den)] == expected
    assert [0] + [s for s in range(20, 172801, 20) if history_output_due(
        s//20-1, 45, history_interval2_steps=180,
        history_interval_change_step=3240)] == expected
    schedule = _history_output_schedule(start_time=datetime(2026, 1, 1),
        run_seconds=172800, cadence_seconds=900,
        interval2_seconds=3600, change_seconds=64800)
    assert [row[0] for row in schedule] == expected
    assert spec.next_history_offset(63900 * clock.tick_den) == 64800 * clock.tick_den
    assert spec.next_history_offset(64800 * clock.tick_den) == 68400 * clock.tick_den
    assert spec.next_history_offset(65700 * clock.tick_den) == 68400 * clock.tick_den
    assert len(expected) == 103
    from gpuwm.disk_budget import history_frames
    assert history_frames(exp, exp.root) == len(expected)
    assert history_frames(exp, exp.root, after_seconds=64800) == 30
    assert history_frames(exp, exp.root, after_seconds=63900) == 31
    assert _history_output_schedule(start_time=datetime(2026, 1, 1),
        run_seconds=172800, cadence_seconds=900,
        interval2_seconds=3600, change_seconds=64800,
        after_seconds=64800) == tuple(row for row in schedule if row[0] > 64800)


def test_importer_preserves_history_switch_past_short_run(tmp_path):
    from gpuwm.namelist_import import import_namelists
    from test_namelist_import import WPS_TEXT, INPUT_TEXT
    wps = tmp_path / "namelist.wps"
    inp = tmp_path / "namelist.input"
    wps.write_text(WPS_TEXT)
    inp.write_text(INPUT_TEXT.replace(" history_interval = 60, 15,",
        " history_interval = 15, 15,\n history_interval2 = 60, 60,\n history_interval_change = 1080, 1080,"))
    text, report = import_namelists(wps, inp)
    doc = tomllib.loads(text)
    assert all(d["history_interval2_s"] == 3600 for d in doc["domain"])
    assert all(d["history_interval_change_s"] == 64800 for d in doc["domain"])
    assert not any(s.key == "history_interval2" for s in report.substitutions)


from conftest import requires_gpu

@requires_gpu
def test_precipitation_gpu_windows_and_history_independence():
    import cupy as cp
    import numpy as np
    from types import SimpleNamespace
    from gpuwm.config import RunConfig
    from gpuwm.core.output_diagnostics import initialize, update_precipitation
    from gpuwm.io.history_layout import live_state_history_fields
    from gpuwm.core.uh_diag import reset_up_heli_max
    cfg = RunConfig(nx=2, ny=2, nz=3, dx=3000., dy=3000.,
                    ztop=10000., dt=20., run_seconds=3600.,
                    prec_acc_dt=60., prec_acc_dt1=15., mp_physics=28)
    arrays = {}
    def scratch(shape, slot):
        if slot not in arrays:
            arrays[slot] = cp.zeros(shape, dtype=cp.float32)
        return arrays[slot]
    plane=lambda values: cp.asarray(values, dtype=cp.float32).reshape(2,2)
    state = SimpleNamespace(_scratch=arrays, scratch=scratch,
        existing_scratch=arrays.get, elapsed_seconds=0., physics=SimpleNamespace(
        cu_raincv=plane([1,2,3,4]), fields={"t2":plane([270,280,270,280])},
        microphysics=SimpleNamespace(rainncv=plane([4,3,2,1]),
            snowncv=plane([.5,0,1,0]), graupelncv=plane([0,.25,0,.5]))))
    initialize(state,cfg)
    for second in (0., 20., 880.):
        state.elapsed_seconds=second
        update_precipitation(state,cfg)
    np.testing.assert_array_equal(cp.asnumpy(arrays["prec_acc_c"]), [[3,6],[9,12]])
    np.testing.assert_array_equal(cp.asnumpy(arrays["snow_acc_nc"]), [[4.5,0],[12,0]])
    # A history read and its ordinary diagnostics reset do not touch buckets.
    reset_up_heli_max(state)
    assert live_state_history_fields(state)["PREC_ACC_NC1"] is arrays["prec_acc_nc1"]
    state.elapsed_seconds=900.
    update_precipitation(state,cfg)
    np.testing.assert_array_equal(cp.asnumpy(arrays["prec_acc_nc1"]), [[4,3],[2,1]])
    np.testing.assert_array_equal(cp.asnumpy(arrays["prec_acc_nc"]), [[16,12],[8,4]])
    state.elapsed_seconds=3600.
    update_precipitation(state,cfg)
    for suffix in ("", "1"):
        np.testing.assert_array_equal(cp.asnumpy(arrays["prec_acc_c"+suffix]), [[1,2],[3,4]])
        np.testing.assert_array_equal(cp.asnumpy(arrays["snow_acc_nc"+suffix]), [[1.5,0],[4,0]])


def test_precipitation_restart_and_shape_inventory(monkeypatch,tmp_path):
    import numpy as np
    from test_restart import _cfg, _fill_setup, _shim_state
    from gpuwm.io import restart
    from gpuwm.io.history_layout import produced_history_shapes
    cfg=_cfg(prec_acc_dt=60.,prec_acc_dt1=15.)
    source=_shim_state(cfg,monkeypatch)
    _fill_setup(source)
    source.existing_scratch("prec_acc_nc1")[...] = 7.25
    path=restart.write_restart(tmp_path/"buckets.npz",source,cfg)
    target=_shim_state(cfg,monkeypatch)
    _fill_setup(target)
    restart.restore_restart(path,target,cfg)
    np.testing.assert_array_equal(target.existing_scratch("prec_acc_nc1"),7.25)
    shapes=produced_history_shapes(cfg)
    assert shapes["PREC_ACC_NC1"] == (cfg.ny,cfg.nx)
    from gpuwm.core import preflight
    from gpuwm.core.output_diagnostics import scratch_shapes
    for slot,shape in scratch_shapes(cfg).items():
        assert preflight.scratch_slot_registry(cfg)[slot] == shape
        assert preflight.scratch_slot_lifetime(slot).kind == "carrying"
        assert restart.classify_scratch_slot(slot) in ("serialize","rebuild")


def test_precipitation_native_route_round_trip(tmp_path):
    from test_route_round_trip_fields import _emit, _replay
    _,paths=_emit(tmp_path,"precipitation-windows",prec_acc_dt=90.,prec_acc_dt1=10.)
    cfg=_replay(paths).root.run
    assert (cfg.prec_acc_dt,cfg.prec_acc_dt1)==(90.,10.)
