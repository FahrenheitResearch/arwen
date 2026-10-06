"""Selected history must fit through direct runners and stream capacity review."""

import dataclasses
from datetime import datetime
from types import SimpleNamespace

import pytest

from gpuwm import disk_budget, output_disk, stream
from gpuwm.config import RunConfig
from gpuwm.experiment import experiment_from_run_config
from gpuwm.io.history_selection import HistorySelection
from gpuwm.resume import KEEP_CHECKPOINTS_ENV


def _experiment():
    cfg = RunConfig(nx=20, ny=16, nz=8, dx=3000.0, dy=3000.0,
                    ztop=12000.0, dt=5.0, run_seconds=600.0,
                    output_interval_s=60.0, restart_interval_s=300.0,
                    moist=True, mp_physics=6)
    return experiment_from_run_config(cfg, datetime(2026, 1, 1))


def _trimmed(exp):
    return dataclasses.replace(
        exp, output=HistorySelection.from_mapping({"history_vars": ["T2"]}))


def test_trimmed_output_passes_a_disk_that_full_history_cannot_hold(tmp_path, monkeypatch):
    exp = _experiment()
    monkeypatch.delenv(KEEP_CHECKPOINTS_ENV, raising=False)
    full = output_disk.forecast_projection(exp)
    trimmed = output_disk.forecast_projection(_trimmed(exp))
    assert full["checkpoint_bytes"] == trimmed["checkpoint_bytes"]
    assert 0 < trimmed["history_bytes"] < full["history_bytes"]
    room = (full["total_bytes"] + trimmed["total_bytes"]) // 2
    monkeypatch.setattr(disk_budget, "free_bytes", lambda _path: room)

    with pytest.raises(ValueError, match="stop partway when the disk fills"):
        output_disk.require_output_space(exp, tmp_path)
    admitted = output_disk.require_output_space(_trimmed(exp), tmp_path)
    assert admitted["total_bytes"] <= admitted["free_bytes"]


def test_no_history_mode_reserves_checkpoints_without_history_or_pictures(monkeypatch):
    monkeypatch.setenv(KEEP_CHECKPOINTS_ENV, "1")
    projection = output_disk.forecast_projection(
        _experiment(), io_mode="none", render_products="all")
    assert projection["history_bytes"] == projection["picture_bytes"] == 0
    assert projection["total_bytes"] == projection["checkpoint_bytes"] > 0
    assert all(row["history_frames"] == 0 for row in projection["domains"])


@pytest.mark.parametrize("door", ["run", "single", "tree"])
def test_direct_forecast_doors_refuse_before_restore_or_device_work(tmp_path, monkeypatch, door):
    monkeypatch.setattr(disk_budget, "free_bytes", lambda _path: 1)
    exp = _trimmed(_experiment())
    inputs = SimpleNamespace(experiment=exp, physics_receipt={})
    with pytest.raises(ValueError, match="stop partway when the disk fills"):
        if door == "run":
            from gpuwm.runtime import run_experiment

            run_experiment(exp, None, tmp_path / "run")
        elif door == "single":
            from gpuwm.prepared_single_domain_forecast import run_prepared_forecast

            run_prepared_forecast(inputs, output_directory=tmp_path / "single")
        else:
            from gpuwm.prepared_domain_tree_forecast import run_prepared_tree

            run_prepared_tree(inputs, output_directory=tmp_path / "tree", io_mode="history")


def test_stream_capacity_tracks_selected_history_and_history_free_mode(monkeypatch):
    monkeypatch.delenv(KEEP_CHECKPOINTS_ENV, raising=False)
    exp = _experiment()
    full_plan = SimpleNamespace(experiment=exp, io_mode="history")
    trim_plan = SimpleNamespace(experiment=_trimmed(exp), io_mode="history")
    none_plan = SimpleNamespace(experiment=exp, io_mode="none")
    full = stream._estimated_generation_bytes(full_plan, 1)
    trimmed = stream._estimated_generation_bytes(trim_plan, 1)
    no_history = stream._estimated_generation_bytes(none_plan, 1)
    assert full > trimmed > no_history > 0
    extended = dataclasses.replace(exp, run_seconds=3600.0)
    assert full - no_history == output_disk.forecast_projection(extended)["history_bytes"]


def test_gui_fit_carries_the_engine_disk_review():
    from gpuwm.gui.api import describe_fit

    disk = output_disk.forecast_projection(_trimmed(_experiment()))
    fit = describe_fit({"disk": disk}, {"lat": 40.0, "lon": -100.0,
                                      "width_km": 100.0, "height_km": 100.0})
    assert fit["disk"] == disk


def test_frozen_single_domain_history_and_reflectivity_follow_the_priced_window():
    from gpuwm.runtime import (history_output_due, refl_10cm_due,
                               single_history_window_steps)

    exp = _experiment()
    domain = dataclasses.replace(exp.root, history_begin_s=125.0,
                                 history_end_s=400.0)
    exp = dataclasses.replace(exp, domains=(domain,))
    begin, end = single_history_window_steps(
        domain.run, history_begin_s=domain.history_begin_s,
        history_end_s=domain.history_end_s)
    operands = {"history_begin_outer_step": begin,
                "history_end_outer_step": end}
    interval_steps = round(domain.history_interval_s / domain.run.dt)
    due = [step for step in range(round(exp.run_seconds / domain.run.dt))
           if history_output_due(step, interval_steps, **operands)]
    assert [(step + 1) * domain.run.dt for step in due] == [125, 185, 245, 305, 365]
    assert len(due) == output_disk.forecast_projection(exp)["domains"][0]["history_frames"]
    assert [step for step in range(120)
            if refl_10cm_due(step, 0, interval_steps, 1, **operands)] == due
    assert all(not refl_10cm_due(step, 0, interval_steps, 2, **operands)
               for step in due)


def test_frozen_single_domain_writer_applies_the_same_selection(monkeypatch, tmp_path):
    import numpy as np
    from gpuwm import runtime
    from gpuwm.io import wrfout

    cfg = _experiment().root.run
    frame = {"T2": np.full((cfg.ny, cfg.nx), 280.0, dtype=np.float32),
             "T": np.zeros((cfg.nz, cfg.ny, cfg.nx), dtype=np.float32),
             "QCLOUD": np.zeros((cfg.nz, cfg.ny, cfg.nx), dtype=np.float32),
             "XLAT": np.zeros((cfg.ny, cfg.nx), dtype=np.float32)}
    state = SimpleNamespace(qv=None, _streamed_domain=SimpleNamespace(
        history_fields=lambda: dict(frame)))
    prepared = SimpleNamespace(cfg=cfg, initial_result=SimpleNamespace(
        state=state, coord=None), grid=None, static_fields={})
    written = []

    class RecordingWriter:
        def __init__(self, path, **kwargs):
            written.append({"path": path, **kwargs})

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def write_frame(self, when, fields):
            written[-1].update(time=when, fields=fields)

        def complete_output_identity(self):
            return None

    monkeypatch.setattr(wrfout, "WrfoutWriter", RecordingWriter)
    monkeypatch.setattr(runtime, "_metadata_frame", lambda *_args: {})
    monkeypatch.setattr(runtime, "_global_wrf_attrs", lambda *_args, **_kwargs: {})
    start = datetime(2026, 1, 1)
    for selection in (None, HistorySelection.from_mapping({"history_vars": ["T2"]})):
        runtime.write_case_output(prepared, tmp_path, start, start_time=start,
                                  title="forecast", history_selection=selection)
    assert tuple(written[0]["fields"]) == tuple(frame)
    assert written[0]["global_attrs"] == {}
    # Temperature remains structural; selecting T2 drops the cloud volume.
    assert tuple(written[1]["fields"]) == ("T2", "T", "XLAT")
    assert written[1]["global_attrs"]["GPUWM_HISTORY_DROPPED"] == "QCLOUD"
    assert written[1]["fields"]["T2"] is frame["T2"]
