"""An explicit cycle is a checked no-op only at the actual input time."""
from datetime import datetime
import json
from pathlib import Path
from types import SimpleNamespace
import tomllib

import numpy as np
import pytest

from gpuwm import cli, go_cli, input_cycle, runplan


def bundle(tmp_path, **values):
    root = tmp_path / "prepared"
    root.mkdir()
    (root / "proof.json").write_text(json.dumps({
        "schema": "gpuwm-era5-direct-wrf-proof-v2", **values}), encoding="utf-8")
    return root


def checkpoint(tmp_path, **header):
    path = tmp_path / "arbitrary-name.npz"
    np.savez(path, __gpuwm_restart_header__=np.frombuffer(
        json.dumps(header).encode(), dtype=np.uint8))
    return path


@pytest.mark.parametrize("time", ["2020-01-02T03:00:00", "2020-01-02T04:00:00+01:00"])
def test_prepared_cycle_reads_bundle_time(tmp_path, time):
    root = bundle(tmp_path, forcing_times=[time])
    receipt = input_cycle.verify("2020-01-02T03", prepared_root=root)
    assert receipt == {"cycle": "2020-01-02T03", "input_start_time": "2020-01-02T03:00:00",
                       "basis": "prepared_bundle", "action": "no-op", "retimed": False}
    with pytest.raises(ValueError, match="whose times a cycle cannot move"):
        input_cycle.verify("2020-01-02T04", prepared_root=root)


@pytest.mark.parametrize("values", [{}, {"forcing_times": ["2020-01-02T03:00:00"],
                                       "valid_time": "2020-01-02T04:00:00"}])
def test_bundle_without_one_proven_start_is_refused(tmp_path, values):
    with pytest.raises(ValueError, match="whose times a cycle cannot move"):
        input_cycle.verify("2020-01-02T03", prepared_root=bundle(tmp_path, **values))


@pytest.mark.parametrize("tree", [False, True])
def test_checkpoint_compares_its_current_time_not_bundle_origin(tmp_path, tree):
    header = {"elapsed_seconds": 3600.0}
    if tree:
        header.update(domain_start_time="2020-01-02T03:00:00", domain_start_ticks=0, tick_den=1000)
    else:
        header["physics_setup"] = {"radiation": {"start_time": "2020-01-02T03:00:00"}}
    path = checkpoint(tmp_path, **header)
    assert input_cycle.verify("2020-01-02T04", restart=path)["basis"] == "checkpoint"
    with pytest.raises(ValueError, match="whose times a cycle cannot move"):
        input_cycle.verify("2020-01-02T03", restart=path)


def test_checkpoint_without_radiation_uses_bundle_origin_and_header_elapsed(tmp_path):
    root = bundle(tmp_path, forcing_times=["2020-01-02T03:00:00"])
    path = checkpoint(tmp_path, elapsed_seconds=3600.0)
    assert input_cycle.verify("2020-01-02T04", prepared_root=root, restart=path)["retimed"] is False


def test_missing_forcing_defers_only_until_acquisition(tmp_path, monkeypatch):
    forcing = tmp_path / "forcing.nc"
    data = SimpleNamespace(forcing=(forcing,), vtable=tmp_path / "Vtable")
    assert input_cycle.verify("2020-01-02T03", data=data, allow_pending=True) is None
    with pytest.raises(ValueError, match="whose times a cycle cannot move"):
        input_cycle.verify("2020-01-02T03", data=data)
    forcing.write_bytes(b"metadata fixture")
    from gpuwm.ingest import grib
    monkeypatch.setattr(grib, "inspect_era5_forcing_times", lambda paths, table: (datetime(2020, 1, 2, 4),))
    with pytest.raises(ValueError, match="actual input start is 2020-01-02T04"):
        input_cycle.verify("2020-01-02T03", data=data, allow_pending=True)
    assert input_cycle.verify("2020-01-02T04", data=data)["basis"] == "case_data"
    with pytest.raises(ValueError, match="configured start differs"):
        input_cycle.verify("2020-01-02T04", data=data, launch_start=datetime(2020, 1, 2, 5))
    restart = checkpoint(tmp_path, elapsed_seconds=3600.0)
    with pytest.raises(ValueError, match="configured start differs"):
        input_cycle.verify("2020-01-02T05", data=data, restart=restart,
                           launch_start=datetime(2020, 1, 2, 5))


def test_equal_fetch_hint_cannot_override_actual_bundle_time(tmp_path):
    root = bundle(tmp_path, forcing_times=["2020-01-02T03:00:00"])
    args = cli.build_parser().parse_args([
        "go", str(tmp_path / "config.toml"), "--prepared-root", str(root),
        "--cycle", "2020-01-02T04"])
    with pytest.raises(go_cli.GoRefusal, match="whose times a cycle cannot move"):
        go_cli._flag_cycle(args, {"fetch": {"source": "era5", "cycle": "2020-01-02T04"}})
    args.cycle = "2020-01-02T03"
    assert go_cli._flag_cycle(args, {"fetch": {"source": "era5", "cycle": "2020-01-02T04"}}) is None
    assert args.input_cycle == "2020-01-02T03"


def test_latest_and_malformed_assertions_are_never_guessed(tmp_path):
    root = bundle(tmp_path, forcing_times=["2020-01-02T03:00:00"])
    for cycle in ("latest", "yesterday"):
        with pytest.raises(ValueError, match="whose times a cycle cannot move"):
            input_cycle.verify(cycle, prepared_root=root)


def test_assertion_is_distinct_from_retiming_option(tmp_path):
    assert runplan._run_option("input_cycle", "2020-01-02T03", tmp_path) == "2020-01-02T03"
    with pytest.raises(runplan.PlanError):
        runplan._run_option("input_cycle", "latest", tmp_path)
    assert "input_cycle" in runplan.ROUTES["experiment"].run_options
    assert "cycle" not in runplan.ROUTES["experiment"].run_options


def test_missing_declared_fetch_is_deferred_but_readiness_cannot_claim_equality(tmp_path):
    from test_case_data import make_case_toml, _CASE_DATA_TOML

    case_data = _CASE_DATA_TOML.replace(
        'forcing = ["forcing/era5_a.grb", "forcing/era5_b.grb"]',
        'forcing = ["era5-combined.nc"]')
    config = make_case_toml(tmp_path, case_data=case_data)
    payload = tomllib.loads(config.read_text())
    payload["fetch"] = {"source": "era5", "cycle": "2020-01-02T03", "hours": 1,
                        "era5_provider": "arco", "area": "34,-99,36,-97", "out": str(tmp_path)}
    args = cli.build_parser().parse_args(["go", str(config), "--cycle", "2020-01-02T03"])
    assert go_cli._flag_cycle(args, payload) is None
    assert args.input_cycle == "2020-01-02T03"
    assert not (tmp_path / "era5-combined.nc").exists()
    args.readiness = True
    with pytest.raises(go_cli.GoRefusal, match="whose times a cycle cannot move"):
        go_cli._flag_cycle(args, payload)


@pytest.mark.parametrize("elapsed", [True, -1, float("inf"), 3600.1])
def test_checkpoint_invalid_or_fractional_time_is_not_rounded_to_a_cycle(tmp_path, elapsed):
    path = checkpoint(tmp_path, elapsed_seconds=elapsed,
        physics_setup={"radiation": {"start_time": "2020-01-02T03:00:00"}})
    with pytest.raises(ValueError, match="whose times a cycle cannot move"):
        input_cycle.verify("2020-01-02T04", restart=path)


@pytest.mark.parametrize("actual", [3, 4])
def test_fetch_checks_actual_time_before_preparation_and_records_noop(
        tmp_path, monkeypatch, actual):
    from dataclasses import replace
    from gpuwm import capabilities
    from gpuwm.ingest import grib
    from test_case_data import make_case_toml

    config = make_case_toml(tmp_path)
    text = config.read_text()
    import re
    text = re.sub(r"start_time\s*=.*", 'start_time = 2020-01-02T03:00:00', text)
    config.write_text(text)
    plan = runplan.build_plan({"schema": runplan.PLAN_SCHEMA, "name": "input-time",
        "route": "experiment", "config": {"path": str(config)},
        "output_root": str(tmp_path / "run"),
        "fetch": {"args": ["--source", "era5", "--cycle", "2020-01-02T03", "--hours", "1",
                           "--area", "34,-99,36,-97", "--out", str(tmp_path / "download")]},
        "run_options": {"input_cycle": "2020-01-02T03", "render_products": "none"}},
        source="test", base_dir=tmp_path, sha256="0" * 64)
    order = []
    monkeypatch.setattr(capabilities, "require", lambda *a, **k: None)
    monkeypatch.setattr(runplan, "disk_admission_refusal", lambda *a, **k: None)
    def fetched(*args, **kwargs):
        order.append("fetch")
        return {}
    def times(paths, table):
        order.append("verify")
        return (datetime(2020, 1, 2, actual),)
    def prepare(*args, **kwargs):
        order.append("prepare")
        return {"completed_seconds": 0.0}
    monkeypatch.setattr(runplan, "_run_fetch", fetched)
    monkeypatch.setattr(grib, "inspect_era5_forcing_times", times)
    monkeypatch.setitem(runplan.ROUTES, "experiment", replace(runplan.ROUTES["experiment"], execute=prepare))
    plan.run_dir.mkdir()
    with runplan.EventStream(plan.run_dir / runplan.EVENTS_FILENAME, mirror=None) as events:
        code = runplan.execute_plan(plan, events=events)
    records = runplan.read_events(plan.run_dir / runplan.EVENTS_FILENAME)
    manifest = json.loads((plan.run_dir / runplan.MANIFEST_FILENAME).read_text())
    if actual == 3:
        assert code == 0, records[-1].get("message", records[-1])
        assert order == ["fetch", "verify", "prepare"]
        assert manifest["input_cycle"]["action"] == "no-op"
        assert records[-1]["summary"]["input_cycle"] == manifest["input_cycle"]
    else:
        assert code == 1
        assert order == ["fetch", "verify"]
        assert "whose times a cycle cannot move" in records[-1]["message"]
        assert "input_cycle" not in manifest
