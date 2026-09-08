"""Duration changes preserve science and bind a source-aligned closing window."""
from datetime import datetime, timedelta
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tomllib

import pytest

from gpuwm import cli, companion_forcing, companion_domains, runplan
from gpuwm.namelist_import import parse_namelist_text
from gpuwm.starter_template import render_tables
from test_companion_forcing import candidate


def schedule_request(tmp_path, monkeypatch, seconds):
    forcing = candidate(tmp_path, monkeypatch)
    return {key: forcing[key] for key in ("config_path", "expected_sha256", "output_path")} | {
        "schema": companion_forcing.SCHEDULE_REQUEST_SCHEMA, "run_seconds": seconds}


def supplied_times(request, monkeypatch, *, hours=18, cadence=6):
    from gpuwm.ingest import grib
    path = Path(request["config_path"])
    raw = tomllib.loads(path.read_text())
    for name in raw["case_data"]["forcing"]:
        full = path.parent / name
        full.parent.mkdir(exist_ok=True)
        full.write_bytes(b"header inventory supplied by test seam")
    (path.parent / raw["case_data"]["vtable"]).write_text("fixture mapping")
    start = raw["experiment"]["start_time"]
    times = tuple(start + timedelta(hours=index) for index in range(0, hours + 1, cadence))
    monkeypatch.setattr(grib, "inspect_era5_forcing_times", lambda *args: times)


@pytest.mark.parametrize("seconds,cadence,closing", [(1800, 6, 6), (21600, 6, 6), (900, 6, 6), (900, 3, 3)])
def test_forcing_save_applies_duration_and_pads_only_the_source_window(
        tmp_path, monkeypatch, seconds, cadence, closing):
    request = candidate(tmp_path, monkeypatch)
    request.update(run_seconds=seconds, cadence_hours=cadence,
                   product_type="reanalysis", member=None)
    before = Path(request["config_path"]).read_bytes()
    result = companion_forcing.edit_configuration(request)
    raw = tomllib.loads(Path(result["config_path"]).read_text())
    old = tomllib.loads(before.decode())
    assert raw["experiment"]["run_seconds"] == seconds
    assert raw["fetch"]["hours"] == closing
    assert raw["shared"] == old["shared"] and raw["domain"] == old["domain"]
    assert Path(request["config_path"]).read_bytes() == before
    wps = parse_namelist_text(Path(result["wps_path"]).read_text())
    expected_end = old["experiment"]["start_time"] + timedelta(seconds=seconds)
    assert wps["share"]["end_date"] == [expected_end.strftime("%Y-%m-%d_%H:%M:%S")]
    assert wps["share"]["interval_seconds"] == [cadence * 3600]
    assert result["selection"]["experiment_run_seconds"] == seconds
    assert result["source_sha256"] == request["expected_sha256"]


@pytest.mark.parametrize("seconds", [1800, 21600, 900])
def test_schedule_keeps_complete_covering_tuple_and_acquisition_identity(
        tmp_path, monkeypatch, seconds):
    request = schedule_request(tmp_path, monkeypatch, seconds)
    supplied_times(request, monkeypatch)
    source = Path(request["config_path"])
    before = source.read_bytes()
    old = tomllib.loads(before.decode())
    result = companion_forcing.edit_schedule(request)
    raw = tomllib.loads(Path(result["config_path"]).read_text())
    assert raw["experiment"] == {**old["experiment"], "run_seconds": seconds}
    assert raw["shared"] == old["shared"] and raw["domain"] == old["domain"]
    assert raw["case_data"]["forcing"] == [str((source.parent / name).resolve()) for name in old["case_data"]["forcing"]]
    assert raw["fetch"] == old["fetch"]  # keep the wider request's exact identity
    assert result["fetch_argv"] is None
    assert result["schedule"]["forcing_policy"] == "preserved-supplied-forcing"
    assert result["schedule"]["retained_forcing_intervals"] == 3
    assert result["schedule"]["boundary_window_hours"] == 6
    assert result["source_path"] == str(source.resolve())
    assert result["source_sha256"] == hashlib.sha256(before).hexdigest()
    assert source.read_bytes() == before
    assert not (tmp_path / ".forcing-cache").exists()
    wps = parse_namelist_text(Path(result["wps_path"]).read_text())
    assert wps["share"]["end_date"] == [(old["experiment"]["start_time"] + timedelta(seconds=seconds)).strftime("%Y-%m-%d_%H:%M:%S")]


def test_longer_schedule_selects_new_cache_and_full_closing_request(tmp_path, monkeypatch):
    request = schedule_request(tmp_path, monkeypatch, 19 * 3600)
    supplied_times(request, monkeypatch)
    original = Path(request["config_path"]).read_bytes()
    result = companion_forcing.edit_schedule(request)
    raw = tomllib.loads(Path(result["config_path"]).read_text())
    assert result["schedule"]["forcing_policy"] == "new-request-cache"
    assert result["schedule"]["boundary_window_hours"] == 24
    assert raw["experiment"]["run_seconds"] == 19 * 3600
    assert raw["fetch"]["hours"] == 24 and raw["fetch"]["retrieve"] is True
    assert raw["case_data"]["forcing"] == [str(Path(raw["fetch"]["out"]) / "era5-combined.grib")]
    args = cli.build_parser().parse_args(result["fetch_argv"])
    assert args.hours == 24 and args.cadence == 6 and args.retrieve
    assert Path(request["config_path"]).read_bytes() == original
    assert not Path(raw["fetch"]["out"]).exists()


def test_schedule_cli_returns_bound_new_configuration(tmp_path, monkeypatch, capsys):
    request = schedule_request(tmp_path, monkeypatch, 900)
    path = tmp_path / "schedule-request.json"
    path.write_text(json.dumps(request))
    args = cli.build_parser().parse_args(["companion-forcing", "--schedule-request", str(path)])
    assert args.func(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["schema"] == companion_forcing.SCHEDULE_RESULT_SCHEMA
    assert result["source_sha256"] == request["expected_sha256"]
    assert result["configuration"]["experiment"]["run_seconds"] == 900
    assert result["schedule"]["boundary_window_hours"] == 6
    assert not result["forecast_started"] and not result["acquisition_started"]


@pytest.mark.parametrize("seconds", [True, 0, -1, float("nan"), float("inf"), "900"])
def test_invalid_duration_never_publishes(tmp_path, monkeypatch, seconds):
    request = schedule_request(tmp_path, monkeypatch, seconds)
    with pytest.raises(ValueError, match="finite positive"):
        companion_forcing.edit_schedule(request)
    assert not Path(request["output_path"]).exists()
    assert not (tmp_path / "candidate.namelist.wps").exists()


def test_stale_schedule_request_preserves_every_file(tmp_path, monkeypatch):
    request = schedule_request(tmp_path, monkeypatch, 1800)
    request["expected_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="changed"):
        companion_forcing.edit_schedule(request)
    assert not Path(request["output_path"]).exists()


def test_short_eda_schedule_retains_the_full_member_request_identity(tmp_path, monkeypatch):
    request = candidate(tmp_path, monkeypatch)
    selected = companion_forcing.edit_configuration(request)
    source = Path(selected["config_path"])
    old = tomllib.loads(source.read_text())
    forcing = Path(old["case_data"]["forcing"][0])
    forcing.parent.mkdir(parents=True)
    forcing.write_bytes(b"encoded member inventory is supplied at the native test seam")
    Path(old["case_data"]["vtable"]).write_text("fixture mapping")
    from gpuwm.ingest import grib
    start = old["experiment"]["start_time"]
    monkeypatch.setattr(grib, "inspect_era5_forcing_times",
        lambda *args: tuple(start + timedelta(hours=index) for index in range(0, 16, 3)))
    changed = companion_forcing.edit_schedule({"schema": companion_forcing.SCHEDULE_REQUEST_SCHEMA,
        "config_path": str(source), "expected_sha256": selected["config_sha256"],
        "output_path": str(tmp_path / "quarter-hour.toml"), "run_seconds": 900})
    raw = tomllib.loads(Path(changed["config_path"]).read_text())
    assert raw["fetch"] == old["fetch"]
    assert raw["case_data"]["forcing"] == old["case_data"]["forcing"]
    assert changed["schedule"]["boundary_window_hours"] == 3
    from gpuwm.case_data import build_case_data
    data = build_case_data(raw["case_data"], source=changed["config_path"],
                           base_dir=tmp_path, require_inputs=False, require_met_inputs=False)
    fetch_args = runplan.declared_forcing_fetch(raw, data)
    assert fetch_args[fetch_args.index("--hours") + 1] == "15"
    assert fetch_args[fetch_args.index("--member") + 1] == "7"


def test_hrrr_schedule_regenerates_and_reimports_every_native_companion(tmp_path, monkeypatch):
    from test_hrrr_configured_physics import _case
    from gpuwm.hrrr_route_inputs import route_input_paths, verify_round_trip
    from gpuwm.experiment import load_experiment
    exp, _, source, _, _ = _case(tmp_path)
    raw = tomllib.loads(source.read_text())
    raw["fetch"] = {"source": "hrrr", "cycle": exp.start_time.strftime("%Y-%m-%dT%H"),
                    "hours": 1, "cadence": 1}
    source.write_text(render_tables(raw))
    monkeypatch.setattr(companion_domains, "native_domain_outlines", lambda exp: [])
    result = companion_forcing.edit_schedule({"schema": companion_forcing.SCHEDULE_REQUEST_SCHEMA,
        "config_path": str(source), "expected_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "output_path": str(tmp_path / "half-hour.toml"), "run_seconds": 1800})
    output = Path(result["config_path"])
    changed = load_experiment(output)
    paths = route_input_paths(output)
    assert all(path.is_file() for path in paths.values())
    assert set(result["route_companions"]) == {str(path) for path in paths.values()}
    verify_round_trip(changed, paths["wps_namelist"], paths["namelist_input"])
    assert changed.run_seconds == 1800
    assert changed.root.run == replace(exp.root.run, run_seconds=1800)
    assert result["schedule"]["boundary_window_hours"] == 1
