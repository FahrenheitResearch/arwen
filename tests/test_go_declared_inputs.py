"""Declared inputs reach the existing experiment runtime and shared renderer."""
import json
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from gpuwm import capabilities, go_cli, runplan, runtime
from gpuwm.cli import main
from test_case_data import make_case_toml
from test_runplan import _stub_run_experiment


def case(tmp_path, *, domains=1, source=None):
    path = make_case_toml(tmp_path)
    text = path.read_text()
    if domains == 2:
        text = text.replace("nx = 24", "nx = 60").replace("ny = 20", "ny = 54")
        text += """
[[domain]]
grid_id = 2
parent_id = 1
i_parent_start = 12
j_parent_start = 12
parent_grid_ratio = 3
parent_time_step_ratio = 3
nx = 30
ny = 27
time_step = 20
dx = 4000.0
history_interval_s = 3600.0
"""
    if source is not None:
        text += f'\n[fetch]\nsource = "{source}"\ncycle = "1999-05-03T12"\nhours = 6\n'
    path.write_text(text)
    return path


@pytest.mark.parametrize("domains,source", [(1, None), (2, "gfs")])
def test_actual_cli_dry_run_names_declared_route_and_replayable_command(
        tmp_path, domains, source):
    config = case(tmp_path, domains=domains, source=source)
    before = config.read_bytes()
    out = tmp_path / "new run"
    override = tmp_path / "different geography"
    override.mkdir()
    argv = ["go", str(config), "--outdir", str(out), "--geog-root", str(override),
            "--products", "none", "--run-stamp", "off"]
    result = subprocess.run([sys.executable, "-m", "gpuwm.cli", *argv, "--dry-run"],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"{domains} domain(s); prepare -> forecast" in result.stdout
    assert "fetch ->" not in result.stdout
    if source is not None:
        assert f"go: {source}," in result.stdout
    command = next(line.removeprefix("Run: ") for line in result.stdout.splitlines()
                   if line.startswith("Run: "))
    tokens = shlex.split(command)
    assert tokens[0:2] == ["gpuwm", "go"]
    again = subprocess.run([sys.executable, "-m", "gpuwm.cli", *tokens[1:], "--dry-run"],
                           capture_output=True, text=True, timeout=60)
    assert again.returncode == 0, again.stdout + again.stderr
    assert not out.exists()
    assert config.read_bytes() == before


def test_declared_inputs_refuse_an_unused_download_directory_before_creation(tmp_path, capsys):
    config = case(tmp_path)
    output = tmp_path / "no-write"
    assert main(["go", str(config), "--outdir", str(output), "--data-dir",
                 str(tmp_path / "unused-cache"), "--dry-run"]) == 2
    error = capsys.readouterr().err
    assert "--data-dir is unused" in error and "[case_data].forcing" in error
    assert not output.exists()


@pytest.fixture
def launch_seams(monkeypatch):
    monkeypatch.setattr(capabilities, "require_for_command", lambda *a, **k: None)
    monkeypatch.setattr(capabilities, "require", lambda *a, **k: None)
    monkeypatch.setattr(go_cli, "_require_forecast_device", lambda: None)
    monkeypatch.setattr(go_cli, "geography_refusal", lambda path: None)
    monkeypatch.setattr(go_cli, "render_extra_missing", lambda: None)
    monkeypatch.setattr(runplan, "_run_fetch", lambda *a, **k: pytest.fail("unexpected fetch"))
    calls = []
    monkeypatch.setattr(runplan.RunObserver, "arm_first_products",
                        lambda self, plan: calls.append(("early", plan.copy())))
    actual_stub = _stub_run_experiment()

    def forecast(exp, data, outdir, **kwargs):
        calls.append(("forecast", data))
        print("internal runtime detail")
        return actual_stub(exp, data, outdir, **kwargs)

    def render(label, argv, **kwargs):
        assert label == "render"
        calls.append(("render", list(argv)))
        directory = Path(argv[argv.index("--out") + 1])
        directory.mkdir(parents=True)
        (directory / "fixture.png").write_bytes(b"rendered fixture")

    monkeypatch.setattr(runtime, "run_experiment", forecast)
    monkeypatch.setattr(go_cli, "_run_stage", render)
    return calls


@pytest.mark.parametrize("products", [None, "none", "t2"])
def test_human_launch_uses_existing_runtime_and_shared_render_stage(
        tmp_path, monkeypatch, capsys, launch_seams, products):
    config = case(tmp_path)
    before = config.read_bytes()
    out = tmp_path / "launch"
    geog = tmp_path / "override"
    geog.mkdir()
    if products == "none":
        monkeypatch.setattr(go_cli, "render_extra_missing",
                            lambda: pytest.fail("no-picture launch queried renderer"))
    argv = ["go", str(config), "--outdir", str(out), "--run-stamp", "off",
            "--geog-root", str(geog), "--no-memory-gate"]
    if products is not None:
        argv += ["--products", products]
    assert main(argv) == 0
    terminal = capsys.readouterr()
    assert "go: complete" in terminal.out
    assert "internal runtime detail" not in terminal.out
    assert "internal runtime detail" in (out / "launch.log").read_text()
    names = [entry[0] for entry in launch_seams]
    assert names == (["forecast"] if products == "none" else ["early", "forecast", "render"])
    data = next(value for label, value in launch_seams if label == "forecast")
    assert data.geog_root == geog
    assert data.forcing[0] == config.parent / "forcing/era5_a.grb"
    assert config.read_bytes() == before
    events = runplan.read_events(out / runplan.EVENTS_FILENAME)
    resolved = next(event for event in events if event["event"] == "resolved_plan")
    assert resolved["configuration"]["case_data"]["geog_root"] == str(geog)
    assert resolved["run_options"]["geog_root"] == str(geog)
    assert any(row["basis"] == "run_options.geog_root"
               for row in resolved["automatic_resolutions"])
    assert events[-1]["event"] == "completed"
    assert events[-1]["summary"]["wrfout_count"] == 2
    if products != "none":
        early = launch_seams[0][1]
        assert early["wrfout_dir"] == out and early["render"] == out / "png"
        command = launch_seams[-1][1]
        assert command[command.index("--products") + 1] == (products or "all")
        assert sum(Path(value).name.startswith("wrfout_d") for value in command) == 2
        assert (out / "png/fixture.png").is_file()


@pytest.mark.parametrize("options,expected", [({}, ["forecast"]),
                                             ({"render_products": "none"}, ["forecast"]),
                                             ({"render_products": "t2"}, ["early", "forecast", "render"])])
def test_experiment_run_plan_keeps_its_no_render_default(
        tmp_path, launch_seams, options, expected):
    config = case(tmp_path)
    out = tmp_path / "plan-run"
    plan = runplan.build_plan({"schema": runplan.PLAN_SCHEMA, "name": "declared",
                              "route": "experiment", "config": {"path": str(config)},
                              "output_root": str(out), "run_options": options},
                             source="test", base_dir=tmp_path, sha256="a" * 64)
    with runplan.EventStream(out / runplan.EVENTS_FILENAME, mirror=None) as events:
        assert runplan.execute_plan(plan, events=events) == 0
    assert [entry[0] for entry in launch_seams] == expected


@pytest.mark.parametrize("stage", ["forecast", "render"])
def test_declared_failure_is_actionable_and_preserves_details(
        tmp_path, monkeypatch, capsys, launch_seams, stage):
    config = case(tmp_path)
    out = tmp_path / "failed"
    if stage == "forecast":
        monkeypatch.setattr(runtime, "run_experiment", _stub_run_experiment(fail_at="forecast"))
    else:
        monkeypatch.setattr(go_cli, "_render_stage", lambda *a, **k: False)
    assert main(["go", str(config), "--outdir", str(out), "--run-stamp", "off",
                 "--no-memory-gate"]) == 1
    output = capsys.readouterr()
    assert "Details:" in output.err and "go: complete" not in output.out
    assert (out / "launch.log").is_file()
    assert list(out.glob("wrfout_d*"))
    events = runplan.read_events(out / runplan.EVENTS_FILENAME)
    assert events[-1]["event"] == "failed"
    if stage == "render":
        assert "requested pictures were not produced" in output.err
        assert "render the saved forecast" in output.err


def test_declared_missing_inputs_keep_existing_runplan_gate(tmp_path, launch_seams, capsys):
    config = make_case_toml(tmp_path, files=False)
    out = tmp_path / "missing"
    assert main(["go", str(config), "--outdir", str(out), "--run-stamp", "off",
                 "--products", "none", "--no-memory-gate"]) == 1
    assert "declared input(s) this run needs are not on disk" in capsys.readouterr().err
    assert launch_seams == []


def test_renderer_unavailable_refuses_before_forecast_and_output_creation(
        tmp_path, monkeypatch, capsys, launch_seams):
    config = case(tmp_path)
    out = tmp_path / "no-renderer"
    monkeypatch.setattr(go_cli, "render_extra_missing", lambda: "missing native binary")
    assert main(["go", str(config), "--outdir", str(out), "--run-stamp", "off",
                 "--no-memory-gate"]) == 2
    assert "Rust renderer is unavailable" in capsys.readouterr().err
    assert launch_seams == [] and not out.exists()


def test_actual_first_frame_worker_and_finalizer_share_the_experiment_output(
        tmp_path, monkeypatch, capsys):
    from gpuwm import first_products
    from test_first_products import _stand_in_renderer

    config = case(tmp_path)
    out = tmp_path / "early-run"
    monkeypatch.setattr(capabilities, "require_for_command", lambda *a, **k: None)
    monkeypatch.setattr(capabilities, "require", lambda *a, **k: None)
    monkeypatch.setattr(go_cli, "_require_forecast_device", lambda: None)
    monkeypatch.setattr(go_cli, "geography_refusal", lambda path: None)
    monkeypatch.setattr(go_cli, "render_extra_missing", lambda: None)
    monkeypatch.setattr(first_products, "_run_render", _stand_in_renderer())
    monkeypatch.setattr(runtime, "run_experiment", _stub_run_experiment(frames=1))
    monkeypatch.setattr(go_cli, "_run_stage", lambda *a, **k: pytest.fail(
        "the digest-proven first frame must not be drawn again"))
    assert main(["go", str(config), "--outdir", str(out), "--run-stamp", "off",
                 "--products", "refl,t2", "--no-memory-gate"]) == 0
    records = runplan.read_events(out / runplan.EVENTS_FILENAME)
    first = next(event for event in records if event["event"] == "first_products_ready")
    assert Path(first["frame"]).parent == out
    assert all(Path(name).parent == out / "png" for name in first["paths"])
    assert (out / "png" / first_products.FIRST_PRODUCTS_RECEIPT).is_file()
    assert records[-1]["event"] == "completed"
    assert "first pictures ready" in capsys.readouterr().out


def test_memory_gate_reuses_the_resolved_experiment_for_a_geography_override(
        tmp_path, monkeypatch, capsys, launch_seams):
    from gpuwm.core import preflight
    from datetime import datetime
    from gpuwm.ingest import grib

    config = case(tmp_path)
    (tmp_path / "GEOG").rmdir()
    override = tmp_path / "real-geography"
    override.mkdir()
    monkeypatch.setattr(preflight, "_load_experiment_any", lambda *a, **k: pytest.fail(
        "the resolved experiment must not be reloaded against the old geography"))
    monkeypatch.setattr(preflight, "device_memory_probe_subprocess", lambda: None)
    monkeypatch.setattr(preflight, "device_memory_probe_reason", lambda: "test device absent")
    monkeypatch.setattr(grib, "inspect_era5_forcing_times", lambda *a, **k: (
        datetime(1999, 5, 3, 12), datetime(1999, 5, 3, 18)))
    assert main(["go", str(config), "--outdir", str(tmp_path / "override-run"),
                 "--run-stamp", "off", "--products", "none",
                 "--geog-root", str(override)]) == 0
    assert next(value for label, value in launch_seams if label == "forecast").geog_root == override


def test_malformed_case_table_cannot_fall_through_to_the_prepared_route(tmp_path, capsys):
    config = tmp_path / "malformed.toml"
    config.write_text('case_data = []\n')
    assert main(["go", str(config), "--dry-run"]) == 2
    assert "go: complete" not in capsys.readouterr().out
