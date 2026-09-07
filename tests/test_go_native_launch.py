"""Human launch uses the executable native routes declared by the registry."""

from __future__ import annotations

import dataclasses
import json
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from gpuwm import capabilities, fetch_routes, go_cli, runplan, source_adapters
from gpuwm.cli import build_parser, main


def emit(tmp_path, source="icon-eu", ladder="12", *, subprocess_door=False):
    config = tmp_path / "weather area.toml"
    args = ["domain", "--point=50.1,8.7" if source != "hrrr" else
            "--point=39,-98", "--card", "16gb", "--ladder", ladder,
            "--source", source, "--cycle", "2026-08-18T06", "--hours", "3",
            "--out", str(config)]
    if subprocess_door:
        result = subprocess.run([sys.executable, "-m", "gpuwm.cli", *args],
                                capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stdout + result.stderr
        return config, result.stdout
    assert main(args) == 0
    return config


@pytest.mark.parametrize("source,ladder", [("icon-eu", "12"),
                                            ("icon-eu", "12-3"),
                                            ("hrrr", "12")])
def test_native_config_has_a_runnable_human_next_command(
        tmp_path, source, ladder):
    """Real CLI processes, including paths with spaces; no bridges or GPU."""
    config, printed = emit(tmp_path, source, ladder, subprocess_door=True)
    line = next("gpuwm go " + line.split("gpuwm go ", 1)[1]
                for line in printed.splitlines() if "gpuwm go " in line
                and not line.lstrip().startswith("#"))
    tokens = shlex.split(line, comments=True)
    parsed = build_parser().parse_args(tokens[1:])
    assert parsed.func is go_cli.go_main
    assert Path(parsed.config) == config
    result = subprocess.run(
        [sys.executable, "-m", "gpuwm.cli", *tokens[1:], "--dry-run"],
        capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"go: {source}, {2 if '-' in ladder else 1} domain(s)" in result.stdout
    assert "sha256" not in result.stdout
    launch = next(line.removeprefix("Run: ") for line in result.stdout.splitlines()
                  if line.startswith("Run: "))
    rerun = shlex.split(launch)
    assert build_parser().parse_args(rerun[1:]).func is go_cli.go_main
    assert "--dry-run" not in rerun
    # A second real process consumes the EXACT printed argv, adding only dry-run.
    again = subprocess.run(
        [sys.executable, "-m", "gpuwm.cli", *rerun[1:], "--dry-run"],
        capture_output=True, text=True, timeout=60)
    assert again.returncode == 0, again.stdout + again.stderr
    assert not (tmp_path / "weather area-go").exists()


def test_registry_extension_reaches_go_without_adding_a_source_name(
        tmp_path, monkeypatch, capsys):
    config = emit(tmp_path)
    donor = source_adapters.get_source_adapter("icon-eu")
    name = "future-native-model"
    graft = dataclasses.replace(donor, source_id=name, aliases=())
    monkeypatch.setattr(source_adapters, "_ADAPTERS",
                        (*source_adapters.source_adapters(), graft))
    monkeypatch.setattr(source_adapters, "_ALIASES",
                        {**source_adapters._ALIASES, name: graft})
    monkeypatch.setattr(fetch_routes, "_ROUTES",
                        {**fetch_routes._ROUTES,
                         name: fetch_routes._ROUTES[donor.source_id]})
    config.write_text(config.read_text().replace('source = "icon-eu"',
                                               f'source = "{name}"'))
    capsys.readouterr()
    assert main(["go", str(config), "--dry-run"]) == 0
    assert f"go: {name}" in capsys.readouterr().out
    assert runplan.prepared_chain_for_source(name) == "prepared:staged"


def test_unsupported_source_refuses_before_creating_output(
        tmp_path, capsys):
    config = emit(tmp_path)
    config.write_text(config.read_text().replace('source = "icon-eu"',
                                               'source = "missing-model"'))
    capsys.readouterr()
    out = tmp_path / "uncreated"
    assert main(["go", str(config), "--outdir", str(out), "--dry-run"]) == 2
    error = capsys.readouterr().err
    assert "missing-model has no automatic native launch route" in error
    assert "Next: gpuwm sources" in error
    assert not out.exists()
    build_parser().parse_args(["sources"])


@pytest.mark.parametrize("with_supplement", [False, True])
def test_go_executes_the_existing_staged_chain_and_carries_the_manifest(
        tmp_path, monkeypatch, capsys, with_supplement):
    """Stub only expensive stages: real plan, route selection and handoff binding."""
    from gpuwm import stage_cli

    allow_launch_resources(monkeypatch)
    config = emit(tmp_path)
    data = tmp_path / "shared data"
    geog = tmp_path / "GEOG"
    geog.mkdir()
    out = tmp_path / "forecast"
    observed = []
    handoff = ["--source", "icon-eu", "--input-list", str(data / "inputs.txt")]
    donor = tmp_path / "declared donor.grib"
    donor.write_bytes(b"explicit donor bytes")
    extra = ["--supplement", f"surface={donor}"] if with_supplement else []

    def fetch(argv, run_dir, **kwargs):
        observed.append(("fetch", list(argv)))
        data.mkdir()
        (data / fetch_routes.PREP_ARGUMENTS_NAME).write_text(json.dumps({
            "schema": fetch_routes.PREP_ARGUMENTS_SCHEMA,
            "source": "icon-eu", "prep_source": "icon-eu", "argv": handoff,
            "unbound_supplement_roles": ["surface"] if with_supplement else [],
            "member": None, "member_set": None}))
        return {}

    def prep(argv):
        observed.append(("prepare", list(argv)))
        root = Path(argv[argv.index("--output-root") + 1])
        root.mkdir(parents=True)
        (root / "proof.json").write_text("{}")

    def bundle(root):
        return {"document": Path(root) / "proof.json", "schema": "probe",
                "source": "icon-eu", "layout": "single", "domains": 1,
                "payload": {}}

    def command(receipt, **options):
        observed.append(("binding", receipt, options))
        return [sys.executable, "-m", "runner", "--source", receipt["source"]]

    def forecast(argv, *, layout, observer):
        observed.append(("forecast", list(argv), layout))
        print("internal digest binding details")
        observer.warn("test_warning", "A source field uses its declared fallback.")

    monkeypatch.setattr(capabilities, "require_for_command", lambda *a, **k: None)
    monkeypatch.setattr(capabilities, "require", lambda *a, **k: None)
    monkeypatch.setattr(runplan, "_run_fetch", fetch)
    monkeypatch.setattr(runplan, "_run_prep", prep)
    monkeypatch.setattr(runplan, "_staged_forecast", forecast)
    monkeypatch.setattr(stage_cli, "resolve_bundle", bundle)
    monkeypatch.setattr(stage_cli, "sim_command", command)
    monkeypatch.setattr(go_cli, "_render_stage",
                        lambda plan, **kw: observed.append(("render", plan)))
    capsys.readouterr()
    assert main(["go", str(config), "--outdir", str(out), "--run-stamp", "off",
                 "--data-dir", str(data), "--geog-root", str(geog),
                 "--products", "none", *extra]) == 0
    terminal = capsys.readouterr()
    assert "go: complete" in terminal.out
    assert "internal digest" not in terminal.out
    assert "declared fallback" in terminal.err
    assert "internal digest" in (out / "launch.log").read_text()
    assert [item[0] for item in observed] == ["fetch", "prepare", "binding",
                                            "forecast", "render"]
    fetch_argv = observed[0][1]
    assert fetch_argv[fetch_argv.index("--source") + 1] == "icon-eu"
    assert Path(fetch_argv[fetch_argv.index("--out") + 1]) == data
    prep_argv = observed[1][1]
    assert prep_argv[:len(handoff)] == handoff
    assert prep_argv[len(handoff):len(handoff) + len(extra)] == extra
    assert Path(prep_argv[prep_argv.index("--geog-root") + 1]) == geog
    receipt, binding = observed[2][1:]
    assert receipt["source"] == "icon-eu"
    assert Path(binding["experiment_config"]) == config
    assert binding["wps_namelist"].is_file()
    events = runplan.read_events(out / runplan.EVENTS_FILENAME)
    assert events[-1]["event"] == "completed"
    assert observed[-1][1]["render_products"] == "none"


def test_failed_native_stage_reports_action_and_keeps_details(
        tmp_path, monkeypatch, capsys):
    from gpuwm.explain import layered

    allow_launch_resources(monkeypatch)
    config = emit(tmp_path)
    out = tmp_path / "failed"
    monkeypatch.setattr(capabilities, "require_for_command", lambda *a, **k: None)
    monkeypatch.setattr(capabilities, "require", lambda *a, **k: None)

    def fetch(*a, **k):
        print("internal failure diagnostics")
        raise runplan.PlanError(layered("The source is unavailable. Try a previous cycle.",
                                        "Long decoder explanation."))

    monkeypatch.setattr(runplan, "_run_fetch", fetch)
    capsys.readouterr()
    assert main(["go", str(config), "--outdir", str(out), "--run-stamp", "off"]) == 1
    output = capsys.readouterr()
    assert "Try a previous cycle" in output.err
    assert "Details:" in output.err
    assert "Long decoder explanation" not in output.err
    assert "internal failure diagnostics" in (out / "launch.log").read_text()
    assert runplan.read_events(out / runplan.EVENTS_FILENAME)[-1]["event"] == "failed"


def allow_launch_resources(monkeypatch):
    from types import SimpleNamespace
    from gpuwm import doctor
    monkeypatch.setattr(doctor, "_cuda_headers_check", lambda: SimpleNamespace(status="verified"))
    monkeypatch.setattr(go_cli, "memory_gate", lambda *a, **k: {
        "verdict": "fits", "refuse": False, "warn": False})
    monkeypatch.setattr(go_cli, "geography_refusal", lambda *a: None)
    monkeypatch.setattr(go_cli, "render_extra_missing", lambda: None)
    monkeypatch.setattr(capabilities, "require_for_command", lambda *a, **k: None)
    monkeypatch.setattr(capabilities, "require", lambda *a, **k: None)


@pytest.mark.parametrize("missing", ["memory", "geography", "renderer"])
def test_native_launch_checks_resources_before_fetch(tmp_path, monkeypatch, capsys, missing):
    config = emit(tmp_path)
    allow_launch_resources(monkeypatch)
    if missing == "memory":
        monkeypatch.setattr(go_cli, "memory_gate", lambda *a, **k: {
            "verdict": "forecast needs 40 GiB", "refuse": True, "warn": False})
    elif missing == "geography":
        monkeypatch.setattr(go_cli, "geography_refusal", lambda *a: "Missing geography. Next: gpuwm fetch-geog")
    else:
        monkeypatch.setattr(go_cli, "render_extra_missing", lambda: "ABI mismatch")
    monkeypatch.setattr(runplan, "execute_plan", lambda *a, **k: pytest.fail("must refuse before executing"))
    out = tmp_path / "not-created"
    capsys.readouterr()
    assert main(["go", str(config), "--outdir", str(out), "--run-stamp", "off"]) == 2
    assert not out.exists()
    message = capsys.readouterr().err
    assert {"memory": "40 GiB", "geography": "fetch-geog", "renderer": "gpuwm setup"}[missing] in message


def test_native_no_memory_gate_and_no_products_are_honored(tmp_path, monkeypatch):
    config = emit(tmp_path)
    allow_launch_resources(monkeypatch)
    monkeypatch.setattr(go_cli, "memory_gate", lambda *a, **k: pytest.fail("memory check was disabled"))
    monkeypatch.setattr(go_cli, "render_extra_missing", lambda: pytest.fail("no products were requested"))
    monkeypatch.setattr(runplan, "execute_plan", lambda *a, **k: 0)
    assert main(["go", str(config), "--outdir", str(tmp_path / "run"),
                 "--products", "none", "--no-memory-gate"]) == 0


@pytest.mark.parametrize("explain", [False, True])
def test_real_adapter_output_is_logged_and_failure_remedy_is_visible(
        tmp_path, monkeypatch, capsys, explain):
    from gpuwm import source_cli
    config = emit(tmp_path)
    allow_launch_resources(monkeypatch)
    out = tmp_path / "real-child"
    def fetch(*a, **k):
        script = (
            "import sys; print('INTERNAL PREP RECEIPT'); "
            "print('warning: source field fallback', file=sys.stderr); "
            "print('Input manifest names a different directory.', file=sys.stderr); "
            "print('Next: use a new manifest path.', file=sys.stderr); sys.exit(78)"
        )
        code = source_cli._run_native_adapter([sys.executable, "-c", script])
        raise runplan.StageExitError("prepare", code)
    monkeypatch.setattr(runplan, "_run_fetch", fetch)
    capsys.readouterr()
    args = ["go", str(config), "--outdir", str(out), "--run-stamp", "off"]
    assert main(args + (["--explain"] if explain else [])) == 1
    terminal = capsys.readouterr()
    assert "Next: use a new manifest path" in terminal.err
    assert "prepare failed (exit 78)" in terminal.err
    assert "source field fallback" in terminal.err
    assert ("INTERNAL PREP RECEIPT" in terminal.out) == explain
    assert "INTERNAL PREP RECEIPT" in (out / "launch.log").read_text()
    assert "Input manifest" in (out / "launch.log").read_text()
    assert source_cli._ADAPTER_OUTPUT.get() is None
    events = runplan.read_events(out / runplan.EVENTS_FILENAME)
    assert events[-1]["exit_code"] == 78


def test_requested_render_failure_cannot_be_reported_complete(tmp_path, monkeypatch):
    class Observer:
        def enter_stage(self, *a, **k):
            pass
    monkeypatch.setattr(go_cli, "_render_stage", lambda *a, **k: False)
    monkeypatch.setattr(runplan, "_chain_render_plan", lambda *a, **k: {
        "render_products": "t2", "run": tmp_path, "png": tmp_path / "png"})
    monkeypatch.setattr(go_cli, "render_command", lambda *a: ["gpuwm", "render", "saved-forecast"])
    monkeypatch.setattr(runplan, "_chain_summary", lambda *a, **k: pytest.fail("must not report complete"))
    with pytest.raises(runplan.PlanError, match="requested pictures were not produced"):
        runplan._chain_render(None, forecast_dir=tmp_path, run_dir=tmp_path, observer=Observer())


@pytest.mark.parametrize("source", ["gfs", "icon-eu"])
def test_kernel_failure_stops_every_launch_before_fetch(tmp_path, monkeypatch, capsys, source):
    from types import SimpleNamespace
    from gpuwm import doctor
    config = emit(tmp_path, source=source)
    allow_launch_resources(monkeypatch)
    monkeypatch.setattr(go_cli, "resolve_bridge", lambda: tmp_path / "bridge")
    monkeypatch.setattr(doctor, "_cuda_headers_check", lambda: SimpleNamespace(
        status="missing", brief="CUDA headers are missing", detail="NVRTC could not find cuda_fp16.h",
        action="install the matching CUDA runtime headers", remedy="header installation detail"))
    monkeypatch.setattr(runplan, "execute_plan", lambda *a, **k: pytest.fail("must not execute"))
    monkeypatch.setattr(go_cli, "_run_stage", lambda *a, **k: pytest.fail("must not fetch"))
    out = tmp_path / "no-output"
    capsys.readouterr()
    assert main(["go", str(config), "--outdir", str(out), "--run-stamp", "off"]) == 2
    assert "install the matching CUDA runtime headers" in capsys.readouterr().err
    assert not out.exists()


def test_unpriced_source_still_checks_forecast_memory(tmp_path, monkeypatch):
    from gpuwm.core import preflight
    config = emit(tmp_path)
    monkeypatch.setattr(preflight, "device_memory_probe_subprocess", lambda: {
        "free_bytes": 128 * 1024 ** 2, "total_bytes": 128 * 1024 ** 2, "profile": None})
    gate = go_cli.memory_gate({"config": config, "source": "icon-eu"})
    assert gate["refuse"]
    assert gate["phases"].forecast_envelope_bytes > gate["free_bytes"]
    assert not gate["phases"].ingest_priced
    assert "NOT PRICED" in gate["verdict"]


def test_go_log_failure_keeps_the_adapter_single_execution(tmp_path, monkeypatch, capsys):
    import errno
    from gpuwm import source_cli
    from gpuwm.command_output import DiagnosticLog
    config = emit(tmp_path)
    allow_launch_resources(monkeypatch)
    out = tmp_path / "log-failure"
    counter = tmp_path / "attempts"
    actual_write = DiagnosticLog.write
    def disk_full(self, text):
        if self.failure is None:
            self._failed(OSError(errno.ENOSPC, "diagnostic log is full"))
        return actual_write(self, text)
    monkeypatch.setattr(DiagnosticLog, "write", disk_full)
    def fetch(*a, **k):
        script = "import pathlib,sys; pathlib.Path(sys.argv[1]).open('a').write('ran\\n'); print('receipt')"
        code = source_cli._run_native_adapter([sys.executable, "-c", script, str(counter)])
        raise runplan.StageExitError("prepare", code)
    monkeypatch.setattr(runplan, "_run_fetch", fetch)
    capsys.readouterr()
    assert main(["go", str(config), "--outdir", str(out), "--run-stamp", "off"]) == 74
    output = capsys.readouterr()
    assert counter.read_text().splitlines() == ["ran"]
    assert "diagnostic log is full" in output.err
    assert "Traceback" not in output.err
    assert "go: complete" not in output.out
    assert source_cli._ADAPTER_OUTPUT.get() is None
