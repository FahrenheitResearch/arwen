"""Public resume reuses the existing strict runners and retains original inputs."""
from __future__ import annotations

import hashlib
import json
import shlex
from pathlib import Path

import pytest

from gpuwm import capabilities, go_cli, runplan, runtime, stage_cli
from gpuwm.cli import build_parser, main
from test_go_declared_inputs import case
from test_runplan import _stub_run_experiment
from test_stage_seams import _single_domain_bundle, _tree_bundle


def plan_for(config, output, **options):
    return runplan.build_plan({
        "schema": runplan.PLAN_SCHEMA, "name": "existing prepared input",
        "route": "prepared", "config": {"path": str(config)},
        "output_root": str(output), "run_options": {
            "render_products": "none", **{key: str(value) for key, value in options.items()}},
    }, source="test plan", base_dir=config.parent, sha256="a" * 64)


@pytest.fixture
def no_preparation(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("an existing prepared launch performed preprocessing or input acquisition")
    monkeypatch.setattr(capabilities, "require_for_command", lambda *a, **k: None)
    monkeypatch.setattr(capabilities, "require", lambda *a, **k: None)
    monkeypatch.setattr(go_cli, "_require_forecast_device", lambda: None)
    monkeypatch.setattr(runplan.RunObserver, "arm_first_products", lambda *a, **k: None)
    for name in ("_run_fetch", "_prepare_stage", "_clear_forecast_output"):
        monkeypatch.setattr(runplan, name, forbidden)
    for name in ("resolve_bridge", "geography_refusal", "memory_gate"):
        monkeypatch.setattr(go_cli, name, forbidden)


@pytest.mark.parametrize("source", ["gfs", "hrrr", "icon-eu"])
@pytest.mark.parametrize("restart", [False, True])
@pytest.mark.parametrize("door", ["go", "run-plan"])
def test_public_go_reuses_bundle_for_every_preparation_chain(
        tmp_path, monkeypatch, capsys, no_preparation, source, restart, door):
    from gpuwm import prepared_domain_tree_forecast as runner
    config = case(tmp_path, domains=2, source=source)
    before = config.read_bytes()
    prepared = _tree_bundle(tmp_path / "existing prepared", domains=2)
    old = tmp_path / "previous run"
    old.mkdir()
    checkpoint = old / "root.npz"
    checkpoint.write_bytes(b"test checkpoint operand; reader is exercised separately")
    output = tmp_path / "new run"
    calls = []

    def existing_runner(argv, *, observer):
        parsed = runner.build_parser().parse_args(argv)
        calls.append(parsed)
        assert parsed.experiment_config == config
        assert parsed.experiment_config_sha256 == hashlib.sha256(before).hexdigest()
        assert parsed.prepared_root == prepared
        assert parsed.preparation_receipt_sha256 == hashlib.sha256(
            (prepared / "proof.json").read_bytes()).hexdigest()
        assert parsed.restart == (checkpoint if restart else None)
        assert parsed.outdir == output / "chain" / "run"
        assert config.read_bytes() == before and checkpoint.is_file()
        evidence = parsed.outdir / "evidence"
        evidence.mkdir(parents=True)
        (evidence / "run-receipt.json").write_text(json.dumps({
            "status": "PASS", "restart_contract": {
                "restart_input": str(checkpoint) if restart else None}}))
        return 0

    monkeypatch.setattr(runner, "main", existing_runner)
    argv = ["go", str(config), "--prepared-root", str(prepared), "--outdir", str(output),
            "--run-stamp", "off", "--products", "none"]
    if restart:
        argv += ["--restart", str(checkpoint)]
    if door == "run-plan":
        options = {"prepared_root": str(prepared), "render_products": "none"}
        if restart:
            options["restart"] = str(checkpoint)
        document = tmp_path / "public-plan.json"
        document.write_text(json.dumps({
            "schema": runplan.PLAN_SCHEMA, "name": "public resume", "route": "prepared",
            "config": {"path": str(config)}, "output_root": str(output), "run_options": options}))
        argv = ["run-plan", str(document)]
    assert main(argv) == 0
    assert len(calls) == 1 and config.read_bytes() == before
    events = runplan.read_events(output / runplan.EVENTS_FILENAME)
    completed = next(row for row in events if row["event"] == "completed")
    assert completed["summary"]["restarted"] is restart
    assert completed["summary"]["restart_input"] == (str(checkpoint) if restart else None)
    assert "fetch ->" not in capsys.readouterr().out


def test_go_case_data_restart_reaches_existing_experiment_runtime(
        tmp_path, monkeypatch, no_preparation):
    config = case(tmp_path)
    checkpoint = tmp_path / "earlier.npz"
    checkpoint.write_bytes(b"checkpoint operand")
    called = []
    stub = _stub_run_experiment()
    monkeypatch.setattr(go_cli, "geography_refusal", lambda path: None)

    def execute(exp, data, outdir, **kwargs):
        called.append(kwargs["restart"])
        return stub(exp, data, outdir, **kwargs)

    monkeypatch.setattr(runtime, "run_experiment", execute)
    assert main(["go", str(config), "--restart", str(checkpoint), "--products", "none",
                 "--no-memory-gate", "--run-stamp", "off", "--outdir", str(tmp_path / "new")]) == 0
    assert called == [checkpoint]


def test_prepared_restart_without_locator_is_refused_before_any_work(tmp_path, capsys):
    config = case(tmp_path, source="gfs")
    # Use the prepared route, not the case_data route in this fixture.
    with pytest.raises(runplan.PlanError, match="prepared_root"):
        plan_for(config, tmp_path / "out", restart=tmp_path / "old.npz")
    text = config.read_text().split("[case_data]", 1)[0]
    config.write_text(text)
    assert main(["go", str(config), "--restart", "old.npz", "--outdir",
                 str(tmp_path / "out"), "--dry-run"]) == 2
    assert "prepared_root" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("location", ["same", "inside", "above", "prepared"])
def test_checkpoint_set_and_bundle_output_overlap_is_refused(tmp_path, location):
    config = case(tmp_path)
    old = tmp_path / "old"
    old.mkdir()
    checkpoint = old / "root.npz"
    checkpoint.write_bytes(b"original")
    prepared = _tree_bundle(tmp_path / "prepared")
    output = {"same": old, "inside": old / "new", "above": tmp_path,
              "prepared": prepared / "new"}[location]
    with pytest.raises(runplan.PlanError, match="overlaps"):
        plan_for(config, output, prepared_root=prepared, restart=checkpoint)
    assert checkpoint.read_bytes() == b"original"
    assert list(old.iterdir()) == [checkpoint]


def test_nonempty_prepared_output_is_not_superseded(tmp_path):
    config = case(tmp_path)
    prepared = _tree_bundle(tmp_path / "prepared")
    output = tmp_path / "out"
    output.mkdir()
    (output / "report.json").write_bytes(b"earlier run")
    with pytest.raises(runplan.PlanError, match="fresh output"):
        plan_for(config, output, prepared_root=prepared)
    assert (output / "report.json").read_bytes() == b"earlier run"


def test_existing_single_bundle_keeps_its_exact_wps_operand(
        tmp_path, monkeypatch, no_preparation):
    from gpuwm import prepared_single_domain_forecast as runner
    config = case(tmp_path)
    prepared = _single_domain_bundle(tmp_path / "prepared")
    wps = tmp_path / "original authority.wps"
    wps.write_bytes(b"exact WPS authority")
    calls = []

    def execute(argv, *, observer):
        calls.append(argv)
        assert argv[argv.index("--wps-namelist") + 1] == str(wps)
        assert argv[argv.index("--experiment-config") + 1] == str(config)
        return 0

    monkeypatch.setattr(runner, "main", execute)
    assert main(["go", str(config), "--prepared-root", str(prepared),
                 "--wps-namelist", str(wps), "--outdir", str(tmp_path / "new"),
                 "--run-stamp", "off", "--products", "none"]) == 0
    assert len(calls) == 1


def test_single_bundle_restart_reaches_existing_single_parser(tmp_path, capsys):
    from gpuwm import prepared_single_domain_forecast as runner
    config = case(tmp_path)
    prepared = _single_domain_bundle(tmp_path / "prepared")
    wps = tmp_path / "namelist.wps"
    wps.write_text("&share /\n")
    bundle = stage_cli.resolve_bundle(prepared)
    command = stage_cli.sim_command(bundle, experiment_config=config, wps_namelist=wps,
        outdir=tmp_path / "unused", restart=tmp_path / "old.npz")
    parsed = runner.build_parser().parse_args(command[3:])
    assert parsed.restart == tmp_path / "old.npz"
    assert parsed.experiment_config == config and parsed.wps_namelist == wps
    assert not (tmp_path / "unused").exists()


def test_prepared_dry_run_replays_real_parser_and_preserves_config(tmp_path, capsys):
    config = case(tmp_path, domains=2)
    before = config.read_bytes()
    prepared = _tree_bundle(tmp_path / "existing bundle", domains=2)
    checkpoint = tmp_path / "old" / "checkpoint.npz"
    argv = ["go", str(config), "--prepared-root", str(prepared), "--restart", str(checkpoint),
            "--outdir", str(tmp_path / "unused"), "--products", "none", "--dry-run"]
    assert main(argv) == 0
    out = capsys.readouterr().out
    replay = shlex.split(next(line[5:] for line in out.splitlines() if line.startswith("Run: ")))
    parsed = build_parser().parse_args(replay[1:])
    assert parsed.restart == checkpoint and parsed.prepared_root == prepared
    assert "verify prepared bundle -> restore" in out and "fetch ->" not in out
    assert not (tmp_path / "unused").exists() and config.read_bytes() == before

@pytest.mark.parametrize("damage", ["source", "child missing"])
def test_public_prepared_launch_retains_strict_child_source_and_identity_checks(
        tmp_path, monkeypatch, capsys, no_preparation, damage):
    from gpuwm import prepared_domain_tree_forecast as runner
    from test_prepared_domain_tree_forecast import _synthetic_prepared_tree
    prepared, receipt, config = _synthetic_prepared_tree(tmp_path, monkeypatch)
    child = prepared / "hierarchy-artifacts" / "domains" / "d02" / "prepared-cache" / "header.json"
    if damage == "source":
        header = json.loads(child.read_text())
        header["identity"]["source_identity"]["adapter"] = "different-provider"
        child.write_text(json.dumps(header))
    else:
        child.unlink()
    monkeypatch.setattr(runner, "run_prepared_tree", lambda *a, **k:
                        pytest.fail("invalid prepared input reached GPU execution"))
    assert main(["go", str(config), "--prepared-root", str(prepared),
                 "--outdir", str(tmp_path / "failed"), "--run-stamp", "off",
                 "--products", "none"]) != 0
    error = (tmp_path / "failed" / "launch.log").read_text()
    assert ("source identities differ" in error if damage == "source" else "header.json" in error)


@pytest.mark.parametrize("key", ["data_dir", "geog_root"])
def test_prepared_plan_retains_and_reports_unused_preparation_paths(tmp_path, key):
    config = case(tmp_path)
    prepared = _tree_bundle(tmp_path / "prepared")
    plan = plan_for(config, tmp_path / "new", prepared_root=prepared, **{key: tmp_path / "donor"})
    resolution, _, _ = runplan.resolve_plan(plan, require_inputs=False)
    assert plan.run_options[key] == str(tmp_path / "donor")
    assert any(key in warning["action"] and "retained but unused" in warning["action"]
               for warning in resolution["warnings"])
    assert next(row["value"] for row in resolution["automatic_resolutions"]
                if row["key"] == "unused_preparation_paths") == {key: str(tmp_path / "donor")}
    assert not (tmp_path / "new").exists() and not (tmp_path / "donor").exists()


def test_already_complete_tree_summary_uses_published_time_and_restore_receipt(tmp_path):
    evidence = tmp_path / "chain" / "run" / "evidence"
    evidence.mkdir(parents=True)
    (evidence / "progress.json").write_text(json.dumps({
        "status": "PASS", "model_elapsed_seconds": 288., "frame_count": 0}))
    checkpoint = str(tmp_path / "previous" / "root.npz")
    (evidence / "run-receipt.json").write_text(json.dumps({
        "status": "PASS", "restart_contract": {"restart_input": checkpoint}}))
    result = runplan._chain_summary(tmp_path / "chain")
    assert result["completed_seconds"] == 288.
    assert result["wrfout_count"] == 0 and result["status"] == "PASS"
    assert result["restarted"] and result["restart_input"] == checkpoint
    assert result["report"] == str(evidence / "run-receipt.json")


@pytest.mark.parametrize("authority", ["config", "wps"])
def test_prepared_output_cannot_overwrite_config_or_wps(tmp_path, authority):
    config = case(tmp_path)
    prepared = _tree_bundle(tmp_path / "prepared")
    wps = tmp_path / "exact.wps"
    wps.write_bytes(b"exact WPS input")
    output = config if authority == "config" else wps
    before = output.read_bytes()
    with pytest.raises(runplan.PlanError, match="overlaps"):
        plan_for(config, output, prepared_root=prepared, wps_namelist=wps)
    assert output.read_bytes() == before


@pytest.mark.parametrize("late_payload", [False, True])
def test_prepared_execution_rechecks_freshness_allowing_active_telemetry(
        tmp_path, monkeypatch, no_preparation, late_payload):
    from gpuwm import prepared_domain_tree_forecast as runner
    from gpuwm.supervisor import HEARTBEAT_NAME
    config = case(tmp_path, domains=2)
    prepared = _tree_bundle(tmp_path / "prepared", domains=2)
    output = tmp_path / "new"
    plan = plan_for(config, output, prepared_root=prepared)
    output.mkdir()
    (output / HEARTBEAT_NAME).write_text('{"stage":"starting"}')
    (output / "launch.log").write_text("this launch owns its log\n")
    if late_payload:
        (output / "wrfout_previous").write_bytes(b"existing forecast")
    calls = []
    monkeypatch.setattr(runner, "main", lambda argv, **kw: calls.append(argv) or 0)
    with runplan.EventStream(output / runplan.EVENTS_FILENAME, mirror=None) as events:
        if late_payload:
            with pytest.raises(runplan.PlanError, match="already contains"):
                runplan.execute_plan(plan, events=events)
            assert not (output / runplan.MANIFEST_FILENAME).exists()
            assert (output / "wrfout_previous").read_bytes() == b"existing forecast"
        else:
            assert runplan.execute_plan(plan, events=events) == 0
    assert len(calls) == (0 if late_payload else 1)


def test_go_prepared_hints_are_advisory_and_survive_into_plan(tmp_path, capsys):
    config = case(tmp_path, domains=2)
    prepared = _tree_bundle(tmp_path / "prepared", domains=2)
    data, geog = tmp_path / "absent data", tmp_path / "absent geography"
    assert main(["go", str(config), "--prepared-root", str(prepared),
                 "--data-dir", str(data), "--geog-root", str(geog),
                 "--outdir", str(tmp_path / "new"), "--dry-run"]) == 0
    output = capsys.readouterr()
    assert "data_dir, geog_root are retained but unused" in output.err
    replay = shlex.split(next(line[5:] for line in output.out.splitlines() if line.startswith("Run: ")))
    parsed = build_parser().parse_args(replay[1:])
    assert parsed.data_dir == data and parsed.geog_root == geog
    assert not data.exists() and not geog.exists() and not (tmp_path / "new").exists()


@pytest.mark.parametrize("health_debug", [False, True])
def test_public_plan_forwards_exact_health_diagnostic_operand(
        tmp_path, monkeypatch, no_preparation, health_debug):
    from gpuwm import prepared_domain_tree_forecast as runner
    config = case(tmp_path, domains=2)
    prepared = _tree_bundle(tmp_path / "prepared", domains=2)
    output = tmp_path / "new"
    document = tmp_path / "plan.json"
    document.write_text(json.dumps({
        "schema": runplan.PLAN_SCHEMA, "name": "health operand", "route": "prepared",
        "config": {"path": str(config)}, "output_root": str(output),
        "run_options": {"prepared_root": str(prepared), "render_products": "none",
                        "health_debug": health_debug}}))
    seen = []
    def execute(argv, **kwargs):
        seen.append(runner.build_parser().parse_args(argv).health_debug)
        return 0
    monkeypatch.setattr(runner, "main", execute)
    assert main(["run-plan", str(document)]) == 0
    assert seen == [health_debug]
    assert config.is_file()


def test_single_prepared_diagnostic_operand_reaches_real_parser(tmp_path):
    from gpuwm import prepared_single_domain_forecast as runner
    config = case(tmp_path)
    prepared = _single_domain_bundle(tmp_path / "prepared")
    wps = tmp_path / "namelist.wps"
    wps.write_text("&share /\n")
    bundle = stage_cli.resolve_bundle(prepared)
    command = stage_cli.sim_command(bundle, experiment_config=config, wps_namelist=wps,
                                  outdir=tmp_path / "new", health_debug=True)
    assert runner.build_parser().parse_args(command[3:]).health_debug
    assert not (tmp_path / "new").exists()


def test_chain_summary_counts_history_without_readiness_receipts(tmp_path):
    forecast = tmp_path / "chain" / "run"
    (forecast / "wrfout").mkdir(parents=True)
    (forecast / "ready").mkdir()
    for domain in (1, 2):
        name = f"wrfout_d{domain:02d}_2026-05-29_18_03_00"
        (forecast / "wrfout" / name).write_bytes(b"history file")
        (forecast / "ready" / (name + ".json")).write_text('{"status":"committed"}')
    (forecast / "wrfout_unused_directory").mkdir()
    assert runplan._chain_summary(tmp_path / "chain")["wrfout_count"] == 2
