"""``gpuwm go --cycle`` and run-plan ``run_options.cycle`` (A136 L2, DESIGN 3.1, 3.7).

A site schedule launches a named cycle, never ``latest``: launched a minute
early, ``latest`` is still the previous cycle and runs a stale forecast.  The
config's own cycle is moved by re-timing it (start time, delayed nests, the
namelists), the way a saved setup is started at a new cycle.
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import tomllib

import pytest

from gpuwm import cli, fetch, go_cli, runplan
from gpuwm import companion_setups as setups
from gpuwm.hrrr_prepared_bundle import render_wps_namelist
from gpuwm.namelist_import import parse_namelist_text
from gpuwm.static import rust_bridge
from gpuwm.toml_document import emit_experiment_toml

from test_companion_setups import era5_case, gfs_case

#: The retimed configuration is published through the same route-file
#: helpers the saved-setup start uses, which need the native static-fields
#: bridge (see tests/test_companion_setups.py).
needs_bridge = pytest.mark.skipif(
    rust_bridge.unavailable_reason() is not None,
    reason="the native static-fields bridge is not built, and a re-timed "
           f"configuration is published through it: {rust_bridge.unavailable_reason()}")

NEW_CYCLE = datetime(2013, 6, 1, 0)


def delayed_gfs_case(tmp_path):
    """The GFS fixture as a 6 h run with d03 starting 3 h into it (a
    delayed start sits on the 3 h forcing cadence)."""

    source, raw = gfs_case(tmp_path)
    raw["experiment"]["run_seconds"] = 21600.0
    raw["domain"][2]["start_time"] = datetime(2013, 5, 31, 21)
    source.write_text(emit_experiment_toml(raw), encoding="utf-8")
    from gpuwm import companion_domains as editor

    source.with_suffix(".namelist.wps").write_text(
        render_wps_namelist(editor._build(raw, source)), encoding="utf-8")
    return source, raw


@needs_bridge
def test_a_config_moves_to_a_cycle_with_its_delayed_nest_and_namelist(tmp_path):
    source, original = delayed_gfs_case(tmp_path)
    before = source.read_bytes()
    out = tmp_path / "go" / "cycles" / "2013-06-01T00" / source.name
    result = setups.retime_to_cycle(source, NEW_CYCLE, out)
    assert source.read_bytes() == before
    moved = tomllib.loads(out.read_text(encoding="utf-8"))
    assert moved["fetch"]["cycle"] == "2013-06-01T00"
    assert moved["experiment"]["start_time"] == NEW_CYCLE
    # The delayed nest keeps its 3 h offset into the run.
    assert moved["domain"][2]["start_time"] == datetime(2013, 6, 1, 3)
    assert "start_time" not in moved["domain"][1]
    for table in ("shared", "projection"):
        assert moved[table] == original[table]
    for key in ("run_seconds", "restart_interval_s"):
        assert moved["experiment"][key] == original["experiment"][key]
    assert moved["fetch"]["hours"] == original["fetch"]["hours"]
    assert {change["field"] for change in result["changes"]} == {
        "fetch.cycle", "fetch.out", "experiment.start_time", "domain[2].start_time"}
    wps = parse_namelist_text(out.with_suffix(".namelist.wps").read_text(encoding="utf-8"))
    # The namelist is rendered again, not copied: every domain's metgrid
    # window moved with the cycle (WPS takes each nest's analysis at the
    # root's start, as the original's did).
    assert wps["share"]["start_date"] == ["2013-06-01_00:00:00"] * 3
    assert wps["share"]["end_date"] == ["2013-06-01_06:00:00"] * 3
    # A launch again of the same config and cycle rewrites nothing.
    assert setups.retime_to_cycle(source, NEW_CYCLE, out)["written"] == []


@needs_bridge
def test_a_saved_setup_with_a_delayed_nest_starts_at_a_new_cycle(tmp_path):
    """The saved-setup start moves a delayed nest with the run, as the
    cycle re-timing does; left at the saved date it fell outside the new
    window and the start was refused."""

    source, _raw = delayed_gfs_case(tmp_path)
    library = tmp_path / "setups"
    setups.save_setup(config_path=source, library=library, name="Delayed")
    result = setups.start_setup(
        setup_path=library / "delayed" / "setup.toml", cycle="2013-06-01T00",
        hours=6.0, forecast_start_hour=0, name="Next",
        out=tmp_path / "forecasts" / "next.toml")
    written = tomllib.loads(Path(result["config_path"]).read_text(encoding="utf-8"))
    assert written["experiment"]["start_time"] == NEW_CYCLE
    assert written["domain"][2]["start_time"] == datetime(2013, 6, 1, 3)
    assert "domain[2].start_time" in {change["field"] for change in result["changes"]}


@needs_bridge
def test_a_config_with_declared_inputs_is_not_moved(tmp_path):
    source, _raw = era5_case(tmp_path)
    with pytest.raises(ValueError, match=r"\[case_data\].*cannot move"):
        setups.retime_to_cycle(source, NEW_CYCLE, tmp_path / "x" / source.name)
    assert not (tmp_path / "x").exists()


@needs_bridge
def test_a_date_time_the_cycle_cannot_move_is_refused(tmp_path):
    source, raw = gfs_case(tmp_path)
    text = source.read_text(encoding="utf-8").replace(
        "[fetch]", "[fetch]\nnot_a_run_time = 2013-05-31T18:00:00", 1)
    source.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="fetch.not_a_run_time"):
        setups.retime_to_cycle(source, NEW_CYCLE, tmp_path / "x" / source.name)


# ---------------------------------------------------------------------------
# gpuwm go --cycle
# ---------------------------------------------------------------------------

def gefs_config(tmp_path):
    config = tmp_path / "gefs.toml"
    config.write_text('[fetch]\nsource = "gefs"\ncycle = "2026-09-29T12"\n'
                      'hours = 12\ncadence = 3\n', encoding="utf-8")
    return config


def test_go_readiness_answers_the_cycle_the_flag_names(tmp_path, monkeypatch, capsys):
    config = gefs_config(tmp_path)
    asked = []

    def head(url, *args, **kwargs):
        asked.append(url)
        return "gefs.20260929/18/" in url

    monkeypatch.setattr(fetch, "_head_answer", head)
    args = cli.build_parser().parse_args(
        ["go", str(config), "--cycle", "2026-09-29T18", "--readiness"])
    assert go_cli.go_main(args) == 0
    document = json.loads(capsys.readouterr().out)
    assert document["cycle"] == "2026-09-29T18" and document["state"] == "ready"
    assert asked and all("/12/" not in url for url in asked)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["gefs.toml"]


def test_go_readiness_asks_for_no_forecast_runtime(tmp_path, monkeypatch, capsys):
    """A site asks readiness from a scheduler that may have no card or
    CuPy; the answer runs nothing, like --dry-run, so the front door's
    runtime preflight does not refuse it."""

    from gpuwm import capabilities

    monkeypatch.setattr(capabilities, "require_for_command", lambda command: pytest.fail(
        f"gpuwm {command} --readiness asked for the forecast runtime"))
    monkeypatch.setattr(fetch, "_head_answer",
                        lambda url, *a, **k: "gefs.20260929/18/" in url)
    assert cli.main(["go", str(gefs_config(tmp_path)), "--cycle", "2026-09-29T18",
                     "--readiness"]) == 0
    assert json.loads(capsys.readouterr().out)["state"] == "ready"


def test_go_cycle_is_refused_on_a_run_with_declared_inputs(tmp_path):
    config = tmp_path / "declared.toml"
    config.write_text('[fetch]\nsource = "gefs"\ncycle = "2026-09-29T12"\n'
                      'hours = 12\n[case_data]\nforcing = ["a.grib2"]\n',
                      encoding="utf-8")
    args = cli.build_parser().parse_args(
        ["go", str(config), "--cycle", "2026-09-29T18"])
    with pytest.raises(go_cli.GoRefusal, match="whose times a cycle cannot move"):
        go_cli.go_main(args)


def test_go_cycle_is_refused_when_it_is_not_a_cycle(tmp_path):
    args = cli.build_parser().parse_args(
        ["go", str(gefs_config(tmp_path)), "--cycle", "yesterday"])
    with pytest.raises(go_cli.GoRefusal, match="YYYY-MM-DDTHH"):
        go_cli.go_main(args)


@needs_bridge
def test_go_cycle_launches_the_retimed_config_in_the_originals_folder(
        tmp_path, monkeypatch):
    source, _raw = delayed_gfs_case(tmp_path)
    launched = {}

    def launch(args, *, observer=None):
        launched["config"] = Path(args.config)
        launched["outdir"] = args.outdir
        return 0

    monkeypatch.setattr(go_cli, "_go_prepared_main", launch)
    monkeypatch.setattr(go_cli, "_registered_launch",
                        lambda args, **kwargs: launch(args))
    args = cli.build_parser().parse_args(
        ["go", str(source), "--cycle", "2013-06-01T00"])
    assert go_cli.go_main(args) == 0
    case_root = source.parent / f"{source.stem}-go"
    assert Path(launched["outdir"]) == case_root
    assert launched["config"] == case_root / "cycles" / "2013-06-01T00" / source.name
    moved = tomllib.loads(launched["config"].read_text(encoding="utf-8"))
    assert moved["experiment"]["start_time"] == NEW_CYCLE
    # The config's own cycle launches the config itself.
    launched.clear()
    args = cli.build_parser().parse_args(
        ["go", str(source), "--cycle", "2013-05-31T18"])
    assert go_cli.go_main(args) == 0
    assert launched["config"] == source


@needs_bridge
def test_go_cycle_latest_is_resolved_under_the_posting_rule(tmp_path, monkeypatch):
    source, _raw = gfs_case(tmp_path)
    seen = {}

    def resolve(source_id, last_hour, **options):
        seen["options"] = options
        return NEW_CYCLE

    def launch(args, **kwargs):
        seen["config"] = Path(args.config)
        return 0

    monkeypatch.setattr(fetch, "resolve_latest_cycle", resolve)
    monkeypatch.setattr(go_cli, "_go_prepared_main", launch)
    monkeypatch.setattr(go_cli, "_registered_launch", launch)
    args = cli.build_parser().parse_args(["go", str(source), "--cycle", "latest"])
    assert go_cli.go_main(args) == 0
    # As posted by default: the newest cycle whose start needs are posted.
    assert seen["options"].get("as_posted") is True
    assert seen["config"].parent.name == "2013-06-01T00"
    args = cli.build_parser().parse_args(
        ["go", str(source), "--cycle", "latest", "--whole-cycle"])
    assert go_cli.go_main(args) == 0
    assert not seen["options"].get("as_posted")


# ---------------------------------------------------------------------------
# run-plan run_options.cycle
# ---------------------------------------------------------------------------

def plan_for(config, tmp_path, route="prepared", **options):
    return runplan.build_plan(
        {"schema": runplan.PLAN_SCHEMA, "name": "cycle", "route": route,
         "config": {"path": str(config)}, "output_root": str(tmp_path / "run"),
         "run_options": options},
        source="test", base_dir=tmp_path, sha256="0" * 64)


def test_run_options_cycle_is_checked_when_the_plan_is_built(tmp_path):
    config = gefs_config(tmp_path)
    assert plan_for(config, tmp_path, cycle="latest").run_options["cycle"] == "latest"
    assert plan_for(config, tmp_path, cycle="2026-09-29T18").run_options["cycle"] == "2026-09-29T18"
    with pytest.raises(runplan.PlanError, match="YYYY-MM-DDTHH"):
        plan_for(config, tmp_path, cycle="2026-09-29 18Z")
    # The config-driven route runs declared files, whose times a cycle
    # cannot move, so the option is not one it takes.
    with pytest.raises(runplan.PlanError, match="cycle"):
        plan_for(config, tmp_path, route="experiment", cycle="2026-09-29T18")


def test_run_plan_readiness_answers_run_options_cycle(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "_head_answer",
                        lambda url, *a, **k: "gefs.20260929/18/" in url)
    plan = plan_for(gefs_config(tmp_path), tmp_path, cycle="2026-09-29T18")
    document, code = runplan.plan_readiness(plan)
    assert (code, document["cycle"], document["state"]) == (0, "2026-09-29T18", "ready")


@needs_bridge
def test_run_options_cycle_retimes_a_config_path_plan_into_the_run(tmp_path):
    source, _raw = delayed_gfs_case(tmp_path)
    plan = plan_for(source, tmp_path, cycle="2013-06-01T00")
    destination = tmp_path / "run"
    moved_plan = runplan.plan_at_cycle(plan, destination)
    assert moved_plan.config_path == (
        destination / runplan.CYCLE_CONFIG_DIRNAME / source.name).resolve()
    moved = tomllib.loads(moved_plan.config_path.read_text(encoding="utf-8"))
    assert moved["fetch"]["cycle"] == "2013-06-01T00"
    assert moved["domain"][2]["start_time"] == datetime(2013, 6, 1, 3)
    assert moved_plan.automatic_resolutions[-1]["basis"] == "run_options.cycle"
    # The config's own cycle leaves the plan on the config.
    same = runplan.plan_at_cycle(plan_for(source, tmp_path, cycle="2013-05-31T18"),
                                 tmp_path / "other")
    assert same.config_path == source.resolve()
    assert not (tmp_path / "other").exists()


def test_run_options_cycle_is_refused_on_an_inline_config(tmp_path):
    plan = runplan.build_plan(
        {"schema": runplan.PLAN_SCHEMA, "name": "inline", "route": "prepared",
         "config": {"inline": gefs_config(tmp_path).read_text(encoding="utf-8")},
         "output_root": str(tmp_path / "run"),
         "run_options": {"cycle": "2026-09-29T18"}},
        source="test", base_dir=tmp_path, sha256="0" * 64)
    with pytest.raises(runplan.PlanError, match="inline configuration has no"):
        runplan.plan_at_cycle(plan, tmp_path / "run")
