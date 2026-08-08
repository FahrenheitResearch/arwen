"""The machine-facing front door: plan in, typed events out.

Every test here is CPU-only.  The one thing this module cannot exercise
without a card is the integration itself, so the GPU half is replaced at
the seam the front door actually uses -- ``runtime.run_experiment``,
driving the REAL :class:`gpuwm.runplan.RunObserver` through the REAL
progress protocol.  What that buys is that the observer's contract is
under test rather than mocked away: a stub that called ``preparing`` with
a phase the mapping table does not know, or emitted progress before the
forecast stage opened, would fail here the same way the live pipeline
would.

The config is ``tests/test_case_data.py``'s fixture pair, reused rather
than copied: it is the smallest TOML that loads through the same
``load_experiment_case_bytes`` seam ``gpuwm run`` uses, it names no case,
and reusing it means a schema change breaks one fixture instead of two.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from gpuwm import explain
from gpuwm.runplan import (EVENT_SCHEMA, EVENT_TAGS, EVENTS_FILENAME,
                           MANIFEST_FILENAME, MANIFEST_SCHEMA, PLAN_SCHEMA,
                           STAGES, EventStream, PlanError, RunObserver,
                           build_plan, execute_plan, load_plan,
                           probe_environment, read_events, resolve_plan,
                           run_plan_main)
from test_case_data import make_case_toml


def _plan_document(config_path: Path, run_dir: Path, **overrides):
    document = {
        "schema": PLAN_SCHEMA,
        "name": "front-door-fixture",
        "route": "experiment",
        "config": {"path": str(config_path)},
        "output_root": str(run_dir),
    }
    document.update(overrides)
    return document


def _write_plan(tmp_path, config_path, run_dir, **overrides) -> Path:
    path = tmp_path / "plan.json"
    path.write_text(
        json.dumps(_plan_document(config_path, run_dir, **overrides)),
        encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# The plan document
# ---------------------------------------------------------------------------


def test_a_plan_round_trips_through_the_loader_with_paths_made_absolute(
        tmp_path):
    config = make_case_toml(tmp_path)
    run_dir = tmp_path / "run"
    plan = load_plan(_write_plan(tmp_path, config, run_dir))

    assert plan.name == "front-door-fixture"
    assert plan.route == "experiment"
    assert plan.config_path == config
    assert plan.run_dir == run_dir
    assert len(plan.sha256) == 64
    # Every run option the route declares is resolved, present or not.
    assert set(plan.run_options) == {
        "device", "dry_run", "restart", "health_debug"}
    assert plan.run_options["dry_run"] is False


def test_relative_paths_resolve_against_the_plans_own_directory(tmp_path):
    nested = tmp_path / "plans"
    nested.mkdir()
    make_case_toml(tmp_path)
    (tmp_path / "case.toml").rename(nested / "case.toml")
    path = nested / "plan.json"
    path.write_text(json.dumps({
        "schema": PLAN_SCHEMA, "name": "relative", "route": "experiment",
        "config": {"path": "case.toml"}, "output_root": "runs/one",
    }), encoding="utf-8")

    plan = load_plan(path)
    assert plan.config_path == nested / "case.toml"
    assert plan.run_dir == (nested / "runs" / "one").resolve()


def test_an_unknown_top_level_key_is_refused_and_names_the_known_ones(
        tmp_path):
    with pytest.raises(PlanError) as refusal:
        build_plan({"schema": PLAN_SCHEMA, "name": "x", "route": "experiment",
                    "config": {"inline": "x = 1"}, "outputroot": "typo"},
                   source="probe.json", base_dir=tmp_path, sha256="0" * 64)
    text = str(refusal.value)
    assert "'outputroot'" in text
    assert "output_root" in text
    # The consequence, not just the fact -- the repo's refusal voice.
    assert "no key is ignored" in text


def test_an_unknown_schema_id_is_refused_naming_both_ids(tmp_path):
    with pytest.raises(PlanError) as refusal:
        build_plan({"schema": "gpuwm.run-plan.v2", "name": "x",
                    "route": "experiment", "config": {"inline": "x = 1"}},
                   source="probe.json", base_dir=tmp_path, sha256="0" * 64)
    text = str(refusal.value)
    assert PLAN_SCHEMA in text and "gpuwm.run-plan.v2" in text


def test_an_unknown_route_is_refused_and_lists_the_routes_this_build_has(
        tmp_path):
    with pytest.raises(PlanError) as refusal:
        build_plan({"schema": PLAN_SCHEMA, "name": "x", "route": "teleport",
                    "config": {"inline": "x = 1"}},
                   source="probe.json", base_dir=tmp_path, sha256="0" * 64)
    assert "experiment" in str(refusal.value)


def test_config_must_carry_exactly_one_of_the_three_spellings(tmp_path):
    for config, expected in (
            ({}, "[]"),
            ({"path": "a.toml", "inline": "x = 1"}, "['inline', 'path']"),
            ({"inline": "x = 1", "intent": {}}, "['inline', 'intent']")):
        with pytest.raises(PlanError) as refusal:
            build_plan({"schema": PLAN_SCHEMA, "name": "x",
                        "route": "experiment", "config": config},
                       source="probe.json", base_dir=tmp_path,
                       sha256="0" * 64)
        text = str(refusal.value)
        # The refusal names exactly which spellings it found, so the
        # reader does not have to work out which two collided.
        assert expected in text
        assert "'path'" in text and "'inline'" in text and "'intent'" in text


def test_a_run_option_the_route_does_not_declare_is_refused(tmp_path):
    with pytest.raises(PlanError) as refusal:
        build_plan({"schema": PLAN_SCHEMA, "name": "x", "route": "experiment",
                    "config": {"inline": "x = 1"},
                    "run_options": {"overclock": True}},
                   source="probe.json", base_dir=tmp_path, sha256="0" * 64)
    assert "overclock" in str(refusal.value)


def test_a_fetch_argument_list_is_checked_against_gpuwm_fetchs_own_parser(
        tmp_path):
    with pytest.raises(PlanError) as refusal:
        build_plan({"schema": PLAN_SCHEMA, "name": "x", "route": "experiment",
                    "config": {"inline": "x = 1"},
                    "fetch": {"args": ["--not-a-fetch-flag"]}},
                   source="probe.json", base_dir=tmp_path, sha256="0" * 64)
    assert "fetch.args" in str(refusal.value)


def test_defaulted_plan_keys_are_reported_never_applied_silently(tmp_path):
    plan = build_plan(
        {"schema": PLAN_SCHEMA, "name": "x", "route": "experiment",
         "config": {"inline": "x = 1"}},
        source="probe.json", base_dir=tmp_path, sha256="0" * 64)
    keys = {(entry["scope"], entry["key"])
            for entry in plan.automatic_resolutions}
    assert ("plan", "output_root") in keys
    assert ("run_options", "dry_run") in keys
    assert ("run_options", "device") in keys


# ---------------------------------------------------------------------------
# Resolution through the real config seam
# ---------------------------------------------------------------------------


def test_the_envelope_resolves_through_the_real_config_loader(tmp_path):
    config = make_case_toml(tmp_path)
    plan = load_plan(_write_plan(tmp_path, config, tmp_path / "run"))

    resolution, exp, data = resolve_plan(plan)

    # The snapshot is the objects the model will run, not a re-read.
    assert resolution["configuration"]["experiment"]["name"] == exp.name
    assert resolution["configuration"]["case_data"]["output_title"] == (
        data.output_title)
    assert resolution["plan"]["config_sha256"] == (
        resolution["plan"]["config_sha256"])
    # It is JSON, all the way down -- no consumer needs a Python type.
    json.dumps(resolution)


def test_an_inline_config_takes_the_same_route_as_a_config_on_disk(tmp_path):
    config = make_case_toml(tmp_path)
    text = config.read_text(encoding="utf-8")
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({
        "schema": PLAN_SCHEMA, "name": "inline", "route": "experiment",
        "config": {"inline": text}, "output_root": str(tmp_path / "run"),
    }), encoding="utf-8")

    inline_resolution, inline_exp, _ = resolve_plan(load_plan(path))
    file_resolution, file_exp, _ = resolve_plan(
        load_plan(_write_plan(tmp_path, config, tmp_path / "run")))

    assert inline_exp == file_exp
    assert (inline_resolution["configuration"]
            == file_resolution["configuration"])


def test_the_derived_timestep_of_every_domain_is_reported_out_loud(tmp_path):
    config = make_case_toml(tmp_path)
    resolution, exp, _ = resolve_plan(
        load_plan(_write_plan(tmp_path, config, tmp_path / "run")))

    steps = [entry for entry in resolution["automatic_resolutions"]
             if entry["key"] == "dt"]
    assert len(steps) == len(exp.domains)
    root = steps[0]
    assert root["basis"] == "declared_time_step"
    assert root["value"] == pytest.approx(float(exp.domains[0].run.dt))
    # The exact rational is carried beside the binary64 image, because
    # the two are genuinely different numbers.
    assert root["exact"]


def test_a_schema_default_the_config_did_not_spell_is_reported(tmp_path):
    config = make_case_toml(tmp_path)
    resolution, _exp, _ = resolve_plan(
        load_plan(_write_plan(tmp_path, config, tmp_path / "run")))
    defaults = {entry["key"] for entry in resolution["automatic_resolutions"]
                if entry.get("basis") == "schema_default"
                and entry["scope"] == "experiment"}
    # The fixture spells none of these; all three change the model.
    assert {"feedback", "blend_width", "spec_bdy_width"} <= defaults


def test_a_library_warning_reaches_the_stream_as_fields(tmp_path):
    captured: list[dict[str, str]] = []
    from gpuwm.runplan import collect_warnings

    with collect_warnings(captured):
        explain.warn("something worth saying", "and why it is worth it")
    assert captured == [{"action": "something worth saying",
                         "why": "and why it is worth it"}]

    # Detached again: the sink stops receiving, and the observer list is
    # not left holding a reference to a dead test's list.
    explain.warn("after", "detach")
    assert len(captured) == 1


def test_a_warning_observer_that_raises_cannot_fail_the_run():
    def hostile(record):
        raise RuntimeError("observer exploded")

    explain.add_warning_observer(hostile)
    try:
        explain.warn("the run continues")  # must not raise
    finally:
        explain.remove_warning_observer(hostile)


# ---------------------------------------------------------------------------
# The intent route
# ---------------------------------------------------------------------------


_INTENT = {"point": "39,-98", "source": "era5", "cycle": "2024-05-03T12",
           "hours": 1, "vram_gib": 24}


def _intent_plan(tmp_path, run_dir, **intent):
    document = {
        "schema": PLAN_SCHEMA, "name": "intent-fixture",
        "route": "experiment", "config": {"intent": {**_INTENT, **intent}},
        "output_root": str(run_dir),
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_an_intent_block_becomes_the_wizards_own_argv(tmp_path):
    from gpuwm.runplan import intent_arguments

    arguments = intent_arguments(_INTENT, out=tmp_path / "c.toml")
    assert "--point" in arguments and "39,-98" in arguments
    assert "--cycle" in arguments and "2024-05-03T12" in arguments
    assert "--vram-gib" in arguments and "24" in arguments
    # run-plan owns where the config lands; a plan cannot redirect it.
    assert arguments[-2:] == ["--out", str(tmp_path / "c.toml")]


def test_an_intent_key_the_wizard_has_no_flag_for_is_refused(tmp_path):
    with pytest.raises(PlanError) as refusal:
        build_plan({"schema": PLAN_SCHEMA, "name": "x", "route": "experiment",
                    "config": {"intent": {**_INTENT, "nx": 400}}},
                   source="probe.json", base_dir=tmp_path, sha256="0" * 64)
    text = str(refusal.value)
    assert "'nx'" in text
    # And it names the keys that DO exist, so a front end can correct.
    assert "point" in text and "ladder" in text


def test_an_intent_without_a_place_is_refused(tmp_path):
    with pytest.raises(PlanError) as refusal:
        build_plan({"schema": PLAN_SCHEMA, "name": "x", "route": "experiment",
                    "config": {"intent": {"cycle": "latest"}}},
                   source="probe.json", base_dir=tmp_path, sha256="0" * 64)
    assert "no default place" in str(refusal.value)


def test_an_intent_source_the_route_cannot_run_is_refused_with_the_reason(
        tmp_path):
    """gfs/hrrr emissions carry no [case_data]; say so, not 'missing table'."""

    with pytest.raises(PlanError) as refusal:
        build_plan({"schema": PLAN_SCHEMA, "name": "x", "route": "experiment",
                    "config": {"intent": {**_INTENT, "source": "gfs"}}},
                   source="probe.json", base_dir=tmp_path, sha256="0" * 64)
    text = str(refusal.value)
    assert "case_data" in text
    assert "era5" in text
    assert "domain_wizard.py" in text  # the engine's own citation


def test_the_wizard_writes_the_config_and_resolution_carries_it_verbatim(
        tmp_path):
    run_dir = tmp_path / "run"
    plan = load_plan(_intent_plan(tmp_path, run_dir))
    resolution, exp, data = resolve_plan(plan, require_inputs=False)

    assert resolution["plan"]["config_kind"] == "intent"
    # The generated text is carried whole: the caller never typed it.
    assert "Emitted by `gpuwm domain`" in resolution["generated_config"]
    assert "[case_data]" in resolution["generated_config"]
    assert exp.name and data.output_title
    # A query-mode resolution generates into a throwaway directory.
    assert not run_dir.exists()


def test_the_generation_itself_is_an_automatic_resolution(tmp_path):
    plan = load_plan(_intent_plan(tmp_path, tmp_path / "run"))
    resolution, _exp, _data = resolve_plan(plan, require_inputs=False)
    entries = {(e["scope"], e["key"]) for e
               in resolution["automatic_resolutions"]}
    assert ("config", "generated_by") in entries
    assert ("config", "generated_config") in entries
    # Domain size is FITTED, never typed -- said out loud, per domain.
    fitted = [e for e in resolution["automatic_resolutions"]
              if e["key"] == "nx_ny"]
    assert fitted and fitted[0]["basis"] == "fitted_to_vram_budget"


def test_resolve_and_estimate_both_work_on_an_intent_plan(tmp_path, capsys):
    """Studio's live estimate strip runs on intent, before any config."""

    from gpuwm.cli import build_parser

    plan_path = _intent_plan(tmp_path, tmp_path / "run")
    assert run_plan_main(build_parser().parse_args(
        ["run-plan", "--resolve", str(plan_path)])) == 0
    resolved = json.loads(capsys.readouterr().out)
    assert resolved["generated_config"]

    assert run_plan_main(build_parser().parse_args(
        ["run-plan", "--estimate", str(plan_path)])) == 0
    estimate = json.loads(capsys.readouterr().out)
    assert estimate["vram"]["estimate_bytes"] > 0
    assert estimate["disk"]["total_frames"] > 0


def test_resolve_reports_the_engines_own_minimum_domain_size(tmp_path):
    plan = load_plan(_intent_plan(tmp_path, tmp_path / "run"))
    resolution, _exp, _data = resolve_plan(plan, require_inputs=False)
    floor = resolution["domain_size_floor"]
    assert floor["root_mass_points"] == {"nx": 60, "ny": 48}
    assert floor["nest_span_mass_points"] == 12
    assert floor["clearance_rows"] == 10
    assert "FITTED" in floor["basis"]


def test_the_floor_is_derived_from_the_wizard_not_transcribed(monkeypatch):
    """Move the wizard's bracket; the reported floor must move with it."""

    from gpuwm import domain_wizard
    from gpuwm.runplan import domain_size_floor

    monkeypatch.setattr(domain_wizard, "_MIN_SCALE", 1.0)
    assert domain_size_floor()["root_mass_points"] == {"nx": 110, "ny": 88}


def test_a_shape_that_cannot_fit_refuses_with_the_engines_words_and_the_floor(
        tmp_path):
    """The fit refusal a front end most needs the numbers from."""

    plan = load_plan(_intent_plan(
        tmp_path, tmp_path / "run", ladder="12-3-1-0.5", vram_gib=4))
    with pytest.raises(PlanError) as refusal:
        resolve_plan(plan, require_inputs=False)
    text = str(refusal.value)
    assert "does not fit" in text
    assert "minimum layout" in text        # the wizard's own sentence
    assert "root_mass_points" in text      # the structured floor beside it


# ---------------------------------------------------------------------------
# The prepared route (the credential-free golden path)
# ---------------------------------------------------------------------------


_GFS_INTENT = {"point": "35.2,-97.4", "source": "gfs", "root_dx_km": 3,
               "cycle": "2024-05-03T12", "hours": 6, "vram_gib": 24}


def _prepared_plan(tmp_path, run_dir, **intent):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({
        "schema": PLAN_SCHEMA, "name": "golden-path", "route": "prepared",
        "config": {"intent": {**_GFS_INTENT, **intent}},
        "output_root": str(run_dir),
    }), encoding="utf-8")
    return path


def test_a_gfs_intent_is_accepted_on_the_prepared_route(tmp_path):
    """The refusal on the experiment route is not a global one."""

    plan = load_plan(_prepared_plan(tmp_path, tmp_path / "run"))
    assert plan.route == "prepared"
    assert plan.config_intent["source"] == "gfs"


def test_the_prepared_route_resolves_a_config_with_no_case_data(tmp_path):
    """GFS emissions carry no [case_data]; the route says so, not fails."""

    plan = load_plan(_prepared_plan(tmp_path, tmp_path / "run"))
    resolution, exp, data = resolve_plan(plan, require_inputs=False)

    assert data is None
    assert resolution["configuration"]["case_data"] is None
    assert resolution["configuration"]["experiment"]["name"]
    assert resolution["declared_inputs"] == []
    # And the generated config really is the gfs one.
    assert "[fetch]" in resolution["generated_config"]
    assert 'source = "gfs"' in resolution["generated_config"]


def test_estimate_works_on_a_prepared_route_plan(tmp_path, capsys):
    from gpuwm.cli import build_parser

    plan_path = _prepared_plan(tmp_path, tmp_path / "run")
    assert run_plan_main(build_parser().parse_args(
        ["run-plan", "--estimate", str(plan_path)])) == 0
    document = json.loads(capsys.readouterr().out)
    assert document["vram"]["estimate_bytes"] > 0
    assert document["disk"]["total_frames"] > 0


def test_the_prepared_route_declares_the_data_dir_option(tmp_path):
    """It downloads its own inputs, so it takes where they land."""

    from gpuwm.runplan import ROUTES

    assert "data_dir" in ROUTES["prepared"].run_options
    # The experiment route declares its inputs in [case_data] instead.
    assert "data_dir" not in ROUTES["experiment"].run_options
    plan = load_plan(_prepared_plan(tmp_path, tmp_path / "run"))
    assert plan.run_options["data_dir"] is None


def test_go_stage_labels_all_map_onto_a_run_plan_stage():
    """A label go can emit that this front door cannot place is a hole."""

    from gpuwm.runplan import STAGES, _GO_STAGES

    assert set(_GO_STAGES.values()) <= set(STAGES)
    # Every label go actually uses is covered.
    assert set(_GO_STAGES) == {"authority", "fetch", "manifest", "prepare",
                               "forecast", "render"}


def test_the_chain_observer_renders_go_stages_as_run_plan_stages(tmp_path):
    """Drive the observer with go's own hook vocabulary."""

    from gpuwm.runplan import _GoObserver

    with EventStream(tmp_path / EVENTS_FILENAME, mirror=None) as events:
        observer = RunObserver(events)
        chain = _GoObserver(observer)
        chain.stage_begin(label="authority", command=["x"])
        chain.stage_end(label="authority", exit_code=0, ok=True,
                        elapsed_seconds=1.0, progress=None)
        chain.stage_begin(label="fetch", command=["x"])
        chain.stage_begin(label="forecast", command=["x"])
        chain.stage_heartbeat(
            label="forecast", elapsed_seconds=40.0,
            progress={"status": "RUNNING", "model_elapsed_seconds": 1200.0})
        observer.finish_stage()

    records = read_events(tmp_path / EVENTS_FILENAME)
    stages = [r["stage"] for r in records if r["event"] == "stage_started"]
    # prepare opens, closes for the download, and opens again: go's real
    # order, not a flattened one.
    assert stages == ["prepare", "fetch", "forecast"]
    progress = next(r for r in records if r["event"] == "model_progress")
    assert progress["model_seconds"] == 1200.0
    assert progress["speed_x"] == 30.0
    assert progress["step_ms"] is None
    # A polled sample says so, so a consumer never mistakes it for a
    # per-step one.
    assert progress["source"] == "stage_progress_file"


def test_a_failed_chain_stage_is_named_in_a_warning(tmp_path):
    from gpuwm.runplan import _GoObserver

    with EventStream(tmp_path / EVENTS_FILENAME, mirror=None) as events:
        chain = _GoObserver(RunObserver(events))
        chain.stage_begin(label="prepare", command=["x"])
        chain.stage_end(label="prepare", exit_code=2, ok=False,
                        elapsed_seconds=3.0, progress=None)
    warning = next(r for r in read_events(tmp_path / EVENTS_FILENAME)
                   if r["event"] == "warning")
    assert warning["code"] == "chain_stage_failed"
    assert warning["stage"] == "prepare"
    assert warning["exit_code"] == 2


def test_go_runs_the_forecast_in_process_only_for_an_observer(monkeypatch,
                                                              tmp_path):
    """The subprocess default is what keeps a CUDA failure in one stage."""

    import gpuwm.go_cli as go_cli

    plan = {"runner": "gpuwm.prepared_single_domain_forecast",
            "source": "gfs", "prepared": tmp_path / "prep",
            "authority": tmp_path / "auth", "run": tmp_path / "run",
            "profile": None}
    digests = {"proof": "a", "source_manifest": "b",
               "prepared_content": "c"}
    spawned = []
    monkeypatch.setattr(go_cli, "_run_stage",
                        lambda label, command, **kw: spawned.append(label))

    go_cli._run_forecast(plan, digests, explain=False, observer=None)
    assert spawned == ["forecast"]

    # With an observer it imports the runner and calls main(argv,
    # observer=...) instead -- same command, no second argument set.
    seen = {}

    class Runner:
        @staticmethod
        def main(argv, *, observer):
            seen["argv"] = argv
            seen["observer"] = observer
            return 0

    monkeypatch.setattr("importlib.import_module", lambda name: Runner)
    sentinel = object()
    go_cli._run_forecast(plan, digests, explain=False, observer=sentinel)
    assert spawned == ["forecast"]          # nothing more was spawned
    assert seen["observer"] is sentinel
    assert seen["argv"][:2] == ["--source", "gfs"]
    assert "--proof-sha256" in seen["argv"]


def test_go_notifications_are_off_without_an_observer():
    """Every existing gpuwm go caller must be unaffected."""

    from gpuwm.go_cli import _notify

    # No observer: nothing happens, and nothing raises.
    _notify(None, "stage_begin", label="x")

    class Hostile:
        def stage_begin(self, **_):
            raise RuntimeError("observer exploded")

    # A hostile observer cannot take a stage down either.
    _notify(Hostile(), "stage_begin", label="x")


@pytest.mark.parametrize("runner", ["gpuwm.prepared_single_domain_forecast",
                                    "gpuwm.prepared_domain_tree_forecast"])
def test_both_runner_mains_accept_an_observer(runner):
    import importlib
    import inspect

    module = importlib.import_module(runner)
    parameter = inspect.signature(module.main).parameters["observer"]
    assert parameter.default is None
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY


def test_a_passing_prepared_run_ends_with_completed_not_failed(
        tmp_path, monkeypatch):
    """The severity-one regression: every good run announced failure.

    The route reported ``completed_seconds: None``; that reached
    ``heartbeat.complete``, whose ``float()`` raised INSIDE the arm that
    emits ``failed`` -- after the chain had already printed its validity
    PASS.  A consumer that trusts the contract marked every good run
    failed.
    """

    import gpuwm.go_cli as go_cli

    chain = tmp_path / "run" / "chain"
    forecast = chain / "run"

    def fake_go_main(args, *, observer=None):
        forecast.mkdir(parents=True, exist_ok=True)
        # The real filename function: WRF spells the valid time with
        # colons, which Windows will not accept in a path, so the
        # product substitutes underscores and a fixture must not
        # invent its own spelling.
        from datetime import datetime as _dt

        from gpuwm.io.wrfout import wrfout_filename
        (forecast / wrfout_filename(_dt(2024, 5, 3, 12), 1)).write_bytes(
            b"f")
        (forecast / "progress.json").write_text(json.dumps({
            "schema": "gpuwm-prepared-single-domain-progress-v1",
            "status": "PASS", "model_elapsed_seconds": 21600.0,
            "frame_count": 13}), encoding="utf-8")
        (forecast / "report.json").write_text(
            json.dumps({"status": "PASS"}), encoding="utf-8")
        return 0

    monkeypatch.setattr(go_cli, "go_main", fake_go_main)
    plan = load_plan(_prepared_plan(tmp_path, tmp_path / "run"))
    run_dir = plan.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    with EventStream(run_dir / EVENTS_FILENAME, mirror=None) as events:
        code = execute_plan(plan, events=events)

    events = read_events(run_dir / EVENTS_FILENAME)
    assert code == 0, [r for r in events if r["event"] == "failed"]
    assert events[-1]["event"] == "completed"
    summary = events[-1]["summary"]
    assert summary["completed_seconds"] == 21600.0
    assert summary["status"] == "PASS"
    assert summary["nan_free"] is True
    assert summary["wrfout_count"] == 1


def test_a_summary_with_no_completion_number_cannot_crash_the_boundary():
    """Defence in depth for the same class of bug on a future route."""

    from gpuwm.runplan import _finite_seconds

    assert _finite_seconds(None) == 0.0
    assert _finite_seconds("21600") == 0.0
    assert _finite_seconds(True) == 0.0
    assert _finite_seconds(float("nan")) == 0.0
    assert _finite_seconds(float("inf")) == 0.0
    assert _finite_seconds(-5.0) == 0.0
    assert _finite_seconds(21600) == 21600.0


def test_every_intent_key_declares_how_it_reaches_the_chain():
    """The audit, made permanent.

    geog_root, data_dir, forcing and vtable were all accepted at intent
    level and then dropped on the prepared route -- the wizard writes
    the last three into [case_data], which it does not emit for gfs.  A
    plan naming a non-default geography tree ran against the default
    one, silently.
    """

    from gpuwm.runplan import _INTENT_DELIVERY, _INTENT_FLAGS

    assert set(_INTENT_FLAGS) == set(_INTENT_DELIVERY)
    for key, delivery in _INTENT_DELIVERY.items():
        assert delivery in ("config", "case_data") or \
            delivery.startswith("go:"), (key, delivery)


def test_go_delivered_intent_keys_are_forwarded_to_the_chain(
        tmp_path, monkeypatch):
    import gpuwm.go_cli as go_cli

    seen = {}

    def fake_go_main(args, *, observer=None):
        seen["geog_root"] = getattr(args, "geog_root", None)
        seen["data_dir"] = getattr(args, "data_dir", None)
        return 1          # stop before the summary; forwarding is the point

    monkeypatch.setattr(go_cli, "go_main", fake_go_main)
    geog = tmp_path / "MY_GEOG"
    geog.mkdir()
    data = tmp_path / "MY_DATA"
    plan = load_plan(_prepared_plan(tmp_path, tmp_path / "run",
                                    geog_root=str(geog),
                                    data_dir=str(data)))
    run_dir = plan.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    with EventStream(run_dir / EVENTS_FILENAME, mirror=None) as events:
        execute_plan(plan, events=events)

    assert Path(seen["geog_root"]) == geog
    assert Path(seen["data_dir"]) == data


def test_a_run_option_beats_the_intents_copy_of_the_same_key(
        tmp_path, monkeypatch):
    import gpuwm.go_cli as go_cli

    seen = {}
    monkeypatch.setattr(go_cli, "go_main",
                        lambda args, *, observer=None: seen.update(
                            geog_root=getattr(args, "geog_root", None)) or 1)
    intent_geog = tmp_path / "FROM_INTENT"
    intent_geog.mkdir()
    option_geog = tmp_path / "FROM_OPTION"
    option_geog.mkdir()
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({
        "schema": PLAN_SCHEMA, "name": "p", "route": "prepared",
        "config": {"intent": {**_GFS_INTENT,
                              "geog_root": str(intent_geog)}},
        "run_options": {"geog_root": str(option_geog)},
        "output_root": str(tmp_path / "run"),
    }), encoding="utf-8")
    plan = load_plan(path)
    plan.run_dir.mkdir(parents=True, exist_ok=True)
    with EventStream(plan.run_dir / EVENTS_FILENAME, mirror=None) as events:
        execute_plan(plan, events=events)
    assert Path(seen["geog_root"]) == option_geog


def test_an_intent_key_the_prepared_route_cannot_deliver_is_refused(
        tmp_path):
    """Accepted-then-dropped is the failure mode; refuse instead."""

    for key, value in (("forcing", ["a.grib"]), ("vtable", "V.tbl")):
        with pytest.raises(PlanError) as refusal:
            build_plan({"schema": PLAN_SCHEMA, "name": "x",
                        "route": "prepared",
                        "config": {"intent": {**_GFS_INTENT, key: value}}},
                       source="probe.json", base_dir=tmp_path,
                       sha256="0" * 64)
        text = str(refusal.value)
        assert key in text
        assert "silently dropped" in text


# ---------------------------------------------------------------------------
# HRRR on the prepared route
# ---------------------------------------------------------------------------


_HRRR_INTENT = {"point": "35.2,-97.4", "source": "hrrr", "root_dx_km": 3,
                "cycle": "2024-05-03T12", "hours": 6, "vram_gib": 24}


def _hrrr_plan(tmp_path, run_dir, **overrides):
    intent = {**_HRRR_INTENT, **overrides.pop("intent", {})}
    document = {
        "schema": PLAN_SCHEMA, "name": "hrrr-plan", "route": "prepared",
        "config": {"intent": intent}, "output_root": str(run_dir),
        **overrides,
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_an_hrrr_intent_resolves_through_the_wizard(tmp_path):
    plan = load_plan(_hrrr_plan(tmp_path, tmp_path / "run"))
    resolution, exp, data = resolve_plan(plan, require_inputs=False)

    assert data is None
    assert len(exp.domains) == 1
    assert exp.domains[0].run.dx == 3000.0
    assert 'source = "hrrr"' in resolution["generated_config"]


def test_an_hrrr_intent_estimates(tmp_path, capsys):
    from gpuwm.cli import build_parser

    plan_path = _hrrr_plan(tmp_path, tmp_path / "run")
    assert run_plan_main(build_parser().parse_args(
        ["run-plan", "--estimate", str(plan_path)])) == 0
    document = json.loads(capsys.readouterr().out)
    assert document["vram"]["estimate_bytes"] > 0
    assert document["disk"]["total_frames"] > 0


def test_the_wizard_writes_every_file_the_hrrr_route_reads(tmp_path):
    """Four files beside the config; the HRRR tools read them, not the TOML."""

    from gpuwm.hrrr_route_inputs import route_input_paths
    from gpuwm.runplan import generate_intent_config

    plan = load_plan(_hrrr_plan(tmp_path, tmp_path / "run"))
    generated, _ = generate_intent_config(plan, destination=tmp_path / "gen")
    for role, path in route_input_paths(generated).items():
        assert path.is_file(), role


def test_the_prepared_route_refuses_a_multi_domain_hrrr_plan(tmp_path):
    """Naming the limitation, and the chain that does run it."""

    from gpuwm.runplan import _hrrr_chain

    plan = load_plan(_hrrr_plan(tmp_path, tmp_path / "run"))

    class _Exp:
        domains = (object(), object())

    with pytest.raises(PlanError) as refusal:
        _hrrr_chain(plan, config_path=tmp_path / "c.toml", exp=_Exp(),
                    observer=None, run_dir=tmp_path / "run")
    text = str(refusal.value)
    assert "2-domain tree" in text
    assert "single-domain only" in text
    # It names the chain that DOES run a tree rather than just refusing.
    assert "hrrr_hierarchy_direct" in text
    assert "prepared-tree-forecast" in text


def test_a_source_the_prepared_route_cannot_drive_is_refused(tmp_path):
    with pytest.raises(PlanError) as refusal:
        build_plan({"schema": PLAN_SCHEMA, "name": "x", "route": "prepared",
                    "config": {"intent": {**_HRRR_INTENT,
                                          "source": "era5"}}},
                   source="probe.json", base_dir=tmp_path, sha256="0" * 64)
    text = str(refusal.value)
    assert "era5" in text and "experiment" in text


def _staged_hrrr_chain(tmp_path, monkeypatch, **plan_overrides):
    """Assemble the HRRR chain without executing any stage."""

    import gpuwm.go_cli as go_cli
    import gpuwm.runplan as runplan_module

    staged = []
    monkeypatch.setattr(
        go_cli, "run_stage",
        lambda label, command, **kw: staged.append((label, list(command))))

    def fake_fetch(arguments, run_dir):
        out = Path(arguments[arguments.index("--out") + 1])
        out.mkdir(parents=True, exist_ok=True)
        (out / "SHA256SUMS").write_text("x", encoding="utf-8")
        staged.append(("fetch", list(arguments)))
        return {}

    monkeypatch.setattr(runplan_module, "_run_fetch", fake_fetch)
    geog = tmp_path / "GEOG"
    geog.mkdir(exist_ok=True)
    plan_overrides.setdefault("run_options", {"geog_root": str(geog)})
    plan = load_plan(_hrrr_plan(tmp_path, tmp_path / "run",
                                **plan_overrides))
    plan.run_dir.mkdir(parents=True, exist_ok=True)
    with EventStream(plan.run_dir / EVENTS_FILENAME, mirror=None) as events:
        execute_plan(plan, events=events)
    return staged, read_events(plan.run_dir / EVENTS_FILENAME)


def test_the_hrrr_chain_passes_the_wps_namelist_to_the_preparer(
        tmp_path, monkeypatch):
    """The whole point of this increment.

    The runner's HRRR manifest role inventory requires a wps_namelist
    role, and the preparer only records it if handed the file.  The
    wizard's printed chain never passed it, so the bundle could not be
    read by the single-domain runner at all.
    """

    staged, _events = _staged_hrrr_chain(tmp_path, monkeypatch)
    prepare = next(cmd for label, cmd in staged if label == "prepare")

    assert "--wps-namelist" in prepare
    namelist = Path(prepare[prepare.index("--wps-namelist") + 1])
    assert namelist.name.endswith(".namelist.wps")
    assert namelist.is_file()


def test_the_hrrr_prepare_command_is_the_wizards_printed_one(
        tmp_path, monkeypatch):
    """Every flag the documented chain passes, and the cycle it spells."""

    staged, _events = _staged_hrrr_chain(tmp_path, monkeypatch)
    prepare = next(cmd for label, cmd in staged if label == "prepare")

    assert prepare[1:3] == ["-m", "tools.prepare_hrrr_wrf"]
    for flag in ("--source-root", "--source-manifest",
                 "--source-manifest-sha256", "--domain-spec",
                 "--namelist-input", "--geog-root", "--cycle",
                 "--run-seconds", "--history-interval-seconds",
                 "--skip-stock-wrf-export", "--output-root"):
        assert flag in prepare, flag
    # [fetch] spells the cycle YYYY-MM-DDTHH; the preparer takes
    # YYYY-MM-DD_HH:MM:SS.  Converted, not sliced.
    assert prepare[prepare.index("--cycle") + 1] == "2024-05-03_12:00:00"
    assert prepare[prepare.index("--run-seconds") + 1] == "21600"
    # The digest is the real one, of the file the fetch actually wrote.
    import hashlib

    manifest = Path(prepare[prepare.index("--source-manifest") + 1])
    assert prepare[prepare.index("--source-manifest-sha256") + 1] == \
        hashlib.sha256(manifest.read_bytes()).hexdigest()


def test_the_hrrr_fetch_argv_comes_from_the_configs_own_fetch_hints(
        tmp_path, monkeypatch):
    staged, _events = _staged_hrrr_chain(tmp_path, monkeypatch)
    fetch = next(cmd for label, cmd in staged if label == "fetch")

    assert fetch[fetch.index("--source") + 1] == "hrrr"
    assert fetch[fetch.index("--cycle") + 1] == "2024-05-03T12"
    assert fetch[fetch.index("--hours") + 1] == "6"
    assert "--area" in fetch          # the wizard sized it
    # And it is a valid `gpuwm fetch` argv, by that parser's own reckoning.
    from gpuwm.cli import build_parser

    build_parser().parse_args(["fetch", *fetch])


def test_the_hrrr_chain_emits_the_stages_in_the_documented_order(
        tmp_path, monkeypatch):
    _staged, events = _staged_hrrr_chain(tmp_path, monkeypatch)
    started = [r["stage"] for r in events if r["event"] == "stage_started"]
    assert started[:2] == ["fetch", "prepare"]


def test_the_physics_profile_is_passed_only_when_the_plan_states_it(
        tmp_path, monkeypatch):
    """The route owns its physics gate; this layer must not invent one."""

    staged, _ = _staged_hrrr_chain(tmp_path, monkeypatch)
    prepare = next(cmd for label, cmd in staged if label == "prepare")
    assert "--physics-profile" not in prepare

    second = tmp_path / "stated"
    second.mkdir()
    staged, _ = _staged_hrrr_chain(
        second, monkeypatch,
        run_options={"geog_root": str(second / "GEOG"),
                     "physics_profile": "wsm6-ysu-mm5-noah-no-radiation-v1"})
    prepare = next(cmd for label, cmd in staged if label == "prepare")
    assert prepare[prepare.index("--physics-profile") + 1] == \
        "wsm6-ysu-mm5-noah-no-radiation-v1"


# ---------------------------------------------------------------------------
# Selective rendering
# ---------------------------------------------------------------------------


def _go_plan(tmp_path, **extra):
    return {"run": tmp_path / "run", "render": tmp_path / "png", **extra}


def test_render_products_reaches_the_render_command_verbatim(tmp_path):
    """The renderer owns the vocabulary; this passes the spec through."""

    from gpuwm.go_cli import render_command

    frames = [tmp_path / "wrfout_d01_0001"]
    plain = render_command(_go_plan(tmp_path), frames)
    assert "--products" not in plain          # default set, unchanged

    filtered = render_command(
        _go_plan(tmp_path, render_products="sbcape,srh_0_1km"), frames)
    assert filtered[filtered.index("--products") + 1] == "sbcape,srh_0_1km"
    # Everything else about the command is identical.
    assert filtered[:len(plain)] == plain


def test_render_products_none_skips_the_stage_without_running_it(
        tmp_path, monkeypatch, capsys):
    import gpuwm.go_cli as go_cli

    ran = []
    monkeypatch.setattr(go_cli, "_run_stage",
                        lambda label, command, **kw: ran.append(label))

    assert go_cli._render_stage(
        _go_plan(tmp_path, render_products="none"), explain=False) is False
    assert ran == []
    assert "render skipped" in capsys.readouterr().out

    # Spelled any way a person would spell it.
    assert go_cli._render_stage(
        _go_plan(tmp_path, render_products=" NONE "), explain=False) is False
    assert ran == []


def test_the_default_render_stage_is_untouched_when_no_filter_is_given(
        tmp_path, monkeypatch):
    """`gpuwm go` itself sets nothing, so its behaviour cannot move."""

    import gpuwm.go_cli as go_cli

    ran = []
    monkeypatch.setattr(go_cli, "render_extra_missing", lambda: None)
    monkeypatch.setattr(go_cli, "wrfout_frames",
                        lambda plan: [tmp_path / "wrfout_d01_0001"])
    monkeypatch.setattr(
        go_cli, "_run_stage",
        lambda label, command, **kw: ran.append((label, list(command))))

    assert go_cli._render_stage(_go_plan(tmp_path), explain=False) is True
    assert ran[0][0] == "render"
    assert "--products" not in ran[0][1]


def test_render_products_is_a_run_option_on_the_prepared_route(tmp_path):
    from gpuwm.runplan import ROUTES

    assert "render_products" in ROUTES["prepared"].run_options
    # Not an intent key: intent mirrors `gpuwm domain` flags one for
    # one, and the wizard writes configs, not pictures.
    from gpuwm.runplan import _INTENT_FLAGS

    assert "render_products" not in _INTENT_FLAGS

    plan = load_plan(_prepared_plan(
        tmp_path, tmp_path / "run",
        ))
    assert plan.run_options["render_products"] is None


def test_the_run_option_is_stamped_onto_the_namespace_go_reads(
        tmp_path, monkeypatch):
    import gpuwm.go_cli as go_cli

    seen = {}
    monkeypatch.setattr(
        go_cli, "go_main",
        lambda args, *, observer=None: seen.update(
            products=getattr(args, "render_products", None)) or 1)

    path = tmp_path / "plan.json"
    path.write_text(json.dumps({
        "schema": PLAN_SCHEMA, "name": "p", "route": "prepared",
        "config": {"intent": dict(_GFS_INTENT)},
        "run_options": {"render_products": "refl,t2"},
        "output_root": str(tmp_path / "run"),
    }), encoding="utf-8")
    plan = load_plan(path)
    plan.run_dir.mkdir(parents=True, exist_ok=True)
    with EventStream(plan.run_dir / EVENTS_FILENAME, mirror=None) as events:
        execute_plan(plan, events=events)
    assert seen["products"] == "refl,t2"


def test_both_chains_honour_the_same_render_filter(tmp_path, monkeypatch):
    """The HRRR chain renders too, so the option cannot mean two things."""

    import gpuwm.go_cli as go_cli
    import gpuwm.runplan as runplan_module

    rendered = []
    monkeypatch.setattr(go_cli, "render_extra_missing", lambda: None)
    monkeypatch.setattr(go_cli, "wrfout_frames",
                        lambda plan: [tmp_path / "wrfout_d01_0001"])
    monkeypatch.setattr(
        go_cli, "_run_stage",
        lambda label, command, **kw: rendered.append(list(command)))

    # The HRRR arm builds the same render plan go does, from its own
    # forecast directory.
    stage = runplan_module._GoObserver
    assert stage is not None
    go_cli._render_stage(
        {"run": tmp_path / "run", "render": tmp_path / "png",
         "render_products": "composite_reflectivity"},
        explain=False)
    assert rendered[0][rendered[0].index("--products") + 1] ==         "composite_reflectivity"


def test_the_catalog_query_returns_the_renderers_real_list(capsys):
    """Asked, never transcribed -- and the parse is checked."""

    from gpuwm.cli import build_parser
    from gpuwm.runplan import CATALOG_SCHEMA

    assert run_plan_main(
        build_parser().parse_args(["run-plan", "--catalog"])) == 0
    document = json.loads(capsys.readouterr().out)

    assert document["schema"] == CATALOG_SCHEMA
    assert document["engine"] in ("rust", "matplotlib")
    assert document["skip_token"] == "none"
    products = document["products"]
    assert products and all(entry["name"] for entry in products)
    # No header or footer line leaked in as a product.
    names = [entry["name"] for entry in products]
    assert not any(" " in name for name in names)
    assert not any(name.startswith(("group keywords", "selectable_slugs"))
                   for name in names)
    # A disagreement with the renderer's own count is reported, not hidden.
    assert "parse_warning" not in document


def test_the_catalog_needs_no_plan_and_writes_nothing(tmp_path):
    from gpuwm.runplan import render_catalog

    document = render_catalog()
    assert document["spec"]
    assert list(tmp_path.iterdir()) == []


def test_a_matplotlib_only_box_still_answers_the_catalog(monkeypatch):
    """The fallback engine's five products are the honest answer there."""

    import gpuwm.render as render_module

    monkeypatch.setattr(render_module, "_resolve_engine",
                        lambda requested: ("matplotlib", "no rust renderer"))
    from gpuwm.runplan import render_catalog

    document = render_catalog()
    assert document["engine"] == "matplotlib"
    assert [entry["name"] for entry in document["products"]] == \
        list(render_module.PRODUCTS)


# ---------------------------------------------------------------------------
# Output cadence
# ---------------------------------------------------------------------------


def _emit(tmp_path, extra):
    """Run the real wizard and return its emitted history intervals."""

    import contextlib
    import io as _io
    import tomllib as _tomllib

    from gpuwm.cli import build_parser
    from gpuwm.domain_wizard import domain_main

    out = tmp_path / "c.toml"
    args = build_parser().parse_args([
        "domain", "--point", "39,-98", "--source", "era5", "--cycle",
        "2024-05-03T12", "--hours", "1", "--vram-gib", "24",
        "--ladder", "12-3", "--out", str(out), *extra])
    args.interactive = False
    with contextlib.redirect_stdout(_io.StringIO()):
        assert domain_main(args) == 0
    domains = _tomllib.load(_io.BytesIO(out.read_bytes()))["domain"]
    return [d["history_interval_s"] for d in domains]


def test_the_wizards_default_cadence_is_unchanged(tmp_path):
    """The flag is a default, not a behaviour change."""

    from gpuwm.domain_wizard import (DEFAULT_NEST_HISTORY_INTERVAL_S,
                                     DEFAULT_ROOT_HISTORY_INTERVAL_S)

    assert _emit(tmp_path, []) == [DEFAULT_ROOT_HISTORY_INTERVAL_S,
                                   DEFAULT_NEST_HISTORY_INTERVAL_S]
    assert (DEFAULT_ROOT_HISTORY_INTERVAL_S,
            DEFAULT_NEST_HISTORY_INTERVAL_S) == (3600.0, 900.0)


def test_the_cadence_flags_reach_the_emitted_config(tmp_path):
    assert _emit(tmp_path, ["--history-interval", "600",
                            "--nest-history-interval", "300"]) == [600.0, 300.0]


def test_a_root_cadence_alone_leaves_the_nest_on_its_default(tmp_path):
    assert _emit(tmp_path, ["--history-interval", "1800"]) == [1800.0, 900.0]


def test_a_cadence_off_the_step_grid_is_refused_before_anything_is_written(
        tmp_path):
    """The engine's own rule, on the wizard's own pre-write round trip."""

    import contextlib
    import io as _io

    from gpuwm.cli import build_parser
    from gpuwm.domain_wizard import domain_main

    out = tmp_path / "c.toml"
    args = build_parser().parse_args([
        "domain", "--point", "39,-98", "--source", "era5", "--cycle",
        "2024-05-03T12", "--hours", "1", "--vram-gib", "24",
        "--out", str(out), "--history-interval", "7"])
    args.interactive = False
    with contextlib.redirect_stdout(_io.StringIO()):
        with pytest.raises(ValueError) as refusal:
            domain_main(args)
    text = str(refusal.value)
    assert "history_interval_s" in text
    assert "whole number of that domain's steps" in text
    assert not out.exists()      # refused BEFORE the file landed


def test_cadence_is_an_intent_key_on_both_routes(tmp_path):
    plan = load_plan(_intent_plan(tmp_path, tmp_path / "run",
                                  history_interval_s=1800))
    resolution, exp, _ = resolve_plan(plan, require_inputs=False)
    assert exp.domains[0].history_interval_s == 1800.0
    assert "history_interval_s = 1800.0" in resolution["generated_config"]


def test_a_bad_intent_cadence_is_a_plan_refusal_not_a_crash(tmp_path):
    plan = load_plan(_intent_plan(tmp_path, tmp_path / "run",
                                  history_interval_s=7))
    with pytest.raises(PlanError) as refusal:
        resolve_plan(plan, require_inputs=False)
    assert "history_interval_s" in str(refusal.value)


# ---------------------------------------------------------------------------
# Cycle "latest"
# ---------------------------------------------------------------------------


def test_latest_is_resolved_to_a_concrete_cycle_before_the_fetch_runs(
        monkeypatch):
    from datetime import datetime as _dt

    import gpuwm.fetch as fetch
    from gpuwm.runplan import resolve_fetch_cycle

    monkeypatch.setattr(fetch, "resolve_latest_cycle",
                        lambda source, last_hour: _dt(2026, 8, 7, 12))
    arguments, resolutions, warnings = resolve_fetch_cycle(
        ["--source", "gfs", "--cycle", "latest", "--hours", "6",
         "--out", "data"])

    assert "latest" not in arguments
    assert arguments[arguments.index("--cycle") + 1] == "2026-08-07T12"
    assert resolutions[0] == {
        "scope": "fetch", "key": "cycle", "value": "2026-08-07T12",
        "basis": "resolved_latest", "note": resolutions[0]["note"]}
    assert "complete by construction" in resolutions[0]["note"]


def test_an_explicit_cycle_is_left_exactly_as_written(monkeypatch):
    import gpuwm.fetch as fetch
    from gpuwm.runplan import resolve_fetch_cycle

    def refuse(*a, **k):
        raise AssertionError("an explicit cycle must not probe the mirrors")

    monkeypatch.setattr(fetch, "resolve_latest_cycle", refuse)
    arguments, resolutions, warnings = resolve_fetch_cycle(
        ["--source", "gfs", "--cycle", "2026-08-07T00", "--hours", "6"])
    assert arguments[arguments.index("--cycle") + 1] == "2026-08-07T00"
    assert resolutions == [] and warnings == []


def test_latest_is_matched_case_insensitively(monkeypatch):
    from datetime import datetime as _dt

    import gpuwm.fetch as fetch
    from gpuwm.runplan import resolve_fetch_cycle

    monkeypatch.setattr(fetch, "resolve_latest_cycle",
                        lambda source, last_hour: _dt(2026, 8, 7, 12))
    arguments, resolutions, _ = resolve_fetch_cycle(
        ["--source", "hrrr", "--cycle", "Latest", "--hours", "3"])
    assert arguments[arguments.index("--cycle") + 1] == "2026-08-07T12"
    assert resolutions


def test_a_stale_latest_cycle_is_a_warning_never_a_refusal(monkeypatch):
    """Newer cycles publishing = the run starts older than you think."""

    from datetime import datetime as _dt, timedelta as _td

    import gpuwm.fetch as fetch
    import gpuwm.runplan as runplan_module
    from gpuwm.runplan import resolve_fetch_cycle

    stale = _dt(2026, 8, 7, 0)
    monkeypatch.setattr(fetch, "resolve_latest_cycle",
                        lambda source, last_hour: stale)
    monkeypatch.setattr(
        runplan_module, "datetime",
        type("C", (_dt,), {"now": classmethod(
            lambda cls, tz=None: stale + _td(hours=20))}))

    arguments, resolutions, warnings = resolve_fetch_cycle(
        ["--source", "gfs", "--cycle", "latest", "--hours", "6"])

    assert arguments[arguments.index("--cycle") + 1] == "2026-08-07T00"
    assert warnings and warnings[0]["code"] == "latest_cycle_is_not_the_newest"
    assert warnings[0]["age_hours"] == 20
    assert "not yet published" in warnings[0]["message"]


def test_the_fetch_hours_include_the_forecast_start_lead(monkeypatch):
    """`latest` must cover the END of the window, lead included."""

    from datetime import datetime as _dt

    import gpuwm.fetch as fetch
    from gpuwm.runplan import resolve_fetch_cycle

    seen = {}

    def record(source, last_hour):
        seen["last_hour"] = last_hour
        return _dt(2026, 8, 7, 0)

    monkeypatch.setattr(fetch, "resolve_latest_cycle", record)
    resolve_fetch_cycle(["--source", "gfs", "--cycle", "latest",
                         "--hours", "12", "--forecast-start-hour", "6"])
    assert seen["last_hour"] == 18


# ---------------------------------------------------------------------------
# The event stream
# ---------------------------------------------------------------------------


def test_the_stream_writes_one_envelope_per_line_with_a_dense_sequence(
        tmp_path):
    path = tmp_path / EVENTS_FILENAME
    with EventStream(path, mirror=None) as events:
        events.emit("plan_accepted", name="a")
        events.emit("warning", code="c", message="m")
        events.emit("completed", dry_run=True)

    records = read_events(path)
    assert [record["sequence"] for record in records] == [1, 2, 3]
    assert {record["schema_version"] for record in records} == {EVENT_SCHEMA}
    assert [record["event"] for record in records] == [
        "plan_accepted", "warning", "completed"]
    # Event-specific fields are flattened alongside the envelope, not
    # nested under a payload key.
    assert records[1]["code"] == "c"
    assert all(isinstance(record["emitted_unix_ms"], int)
               for record in records)


def test_an_unknown_event_tag_is_refused_rather_than_written(tmp_path):
    path = tmp_path / EVENTS_FILENAME
    with EventStream(path, mirror=None) as events:
        with pytest.raises(PlanError):
            events.emit("almost_completed")
    assert read_events(path) == []


def test_an_event_may_not_shadow_an_envelope_key(tmp_path):
    with EventStream(tmp_path / EVENTS_FILENAME, mirror=None) as events:
        with pytest.raises(PlanError) as refusal:
            events.emit("warning", sequence=99, code="c", message="m")
    assert "sequence" in str(refusal.value)


def test_a_replayed_stream_is_identical_to_what_was_emitted(tmp_path):
    path = tmp_path / EVENTS_FILENAME
    emitted = []
    with EventStream(path, mirror=None) as events:
        for index in range(20):
            emitted.append(events.emit("model_progress", domain=1,
                                       model_seconds=float(index)))
    assert read_events(path) == emitted


def test_a_second_stream_into_the_same_file_continues_the_sequence(tmp_path):
    """Append must not restart at 1, or the whole file becomes unreadable.

    The file is opened for append, so a resume -- or any caller that
    reuses a run directory -- writes after records that already exist.
    A restarted counter puts a 1 after a 7 and read_events refuses the
    lot as reordered.
    """

    path = tmp_path / EVENTS_FILENAME
    with EventStream(path, mirror=None) as events:
        events.emit("plan_accepted", name="first")
        events.emit("completed", dry_run=True)
    with EventStream(path, mirror=None) as events:
        assert events.sequence == 2
        events.emit("plan_accepted", name="second")

    records = read_events(path)
    assert [r["sequence"] for r in records] == [1, 2, 3]
    assert records[-1]["name"] == "second"


def test_a_torn_final_line_is_refused_unless_the_caller_says_otherwise(
        tmp_path):
    path = tmp_path / EVENTS_FILENAME
    with EventStream(path, mirror=None) as events:
        events.emit("plan_accepted", name="a")
    with path.open("a", encoding="utf-8") as stream:
        stream.write('{"schema_version": "gpuwm.run-p')

    with pytest.raises(PlanError):
        read_events(path)
    assert len(read_events(path, allow_partial_tail=True)) == 1


def test_a_sequence_gap_is_refused_because_it_means_a_lost_line(tmp_path):
    path = tmp_path / EVENTS_FILENAME
    path.write_text("\n".join(json.dumps({
        "schema_version": EVENT_SCHEMA, "sequence": sequence,
        "emitted_unix_ms": 0, "event": "warning", "code": "c",
        "message": "m"}) for sequence in (1, 3)) + "\n", encoding="utf-8")
    with pytest.raises(PlanError) as refusal:
        read_events(path)
    assert "sequence" in str(refusal.value)


# ---------------------------------------------------------------------------
# A run, with the GPU integrate step replaced at the observer seam
# ---------------------------------------------------------------------------


class _StubSummary:
    """What ``run_experiment`` returns, in the shape the route reads."""

    def __init__(self, paths):
        self.wrfout_paths = tuple(paths)
        self.completed_seconds = 3600.0
        self.nan_free = True


def _stub_run_experiment(fail_at=None, frames=2):
    """A run that drives the real observer through the real protocol.

    The phases and the keyword set are the pipeline's own -- copied from
    ``runtime._preparation_progress``'s call sites and the per-step
    ``progress_callback`` -- so this stub is wrong in exactly the ways
    the live pipeline would be wrong, and no others.
    """

    def run_experiment(exp, data, outdir, *, restart=None,
                       progress_callback=None, health_debug=False):
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        paths = []
        for phase in ("quarantine-wrfout", "resolve-schedule",
                      "prepare-case"):
            progress_callback.preparing(phase)
        if fail_at == "prepare":
            raise RuntimeError("preparation refused")
        for phase in ("initialize-health-validator", "cold-start-wrfout",
                      "initial-health-gate"):
            progress_callback.preparing(phase)
        for step in range(1, frames + 1):
            valid = exp.start_time + timedelta(seconds=step * 60.0)
            path = outdir / f"wrfout_d01_{step:03d}"
            path.write_bytes(b"frame")
            paths.append(path)
            progress_callback.output_committed(
                domain=1, valid_time=valid, path=path)
            progress_callback(
                model_elapsed_seconds=float(step * 60),
                outer_step=step, last_durable_wrfout=path,
                last_checkpoint=None, phase="post-d01-sync",
                step_wall_seconds=0.004)
            if fail_at == "forecast" and step == 1:
                raise RuntimeError("the dycore refused")
        return _StubSummary(paths)

    return run_experiment


@pytest.fixture()
def stubbed_runtime(monkeypatch):
    def install(**kwargs):
        from gpuwm import runtime
        monkeypatch.setattr(runtime, "run_experiment",
                            _stub_run_experiment(**kwargs))
    return install


def _run(tmp_path, stubbed_runtime, *, plan_overrides=None, **stub):
    stubbed_runtime(**stub)
    config = make_case_toml(tmp_path)
    run_dir = tmp_path / "run"
    plan = load_plan(_write_plan(tmp_path, config, run_dir,
                                 **(plan_overrides or {})))
    with EventStream(run_dir / EVENTS_FILENAME, mirror=None) as events:
        code = execute_plan(plan, events=events)
    return code, run_dir, read_events(run_dir / EVENTS_FILENAME)


def test_a_completed_run_emits_its_events_in_order_with_a_dense_sequence(
        tmp_path, stubbed_runtime):
    code, run_dir, events = _run(tmp_path, stubbed_runtime)

    assert code == 0
    assert [record["sequence"] for record in events] == list(
        range(1, len(events) + 1))
    tags = [record["event"] for record in events]
    assert tags[0] == "plan_accepted"
    assert tags[1] == "resolved_plan"
    assert tags[-1] == "completed"
    assert set(tags) <= set(EVENT_TAGS)

    # The stages the pipeline's own phases opened, in order, each one
    # started before it finished.
    started = [record["stage"] for record in events
               if record["event"] == "stage_started"]
    finished = [record["stage"] for record in events
                if record["event"] == "stage_finished"]
    assert started == ["prepare", "initialize", "forecast", "finalize"]
    assert finished == started
    assert all(stage in STAGES for stage in started)

    # Every output that landed is announced with its domain, valid time
    # and path -- no consumer re-derives them from a filename.
    committed = [record for record in events
                 if record["event"] == "output_committed"]
    assert len(committed) == 2
    assert committed[0]["domain"] == 1
    assert Path(committed[0]["path"]).is_file()
    datetime.fromisoformat(committed[0]["valid_time"])

    progress = [record for record in events
                if record["event"] == "model_progress"]
    assert [record["outer_step"] for record in progress] == [1, 2]
    assert progress[0]["step_ms"] == pytest.approx(4.0)
    assert progress[-1]["model_seconds"] == 120.0
    assert progress[-1]["wall_seconds"] >= 0.0

    completed = events[-1]
    assert completed["summary"]["wrfout_count"] == 2
    assert completed["outputs_committed"] == 2


def test_an_output_is_announced_only_after_the_progress_that_precedes_it_is(
        tmp_path, stubbed_runtime):
    """Ordering is the contract; a consumer builds a timeline from it."""

    _code, _run_dir, events = _run(tmp_path, stubbed_runtime)
    ordered = [(record["sequence"], record["event"]) for record in events
               if record["event"] in ("output_committed", "model_progress")]
    assert [tag for _, tag in ordered] == [
        "output_committed", "model_progress"] * 2
    assert [sequence for sequence, _ in ordered] == sorted(
        sequence for sequence, _ in ordered)


def test_the_manifest_is_written_before_any_work_and_names_every_stream(
        tmp_path, stubbed_runtime):
    from gpuwm.supervisor import (FAILURE_CAPSULE_SCHEMA, HEARTBEAT_NAME,
                                  HEARTBEAT_SCHEMA)

    _code, run_dir, events = _run(tmp_path, stubbed_runtime)
    manifest = json.loads(
        (run_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))

    assert manifest["schema"] == MANIFEST_SCHEMA
    assert manifest["pid"] > 0
    assert len(manifest["plan_sha256"]) == 64
    assert Path(manifest["events_path"]) == run_dir / EVENTS_FILENAME
    assert manifest["events_schema"] == EVENT_SCHEMA
    # It points at the schemas this module does NOT own, so a front end
    # never has to know their filenames.
    assert Path(manifest["progress_path"]) == run_dir / HEARTBEAT_NAME
    assert manifest["progress_schema"] == HEARTBEAT_SCHEMA
    assert manifest["failure_capsule_schema"] == FAILURE_CAPSULE_SCHEMA
    assert "heartbeat" in manifest["reattach"]

    # The manifest exists by the time the first event says it does.
    assert events[0]["manifest_path"] == str(run_dir / MANIFEST_FILENAME)


def test_reattach_replay_yields_exactly_the_event_list_that_was_emitted(
        tmp_path, stubbed_runtime):
    _code, run_dir, events = _run(tmp_path, stubbed_runtime)
    manifest = json.loads(
        (run_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))

    # A consumer attaching afterwards has only the manifest, and reaches
    # the same history from it.
    replayed = read_events(manifest["events_path"])
    assert replayed == events
    # Twice, because a replay that is not idempotent is not a replay.
    assert read_events(manifest["events_path"]) == replayed


def test_the_supervisors_own_heartbeat_is_the_one_that_gets_written(
        tmp_path, stubbed_runtime):
    """No second progress writer: this front door composes with theirs."""

    from gpuwm.supervisor import HEARTBEAT_NAME, HEARTBEAT_SCHEMA, read_heartbeat

    _code, run_dir, _events = _run(tmp_path, stubbed_runtime)
    heartbeat = read_heartbeat(run_dir / HEARTBEAT_NAME)
    assert heartbeat.schema == HEARTBEAT_SCHEMA
    assert heartbeat.status == "complete"
    assert heartbeat.outer_step == 2
    assert heartbeat.model_elapsed_seconds == 3600.0


def test_a_failed_run_emits_failed_last_and_exits_nonzero(
        tmp_path, stubbed_runtime):
    code, _run_dir, events = _run(tmp_path, stubbed_runtime,
                                  fail_at="forecast")

    assert code != 0
    assert events[-1]["event"] == "failed"
    assert events[-1]["stage"] == "forecast"
    assert events[-1]["error_class"] == "RuntimeError"
    assert "dycore refused" in events[-1]["message"]
    # The stage that was open is closed before the failure, so a
    # consumer's stage timeline has no stage left hanging.
    closing = [record for record in events
               if record["event"] == "stage_finished"]
    assert closing[-1]["stage"] == "forecast"
    assert closing[-1]["outcome"] == "failed"


def test_a_failure_during_preparation_names_the_stage_it_failed_in(
        tmp_path, stubbed_runtime):
    code, _run_dir, events = _run(tmp_path, stubbed_runtime,
                                  fail_at="prepare")
    assert code != 0
    assert events[-1]["event"] == "failed"
    assert events[-1]["stage"] == "prepare"


def test_a_dry_run_resolves_the_plan_and_stops_before_any_device_work(
        tmp_path, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("a dry run must not reach the runtime")

    from gpuwm import runtime
    monkeypatch.setattr(runtime, "run_experiment", refuse)

    config = make_case_toml(tmp_path)
    run_dir = tmp_path / "run"
    plan = load_plan(_write_plan(tmp_path, config, run_dir,
                                 run_options={"dry_run": True}))
    with EventStream(run_dir / EVENTS_FILENAME, mirror=None) as events:
        code = execute_plan(plan, events=events)

    events = read_events(run_dir / EVENTS_FILENAME)
    assert code == 0
    assert events[-1]["event"] == "completed"
    assert events[-1]["dry_run"] is True
    # It still resolved: the whole point of a dry run is the snapshot.
    assert any(record["event"] == "resolved_plan" for record in events)


def test_the_resolved_plan_event_carries_the_snapshot_and_the_resolutions(
        tmp_path, stubbed_runtime):
    _code, _run_dir, events = _run(tmp_path, stubbed_runtime)
    resolved = next(record for record in events
                    if record["event"] == "resolved_plan")
    assert resolved["configuration"]["experiment"]["name"]
    assert resolved["automatic_resolutions"]
    assert any(entry["key"] == "execution_mode"
               for entry in resolved["automatic_resolutions"])


def test_the_stream_is_mirrored_to_stdout_line_for_line(
        tmp_path, stubbed_runtime, capsys):
    stubbed_runtime()
    config = make_case_toml(tmp_path)
    run_dir = tmp_path / "run"
    plan_path = _write_plan(tmp_path, config, run_dir)

    from gpuwm.cli import build_parser
    args = build_parser().parse_args(["run-plan", str(plan_path)])
    code = run_plan_main(args)

    assert code == 0
    printed = [json.loads(line) for line
               in capsys.readouterr().out.splitlines() if line.startswith("{")]
    assert printed == read_events(run_dir / EVENTS_FILENAME)


# ---------------------------------------------------------------------------
# The observer, on its own
# ---------------------------------------------------------------------------


def test_an_unmapped_preparation_phase_is_reported_not_mis_filed(tmp_path):
    with EventStream(tmp_path / EVENTS_FILENAME, mirror=None) as events:
        observer = RunObserver(events)
        observer.enter_stage("prepare")
        observer.preparing("a-phase-nobody-mapped")
        observer.finish_stage()
    records = read_events(tmp_path / EVENTS_FILENAME)
    warning = next(record for record in records
                   if record["event"] == "warning")
    assert warning["code"] == "unmapped_pipeline_phase"
    assert warning["phase"] == "a-phase-nobody-mapped"
    # And it still lands in the open stage's phase list rather than
    # vanishing.
    finished = next(record for record in records
                    if record["event"] == "stage_finished")
    assert "a-phase-nobody-mapped" in finished["phases"]


def test_speed_x_is_null_rather_than_an_infinity_before_any_wall_elapses(
        tmp_path):
    with EventStream(tmp_path / EVENTS_FILENAME, mirror=None) as events:
        observer = RunObserver(events)
        observer(model_elapsed_seconds=0.0, outer_step=0,
                 last_durable_wrfout=None, last_checkpoint=None,
                 phase="initialized-or-restored", step_wall_seconds=0.0)
    progress = next(record for record in read_events(
        tmp_path / EVENTS_FILENAME) if record["event"] == "model_progress")
    assert progress["speed_x"] is None
    assert progress["step_ms"] is None


def test_the_landing_hook_is_optional_on_a_hand_built_writer_shell():
    """The seam must not require the ``__init__`` path to have run.

    ``tests/test_wrfout.py`` stands this writer up field-by-field
    through ``object.__new__`` because a CPU-only harness cannot
    allocate a CuPy stream.  The first version of the landing hook set
    its attributes in ``__init__`` only, so every such shell raised
    AttributeError on the worker thread -- four tests, none of them
    about this feature.  The defaults live on the class for that reason.
    """

    from gpuwm.io.wrfout import AsyncDomainWrfoutWriter

    shell = object.__new__(AsyncDomainWrfoutWriter)
    assert shell.landing_observer is None
    assert shell.grid_id is None


def test_the_landing_hook_can_be_bound_after_the_writers_are_built():
    """Both prepared runners build their closure OVER the writers.

    Their progress closure reports ``writers.paths``, so it cannot exist
    before the object it reads -- a real cycle, which reordering does
    not break.  ``attach_progress_callback`` is the late-binding half,
    and these two nodes pin it: it binds, and it refuses once a frame
    has gone by.
    """

    from gpuwm.io.wrfout import AsyncDomainWrfoutWriter, PerDomainWrfoutWriters

    writers = object.__new__(PerDomainWrfoutWriters)
    shell = object.__new__(AsyncDomainWrfoutWriter)
    shell.paths = []
    shell._pending = 0
    shell._condition = __import__("threading").Condition()
    writers._writers = {1: shell}

    def observed(**_kwargs):
        pass

    carrier = type("Carrier", (), {"output_committed": staticmethod(observed)})()
    writers.attach_progress_callback(carrier)
    assert shell.landing_observer is observed

    # A progress object without the hook binds nothing, rather than
    # binding something that cannot be called.
    writers.attach_progress_callback(lambda **_: None)
    assert shell.landing_observer is None


def test_binding_the_landing_hook_late_is_refused_once_a_frame_has_landed():
    from gpuwm.io.wrfout import AsyncDomainWrfoutWriter, PerDomainWrfoutWriters

    writers = object.__new__(PerDomainWrfoutWriters)
    shell = object.__new__(AsyncDomainWrfoutWriter)
    shell.paths = [Path("wrfout_d01_0001")]
    shell._pending = 0
    shell._condition = __import__("threading").Condition()
    writers._writers = {1: shell}

    with pytest.raises(RuntimeError) as refusal:
        writers.attach_progress_callback(
            type("C", (), {"output_committed": staticmethod(lambda **_: None)})())
    assert "silently miss" in str(refusal.value)


@pytest.mark.parametrize("runner", ["gpuwm.prepared_single_domain_forecast",
                                    "gpuwm.prepared_domain_tree_forecast"])
def test_both_prepared_runners_accept_an_external_observer(runner):
    """The parameter exists and defaults to off, on both sibling routes.

    This is the plumbing a future run-plan route needs; the routes
    themselves are not registered yet.  Signature-level, because the
    runner bodies need a card.
    """

    import importlib
    import inspect

    module = importlib.import_module(runner)
    entry = (module.run_prepared_forecast
             if hasattr(module, "run_prepared_forecast")
             else module.run_prepared_tree)
    parameter = inspect.signature(entry).parameters["observer"]
    assert parameter.default is None
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY


def test_the_observer_works_with_no_heartbeat_at_all(tmp_path):
    """The heartbeat is composed, not required: this must not crash."""

    with EventStream(tmp_path / EVENTS_FILENAME, mirror=None) as events:
        observer = RunObserver(events, heartbeat=None)
        observer.starting()
        observer.preparing("prepare-case")
        observer.complete(1.0)
        observer.failed()
    assert read_events(tmp_path / EVENTS_FILENAME)


# ---------------------------------------------------------------------------
# Query modes
# ---------------------------------------------------------------------------


def test_resolve_prints_one_json_document_and_runs_nothing(
        tmp_path, capsys, monkeypatch):
    from gpuwm import runtime
    monkeypatch.setattr(runtime, "run_experiment", lambda *a, **k: 1 / 0)

    config = make_case_toml(tmp_path)
    plan_path = _write_plan(tmp_path, config, tmp_path / "run")
    from gpuwm.cli import build_parser
    args = build_parser().parse_args(["run-plan", "--resolve", str(plan_path)])

    assert run_plan_main(args) == 0
    document = json.loads(capsys.readouterr().out)
    assert document["schema"] == "gpuwm.run-plan.resolved.v1"
    assert document["configuration"]["experiment"]["name"]
    assert document["automatic_resolutions"]
    # Nothing was created: --resolve answers a question, it does not
    # claim a directory.
    assert not (tmp_path / "run" / EVENTS_FILENAME).exists()


def test_estimate_reports_measured_numbers_and_nulls_the_unmeasured_ones(
        tmp_path, capsys):
    config = make_case_toml(tmp_path)
    plan_path = _write_plan(tmp_path, config, tmp_path / "run")
    from gpuwm.cli import build_parser
    args = build_parser().parse_args(
        ["run-plan", "--estimate", str(plan_path)])

    assert run_plan_main(args) == 0
    document = json.loads(capsys.readouterr().out)
    assert document["schema"] == "gpuwm.run-plan.estimate.v1"
    assert document["vram"]["estimate_bytes"] > 0
    assert document["disk"]["total_frames"] > 0
    # The honest nulls, each with its basis stated rather than a number
    # this package never measured.
    assert document["disk"]["bytes"] is None
    assert document["wall_time"]["seconds"] is None
    assert document["wall_time"]["basis"]
    assert document["download"]["bytes"] is None


def test_probe_answers_without_a_plan_and_without_touching_the_card(
        capsys, monkeypatch):
    import gpuwm.core.preflight as preflight
    import gpuwm.supervisor as supervisor
    from gpuwm.supervisor import GPUIdentity

    monkeypatch.setattr(supervisor, "query_gpus",
                        lambda: (GPUIdentity("GPU-fixture", "999.00",
                                             "Fixture Card", 0),))
    monkeypatch.setattr(preflight, "device_physical_total_bytes",
                        lambda: 32 * 1024 ** 3)
    monkeypatch.setattr(preflight, "device_wide_used_bytes",
                        lambda: 2 * 1024 ** 3)

    from gpuwm.cli import build_parser
    args = build_parser().parse_args(
        ["run-plan", "--probe", "--no-readiness"])

    assert run_plan_main(args) == 0
    document = json.loads(capsys.readouterr().out)
    assert document["schema"] == "gpuwm.run-plan.probe.v1"
    assert document["devices"][0]["name"] == "Fixture Card"
    assert document["devices"][0]["memory_free_bytes"] == 30 * 1024 ** 3
    assert document["readiness"]["collected"] is False
    assert "experiment" in document["routes"]
    assert document["schemas"]["event"] == EVENT_SCHEMA


def test_a_probe_survives_a_machine_with_no_nvidia_smi_at_all(monkeypatch):
    import gpuwm.supervisor as supervisor

    def absent():
        raise supervisor.GPUPreflightError("nvidia-smi unavailable")

    monkeypatch.setattr(supervisor, "query_gpus", absent)
    document = probe_environment(readiness=False)
    assert document["devices"] == []
    assert "nvidia-smi" in document["device_query_error"]


def test_the_module_entry_and_the_subcommand_take_the_same_flags():
    """One parser, two spellings; they cannot drift."""

    from gpuwm.cli import build_parser

    subcommand = build_parser().parse_args(["run-plan", "--probe"])
    assert subcommand.probe is True
    assert subcommand.func.__name__ == "run_plan_main"


# ---------------------------------------------------------------------------
# The real artifact, in a real subprocess
# ---------------------------------------------------------------------------
#
# Everything above drives the front door in-process, which is where the
# contract lives.  These three run the actual command, because two of
# its promises are only true of a process: the exit code, and what
# reaches the terminal.  The first version of `python -m gpuwm.runplan`
# passed every in-process test above and still printed the [[explain]]
# sentinel to stderr on a layered refusal -- it had grown its own
# refusal boundary instead of using the one boundary. Only running it
# showed that.


def _cli(*tokens, cwd):
    """Run the real command in a fresh interpreter, pinned to this tree."""

    import os
    import subprocess
    import sys as _sys

    repo = str(Path(__file__).resolve().parents[1])
    environment = dict(os.environ)
    # The pin is load-bearing: without it a subprocess resolves gpuwm
    # through whatever editable install this interpreter carries, and
    # the test silently exercises a different checkout.
    environment["PYTHONPATH"] = repo + os.pathsep + environment.get(
        "PYTHONPATH", "")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [_sys.executable, "-m", "gpuwm.runplan", *tokens],
        capture_output=True, text=True, cwd=str(cwd), env=environment,
        timeout=300)


def test_the_real_command_exits_zero_and_prints_only_the_event_stream(
        tmp_path):
    config = make_case_toml(tmp_path)
    run_dir = tmp_path / "run"
    plan_path = _write_plan(tmp_path, config, run_dir,
                            run_options={"dry_run": True})

    result = _cli(str(plan_path), cwd=tmp_path)

    assert result.returncode == 0, result.stderr
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    printed = [json.loads(line) for line in lines]
    assert [record["event"] for record in printed] == [
        "plan_accepted", "resolved_plan", "completed"]
    # stdout is the machine channel and carries nothing else at all.
    assert len(printed) == len(lines)
    assert printed == read_events(run_dir / EVENTS_FILENAME)
    assert (run_dir / MANIFEST_FILENAME).is_file()


def test_the_real_command_keeps_stdout_pure_through_a_talking_pipeline(
        tmp_path):
    """The regression the dry-run test could not see.

    A real run reaches code that prints for a person: the pipeline's
    resolved-config report, the feedback advisory, and (on an intent
    plan) the whole wizard.  All of it used to land in the middle of the
    JSONL a consumer is calling json.loads on line by line.  The
    dry-run subprocess test never reaches any of it and passed happily.

    An intent plan is the sharpest version -- the wizard is the
    chattiest thing in the package -- and it only has to get as far as
    resolution to prove the point, so this stays CPU-only.
    """

    run_dir = tmp_path / "run"
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps({
        "schema": PLAN_SCHEMA, "name": "talker", "route": "experiment",
        "config": {"intent": {"point": "39,-98", "source": "era5",
                              "cycle": "2024-05-03T12", "hours": 1,
                              "vram_gib": 24}},
        "output_root": str(run_dir),
    }), encoding="utf-8")

    result = _cli(str(plan_path), cwd=tmp_path)

    # It fails -- the forcing was never fetched -- and that is fine:
    # what is under test is which channel each half went down.
    assert result.returncode == 1
    for line in result.stdout.splitlines():
        if line.strip():
            json.loads(line)   # every stdout line, without exception
    events = [json.loads(line) for line in result.stdout.splitlines()
              if line.strip()]
    assert events[-1]["event"] == "failed"
    # The wizard really did run and really did talk -- to stderr.
    assert "gpuwm domain" in result.stderr
    assert "sizing:" in result.stderr


@pytest.mark.parametrize("failure", [
    "ImportError('No module named cupy')",
    "RuntimeError('CUDA driver version is insufficient')",
    "OSError('cannot load nvrtc64_120_0.dll')",
])
def test_probe_works_on_a_box_whose_cupy_will_not_load(tmp_path, failure):
    """--probe exists to preflight an install, including a broken one.

    An ABSENT CuPy raises ImportError and was always survived.  An
    INSTALLED BUT UNLOADABLE one -- a cupy-cuda12x wheel on a CUDA-13
    box, a missing nvrtc DLL -- raises RuntimeError or OSError from
    inside the import, and gpuwm.cli reaches that import at module
    scope (cli -> downscale -> offline_child -> core.state).  So the
    whole command line died on exactly the installs --probe is for.

    A subprocess, because the guard runs once at import and cannot be
    re-armed in a live interpreter.
    """

    import os
    import subprocess
    import sys as _sys

    repo = str(Path(__file__).resolve().parents[1])
    script = tmp_path / "probe_without_cupy.py"
    script.write_text(
        "import sys\n"
        "class Blocker:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] in ('cupy', 'cupy_backends'):\n"
        f"            raise {failure}\n"
        "        return None\n"
        "sys.meta_path.insert(0, Blocker())\n"
        "from gpuwm.runplan import _module_entry\n"
        "sys.exit(_module_entry(['--probe', '--no-readiness']))\n",
        encoding="utf-8")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = repo + os.pathsep + environment.get(
        "PYTHONPATH", "")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [_sys.executable, str(script)], capture_output=True, text=True,
        cwd=str(tmp_path), env=environment, timeout=300)

    assert result.returncode == 0, result.stderr
    document = json.loads(result.stdout)
    assert document["schema"] == "gpuwm.run-plan.probe.v1"
    assert document["readiness"]["collected"] is False


def test_an_unloadable_cupy_still_fails_loudly_where_it_is_needed(tmp_path):
    """Deferred, not swallowed: the use site names the real cause."""

    import os
    import subprocess
    import sys as _sys

    repo = str(Path(__file__).resolve().parents[1])
    script = tmp_path / "require_cupy.py"
    script.write_text(
        "import sys\n"
        "class Blocker:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] in ('cupy', 'cupy_backends'):\n"
        "            raise RuntimeError('CUDA driver version is insufficient')\n"
        "        return None\n"
        "sys.meta_path.insert(0, Blocker())\n"
        "from gpuwm.core.state import _require_cupy\n"
        "try:\n"
        "    _require_cupy()\n"
        "except RuntimeError as error:\n"
        "    print(error)\n",
        encoding="utf-8")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = repo + os.pathsep + environment.get(
        "PYTHONPATH", "")
    result = subprocess.run(
        [_sys.executable, str(script)], capture_output=True, text=True,
        cwd=str(tmp_path), env=environment, timeout=300)

    assert result.returncode == 0, result.stderr
    # Installed-and-broken is a different problem from absent, with a
    # different fix, so it does not get the "install it" sentence alone.
    assert "CuPy IS installed here but failed to load" in result.stdout
    assert "CUDA driver version is insufficient" in result.stdout
    assert "gpuwm doctor" in result.stdout


def test_the_real_command_refuses_a_bad_plan_at_exit_2_in_one_layer(tmp_path):
    from gpuwm.explain import EXPLAIN_MARK

    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps({
        "schema": "gpuwm.run-plan.v9", "name": "x", "route": "experiment",
        "config": {"inline": "x = 1"}}), encoding="utf-8")

    result = _cli(str(plan_path), cwd=tmp_path)

    assert result.returncode == 2
    assert PLAN_SCHEMA in result.stderr
    # ONE layer reaches the terminal, and the sentinel never does.
    assert EXPLAIN_MARK.strip() not in result.stderr
    assert "--explain" in result.stderr
    assert result.stdout == ""


def test_the_real_command_exits_nonzero_with_failed_as_its_last_line(
        tmp_path):
    config = make_case_toml(tmp_path)
    # A declared input that is not there: it loads far enough to be
    # accepted, then fails inside resolution -- so the failure arrives
    # as an event on an already-open stream, which is the case a
    # consumer has to handle.
    (tmp_path / "Vtable.ERA5").unlink()
    run_dir = tmp_path / "run"

    result = _cli(str(_write_plan(tmp_path, config, run_dir)), cwd=tmp_path)

    assert result.returncode == 1
    events = [json.loads(line) for line in result.stdout.splitlines()
              if line.strip()]
    assert events[-1]["event"] == "failed"
    assert events[-1]["error_class"]
    assert events[-1]["message"]
    assert read_events(run_dir / EVENTS_FILENAME) == events
