"""Every remote command comes from a declared door row, and a resume keeps its route.

CPU-only: no forecast is run, no card is opened and no command is executed.
"""
import json
import sys
import tomllib
from datetime import datetime
from pathlib import Path

import pytest

from gpuwm import remote_plan as rp, remote_worker as rw


@pytest.fixture(autouse=True)
def cpu_only_memory_review(monkeypatch):
    monkeypatch.setattr("gpuwm.remote_plan.memory_review", lambda *_a, **_k:
        {"measured": False, "free_bytes": None, "refuse": False, "warn": True,
         "verdict": "CPU-only entry door contract test"})


def config(tmp_path, name="entry-doors"):
    from gpuwm import domain_wizard as dw
    path = tmp_path / "case.toml"
    path.write_text(dw.render_config(name=name, start_time=datetime(2026, 9, 5), hours=3,
        projection=dw._projection_entries(40, -100, "auto"), dims=dw._dims_for_scale(1, ()), ratios=(),
        fetch_hints=dict(source="gfs", cycle="2026-09-05T00", hours=3, out="data/cache", cadence=3),
        case_data=None), encoding="utf-8")
    return path


def restart(path, *, grid_id=1):
    import numpy as np
    header = {"format_version": 3, "grid_id": grid_id,
              "array_manifest": {"state/u": {"shape": [4], "dtype": "float32"}}}
    with path.open("wb") as stream:
        np.savez(stream, **{"state/u": np.arange(4, dtype=np.float32),
                            "__gpuwm_restart_header__": np.frombuffer(
                                json.dumps(header).encode(), dtype=np.uint8)})
    return path


# ----------------------------------------------------------------- the table

def test_every_command_comes_from_a_declared_door_row():
    assert set(rw.ENTRY_DOORS) == {"go", "run-plan"}
    argv = rw.compose_argv({"door": "go", "document": "/node/case.toml",
                            "flags": [("--outdir", "/node/out"), ("--products", "none")]})
    assert argv == [sys.executable, "-I", "-u", "-m", "gpuwm.cli", "go", "/node/case.toml",
                    "--outdir", "/node/out", "--products", "none"]
    assert rw.compose_argv({"door": "run-plan", "document": "/node/plan.json", "flags": []}) == [
        sys.executable, "-I", "-u", "-m", "gpuwm.cli", "run-plan", "/node/plan.json"]


def test_remote_go_door_carries_the_section_line():
    argv = rw.compose_argv({"door": "go", "document": "/node/config.toml",
                            "flags": [("--products", "xsec:wa"),
                                      ("--section", "40,-100,41,-99")]})
    assert argv[-4:] == ["--products", "xsec:wa", "--section", "40,-100,41,-99"]


def test_remote_go_door_passes_a_signed_line_as_one_argument():
    from gpuwm.cli import build_parser
    section = "-40,100,-41,99"
    argv = rw.compose_argv({"door": "go", "document": "/node/config.toml",
                            "flags": [("--products", "xsec:wa"), ("--section", section)]})
    parsed = build_parser().parse_args(argv[5:])
    assert parsed.render_section == section


def test_remote_go_review_carries_and_records_the_section_line(tmp_path, monkeypatch):
    source = config(tmp_path)
    request = {"schema": "gpuwm.remote.request.v1", "action": "start", "workspace": str(tmp_path),
               "config": str(source), "outdir": str(tmp_path / "new output"),
               "products": "xsec:wa", "section": "40,-100,41,-99", "dry_run": True}
    monkeypatch.setattr(rw, "_ownership_provider", lambda: {})
    review = rw.dispatch(request)["review"]
    assert review["section"] == request["section"]
    assert review["argv"][-2:] == ["--section", request["section"]]
    assert not (tmp_path / "new output").exists()


@pytest.mark.parametrize("section", [None, "", "40,-100,40,-100", "91,0,45,1"])
def test_remote_go_review_refuses_sections_it_cannot_draw(tmp_path, section):
    source = config(tmp_path)
    request = {"action": "start", "config": str(source), "outdir": str(tmp_path / "new output"),
               "products": "xsec:wa", "section": section}
    with pytest.raises(ValueError) as failure:
        rw._review(request, tmp_path)
    assert "after the whole forecast" in str(failure.value)
    assert "--section" in str(failure.value)
    assert not (tmp_path / "new output").exists()


def test_remote_go_section_file_is_resolved_against_the_config(tmp_path):
    source = config(tmp_path)
    line = tmp_path / "line.json"
    line.write_text('{"start":[40,-100],"end":[41,-99]}', encoding="utf-8")
    review, _sources, _snapshots = rw._review({"action": "start", "config": str(source),
        "outdir": str(tmp_path / "new output"), "products": "xsec:wa", "section": "line.json"}, tmp_path)
    assert review["section"] == str(line)
    assert review["argv"][-2:] == ["--section", str(line)]


def test_an_unregistered_door_is_refused_by_name_with_the_registered_set():
    with pytest.raises(ValueError) as failure:
        rw.compose_argv({"door": "nope", "document": "/node/x", "flags": []})
    message = str(failure.value)
    assert "nope" in message and "go" in message and "run-plan" in message


def test_a_switch_carries_no_value_and_an_option_carries_one():
    """The go door's advisory sizing switch is a bare word in the composed command."""
    argv = rw.compose_argv({"door": "go", "document": "/node/case.toml",
                            "flags": [("--outdir", "/node/out"), ("--no-memory-gate", None)]})
    assert argv[-3:] == ["--outdir", "/node/out", "--no-memory-gate"]


def test_a_door_refuses_an_option_it_does_not_carry_by_name():
    with pytest.raises(ValueError) as failure:
        rw.compose_argv({"door": "run-plan", "document": "/node/plan.json",
                         "flags": [("--products", "t2")]})
    message = str(failure.value)
    assert "--products" in message and "run-plan" in message


def test_the_captured_document_is_substituted_by_slot_not_by_index(tmp_path, monkeypatch):
    """A door whose argv grew a leading flag would break an index-based patch."""
    entry = {"door": "go", "document": "/node/case.toml",
             "flags": [("--outdir", "/node/out"), ("--no-memory-gate", None), ("--products", "t2")]}
    argv = rw.compose_argv({**entry, "document": "/captured/inputs/case.toml"})
    assert argv[6] == "/captured/inputs/case.toml"
    assert argv[-2:] == ["--products", "t2"]


# --------------------------------------------- the node-configuration route

def test_the_configuration_route_claims_a_run_folder_instead_of_the_workaround(tmp_path):
    """C-224: the run folder is this run's own, recorded, and not `--run-stamp off`."""
    from gpuwm.run_stamp import is_run_folder
    source = config(tmp_path)
    request = {"schema": "gpuwm.remote.request.v1", "action": "start", "workspace": str(tmp_path),
               "config": str(source), "outdir": str(tmp_path / "new output"), "products": "none"}
    review, _sources, _snapshots = rw._review(request, tmp_path)
    assert review["entry"]["door"] == "go"
    run_root = Path(review["run_root"])
    assert run_root.parent == tmp_path / "new output" and is_run_folder(run_root)
    assert review["argv"] == [sys.executable, "-I", "-u", "-m", "gpuwm.cli", "go", str(source),
                              "--outdir", str(run_root), "--no-memory-gate", "--products", "none"]
    assert "--run-stamp" not in review["argv"]
    assert review["argv"] == rw.compose_argv(review["entry"])


def test_a_second_run_into_one_case_folder_claims_a_sibling_and_the_pointer_names_it(tmp_path):
    """C-224: the case folder collects runs, exactly as the local front door lays them."""
    from gpuwm import run_stamp
    source = config(tmp_path)
    case_folder = tmp_path / "case runs"
    stub = [sys.executable, "-c", "pass"]
    request = {"schema": "gpuwm.remote.request.v1", "action": "start", "workspace": str(tmp_path),
               "config": str(source), "outdir": str(case_folder), "products": "none"}
    first = rw._launch(request, tmp_path, worker_command=stub)["job"]
    assert Path(first["run_root"]).parent == case_folder and run_stamp.is_run_folder(first["run_root"])
    assert run_stamp.latest(case_folder) == Path(first["run_root"])
    # The case folder now exists, and a second run of the same configuration
    # is a sibling beside the first rather than a refusal against it.
    second = rw._launch({**request, "products": "t2"}, tmp_path, worker_command=stub)["job"]
    assert Path(second["run_root"]).parent == case_folder
    assert second["run_root"] != first["run_root"]
    assert run_stamp.latest(case_folder) == Path(second["run_root"])
    assert {path.name for path in case_folder.iterdir() if run_stamp.is_run_folder(path)} >= {
        Path(first["run_root"]).name, Path(second["run_root"]).name}


def test_a_named_run_folder_that_already_exists_is_refused_naming_it(tmp_path):
    source = config(tmp_path)
    named = tmp_path / "runs" / "run-20260907-180000Z"
    named.mkdir(parents=True)
    request = {"schema": "gpuwm.remote.request.v1", "action": "start", "workspace": str(tmp_path),
               "config": str(source), "outdir": str(named)}
    with pytest.raises(ValueError) as failure:
        rw._review(request, tmp_path)
    message = str(failure.value)
    assert str(named) in message and "already exists" in message
    assert "directory above it" in message


def test_an_output_path_that_is_a_file_is_refused_naming_it(tmp_path):
    source = config(tmp_path)
    blocker = tmp_path / "a file"
    blocker.write_text("not a directory")
    request = {"schema": "gpuwm.remote.request.v1", "action": "start", "workspace": str(tmp_path),
               "config": str(source), "outdir": str(blocker)}
    with pytest.raises(ValueError, match="is not a directory"):
        rw._review(request, tmp_path)


def test_a_resume_may_claim_a_sibling_beside_its_parents_run_folder(staged_parent):
    """The refusal is about the parent's own run tree, not the case folder above it."""
    tmp_path, source, plan, output, checkpoint, old = staged_parent
    old["run_root"] = str(output)
    request = {"schema": "gpuwm.remote.request.v1", "action": "resume", "workspace": str(tmp_path),
               "job": "old-job", "outdir": str(output.parent), "dry_run": True}
    review = rw._launch(request, tmp_path)["review"]
    assert Path(review["run_root"]).parent == output.parent
    with pytest.raises(ValueError, match="outside the source job's own run folder"):
        rw._launch({**request, "outdir": str(output)}, tmp_path)


def test_an_output_directory_that_already_names_a_run_folder_is_honoured(tmp_path):
    source = config(tmp_path)
    named = tmp_path / "runs" / "run-20260907-180000Z"
    named.parent.mkdir()
    request = {"schema": "gpuwm.remote.request.v1", "action": "start", "workspace": str(tmp_path),
               "config": str(source), "outdir": str(named)}
    review, _sources, _snapshots = rw._review(request, tmp_path)
    assert review["run_root"] == str(named) == review["outdir"]


# ------------------------------------------------- resuming a staged map plan

@pytest.fixture
def staged_parent(tmp_path, monkeypatch):
    source = config(tmp_path)
    output = tmp_path / "old output"
    output.mkdir()
    checkpoint = restart(output / "gpuwmrst_d01_2026-09-05_01_00_00.npz")
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"schema": "gpuwm.run-plan.v1", "name": "reviewed-map",
        "route": "experiment", "config": {"path": "case.toml"},
        "output_root": str(output), "run_options": {"render_products": "t2,wind10"}}), encoding="utf-8")
    old = {"id": "old-job", "action": "start-plan", "snapshot_config": str(source),
           "snapshot_plan": str(plan), "plan_sha256": "b" * 64, "outdir": str(output),
           "cwd": str(tmp_path), "products": None}
    monkeypatch.setattr(rw, "_directory", lambda workspace, job: tmp_path)
    monkeypatch.setattr(rw, "_record", lambda directory: old)
    monkeypatch.setattr(rw, "_status", lambda directory: {"state": "stopped"})
    return tmp_path, source, plan, output, checkpoint, old


def test_resume_of_a_staged_plan_job_keeps_the_run_plan_route(staged_parent):
    tmp_path, source, plan, output, checkpoint, _old = staged_parent
    request = {"schema": "gpuwm.remote.request.v1", "action": "resume", "workspace": str(tmp_path),
               "job": "old-job", "outdir": str(tmp_path / "new output"), "dry_run": True}
    review = rw._launch(request, tmp_path)["review"]
    assert review["argv"][4:6] == ["gpuwm.cli", "run-plan"]
    assert review["entry"]["door"] == "run-plan"
    assert review["route"] == "staged_plan"
    assert review["checkpoint"] == str(checkpoint)
    assert not (tmp_path / "new output").exists()
    assert list(output.iterdir()) == [checkpoint]


def test_the_resumed_plan_points_at_this_resume_output_and_checkpoint(staged_parent, monkeypatch):
    tmp_path, source, plan, output, checkpoint, _old = staged_parent
    request = {"schema": "gpuwm.remote.request.v1", "action": "resume", "workspace": str(tmp_path),
               "job": "old-job", "outdir": str(tmp_path / "new output")}
    review, _sources, snapshots = rw._review(request, tmp_path)
    assert "plan.json" in snapshots
    resumed = json.loads(snapshots["plan.json"])
    # The resumed plan is pointed at the folder this resume's run claims, which
    # is the folder the job records and every reader of its artifacts resolves.
    assert resumed["output_root"] == review["run_root"]
    assert Path(review["run_root"]).parent == tmp_path / "new output"
    assert resumed["run_options"]["restart"] == str(checkpoint)
    assert review["plan"].endswith("plan.json")
    assert review["plan_sha256"] == rw._sha(snapshots["plan.json"])


def test_resume_of_a_staged_plan_job_keeps_its_render_selection(staged_parent):
    tmp_path, source, plan, output, checkpoint, _old = staged_parent
    request = {"schema": "gpuwm.remote.request.v1", "action": "resume", "workspace": str(tmp_path),
               "job": "old-job", "outdir": str(tmp_path / "new output")}
    review, _sources, snapshots = rw._review(request, tmp_path)
    assert json.loads(snapshots["plan.json"])["run_options"]["render_products"] == "t2,wind10"
    assert "--products" not in review["argv"]


@pytest.mark.parametrize("override", [None, "41,-101,42,-100"])
def test_remote_resume_plan_keeps_or_overrides_the_section_line(staged_parent, override):
    tmp_path, _source, plan, _output, _checkpoint, _old = staged_parent
    document = json.loads(plan.read_text(encoding="utf-8"))
    document["run_options"].update(render_products="xsec:wa", render_section="40,-100,41,-99")
    plan.write_text(json.dumps(document), encoding="utf-8")
    request = {"action": "resume", "job": "old-job", "outdir": str(tmp_path / "new output")}
    if override is not None:
        request["section"] = override
    review, _sources, snapshots = rw._review(request, tmp_path)
    expected = override or "40,-100,41,-99"
    assert json.loads(snapshots["plan.json"])["run_options"]["render_section"] == expected
    assert review["section"] == expected
    assert "--section" not in review["argv"]


def test_a_parent_plan_without_a_render_selection_does_not_gain_one(staged_parent):
    tmp_path, source, plan, output, checkpoint, _old = staged_parent
    plan.write_text(json.dumps({"schema": "gpuwm.run-plan.v1", "name": "reviewed-map",
        "route": "experiment", "config": {"path": "case.toml"},
        "output_root": str(output), "run_options": {}}), encoding="utf-8")
    request = {"schema": "gpuwm.remote.request.v1", "action": "resume", "workspace": str(tmp_path),
               "job": "old-job", "outdir": str(tmp_path / "new output")}
    review, _sources, snapshots = rw._review(request, tmp_path)
    assert "render_products" not in json.loads(snapshots["plan.json"])["run_options"]
    assert review["products"] is None
    assert "--products" not in review["argv"]


@pytest.mark.parametrize("section", [None, "40,-100,41,-99"])
def test_a_configuration_parent_still_resumes_through_the_go_door(tmp_path, monkeypatch, section):
    from gpuwm.toml_document import emit_experiment_toml
    source = config(tmp_path)
    raw = tomllib.loads(source.read_text(encoding="utf-8"))
    raw["case_data"] = {"geog_root": "geography", "forcing": ["forcing.nc"], "vtable": "Vtable",
                        "wps_namelist": "inputs.wps", "sfcp_to_sfcp": True, "output_title": "fixture"}
    (tmp_path / "geography").mkdir()
    for name in ("forcing.nc", "Vtable", "inputs.wps"):
        (tmp_path / name).write_text("fixture")
    source.write_text(emit_experiment_toml(raw), encoding="utf-8")
    output = tmp_path / "old output"
    output.mkdir()
    checkpoint = restart(output / "gpuwmrst_d01_2026-09-05_01_00_00.npz")
    old = {"id": "old-job", "action": "start", "snapshot_config": str(source), "snapshot_plan": None,
           "outdir": str(output), "cwd": str(tmp_path), "products": "t2"}
    if section is not None:
        old.update(products="xsec:wa", section=section)
    monkeypatch.setattr(rw, "_directory", lambda workspace, job: tmp_path)
    monkeypatch.setattr(rw, "_record", lambda directory: old)
    monkeypatch.setattr(rw, "_status", lambda directory: {"state": "stopped"})
    request = {"schema": "gpuwm.remote.request.v1", "action": "resume", "workspace": str(tmp_path),
               "job": "old-job", "outdir": str(tmp_path / "new output"), "dry_run": True}
    review = rw._launch(request, tmp_path)["review"]
    assert review["entry"]["door"] == "go"
    assert review["argv"][-2:] == ["--restart", str(checkpoint)]
    assert review["products"] == old["products"]
    if section is not None:
        assert review["section"] == section
        index = review["argv"].index("--section")
        assert review["argv"][index + 1] == section
    assert review["route"] == "node_config"


def test_a_prepared_route_plan_with_no_recorded_bundle_refuses_at_review(staged_parent):
    tmp_path, source, plan, output, checkpoint, _old = staged_parent
    plan.write_text(json.dumps({"schema": "gpuwm.run-plan.v1", "name": "reviewed-map",
        "route": "prepared", "config": {"path": "case.toml"},
        "output_root": str(output), "run_options": {"render_products": "none"}}), encoding="utf-8")
    request = {"schema": "gpuwm.remote.request.v1", "action": "resume", "workspace": str(tmp_path),
               "job": "old-job", "outdir": str(tmp_path / "new output"), "dry_run": True}
    with pytest.raises(ValueError) as failure:
        rw._launch(request, tmp_path)
    message = str(failure.value)
    assert "prepared bundle that wrote it" in message
    assert "Start a new run from that bundle" in message
    assert not (tmp_path / "new output").exists()


def test_the_staged_plan_door_takes_its_command_and_selection_from_the_review(tmp_path, monkeypatch):
    """launch() composes from exactly the record review() returned."""
    directory = tmp_path / "bundle"
    directory.mkdir()
    entry = {"door": "run-plan", "document": str(directory / "plan.json"), "flags": []}
    review = {"entry": entry, "render_products": "xsec:wa", "render_section": "40,-100,41,-99",
              "memory": {"measured": True, "refuse": False},
              "plan_sha256": "a", "config_sha256": "b", "input_sha256": "c", "geog_root": None}
    monkeypatch.setattr(rp, "review", lambda *_: (dict(review), {"files": [], "geog_root": None}, directory))
    observed = []
    monkeypatch.setattr(rw, "_launch_review", lambda *args: observed.append(args) or {"job": {"id": "x"}})
    rp.launch({"expected_plan_sha256": "a", "expected_config_sha256": "b",
               "expected_input_sha256": "c"}, tmp_path)
    launched = observed[0][2]
    assert launched["argv"] == rw.compose_argv(entry)
    assert launched["argv"][4:6] == ["gpuwm.cli", "run-plan"]
    # The record carries the plan's own render selection instead of a None the
    # status door and a later resume would both read as "no selection".
    assert launched["products"] == "xsec:wa"
    assert launched["section"] == "40,-100,41,-99"


@pytest.fixture
def two_cards(monkeypatch):
    """A node with two cards, so a selection can be told apart from a default."""
    probe = {"devices": [{"index": 0, "uuid": "GPU-aaaa", "name": "card A", "memory_free_bytes": 8_000_000_000},
                         {"index": 1, "uuid": "GPU-bbbb", "name": "card B", "memory_free_bytes": 24_000_000_000}],
             "device_query_basis": "NVML via nvidia-smi fixture"}
    monkeypatch.setattr(rp, "node_device_probe", lambda *_a, **_k: probe)
    return probe


def test_a_resumed_plan_relocates_this_resumes_run_options_into_the_plan(staged_parent, tmp_path, two_cards):
    """C-275: a named input reaches the run instead of being validated and dropped."""
    tmp_path, source, plan, output, checkpoint, _old = staged_parent
    (tmp_path / "node geography").mkdir()
    request = {"schema": "gpuwm.remote.request.v1", "action": "resume", "workspace": str(tmp_path),
               "job": "old-job", "outdir": str(tmp_path / "new output"),
               "products": "sbcape", "device": "1", "geog_root": str(tmp_path / "node geography")}
    review, _sources, snapshots = rw._review(request, tmp_path)
    options = json.loads(snapshots["plan.json"])["run_options"]
    assert options["render_products"] == "sbcape"
    assert options["device"] == "1"
    assert options["geog_root"] == str(tmp_path / "node geography")
    # The run-plan door carries no such flags, so nothing is passed twice.
    assert review["argv"] == rw.compose_argv(review["entry"])
    assert "--products" not in review["argv"] and "--device" not in review["argv"]
    # What the run draws is what this job's own watchers prepare and draw.
    assert review["products"] == "sbcape"
    # C-288: the envelope is priced against the card this run selected, not
    # against the smallest of a set it will not run on.
    assert review["device"] == "1" and review["memory"]["device"]["index"] == 1
    assert review["memory"]["priced_free_bytes"] == 24_000_000_000
    assert "the card this run selected" in review["memory"]["priced_note"]


def test_a_card_this_node_does_not_have_is_refused_naming_the_cards_it_has(tmp_path, two_cards):
    source = config(tmp_path)
    request = {"schema": "gpuwm.remote.request.v1", "action": "start", "workspace": str(tmp_path),
               "config": str(source), "outdir": str(tmp_path / "new output"), "device": "7"}
    with pytest.raises(ValueError) as failure:
        rw._review(request, tmp_path)
    message = str(failure.value)
    assert "no card '7'" in message and "GPU-bbbb" in message
    assert not (tmp_path / "new output").exists()


def test_the_selected_card_is_exported_to_the_forecast_child(tmp_path, two_cards, monkeypatch):
    """C-303: a selector that reaches the record reaches the run's environment."""
    source = config(tmp_path)
    request = {"schema": "gpuwm.remote.request.v1", "action": "start", "workspace": str(tmp_path),
               "config": str(source), "outdir": str(tmp_path / "new output"), "device": "GPU-bbbb"}
    review, _sources, _snapshots = rw._review(request, tmp_path)
    assert review["device"] == "GPU-bbbb"
    seen = {}
    class Child:
        pid = 4321
        def poll(self): return 0
        def wait(self, timeout=None): return 0
    def popen(argv, **kwargs):
        seen.update(kwargs)
        return Child()
    directory = tmp_path / "job"
    directory.mkdir()
    record = {"schema": "gpuwm.remote.job.v1", "id": directory.name, "token": "a" * 64,
              "argv": [sys.executable, "-c", "pass"], "cwd": str(tmp_path), "device": "GPU-bbbb",
              "snapshot_inputs": {}, "outdir": str(tmp_path / "new output")}
    rw._write(directory / "job.json", record)
    monkeypatch.setattr(rw, "_record", lambda *_: record)
    monkeypatch.setattr(rw, "_process", lambda *_a, **_k: {"pid": 1, "started_at": 1})
    monkeypatch.setattr(rw, "_owned_processes", lambda *_a, **_k: [])
    monkeypatch.setattr(rw, "_has_token", lambda *_a, **_k: True)
    monkeypatch.setattr(rw.subprocess, "Popen", popen)
    monkeypatch.setenv(rw.TOKEN_ENV, record["token"])
    rw.run_worker(directory, record["token"])
    assert seen["env"]["CUDA_VISIBLE_DEVICES"] == "GPU-bbbb"
    assert seen["env"][rw.TOKEN_ENV] == record["token"]


def test_a_resume_option_the_plans_route_does_not_carry_names_that_route(staged_parent, tmp_path):
    """C-268: the refusal names the route and its own options, at review."""
    tmp_path, source, plan, output, checkpoint, _old = staged_parent
    from gpuwm import stage_cli
    prepared = tmp_path / "prepared bundle"
    prepared.mkdir()
    # The public receipt relay only; no cache preflight and no forecast here.
    schema = next(name for name, entry in stage_cli._schema_index().items() if entry["layout"] == "tree")
    (prepared / "receipt.json").write_text(json.dumps({"schema": schema}), encoding="utf-8")
    request = {"schema": "gpuwm.remote.request.v1", "action": "resume", "workspace": str(tmp_path),
               "job": "old-job", "outdir": str(tmp_path / "new output"), "prepared_root": str(prepared)}
    with pytest.raises(ValueError) as failure:
        rw._review(request, tmp_path)
    message = str(failure.value)
    assert "'experiment' route" in message and "render_products" in message
    assert not (tmp_path / "new output").exists()


def test_a_staged_plan_job_records_the_wps_authority_the_plan_binds_not_a_phantom_capture(tmp_path, monkeypatch):
    """The run-plan door captures no prepared-inputs namelist, so its record must not name one."""
    directory = tmp_path / "bundle"
    directory.mkdir()
    (directory / "plan.json").write_text("{}", encoding="utf-8")
    (directory / "case.toml").write_text("[experiment]\n", encoding="utf-8")
    node_wps = tmp_path / "node prepared" / "namelist.wps"
    node_wps.parent.mkdir()
    node_wps.write_text("&share\n/\n", encoding="utf-8")
    review = {"entry": {"door": "run-plan", "document": str(directory / "plan.json"), "flags": []},
              "argv": rw.compose_argv({"door": "run-plan", "document": str(directory / "plan.json"), "flags": []}),
              "plan": str(directory / "plan.json"), "config": str(directory / "case.toml"),
              "outdir": str(tmp_path / "out"), "run_root": str(tmp_path / "out"), "cwd": str(directory),
              "geog_root": None, "products": None, "checkpoint": None,
              "prepared_root": str(node_wps.parent), "wps_namelist": str(node_wps), "parent_job": None,
              "runtime": {"version": "fixture"}, "memory": {"advisory": True}}
    snapshots = {"plan.json": b"{}", "case.toml": b"[experiment]\n"}
    recorded = []
    monkeypatch.setattr(rw.subprocess, "Popen", lambda *a, **k: recorded.append(a) or type(
        "P", (), {"poll": lambda self: 0, "returncode": 0})())
    rw._launch_review({"action": "start-plan"}, tmp_path, review, {}, snapshots,
                      worker_command=[sys.executable, "-c", "pass"])
    record = rw._json(next((tmp_path / ".arwen-jobs").glob("*/job.json")))
    assert record["snapshot_wps_namelist"] == str(node_wps)
    assert Path(record["snapshot_wps_namelist"]).is_file()
    assert not (Path(record["snapshot_config"]).parent / "prepared-inputs").exists()
def test_a_retried_launch_request_reconciles_with_the_job_it_already_created(tmp_path):
    """C-287: a retry of one attempt is that attempt, not a second forecast."""
    source = config(tmp_path)
    request = {"schema": "gpuwm.remote.request.v1", "action": "start", "workspace": str(tmp_path),
               "config": str(source), "outdir": str(tmp_path / "new output"), "products": "none",
               "request_id": "a1" * 16}
    stub = [sys.executable, "-c", "pass"]
    first = rw._launch(request, tmp_path, worker_command=stub)
    again = rw._launch(request, tmp_path, worker_command=stub)
    assert again["job"]["id"] == first["job"]["id"]
    assert again["reconciled_request_id"] == request["request_id"]
    jobs = [path for path in (tmp_path / ".arwen-jobs").iterdir() if rw.JOB_ID.fullmatch(path.name)]
    assert len(jobs) == 1
    # The attempt lives on the job record itself; nothing else was written
    # into the durable store, so a listing reads exactly the jobs that exist.
    assert {path.name for path in (tmp_path / ".arwen-jobs").iterdir()} == {jobs[0].name}
    assert rw._record(jobs[0])["request_id"] == request["request_id"]
    assert rw.dispatch({"schema": "gpuwm.remote.request.v1", "action": "list",
                        "workspace": str(tmp_path)})["jobs"] == [rw._status(jobs[0])]
    # A different attempt is a different job: a sibling run beside the first.
    other = rw._launch({**request, "request_id": "b2" * 16}, tmp_path, worker_command=stub)
    assert other["job"]["id"] != first["job"]["id"] and "reconciled_request_id" not in other
    assert Path(other["job"]["run_root"]).parent == Path(first["job"]["run_root"]).parent


def test_a_launch_request_identity_must_be_the_clients_own_32_characters(tmp_path):
    with pytest.raises(ValueError, match="32 hexadecimal characters"):
        rw.reconciled_job({"request_id": "../escape"}, tmp_path)


def test_a_launch_request_without_an_attempt_is_never_reconciled(tmp_path):
    assert rw.reconciled_job({"action": "start"}, tmp_path) is None
    assert rw.reconciled_job({"request_id": "c3" * 16}, tmp_path) is None, "no store, no job"
