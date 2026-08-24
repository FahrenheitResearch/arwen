"""Nested HRRR on run-plan's prepared route.

HRRR's tree is not GFS's with a different source.  rw-wps is not on the
path at all: the root preparation is ``tools.prepare_hrrr_wrf``, a THIRD
stage (``gpuwm.hrrr_hierarchy_direct``) builds d02..dNN from the sealed
d01, and only then does the same tree runner the GFS route already
drives take over.

Two things about the relay are worth stating because they are where
this could have gone wrong:

* the hierarchy writes ``receipt.json`` where rw-wps writes
  ``proof.json``, and the document resolver matches on SCHEMA rather
  than filename order, so it found the new one unchanged;
* the tool prints ``preparation_receipt_sha256`` on stdout, and it is
  not read -- the tool computes it as the sha256 of ``receipt.json``'s
  bytes, so hashing the artifact gives the identical digest without
  making a printed line load-bearing.

CPU-only throughout: the stages are captured at the ``run_stage`` seam
and the runner at its ``observer`` seam.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
from pathlib import Path

import pytest

import gpuwm.go_cli as go_cli
import gpuwm.runplan as runplan_module
from gpuwm.runplan import (PLAN_SCHEMA, generate_intent_config, load_plan,
                           resolve_plan, run_plan_main)

_HRRR_TREE = {"point": "35.2,-97.4", "source": "hrrr", "ladder": "12-3",
              "cycle": "2024-05-03T12", "hours": 6, "vram_gib": 24}


def _plan(tmp_path, **intent):
    geog = tmp_path / "GEOG"
    geog.mkdir(exist_ok=True)
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({
        "schema": PLAN_SCHEMA, "name": "hrrr-tree", "route": "prepared",
        "config": {"intent": {**_HRRR_TREE, **intent}},
        "run_options": {"geog_root": str(geog)},
        "output_root": str(tmp_path / "run")}), encoding="utf-8")
    return load_plan(path)


class _Observer:
    """Swallows the observer protocol; the chain's argv is the subject."""

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def _follow_block(exp) -> str:
    """A one-move itinerary on d02, at a whole number of root steps."""

    return f"""
[relocation]
enabled = true
grid_id = 2

[[relocation.move]]
at_seconds = {float(exp.root.run.dt) * 2:.1f}
di_parent_cells = 1
dj_parent_cells = 0
"""


def _drive(tmp_path, monkeypatch, follow=False, **intent):
    """Run the chain with every stage captured and nothing executed."""

    plan = _plan(tmp_path, **intent)
    with contextlib.redirect_stdout(io.StringIO()):
        config, _ = generate_intent_config(plan, destination=plan.run_dir)
        exp = resolve_plan(plan, require_inputs=False)[1]
    if follow:
        # Grown on the config the wizard just emitted, so the four HRRR
        # route inputs beside it stay the ones this chain reads.
        from gpuwm.experiment import load_experiment

        config.write_text(
            config.read_text(encoding="utf-8") + _follow_block(exp),
            encoding="utf-8")
        exp = load_experiment(config)

    staged: list[tuple[str, list[str]]] = []
    tree_root = plan.run_dir / "chain" / "hrrr-hierarchy"

    def run_stage(label, command, **kwargs):
        staged.append((label, [str(part) for part in command]))
        if label == "hierarchy":
            tree_root.mkdir(parents=True, exist_ok=True)
            (tree_root / "receipt.json").write_text(
                json.dumps({
                    "schema": "gpuwm-native-hrrr-hierarchy-direct-v1",
                    "status": "PASS"}, indent=2, sort_keys=True) + "\n",
                encoding="utf-8")

    def fake_fetch(arguments, run_dir, **_kwargs):
        out = Path(arguments[arguments.index("--out") + 1])
        out.mkdir(parents=True, exist_ok=True)
        (out / "SHA256SUMS").write_text("x", encoding="utf-8")
        staged.append(("fetch", [str(a) for a in arguments]))
        return {}

    captured: dict = {}
    import gpuwm.prepared_domain_tree_forecast as tree

    monkeypatch.setattr(go_cli, "run_stage", run_stage)
    monkeypatch.setattr(runplan_module, "_run_fetch", fake_fetch)
    monkeypatch.setattr(
        tree, "main",
        lambda argv, *, observer=None: captured.update(argv=list(argv)) or 0)
    monkeypatch.setattr(runplan_module, "_chain_render",
                        lambda plan, **kwargs: {"ok": True})

    with contextlib.redirect_stdout(io.StringIO()):
        runplan_module._hrrr_chain(
            plan, config_path=config, exp=exp, observer=_Observer(),
            run_dir=plan.run_dir)
    return staged, captured, tree_root, config


def _stage(staged, label):
    return next(cmd for name, cmd in staged if name == label)


# ---------------------------------------------------------------------------
# The chain
# ---------------------------------------------------------------------------


def test_a_nested_hrrr_plan_is_no_longer_refused(tmp_path, monkeypatch):
    """The refusal this replaces named the chain it could not drive."""

    staged, captured, _, _ = _drive(tmp_path, monkeypatch)
    assert [name for name, _ in staged] == ["fetch", "prepare", "hierarchy"]
    assert captured["argv"]


def test_the_four_stages_run_in_the_documented_order(tmp_path, monkeypatch):
    staged, captured, _, _ = _drive(tmp_path, monkeypatch)

    assert _stage(staged, "prepare")[2] == "tools.prepare_hrrr_wrf"
    assert _stage(staged, "hierarchy")[2] == "gpuwm.hrrr_hierarchy_direct"
    # rw-wps is not on this path at all.
    assert not any("source_cli" in " ".join(cmd) for _, cmd in staged)


def test_the_root_preparation_passes_the_wps_namelist(tmp_path, monkeypatch):
    """The wizard's printed chain omits it; the bundle needs it."""

    staged, _, _, _ = _drive(tmp_path, monkeypatch)
    prepare = _stage(staged, "prepare")
    assert "--wps-namelist" in prepare
    assert Path(prepare[prepare.index("--wps-namelist") + 1]).is_file()


def test_the_hierarchy_stage_passes_all_nine_required_flags(tmp_path,
                                                            monkeypatch):
    """Composed for this tool, not shared with its neighbours."""

    staged, _, _, _ = _drive(tmp_path, monkeypatch)
    hierarchy = _stage(staged, "hierarchy")
    for flag in ("--root-preparation", "--root-domain-spec",
                 "--wps-namelist", "--namelist-input",
                 "--stock-wrf-namelist-input", "--geog-root",
                 "--source-manifest", "--source-manifest-sha256",
                 "--output-root"):
        assert flag in hierarchy, flag

    # The flag rw-wps REJECTS must appear here and nowhere else, which
    # is the whole reason these two argv are not built by one helper.
    assert "--stock-wrf-namelist-input" not in _stage(staged, "prepare")


def test_the_source_manifest_digest_is_of_the_file_the_fetch_wrote(
        tmp_path, monkeypatch):
    staged, _, _, _ = _drive(tmp_path, monkeypatch)
    hierarchy = _stage(staged, "hierarchy")
    manifest = Path(hierarchy[hierarchy.index("--source-manifest") + 1])
    assert hierarchy[hierarchy.index("--source-manifest-sha256") + 1] == \
        hashlib.sha256(manifest.read_bytes()).hexdigest()


def test_the_cycle_is_spelled_the_way_each_tool_takes_it(tmp_path,
                                                         monkeypatch):
    """[fetch] carries YYYY-MM-DDTHH; both native tools take the stamp."""

    staged, _, _, _ = _drive(tmp_path, monkeypatch)
    assert _stage(staged, "fetch")[
        _stage(staged, "fetch").index("--cycle") + 1] == "2024-05-03T12"
    for label in ("prepare", "hierarchy"):
        command = _stage(staged, label)
        assert command[command.index("--cycle") + 1] == "2024-05-03_12:00:00"


def test_a_zero_lead_is_left_off_rather_than_passed_as_zero(tmp_path,
                                                            monkeypatch):
    """The stage raises on a negative lead and on lead-with-valid-time."""

    staged, _, _, _ = _drive(tmp_path, monkeypatch)
    assert "--forecast-start-hour" not in _stage(staged, "hierarchy")
    # And --valid-time, whose combination with a lead the tool refuses,
    # is never composed here at all.
    assert "--valid-time" not in _stage(staged, "hierarchy")


# ---------------------------------------------------------------------------
# The moving nest: the tenth flag, and where it goes
# ---------------------------------------------------------------------------


def test_a_follow_config_puts_the_corridor_flag_on_the_hierarchy_stage(
        tmp_path, monkeypatch):
    """The stage that builds the children is the stage that seals.

    On the GFS chain the corridor flag rides rw-wps.  There is no rw-wps
    here, and the root preparer knows only d01 -- it never reads a child
    geometry and would refuse the flag.  So it belongs to the hierarchy
    stage, which holds --geog-root and the child domains.
    """
    staged, _captured, _tree, _config = _drive(
        tmp_path, monkeypatch, follow=True)

    hierarchy = _stage(staged, "hierarchy")
    assert "--statics-corridor" in hierarchy
    # Bare: the preparation reads that as "every child domain", which is
    # what run-plan's --estimate priced for this plan.
    flag = hierarchy.index("--statics-corridor")
    assert flag == len(hierarchy) - 1 or hierarchy[flag + 1].startswith("--")
    # Not on the root preparation, and not on the fetch.
    assert "--statics-corridor" not in _stage(staged, "prepare")
    assert "--statics-corridor" not in _stage(staged, "fetch")


def test_a_still_nested_hrrr_plan_composes_exactly_what_it_always_did(
        tmp_path, monkeypatch):
    """Token for token, the two arms differ by one flag and nothing else.

    A corridor that leaked into an ordinary nested HRRR run would change
    what every existing plan prepares.
    """
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    still, _c, _t, _cfg = _drive(tmp_path / "a", monkeypatch)
    follow, _c2, _t2, _cfg2 = _drive(tmp_path / "b", monkeypatch,
                                     follow=True)

    def _shape(entries, root):
        return {name: [token.replace(str(root), "<ROOT>")
                       .replace("\\", "/") for token in command]
                for name, command in entries}

    still_shape = _shape(still, tmp_path / "a")
    follow_shape = _shape(follow, tmp_path / "b")
    assert set(still_shape) == set(follow_shape)
    for stage in ("fetch", "prepare"):
        assert still_shape[stage] == follow_shape[stage], stage
    assert (still_shape["hierarchy"]
            == [token for token in follow_shape["hierarchy"]
                if token != "--statics-corridor"])


def test_the_flag_the_chain_composes_is_the_one_the_stage_accepts(
        tmp_path, monkeypatch):
    """Composed argv, parsed by the real tool's own parser.

    A flag spelled correctly here and differently there is exactly the
    failure a composed chain cannot see until it runs -- and this chain
    runs the stage as a subprocess, so argparse's refusal would arrive
    after the fetch and the whole root preparation.
    """
    from gpuwm.hrrr_hierarchy_direct import _parser

    staged, _captured, _tree, _config = _drive(
        tmp_path, monkeypatch, follow=True)
    hierarchy = _stage(staged, "hierarchy")
    # Drop `python -m gpuwm.hrrr_hierarchy_direct`; parse the rest.
    parsed = _parser().parse_args(hierarchy[3:])
    assert parsed.statics_corridor == "all"


def test_the_relay_binds_the_hierarchy_receipt_off_its_bytes(tmp_path,
                                                             monkeypatch):
    staged, captured, tree_root, config = _drive(tmp_path, monkeypatch)
    argv = captured["argv"]

    assert Path(argv[argv.index("--prepared-root") + 1]) == tree_root
    assert argv[argv.index("--preparation-receipt-sha256") + 1] == \
        hashlib.sha256((tree_root / "receipt.json").read_bytes()).hexdigest()
    assert argv[argv.index("--experiment-config-sha256") + 1] == \
        hashlib.sha256(config.read_bytes()).hexdigest()
    # The single-domain binding has no place here.
    assert "--proof-sha256" not in argv
    assert "--prepared-content-sha256" not in argv


def test_the_document_resolver_finds_receipt_json_by_schema(tmp_path):
    """It matched proof.json for GFS; the schema table carries both."""

    root = tmp_path / "hier"
    root.mkdir()
    (root / "receipt.json").write_text(json.dumps({
        "schema": "gpuwm-native-hrrr-hierarchy-direct-v1",
        "status": "PASS"}), encoding="utf-8")
    assert go_cli._hierarchy_document(root).name == "receipt.json"


def test_the_hrrr_hierarchy_schema_is_the_runners_own(tmp_path):
    """Not restated here: read off the runner's table."""

    from gpuwm.prepared_domain_tree_forecast import (HIERARCHY_SCHEMA,
                                                     _HIERARCHY_DOCUMENTS)

    assert HIERARCHY_SCHEMA == "gpuwm-native-hrrr-hierarchy-direct-v1"
    entry = next(e for e in _HIERARCHY_DOCUMENTS
                 if e["filename"] == "receipt.json")
    assert entry["schema"] == HIERARCHY_SCHEMA
    assert entry["status"] == "PASS"


# ---------------------------------------------------------------------------
# Resolve and estimate
# ---------------------------------------------------------------------------


def test_resolve_and_estimate_work_on_a_nested_hrrr_plan(tmp_path, capsys):
    from gpuwm.cli import build_parser

    plan = _plan(tmp_path)
    path = tmp_path / "plan.json"

    assert run_plan_main(build_parser().parse_args(
        ["run-plan", "--resolve", str(path)])) == 0
    resolved = json.loads(capsys.readouterr().out)
    assert len(resolved["configuration"]["experiment"]["domains"]) == 2

    assert run_plan_main(build_parser().parse_args(
        ["run-plan", "--estimate", str(path)])) == 0
    estimate = json.loads(capsys.readouterr().out)
    assert estimate["vram"]["domains"] == 2
    assert len(estimate["disk"]["frames"]) == 2
    assert plan.route == "prepared"


def test_a_single_domain_hrrr_plan_still_takes_the_direct_bundle(
        tmp_path, monkeypatch):
    """The tree branch must not capture the route it shares a prefix with."""

    plan = _plan(tmp_path, ladder="12")
    with contextlib.redirect_stdout(io.StringIO()):
        config, _ = generate_intent_config(plan, destination=plan.run_dir)
        exp = resolve_plan(plan, require_inputs=False)[1]
    assert len(exp.domains) == 1

    staged: list[str] = []
    monkeypatch.setattr(
        go_cli, "run_stage",
        lambda label, command, **kwargs: staged.append(label))

    def fake_fetch(arguments, run_dir, **_kwargs):
        out = Path(arguments[arguments.index("--out") + 1])
        out.mkdir(parents=True, exist_ok=True)
        (out / "SHA256SUMS").write_text("x", encoding="utf-8")
        return {}

    monkeypatch.setattr(runplan_module, "_run_fetch", fake_fetch)
    with contextlib.redirect_stdout(io.StringIO()):
        with pytest.raises(runplan_module.PlanError) as refusal:
            runplan_module._hrrr_chain(
                plan, config_path=config, exp=exp, observer=_Observer(),
                run_dir=plan.run_dir)
    # It stops at the single-domain handoff, never at a hierarchy stage.
    assert "portable bundle" in str(refusal.value)
    assert "hierarchy" not in staged


# ---------------------------------------------------------------------------
# What the first live nested run found
# ---------------------------------------------------------------------------


def test_speed_x_arms_even_when_the_chain_pre_opens_the_forecast_stage(
        tmp_path):
    """A live nested run published 181 events with speed_x null.

    Every prepared chain calls enter_stage("forecast") itself before
    handing the runner over, so arming the wall/model baselines on the
    stage TRANSITION meant they never armed at all.
    """

    import time

    from gpuwm.runplan import EVENTS_FILENAME, EventStream, RunObserver, read_events

    with EventStream(tmp_path / EVENTS_FILENAME, mirror=None) as events:
        observer = RunObserver(events)
        observer.enter_stage("forecast")          # the chain, not the runner
        observer(model_elapsed_seconds=0.0, outer_step=0,
                 last_durable_wrfout=None, last_checkpoint=None,
                 phase="post-d01-sync", step_wall_seconds=0.0)
        time.sleep(0.02)
        observer(model_elapsed_seconds=600.0, outer_step=10,
                 last_durable_wrfout=None, last_checkpoint=None,
                 phase="post-d01-sync", step_wall_seconds=0.002)

    progress = [r for r in read_events(tmp_path / EVENTS_FILENAME)
                if r["event"] == "model_progress"]
    assert progress[-1]["wall_seconds"] > 0.0
    assert progress[-1]["speed_x"] is not None
    assert progress[-1]["speed_x"] > 0.0


def test_the_summary_reads_the_tree_runners_own_receipt(tmp_path):
    """The two runners leave different receipts.

    Reading only the single-domain pair made a COMPLETED nested run
    report completed_seconds 0.0 and status null -- and the heartbeat,
    fed from this summary, published `complete` with a model time of
    zero beside an outer_step of 180.
    """

    from gpuwm.runplan import _chain_summary

    chain = tmp_path / "chain"
    forecast = chain / "run"
    (forecast / "wrfout").mkdir(parents=True)
    for name in ("wrfout_d01_2026-08-08_06_00_00",
                 "wrfout_d02_2026-08-08_06_00_00"):
        (forecast / "wrfout" / name).write_bytes(b"f")
    (forecast / "certification-capsule.json").write_text(json.dumps({
        "schema": "gpuwm.certification-capsule/v1",
        "output": {"frames": [{"path": "a"}, {"path": "b"}]}}),
        encoding="utf-8")

    class _Seen:
        last_model_seconds = 10800.0

    summary = _chain_summary(chain, observer=_Seen())
    assert summary["completed_seconds"] == 10800.0
    assert summary["wrfout_count"] == 2
    assert summary["certification_capsule"] is not None
    # No verdict is invented from the absence of a failure.
    assert summary["status"] is None
    assert "not read as a verdict" in summary["status_basis"]


def test_the_summary_still_prefers_a_real_status_report(tmp_path):
    from gpuwm.runplan import _chain_summary

    chain = tmp_path / "chain"
    forecast = chain / "run"
    forecast.mkdir(parents=True)
    (forecast / "progress.json").write_text(json.dumps({
        "status": "PASS", "model_elapsed_seconds": 21600.0,
        "frame_count": 7}), encoding="utf-8")
    (forecast / "report.json").write_text(
        json.dumps({"status": "PASS"}), encoding="utf-8")

    summary = _chain_summary(chain, observer=None)
    assert summary["status"] == "PASS"
    assert summary["nan_free"] is True
    assert summary["completed_seconds"] == 21600.0
    assert summary["status_basis"] == "the runner's own report"
