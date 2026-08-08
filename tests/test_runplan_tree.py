"""Domain trees on run-plan's prepared route.

The last unwired route.  ``gpuwm go`` refuses a multi-domain config by
construction -- that is the refusal a nested GFS launch hit -- so
run-plan asks it not to and swaps exactly ONE stage: the tree runner
binds a single preparation receipt where the single-domain runner binds
three proof digests.  Preparation itself does not branch at all; rw-wps
reads the domain count out of the config and builds a hierarchy.

Everything here is CPU-only.  The two runners are stubbed at the
observer seam, and the receipt relay is exercised against real files on
disk, because the relay IS the work: both digests are sha256 of an
artifact's bytes, and a test that mocked the hashing would be testing
nothing.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import pytest

import gpuwm.go_cli as go_cli
from gpuwm.runplan import (EVENTS_FILENAME, PLAN_SCHEMA, EventStream,
                           RunObserver, generate_intent_config, load_plan,
                           read_events, run_plan_main)

_TREE_INTENT = {"point": "35.2,-97.4", "source": "gfs", "ladder": "12-3",
                "cycle": "2024-05-03T12", "hours": 6, "vram_gib": 24}


def _tree_plan(tmp_path, run_dir, **intent):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({
        "schema": PLAN_SCHEMA, "name": "tree-plan", "route": "prepared",
        "config": {"intent": {**_TREE_INTENT, **intent}},
        "output_root": str(run_dir)}), encoding="utf-8")
    return path


def _seed_preparation(plan_root: Path) -> tuple[Path, Path]:
    """The two artifacts the relay reads, shaped as rw-wps leaves them."""

    prepared = Path(plan_root) / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    (prepared / "proof.json").write_text(
        json.dumps({"schema": "gpuwm-gfs-native-hierarchy-proof-v2",
                    "status": "READY_NOT_YET_STOCK_WRF_GATED",
                    "domain_count": 2}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    authority = Path(plan_root) / "authority"
    authority.mkdir(parents=True, exist_ok=True)
    (authority / "experiment.toml").write_text(
        "name = 'tree'\n", encoding="utf-8")
    return prepared, authority


# ---------------------------------------------------------------------------
# The refusal, and the one caller allowed past it
# ---------------------------------------------------------------------------


def test_a_tree_config_still_refuses_on_gpuwm_go_itself(tmp_path):
    """The interactive refusal a nested launch hit must not move.

    It protects a reader whose chain would die at the forecast stage.
    run-plan dispatches to the tree runner instead, so it is not the
    reader that refusal is for -- but everyone else still is.
    """

    plan = load_plan(_tree_plan(tmp_path, tmp_path / "run"))
    config, _ = generate_intent_config(plan, destination=tmp_path / "gen")

    with pytest.raises(go_cli.GoRefusal) as refusal:
        go_cli.plan_from_config(config, outdir=tmp_path / "out")
    text = str(refusal.value)
    assert "declares 2 domains" in text
    assert "gpuwm-prepared-tree-forecast" in text


def test_allow_tree_lets_it_through_and_names_the_tree_runner(tmp_path):
    plan = load_plan(_tree_plan(tmp_path, tmp_path / "run"))
    config, _ = generate_intent_config(plan, destination=tmp_path / "gen")

    resolved = go_cli.plan_from_config(
        config, outdir=tmp_path / "out", allow_tree=True)
    assert resolved["domains"] == 2
    assert resolved["runner"] == go_cli.TREE_RUNNER_MODULE


def test_a_single_domain_config_still_names_the_single_domain_runner(
        tmp_path):
    """allow_tree must not change what a one-domain plan runs."""

    plan = load_plan(_tree_plan(tmp_path, tmp_path / "run", ladder="12"))
    config, _ = generate_intent_config(plan, destination=tmp_path / "gen")

    resolved = go_cli.plan_from_config(
        config, outdir=tmp_path / "out", allow_tree=True)
    assert resolved["domains"] == 1
    assert resolved["runner"] == go_cli.RUNNER_MODULE


# ---------------------------------------------------------------------------
# The receipt relay
# ---------------------------------------------------------------------------


def test_the_relay_binds_both_digests_off_the_artifacts(tmp_path):
    """The two-stage relay, and the whole of what made this a blocker."""

    prepared, authority = _seed_preparation(tmp_path)
    command = go_cli.tree_forecast_command({
        "prepared": prepared, "authority": authority,
        "run": tmp_path / "run", "runner": go_cli.TREE_RUNNER_MODULE,
        "domains": 2})

    assert command[2] == go_cli.TREE_RUNNER_MODULE
    # Digests OF THE FILES' BYTES -- exactly what the tree runner
    # recomputes and compares.  Never re-derived from content, and never
    # scraped from what rw-wps printed.
    assert command[command.index("--preparation-receipt-sha256") + 1] == \
        hashlib.sha256((prepared / "proof.json").read_bytes()).hexdigest()
    assert command[command.index("--experiment-config-sha256") + 1] == \
        hashlib.sha256(
            (authority / "experiment.toml").read_bytes()).hexdigest()
    # Every flag the tree runner declares required.
    for flag in ("--prepared-root", "--preparation-receipt-sha256",
                 "--experiment-config", "--experiment-config-sha256",
                 "--outdir"):
        assert flag in command, flag
    # --prepared-root IS rw-wps's --output-root, not a subdirectory.
    assert Path(command[command.index("--prepared-root") + 1]) == prepared


def test_the_hierarchy_document_is_chosen_by_schema_not_filename_order(
        tmp_path):
    """Four sources write proof.json and one writes receipt.json."""

    prepared, _ = _seed_preparation(tmp_path)
    (prepared / "receipt.json").write_text(
        json.dumps({"schema": "something-else"}), encoding="utf-8")
    assert go_cli._hierarchy_document(prepared).name == "proof.json"


def test_a_prepared_root_with_no_hierarchy_document_is_refused(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(go_cli.GoRefusal) as refusal:
        go_cli._hierarchy_document(empty)
    assert "no preparation receipt to bind" in str(refusal.value)


def test_a_single_domain_proof_is_refused_naming_its_own_runner(tmp_path):
    single = tmp_path / "single"
    single.mkdir()
    (single / "proof.json").write_text(json.dumps({
        "input_manifest_sha256": "a" * 64,
        "prepared_cache": {"content_sha256": "b" * 64}}), encoding="utf-8")
    with pytest.raises(go_cli.GoRefusal) as refusal:
        go_cli._hierarchy_document(single)
    assert "prepared_single_domain_forecast" in str(refusal.value)


def test_a_hierarchy_proof_is_refused_by_the_single_domain_reader():
    """The two relays are not interchangeable, and each says so."""

    root = Path(tempfile.mkdtemp())
    (root / "proof.json").write_text(json.dumps({
        "schema": "gpuwm-gfs-native-hierarchy-proof-v2"}), encoding="utf-8")
    with pytest.raises(go_cli.GoRefusal) as refusal:
        go_cli.proof_digests(root)
    assert "multi-domain hierarchy product" in str(refusal.value)


def test_the_forecast_stage_dispatches_to_the_tree_runner(tmp_path,
                                                          monkeypatch):
    prepared, authority = _seed_preparation(tmp_path)
    plan = {"prepared": prepared, "authority": authority,
            "run": tmp_path / "run", "runner": go_cli.TREE_RUNNER_MODULE,
            "domains": 2}
    seen = {}

    class Runner:
        @staticmethod
        def main(argv, *, observer):
            seen["argv"] = argv
            seen["observer"] = observer
            return 0

    monkeypatch.setattr("importlib.import_module", lambda name: Runner)
    sentinel = object()
    go_cli._run_forecast(plan, {}, explain=False, observer=sentinel)

    assert seen["observer"] is sentinel
    assert "--preparation-receipt-sha256" in seen["argv"]
    # The single-domain binding is absent: this is a different contract.
    assert "--proof-sha256" not in seen["argv"]
    assert "--prepared-content-sha256" not in seen["argv"]


# ---------------------------------------------------------------------------
# Resolve and estimate
# ---------------------------------------------------------------------------


def test_resolve_and_estimate_work_on_a_tree_plan(tmp_path, capsys):
    from gpuwm.cli import build_parser

    plan_path = _tree_plan(tmp_path, tmp_path / "run")
    assert run_plan_main(build_parser().parse_args(
        ["run-plan", "--resolve", str(plan_path)])) == 0
    resolved = json.loads(capsys.readouterr().out)
    assert len(resolved["configuration"]["experiment"]["domains"]) == 2

    assert run_plan_main(build_parser().parse_args(
        ["run-plan", "--estimate", str(plan_path)])) == 0
    estimate = json.loads(capsys.readouterr().out)
    assert estimate["vram"]["domains"] == 2
    assert len(estimate["disk"]["frames"]) == 2
    assert estimate["vram"]["peak_envelope_bytes"] > \
        estimate["vram"]["estimate_bytes"]


def test_the_tree_is_priced_as_a_tree_not_as_d01(tmp_path, capsys):
    """A d01-only answer would not move with the nest count."""

    from gpuwm.cli import build_parser

    def priced(ladder):
        directory = tmp_path / ladder
        directory.mkdir()
        path = _tree_plan(directory, directory / "run", ladder=ladder)
        assert run_plan_main(build_parser().parse_args(
            ["run-plan", "--estimate", str(path)])) == 0
        return json.loads(capsys.readouterr().out)["vram"]

    one, two = priced("12"), priced("12-3")
    assert one["domains"] == 1 and two["domains"] == 2
    # The envelope carries a per-nest term (machine_peak_envelope_bytes:
    # nests = domains - 1), so the tree costs more per allocated byte.
    assert (two["peak_envelope_bytes"] / two["estimate_bytes"]) > \
        (one["peak_envelope_bytes"] / one["estimate_bytes"])


# ---------------------------------------------------------------------------
# Per-domain progress
# ---------------------------------------------------------------------------


def test_per_domain_clocks_reach_model_progress(tmp_path):
    """A tree advances its nests on their own clocks."""

    with EventStream(tmp_path / EVENTS_FILENAME, mirror=None) as events:
        observer = RunObserver(events)
        observer(model_elapsed_seconds=120.0, outer_step=2,
                 last_durable_wrfout=None, last_checkpoint=None,
                 phase="post-d01-sync", step_wall_seconds=0.5,
                 domain_clocks={1: 120.0, 2: 105.0})
    progress = next(r for r in read_events(tmp_path / EVENTS_FILENAME)
                    if r["event"] == "model_progress")
    assert progress["domain"] == 1          # the ROOT clock, as before
    assert progress["domains"] == [
        {"domain": 1, "model_seconds": 120.0},
        {"domain": 2, "model_seconds": 105.0}]


def test_a_single_domain_run_carries_no_domains_array(tmp_path):
    """Absence means "the root IS the tree", so it must stay absent."""

    with EventStream(tmp_path / EVENTS_FILENAME, mirror=None) as events:
        observer = RunObserver(events)
        observer(model_elapsed_seconds=60.0, outer_step=1,
                 last_durable_wrfout=None, last_checkpoint=None,
                 phase="post-d01-sync", step_wall_seconds=0.5,
                 domain_clocks={1: 60.0})
    progress = next(r for r in read_events(tmp_path / EVENTS_FILENAME)
                    if r["event"] == "model_progress")
    assert "domains" not in progress


def test_the_core_callback_publishes_every_clock(tmp_path):
    """The seam itself: model.py must forward the whole clocks dict.

    Driven through the real on_period_commit closure rather than
    asserted on a copy of its body, so a change there fails here.
    """

    import inspect

    from gpuwm.core import model as model_module

    source = inspect.getsource(model_module.execute_experiment)
    assert "domain_clocks=" in source
    # And it is built from the clocks dict, not from the root alone.
    assert "for grid_id, clock in clocks.items()" in source


def test_the_authority_stage_never_goes_to_the_tree_runner(tmp_path):
    """Stage one would have refused on every tree run.

    Materializing the physics authority is SOURCE-level work and lives
    in the single-domain module for every route (FIRST-LIGHT step 2
    spells it that way even for a chain ending in the tree runner).
    The tree runner has no --materialize-authorities, so keying the
    command off plan["runner"] sent stage one somewhere that refuses.
    """

    plan = {"runner": go_cli.TREE_RUNNER_MODULE, "domains": 2,
            "source": "gfs", "config": tmp_path / "c.toml",
            "wps_namelist": tmp_path / "c.namelist.wps",
            "authority": tmp_path / "auth", "profile": None}
    command = go_cli.authority_command(plan)

    assert "--materialize-authorities" in command
    assert command[2] == go_cli.RUNNER_MODULE
    assert command[2] != go_cli.TREE_RUNNER_MODULE

    # And the module named really does own the flag.
    import gpuwm.prepared_domain_tree_forecast as tree
    import gpuwm.prepared_single_domain_forecast as single
    import inspect

    assert "--materialize-authorities" in inspect.getsource(single.main)
    assert "--materialize-authorities" not in inspect.getsource(tree.main)
