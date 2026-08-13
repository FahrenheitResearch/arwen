"""``[tiles]`` on the native HRRR chain: the plumbing, and the refusals.

THE DEFECT.  ``[tiles]`` was configurable on this route and unreachable
at the same time.  The single-domain arm hands its forecast the
authority the PREPARER published -- rendered from tables
``tools/hrrr_single_domain_benchmark.py`` builds in code, which are
``{experiment, projection, shared, domain}`` and have never included a
``[tiles]`` -- so a user's block was loaded by run-plan, validated,
reported by ``--resolve``, and then replaced by a document that does not
mention it.  The forecast ran resident.  The only evidence was a line
that never appeared in the log.  Nothing refused, either: the strings
``tiles`` and ``streaming`` appeared zero times in ``gpuwm/runplan.py``.

Two halves are pinned here.  The first is that the user's table now
REACHES the forecast on both arms, and by different routes on purpose:
the single-domain arm carries it as ``--tiles`` because the document it
hands over cannot hold one, and the tree arm carries nothing because the
document it hands over IS the user's config.  The second is the refusal
matrix -- the combinations that cannot stream, refused from the config
alone rather than discovered at the first tile buffer, which on this
chain is a download and two preparations later.

CPU-only.  The chain is captured at its ``run_stage``/``_run_fetch``
seams and the forecast at its ``main`` seam, exactly as
tests/test_runplan_hrrr_tree.py does.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import textwrap
from pathlib import Path

import pytest

import gpuwm.go_cli as go_cli
import gpuwm.prepared_single_domain_forecast as psdf
import gpuwm.runplan as runplan_module
from gpuwm.core.streaming import StreamingOptions
from gpuwm.experiment import load_experiment
from gpuwm.runplan import (PLAN_SCHEMA, PlanError, build_plan, resolve_plan,
                           streaming_decision)

_ROOT = """\
[[domain]]
grid_id = 1
parent_id = 0
i_parent_start = 1
j_parent_start = 1
parent_grid_ratio = 1
parent_time_step_ratio = 1
nx = 100
ny = 80
time_step = 60
dx = 12000.0
history_interval_s = 3600.0
"""

_NEST = """\
[[domain]]
grid_id = 2
parent_id = 1
i_parent_start = 40
j_parent_start = 30
parent_grid_ratio = 3
parent_time_step_ratio = 3
e_we = 61
e_sn = 61
history_interval_s = 900.0
"""

_FOLLOW = """\
[relocation]
enabled = true
grid_id = 2

[[relocation.move]]
at_seconds = 120.0
di_parent_cells = 1
dj_parent_cells = 0
"""


def _config(tmp_path, *, tiles="", nested=False, follow=False,
            source="hrrr", name="exp"):
    """One HRRR-shaped experiment TOML, plus the route's four inputs.

    The four files beside it are EMPTY: ``_hrrr_chain`` checks that each
    exists and passes its path onward, and every stage that reads their
    contents is captured in these tests.  A fixture that authored real
    namelists would be testing the wizard.
    """

    path = tmp_path / f"{name}.toml"
    path.write_text(textwrap.dedent(f"""\
        [experiment]
        name = "synth"
        start_time = 2024-05-03T12:00:00
        run_seconds = 3600.0
        restart_interval_s = 0.0

        [fetch]
        source = "{source}"
        cycle = "2024-05-03T12"
        hours = 6

        [shared]
        nz = 8
        ztop = 12000.0

        """) + tiles + "\n" + _ROOT + ("\n" + _NEST if nested else "")
        + ("\n" + _FOLLOW if follow else ""), encoding="utf-8")
    from gpuwm.hrrr_route_inputs import route_input_paths

    for role_path in route_input_paths(path).values():
        role_path.write_text("", encoding="utf-8")
    return path


def _plan(tmp_path, config_path, **run_options):
    geog = tmp_path / "GEOG"
    geog.mkdir(exist_ok=True)
    return build_plan(
        {"schema": PLAN_SCHEMA, "name": "tiles-fixture", "route": "prepared",
         "config": {"path": str(config_path)},
         "run_options": {"geog_root": str(geog), **run_options},
         "output_root": str(tmp_path / "run")},
        source="plan.json", base_dir=tmp_path, sha256="0" * 64)


class _Observer:
    """Swallows the observer protocol; the chain's argv is the subject."""

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def _drive(tmp_path, monkeypatch, *, tiles="", nested=False):
    """Run the whole HRRR chain with every stage captured, nothing run."""

    config = _config(tmp_path, tiles=tiles, nested=nested)
    plan = _plan(tmp_path, config)
    exp = load_experiment(config)
    staged: list[tuple[str, list[str]]] = []
    prep_root = plan.run_dir / "chain" / "hrrr-root-prep"
    tree_root = plan.run_dir / "chain" / "hrrr-hierarchy"

    def run_stage(label, command, **kwargs):
        staged.append((label, [str(part) for part in command]))
        if label == "prepare":
            _seal_single_domain_bundle(prep_root, config)
        if label == "hierarchy":
            tree_root.mkdir(parents=True, exist_ok=True)
            (tree_root / "receipt.json").write_text(
                json.dumps({
                    "schema": "gpuwm-native-hrrr-hierarchy-direct-v1",
                    "status": "PASS"}, indent=2, sort_keys=True) + "\n",
                encoding="utf-8")

    def fake_fetch(arguments, run_dir):
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
        psdf, "main",
        lambda argv, *, observer=None: captured.update(
            single=[str(part) for part in argv]) or 0)
    monkeypatch.setattr(
        tree, "main",
        lambda argv, *, observer=None: captured.update(
            tree=[str(part) for part in argv]) or 0)
    monkeypatch.setattr(runplan_module, "_hrrr_render",
                        lambda plan, **kwargs: {"ok": True})

    runplan_module._hrrr_chain(
        plan, config_path=config, exp=exp, observer=_Observer(),
        run_dir=plan.run_dir)
    return captured, config, staged


def _seal_single_domain_bundle(prep_root: Path, config: Path) -> None:
    """The artifacts the preparer publishes, as the chain relays them.

    Written by the captured prepare stage rather than by the test body,
    because the chain reads them BETWEEN its stages: the wrapper result,
    the proof beside it, and the three digests that must agree with what
    ``go_cli.proof_digests`` reads back off the same file.
    """

    prep_root.mkdir(parents=True, exist_ok=True)
    proof = prep_root / "proof.json"
    proof.write_text(json.dumps({
        "prepared_cache": {"content_sha256": "c" * 64},
        "input_manifest_sha256": "m" * 64,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    # The PUBLISHED authority: a copy of the user's config under the
    # preparer's own name, which is what the real preparer emits and
    # what the runner's manifest role check binds by name and digest.
    published = prep_root / "experiment.toml"
    published.write_text(config.read_text(encoding="utf-8"), encoding="utf-8")
    namelist = prep_root / "namelist.wps"
    namelist.write_text("", encoding="utf-8")
    (prep_root / "public-wrapper-result.json").write_text(json.dumps({
        "portable_bundle": {
            "prepared_root": str(prep_root),
            "proof_sha256": hashlib.sha256(proof.read_bytes()).hexdigest(),
            "source_manifest_sha256": "m" * 64,
            "prepared_content_sha256": "c" * 64,
            "experiment_config": str(published),
            "wps_namelist": str(namelist),
        }}, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _flag(argv, name):
    return argv[argv.index(name) + 1]


# ---------------------------------------------------------------------------
# The plumbing: the user's table reaching the forecast
# ---------------------------------------------------------------------------


_ON = """\
[tiles]
mode = "on"
tile_nx = 128
tile_ny = 96
store = "device"
host_budget_bytes = 34359738368
"""


def test_the_single_domain_arm_carries_the_users_tiles_to_the_forecast(
        tmp_path, monkeypatch):
    """The hole this closes.

    ``--experiment-config`` on this arm is the authority the PREPARER
    published, and that document has no [tiles] table -- so before this
    flag existed the user's block simply stopped here.
    """

    captured, _config, _staged = _drive(tmp_path, monkeypatch, tiles=_ON)
    argv = captured["single"]

    assert "--tiles" in argv
    assert StreamingOptions.from_mapping(
        json.loads(_flag(argv, "--tiles"))) == StreamingOptions(
            mode="on", tile_nx=128, tile_ny=96, store="device",
            host_budget_bytes=34359738368)


def test_every_key_survives_the_hop_including_the_ones_to_json_drops(
        tmp_path, monkeypatch):
    """``StreamingOptions.to_json`` is NOT the wire form, and must not be.

    It omits every ``None`` and both budget keys, so a forwarding built
    on it would silently drop ``host_budget_bytes`` -- the override that
    exists because ``/proc/meminfo`` reports the HOST's RAM inside a
    container and is not a budget.  The chain sends the whole dataclass.
    """

    captured, _config, _staged = _drive(tmp_path, monkeypatch, tiles=_ON)
    payload = json.loads(_flag(captured["single"], "--tiles"))

    assert payload["host_budget_bytes"] == 34359738368
    assert set(payload) == set(StreamingOptions.__dataclass_fields__)
    assert "host_budget_bytes" not in StreamingOptions(
        mode="on", host_budget_bytes=34359738368).to_json()


def test_an_unconfigured_run_composes_the_argv_it_always_did(
        tmp_path, monkeypatch):
    """Token for token, the two arms differ by the flag and nothing else.

    ``[tiles]`` absent is the OFF contract, and the whole promise of the
    mode is that a run which never mentions it pays nothing for its
    existence -- including in the command line it composes.
    """

    (tmp_path / "off").mkdir()
    (tmp_path / "on").mkdir()
    quiet, _c, _s = _drive(tmp_path / "off", monkeypatch)
    loud, _c2, _s2 = _drive(tmp_path / "on", monkeypatch, tiles=_ON)

    assert "--tiles" not in quiet["single"]

    def _shape(argv, root):
        return [token.replace(str(root), "<ROOT>").replace("\\", "/")
                for token in argv]

    trimmed = list(loud["single"])
    del trimmed[trimmed.index("--tiles"):trimmed.index("--tiles") + 2]
    assert _shape(quiet["single"], tmp_path / "off") == _shape(
        trimmed, tmp_path / "on")


def test_the_tree_arm_carries_tiles_in_the_users_own_config(
        tmp_path, monkeypatch):
    """No flag here, and that is the point rather than an omission.

    The tree arm hands the tree runner the USER'S config -- bound by its
    own digest on the next token -- so ``exp.tiles`` there is already the
    table they typed.  Adding a flag as well would give one table two
    sources on one route, which is how two arms of one chain end up
    streaming differently from the same config.

    GREEN BEFORE THIS CHANGE TOO, deliberately.  It is the regression
    pin on the arm that was already right: what would break it is
    someone "finishing the job" by adding --tiles here as well.
    """

    captured, config, _staged = _drive(
        tmp_path, monkeypatch,
        tiles='[tiles]\nmode = "auto"\n', nested=True)
    argv = captured["tree"]

    assert "--tiles" not in argv
    handed = Path(_flag(argv, "--experiment-config"))
    assert handed == config
    assert load_experiment(handed).tiles == StreamingOptions(mode="auto")
    # And the digest beside it is of that same file, so the runner is
    # reading the bytes this assertion just read.
    assert _flag(argv, "--experiment-config-sha256") == hashlib.sha256(
        handed.read_bytes()).hexdigest()


def test_the_runner_forwards_the_flag_into_its_preflight(tmp_path,
                                                         monkeypatch):
    """``main`` -> ``preflight_prepared_forecast(tiles=...)``, validated.

    Through ``StreamingOptions.from_mapping``, which is the same
    validator the config front door uses -- so the flag and a [tiles]
    table cannot come to disagree about what a legal table is.
    """

    from types import SimpleNamespace

    seen: dict = {}
    monkeypatch.setattr(
        psdf, "preflight_prepared_forecast",
        lambda **kwargs: seen.update(kwargs) or SimpleNamespace(
            physics_receipt={}))
    monkeypatch.setattr(
        psdf, "run_prepared_forecast",
        lambda inputs, **kwargs: {
            "schema": "s", "status": "PASS", "source": "hrrr",
            "run_seconds": 1.0, "history_interval_seconds": 1.0,
            "gridded_output": {"frame_count": 0},
            "input": {"prepared_content_sha256": "c" * 64}})

    assert psdf.main([
        "--source", "hrrr", "--prepared-root", str(tmp_path / "bundle"),
        "--proof-sha256", "0" * 64, "--source-manifest-sha256", "0" * 64,
        "--prepared-content-sha256", "0" * 64,
        "--experiment-config", str(tmp_path / "e.toml"),
        "--wps-namelist", str(tmp_path / "n.wps"),
        "--run-seconds", "60", "--history-interval-seconds", "60",
        "--io-mode", "history", "--outdir", str(tmp_path / "out"),
        "--tiles", json.dumps({"mode": "on", "tile_nx": 64, "tile_ny": 64}),
    ]) == 0
    assert seen["tiles"] == StreamingOptions(
        mode="on", tile_nx=64, tile_ny=64)


def test_the_overlay_lands_after_every_identity_comparison(tmp_path):
    """Order, which is the property that cannot be read off a return value.

    [tiles] is not a domain field and could not move one of the hundred
    and ten the prepared-cache identity compares -- but an execution
    control applied BEFORE the checks it cannot affect is one somebody
    later assumes was checked.
    """

    import inspect

    source = inspect.getsource(psdf.preflight_prepared_forecast)
    overlay = source.index("exp = replace(exp, tiles=tiles)")
    assert source.index(
        "preparation authorities changed during preflight") < overlay
    assert overlay < source.index("return PreparedForecastInputs(")


def test_a_malformed_tiles_flag_is_a_usage_refusal_not_a_traceback(
        tmp_path, capsys):
    """And in the config front door's own vocabulary, not a second one."""

    argv = [
        "--source", "hrrr", "--prepared-root", str(tmp_path / "bundle"),
        "--proof-sha256", "0" * 64, "--source-manifest-sha256", "0" * 64,
        "--prepared-content-sha256", "0" * 64,
        "--experiment-config", str(tmp_path / "e.toml"),
        "--wps-namelist", str(tmp_path / "n.wps"),
        "--run-seconds", "60", "--history-interval-seconds", "60",
        "--io-mode", "history", "--outdir", str(tmp_path / "out"),
    ]

    assert psdf.main(argv + ["--tiles", '{"mode": "sideways"}']) == 2
    assert "mode = 'sideways'" in capsys.readouterr().err

    assert psdf.main(argv + ["--tiles", '{"tile_nx": 64}']) == 2
    # from_mapping's own sentence: half a tiling is not a tiling.
    assert "must be given together" in capsys.readouterr().err

    assert psdf.main(argv + ["--tiles", '{"tiles_nx": 64}']) == 2
    assert "unknown key(s)" in capsys.readouterr().err

    # And nothing was created for the typo.
    assert not (tmp_path / "out").exists()


def test_the_published_hrrr_authority_still_cannot_carry_a_tiles_table():
    """The shape NOT taken, pinned so it is not taken by accident.

    Rendering [tiles] into the preparer's published document was the
    other candidate.  It is refused because that document is HASH-BOUND
    -- the runner compares its name and sha256 against the portable
    source manifest -- so the execution mode would become part of the
    prepared bundle's identity, and a bundle prepared streamed could not
    be re-run resident.  That is the exact coupling
    ``streaming.identity_payload_entry`` returns ``{}`` to prevent.

    GREEN BEFORE THIS CHANGE TOO, deliberately: it pins a status quo
    that the alternative implementation would have broken, so what it
    guards against is the OTHER shape being adopted later by someone who
    reads the flag as a workaround for a missing table.
    """

    from gpuwm.core import streaming
    from gpuwm.experiment_document import (ExperimentDocumentError,
                                           _TABLE_ORDER,
                                           render_experiment_document)

    assert "tiles" not in _TABLE_ORDER
    with pytest.raises(ExperimentDocumentError) as refusal:
        render_experiment_document({
            "experiment": {"name": "x"}, "tiles": {"mode": "on"},
            "domain": [{"grid_id": 1}]})
    assert "tiles" in str(refusal.value)
    # And the reason the flag is sound: the table binds no identity.
    assert streaming.identity_payload_entry(
        StreamingOptions(mode="on")) == {}


# ---------------------------------------------------------------------------
# The refusal matrix, decided from the config alone
# ---------------------------------------------------------------------------


def test_every_chain_declares_what_tiles_does_on_it():
    """A chain cannot be added without answering the question.

    The same completeness rule ``_FOLLOW_STATICS_DELIVERY`` keeps: a
    chain absent from the table would be judged by a guess, and a guess
    here either refuses a config that streams or accepts one whose
    forecast stage will never read the table.
    """

    chains = {runplan_module._chain_key(route, source)
              for route in runplan_module.ROUTES
              for source in (None, "gfs", "hrrr", "era5")}
    assert chains <= set(runplan_module._STREAMING_DELIVERY)
    assert set(runplan_module._STREAMING_DELIVERY) == chains


def test_an_unknown_chain_is_refused_rather_than_guessed(tmp_path):
    exp = load_experiment(_config(tmp_path, tiles=_ON))
    with pytest.raises(PlanError) as refusal:
        streaming_decision(exp, chain="prepared:teleport")
    assert "prepared:teleport" in str(refusal.value)


def test_a_plan_that_configures_no_tiles_says_nothing_about_them(
        tmp_path):
    """The emptiness contract, at this layer too."""

    plan = _plan(tmp_path, _config(tmp_path))
    resolution, exp, _data = resolve_plan(plan, require_inputs=False)

    assert streaming_decision(exp, chain="prepared:hrrr") is None
    assert resolution["tiles"] is None
    assert not [entry for entry in resolution["automatic_resolutions"]
                if entry["key"] == "tiles_delivery"]
    # The schema-default entry is what an unconfigured config gets, and
    # it is a different key on purpose: its value is the whole default
    # StreamingOptions, not a mode.
    assert [entry for entry in resolution["automatic_resolutions"]
            if entry["key"] == "tiles"] == [
        {"scope": "experiment", "key": "tiles",
         "value": dataclasses.asdict(StreamingOptions()),
         "basis": "schema_default"}]


def test_a_nested_plan_that_asks_every_grid_to_stream_is_refused_by_name(
        tmp_path):
    """``mode = "on"`` streams unconditionally, and a nest cannot.

    Without this the plan fetches, prepares twice, builds the whole
    resident tree and THEN stops at the tile builder -- a refusal that
    was knowable from the config before anything was downloaded.
    """

    plan = _plan(tmp_path, _config(tmp_path, tiles=_ON, nested=True))
    with pytest.raises(PlanError) as refusal:
        resolve_plan(plan, require_inputs=False)
    text = str(refusal.value)

    assert "d02" in text and "NEST" in text
    assert "d01" in text          # what CAN stream, named
    assert "mode = 'auto'" in text and "one [[domain]] table" in text


def test_the_moving_domain_is_named_because_a_moving_domain_is_a_nest(
        tmp_path):
    """A [relocation] follow config cannot stream the domain that moves."""

    plan = _plan(tmp_path, _config(tmp_path, tiles=_ON, nested=True,
                                   follow=True))
    with pytest.raises(PlanError) as refusal:
        resolve_plan(plan, require_inputs=False)
    text = str(refusal.value)

    assert "[relocation] follow domain" in text
    assert "d02 is also" in text


def test_auto_on_a_tree_is_accepted_and_the_gap_is_named(tmp_path):
    """The asymmetry is streaming.refuse_unrouted_streaming's, not a new one.

    ``on`` is decidable from the config; ``auto`` asks
    ``tilestream.autoplan`` about a specific card and legitimately
    answers "resident" for a nest that fits.  Asking that here would
    stand a CUDA context up in a front door that has not decided to use
    the device.  What auto does to a nest that does NOT fit is the same
    unfixed gap that function names -- so it is named, not papered over.
    """

    plan = _plan(tmp_path, _config(
        tmp_path, tiles='[tiles]\nmode = "auto"\n', nested=True))
    resolution, exp, _data = resolve_plan(plan, require_inputs=False)

    assert resolution["tiles"] == {
        "chain": "prepared:hrrr", "delivery": "root_only", "mode": "auto",
        "store": "host", "streamable_grid_id": 1, "resident_grid_ids": [2],
        "relocation_grid_id": None}
    note = next(entry for entry in resolution["automatic_resolutions"]
                if entry["key"] == "tiles_delivery")["note"]
    assert "d02 therefore run resident" in note
    assert "does not fit" in note


def test_a_single_domain_plan_streams_and_the_document_says_which_grid(
        tmp_path):
    """A run cannot answer this for itself.

    A grid that declined to stream is ABSENT from the stepper dict, and
    absent is what an unconfigured grid looks like too -- so the front
    end is told before the run or not at all.
    """

    plan = _plan(tmp_path, _config(tmp_path, tiles=_ON))
    resolution, _exp, _data = resolve_plan(plan, require_inputs=False)

    assert resolution["tiles"]["streamable_grid_id"] == 1
    assert resolution["tiles"]["resident_grid_ids"] == []
    entry = next(e for e in resolution["automatic_resolutions"]
                 if e["key"] == "tiles_delivery")
    assert entry["value"] == "on"
    assert entry["basis"] == "experiment_config"
    assert "streams for real" in entry["note"]
    # And the promise that makes the flag sound is stated where an
    # operator reads it, not only in the core's docstring.
    assert "restart identity" in entry["note"]


def test_the_config_driven_route_refuses_in_the_cores_own_words(tmp_path):
    """``gpuwm run`` reads exp.tiles at NO point, so auto is refused too.

    The sentence is ``streaming.refuse_unrouted_streaming``'s, raised
    rather than restated: a route that learns to stream stops refusing
    here on the day it stops refusing there.
    """

    from test_case_data import make_case_toml

    config = make_case_toml(tmp_path)
    config.write_text(
        config.read_text(encoding="utf-8")
        + '\n[tiles]\nmode = "auto"\n', encoding="utf-8")
    plan = build_plan(
        {"schema": PLAN_SCHEMA, "name": "x", "route": "experiment",
         "config": {"path": str(config)},
         "output_root": str(tmp_path / "run")},
        source="plan.json", base_dir=tmp_path, sha256="0" * 64)

    with pytest.raises(PlanError) as refusal:
        resolve_plan(plan, require_inputs=False)
    text = str(refusal.value)
    assert "does not read [tiles] at ANY point" in text
    assert "'experiment'" in text


def test_the_prepared_chains_are_not_refused_for_having_a_nest_under_auto(
        tmp_path):
    """The negative control: `auto` is not quietly turned into `on`."""

    exp = load_experiment(_config(
        tmp_path, tiles='[tiles]\nmode = "auto"\n', nested=True))
    for chain in ("prepared:go", "prepared:hrrr"):
        assert streaming_decision(exp, chain=chain)["refusal"] is None
