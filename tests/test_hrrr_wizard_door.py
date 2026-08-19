"""The wizard's closing block for ``--source hrrr`` names the shipped door.

`gpuwm domain --source hrrr` ends with the commands a reader is meant to
run next, and until now those commands were the route's INTERNALS:

    python -m tools.prepare_hrrr_wrf ...        <- a path a wheel has not got
    python tools/hrrr_single_domain_benchmark.py ...
    python -m gpuwm.hrrr_hierarchy_direct ...
    python -m gpuwm.prepared_domain_tree_forecast ...

Every one of those is a program `gpuwm prep`/`gpuwm sim` already spawn --
MEASURED with `gpuwm prep --source hrrr ... --dry-run`, which prints the
composed line verbatim.  So the wizard was handing a reader the machinery
behind two shipped commands, two of the four spellings do not exist in a
pip install at all, and the two `<printed by ...>` digest placeholders it
told them to fill in are values `gpuwm sim` reads off the bundle itself.

`gpuwm go`'s hrrr refusal and `gpuwm sim`'s "finished but unbindable"
refusal ALREADY spell the real route.  Three hand-written copies of one
route is the defect class here, so the assertions below are about a
single helper: the wizard and both refusals render from it, and a fourth
copy cannot be added without failing a test.
"""

from __future__ import annotations

import re

import pytest

from gpuwm import domain_wizard, go_cli, stage_cli
from gpuwm.cli import main as cli_main

_DIGEST = "0" * 64


def _emit(tmp_path, capsys, name: str, ladder: str) -> tuple[str, str]:
    """``(config path, the wizard's printed closing block)``, for real."""
    out = tmp_path / f"{name}.toml"
    assert cli_main([
        "domain", "--point=35.3,-97.5", "--card", "24gb",
        "--ladder", ladder, "--source", "hrrr",
        "--cycle", "2026-07-29T18", "--hours", "6",
        "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    return str(out), printed


def _commands(printed: str, front_door: str) -> list[list[str]]:
    """Every ``front_door`` command in the block, backslashes joined.

    The wizard prints a shell-shaped, indented, continued command; a
    reader pastes it.  This reads it back the way that reader's shell
    would, so the assertions are about what they actually run.
    """

    joined = re.sub(r"\\\s*\n\s*", " ", printed)
    commands = []
    for line in joined.splitlines():
        text = line.strip()
        if not text.startswith(front_door):
            continue
        text = text.replace("<your WPS_GEOG>", "GEOG")
        text = re.sub(r"<[^>]+>", _DIGEST, text)
        commands.append(text.split())
    return commands


# ---------------------------------------------------------------------------
# What the reader is handed
# ---------------------------------------------------------------------------

_INTERNALS = (
    "python -m tools.prepare_hrrr_wrf",
    "tools/hrrr_single_domain_benchmark.py",
    "python -m gpuwm.hrrr_hierarchy_direct",
    "python -m gpuwm.prepared_domain_tree_forecast",
)


@pytest.mark.parametrize("ladder", ("12", "12-3"))
def test_the_hrrr_wizard_prints_the_shipped_route(tmp_path, capsys, ladder):
    """Both arms: the two stage commands, and none of the machinery."""
    _config, printed = _emit(tmp_path, capsys, "hrrr", ladder)
    assert "gpuwm prep --source hrrr" in printed
    assert "gpuwm sim " in printed
    for internal in _INTERNALS:
        assert internal not in printed, (
            f"the closing block still hands the reader `{internal}`, which "
            "is what `gpuwm prep`/`gpuwm sim` spawn for them")


@pytest.mark.parametrize("ladder", ("12", "12-3"))
def test_no_printed_digest_placeholder_survives_the_forecast_stage(
        tmp_path, capsys, ladder):
    """`gpuwm sim` reads the digests off the bundle; nobody types them."""
    _config, printed = _emit(tmp_path, capsys, "hrrr", ladder)
    sim = [line for line in printed.splitlines() if "gpuwm sim " in line]
    assert sim, printed
    for line in sim:
        assert "<" not in line, (
            f"{line.strip()!r} asks the reader for a value the bundle "
            "already carries")


@pytest.mark.parametrize("ladder", ("12", "12-3"))
def test_every_printed_prep_line_is_one_the_real_front_door_accepts(
        tmp_path, capsys, ladder, monkeypatch):
    """Verified against the artifact: each line, through `gpuwm prep`.

    ``--dry-run`` composes and prints the internal command without
    running it, so this asserts the printed flags are flags this door
    takes AND that the program behind them is the one the route needs.
    """
    _config, printed = _emit(tmp_path, capsys, "hrrr", ladder)
    commands = _commands(printed, "gpuwm prep")
    assert commands, printed
    for command in commands:
        assert cli_main(command[1:] + ["--dry-run"]) == 0, command
        composed = capsys.readouterr().out
        expected = ("gpuwm.hrrr_hierarchy_direct"
                    if "--root-preparation" in command
                    else "prepare_hrrr_wrf")
        assert expected in composed, composed


def test_the_printed_corridor_flag_reaches_the_hierarchy(tmp_path, capsys):
    """A moving nest's corridor must survive the door it is typed at.

    ``--statics-corridor`` seals child-resolution statics, and the tree
    runner refuses a follow bundle prepared without them.  The front
    door accepted the flag and dropped it on the floor, so a pasted
    chain prepared a bundle its own last line refuses.
    """
    from gpuwm.experiment import load_experiment
    from gpuwm.static.corridor import config_declares_follow_source

    config, printed = _emit(tmp_path, capsys, "hrrr-follow", "12-3")
    dt = float(load_experiment(config).root.run.dt)
    text = (open(config, encoding="utf-8").read()
            + "\n[relocation]\nenabled = true\ngrid_id = 2\n\n"
            "[[relocation.move]]\n"
            f"at_seconds = {dt * 2:.1f}\n"
            "di_parent_cells = 1\ndj_parent_cells = 0\n")
    with open(config, "w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
    experiment = load_experiment(config)
    assert config_declares_follow_source(experiment) is True

    chain = domain_wizard.hrrr_route_commands(
        config, experiment, profile=None, data_dir=str(tmp_path / "data"))
    hierarchy = [command for command in _commands(chain, "gpuwm prep")
                 if "--root-preparation" in command]
    assert hierarchy, chain
    assert "--statics-corridor" in hierarchy[0], hierarchy[0]
    assert cli_main(hierarchy[0][1:] + ["--dry-run"]) == 0
    composed = capsys.readouterr().out
    assert "--statics-corridor" in composed, (
        "the front door accepted --statics-corridor and composed a "
        "hierarchy command without it, so the corridor is never sealed")


# ---------------------------------------------------------------------------
# One spelling, three doors
# ---------------------------------------------------------------------------

def test_the_two_stage_route_is_spelled_in_exactly_one_place():
    """The helper exists, and it is what the refusals say."""
    prep, sim = stage_cli.staged_route_commands("hrrr")
    assert prep.startswith("gpuwm prep --source hrrr")
    assert prep.endswith("--output-root DIR")
    assert sim.startswith("gpuwm sim DIR")
    assert "--experiment-config DIR/experiment.toml" in sim
    assert "--wps-namelist DIR/namelist.wps" in sim
    assert "--outdir OUT" in sim


def test_the_sim_line_names_the_files_the_preparation_actually_publishes():
    """The seam's names and the writer's names, asserted equal once.

    ``gpuwm sim`` is told to read ``DIR/experiment.toml`` and
    ``DIR/namelist.wps``; the preparation writes whatever
    ``hrrr_prepared_bundle`` calls them.  A rename on either side would
    produce a printed command that cannot open its own inputs.
    """
    from gpuwm import hrrr_prepared_bundle

    assert (stage_cli.PREPARED_EXPERIMENT_CONFIG
            == hrrr_prepared_bundle.EXPERIMENT_CONFIG_NAME)
    assert (stage_cli.PREPARED_WPS_NAMELIST
            == hrrr_prepared_bundle.WPS_NAMELIST_NAME)


def test_gos_refusal_renders_the_route_from_that_helper(tmp_path):
    """`gpuwm go` on an hrrr config: the same two strings, not a copy."""
    config = tmp_path / "hrrr.toml"
    config.write_text(
        '[experiment]\nname = "x"\n\n[fetch]\nsource = "hrrr"\n'
        '[[domain]]\ngrid_id = 1\n', encoding="utf-8")
    prep, sim = stage_cli.staged_route_commands("hrrr")
    with pytest.raises(go_cli.GoRefusal) as refusal:
        go_cli.plan_from_config(config, outdir=tmp_path / "go")
    assert prep in str(refusal.value)
    assert sim in str(refusal.value)


def test_sims_finished_but_unbindable_refusal_renders_it_too(tmp_path):
    """The other refusal that already spelled the route by hand."""
    import json

    root = tmp_path / "prep"
    root.mkdir()
    (root / "public-wrapper-result.json").write_text(
        json.dumps({"status": "PASS", "portable_bundle": None}),
        encoding="utf-8")
    with pytest.raises(stage_cli.StageRefusal) as refusal:
        stage_cli.resolve_bundle(root)
    prep, sim = stage_cli.staged_route_commands("hrrr")
    assert prep in str(refusal.value)
    assert sim in str(refusal.value)


@pytest.mark.parametrize("module", (domain_wizard, go_cli))
def test_no_second_copy_of_the_route_can_be_written(module):
    """The drift guard: only the helper's module spells the sim line."""
    import inspect

    source = inspect.getsource(module)
    for line in source.splitlines():
        assert not ("gpuwm sim " in line and "--outdir" in line), (
            f"{module.__name__} spells the forecast stage itself: "
            f"{line.strip()!r}.  Render it from "
            "stage_cli.staged_route_commands instead")
