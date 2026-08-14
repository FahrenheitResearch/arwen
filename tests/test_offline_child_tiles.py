"""``[tiles]`` reaches the offline child, from the TOML to the stepper.

THE DEFECT THIS PINS.  The experiment TOML the multi-domain front doors
read has accepted ``[tiles]`` since 2.2.0.  The RunConfig TOML -- the
schema ``gpuwm downscale`` hands to the standalone child -- refused the
block outright as an unknown table, so the one route whose domain is most
likely to outgrow the card it is run on was the one route that could not
ask to stream.  Measured at the 2.2.1 cut: a ``[tiles] mode = 'on'``
appended to a derived child config died in ``load_config`` before any
parent frame was opened.

CPU-side only, deliberately.  The streamed integration itself is proven by
``tilestream/test_join.py`` (bit-exact against the resident arm) and by
the 2.2.2 planner-driven GPU leg; what was missing was never the
transport, it was the wiring, and wiring is what these assert.
"""

from datetime import datetime, timedelta

import pytest

from gpuwm.cli import main as cli_main
from gpuwm.config import load_config, load_streaming_options
from gpuwm.core import streaming
from gpuwm.downscale import _derive_child_run_config, _render_child_toml
from gpuwm.offline_child import OfflineChildContractError
from gpuwm.offline_child_run import _CAPABILITIES
from test_downscale_cli import _PARENT_CONFIG
from test_offline_child import _history


def _child_toml(tmp_path, *, tiles_block: str = "") -> "object":
    parent = {"nx": 20, "ny": 18, "dx": 1000.0, "dy": 1000.0}
    merged = _derive_child_run_config(
        _PARENT_CONFIG, parent=parent, ratio=1, child_nx=12, child_ny=10,
        run_seconds=600.0, output_interval_s=300.0)
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "child.toml"
    path.write_text(_render_child_toml(merged) + tiles_block,
                    encoding="utf-8", newline="\n")
    return path


def test_the_child_schema_accepts_a_tiles_table(tmp_path):
    """RED BEFORE THE FIX: ``unknown table(s) ['tiles']``."""

    path = _child_toml(tmp_path, tiles_block='[tiles]\nmode = "on"\n')
    cfg = load_config(path)
    # Accepted, and NOT merged into the RunConfig: [tiles] is an execution
    # choice whose whole claim is that it changes nothing, so it must not
    # reach the fields a restart identity binds.
    assert not hasattr(cfg, "tiles")
    assert (cfg.nx, cfg.ny) == (12, 10)

    options = load_streaming_options(path)
    assert options.mode == "on"
    assert options.enabled


def test_a_child_config_without_the_block_is_the_shared_off_object(tmp_path):
    """The OFF contract: not a parsed default, the same object."""

    options = load_streaming_options(_child_toml(tmp_path))
    assert options is streaming.OFF
    assert not options.enabled


def test_a_misspelled_tiles_key_is_refused_by_the_run_config_reader(tmp_path):
    """A knob that silently does nothing is how a run gets the wrong mode.

    ``load_config`` validates the block and discards it, so a caller that
    reads only the RunConfig is not the reason a typo survives admission.
    """

    path = _child_toml(
        tmp_path, tiles_block='[tiles]\nmode = "on"\ntile_x = 128\n')
    with pytest.raises(ValueError, match="tile_x"):
        load_config(path)


def test_a_tiling_under_mode_off_is_refused(tmp_path):
    """A surface that is off must be empty, or a mode flips somewhere else."""

    path = _child_toml(
        tmp_path, tiles_block='[tiles]\ntile_nx = 128\ntile_ny = 128\n')
    with pytest.raises(ValueError, match="must be empty"):
        load_streaming_options(path)


def test_off_binds_the_dycore_step_itself(tmp_path):
    """No ``[tiles]`` means no branch at all -- the same function object.

    This is the whole OFF contract, and it is what makes "a child that
    configures nothing is unchanged" a fact rather than a claim.
    """
    from gpuwm.core.dycore import step

    cfg = load_config(_child_toml(tmp_path))
    stepper = streaming.make_stepper(
        None, cfg, load_streaming_options(_child_toml(tmp_path)),
        build=None)
    assert stepper is step


def test_the_standalone_builder_is_the_prepared_builder(tmp_path):
    """A root with no tree still gets the proven builder, not a second one."""

    build = streaming.standalone_domain_builder(grid_id=2)
    assert callable(build)
    # The two facts it answers for a domain that is its own root.
    node = streaming._StandaloneNode(
        cfg=streaming._StandaloneNodeCfg(grid_id=2))
    assert node.parent is None
    assert node.cfg.grid_id == 2


def test_the_runner_declares_that_it_honors_tiles():
    """The capability surface Studio's doctor reads, not a comment."""

    assert _CAPABILITIES["tiles"] == "honored"


# ---------------------------------------------------------------------------
# the front door
# ---------------------------------------------------------------------------


def _parent_archive(tmp_path):
    start = datetime(1974, 4, 3, 12)
    for index in range(3):
        _history(tmp_path / f"wrfout_d03_1974-04-03_{12 + index:02d}_00_00",
                 start + timedelta(hours=index), ny=18, nx=20)
    namelist = tmp_path / "namelist.input"
    namelist.write_text("&physics\n mp_physics = 8,\n/\n", encoding="utf-8")
    return namelist


def _plan(capsys):
    import json

    out = capsys.readouterr().out
    return json.loads(out[out.index("{"):])


def test_the_plan_reports_the_mode_the_child_will_actually_use(
        tmp_path, capsys):
    """Read off the config that will be RUN, not off the flag that was typed.

    The whole front-door claim is that a ``[tiles]`` block in a child config
    is honored, so ``--dry-run`` has to be able to say which mode the run
    will take -- including for a config this command did not write.
    """

    namelist = _parent_archive(tmp_path)
    args = [
        "downscale", str(tmp_path), "--parent-domain", "3",
        "--parent-namelist", str(namelist), "--ratio", "1",
        "--i-parent-start", "4", "--j-parent-start", "4",
        "--accept-parent-cadence",
        "--out", str(tmp_path / "child-run"), "--dry-run"]

    plain = _child_toml(tmp_path / "plain")
    assert cli_main(args + ["--child-config", str(plain)]) == 0
    assert _plan(capsys)["tiles"]["mode"] == "off"

    streamed = _child_toml(tmp_path / "streamed",
                           tiles_block='[tiles]\nmode = "on"\n')
    assert cli_main(args + ["--child-config", str(streamed)]) == 0
    reported = _plan(capsys)["tiles"]
    assert reported["mode"] == "on"
    # Never a fabricated tiling: the planner answers that for the card the
    # run meets, and the plan must not pretend to know it here.
    assert "tile_nx" not in reported and "nbuffers" not in reported


def test_tiles_flag_is_refused_against_a_supplied_child_config(
        tmp_path, capsys):
    """--tiles writes into a config this command DERIVES.

    With --child-config the caller owns the file, and silently rewriting it
    -- or silently ignoring the flag -- are both worse than saying so.
    """

    namelist = _parent_archive(tmp_path)
    assert cli_main([
        "downscale", str(tmp_path), "--parent-domain", "3",
        "--parent-namelist", str(namelist),
        "--child-config", str(_child_toml(tmp_path)), "--ratio", "1",
        "--i-parent-start", "4", "--j-parent-start", "4",
        "--accept-parent-cadence", "--tiles", "on",
        "--out", str(tmp_path / "child-run"), "--dry-run"]) != 0
    err = capsys.readouterr().err
    assert "--child-config" in err and "[tiles]" in err


def test_render_child_toml_emits_a_mode_only_block():
    """Only the mode: a derived config must not carry THIS machine's plan."""

    parent = {"nx": 20, "ny": 18, "dx": 1000.0, "dy": 1000.0}
    merged = _derive_child_run_config(
        _PARENT_CONFIG, parent=parent, ratio=1, child_nx=12, child_ny=10,
        run_seconds=600.0, output_interval_s=300.0)
    text = _render_child_toml(merged, tiles_mode="auto")
    assert '[tiles]\nmode = "auto"' in text
    for pinned in ("tile_nx", "tile_ny", "nbuffers", "halo"):
        assert pinned not in text
    assert _render_child_toml(merged) .count("[tiles]") == 0
