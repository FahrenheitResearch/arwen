"""`gpuwm go --tiles` and the [tiles] a prepared bundle's forecast runs under.

The memory refusal names "[tiles] mode = 'auto' (or --tiles auto)" as its
remedy.  On the two-step form (``--prepare-only``, then
``--prepared-root``) neither half was reachable: the bundle's experiment
config is hash-bound and carries no [tiles] (the preparation binds no
execution mode), and ``gpuwm go`` had no ``--tiles``.  A full 1800x1060
HRRR grid was refused at 107 GiB on a 96 GiB card with nothing the user
could type to stream it.  These tests hold the three ways a prepared
launch now gets its mode, in order, and the one-shot re-statement.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpuwm import go_cli, runplan
from gpuwm.core.streaming import StreamingOptions


def _plan(**options):
    return SimpleNamespace(run_options={"forecast_tiles": None, **options})


def _bundle(tmp_path: Path, layout: str = "single") -> dict:
    root = tmp_path / "prepared"
    root.mkdir(exist_ok=True)
    document = root / "proof.json"
    document.write_text("{}", encoding="utf-8")
    return {"layout": layout, "document": document}


def _exp(mode: str | None):
    tiles = (StreamingOptions.from_mapping({"mode": mode}) if mode
             else StreamingOptions.from_mapping(None))
    return SimpleNamespace(tiles=tiles)


def test_the_go_parser_takes_tiles():
    from gpuwm.cli import build_parser

    args = build_parser().parse_args(["go", "x.toml", "--tiles", "auto"])
    assert args.tiles == "auto"
    with pytest.raises(SystemExit):
        build_parser().parse_args(["go", "x.toml", "--tiles", "sometimes"])


def test_a_prepared_launch_takes_the_flag_then_the_config_then_the_record(
        tmp_path, capsys):
    bundle = _bundle(tmp_path)
    # Nothing asked: resident, as before.
    assert runplan._prepared_launch_tiles(_plan(), _exp(None), bundle) is None
    # The bundle's --prepare-only launch recorded its config's [tiles].
    go_cli._record_launch_tiles(Path(bundle["document"]).parent,
                                _exp("auto"), tmp_path / "n1t.toml")
    assert "recorded beside the prepared bundle" in capsys.readouterr().out
    record = json.loads((Path(bundle["document"]).parent
                         / go_cli.LAUNCH_TILES_NAME).read_text("utf-8"))
    assert record["tiles"]["mode"] == "auto" and record["binds_identity"] is False
    chosen = runplan._prepared_launch_tiles(_plan(), _exp(None), bundle)
    assert chosen.mode == "auto"
    # The launch config's own [tiles] beats the record.
    assert runplan._prepared_launch_tiles(
        _plan(), _exp("on"), bundle).mode == "on"
    # The flag beats both.
    assert runplan._prepared_launch_tiles(
        _plan(forecast_tiles="off"), _exp("on"), bundle).mode == "off"


def test_a_resident_launch_records_nothing(tmp_path):
    root = tmp_path / "prepared"
    root.mkdir()
    go_cli._record_launch_tiles(root, _exp(None), tmp_path / "n1.toml")
    assert not (root / go_cli.LAUNCH_TILES_NAME).exists()


def test_a_tree_bundle_takes_only_an_explicit_flag(tmp_path):
    from gpuwm import stage_cli

    bundle = _bundle(tmp_path, layout="tree")
    assert runplan._prepared_launch_tiles(_plan(), _exp("auto"), bundle) is None
    asked = runplan._prepared_launch_tiles(_plan(forecast_tiles="auto"), _exp(None), bundle)
    with pytest.raises(stage_cli.StageRefusal, match="single-domain runner"):
        stage_cli.streaming_flags("tree", tiles=asked)


def test_the_flag_reaches_the_runner_as_its_own_tiles(tmp_path):
    from gpuwm import stage_cli

    options = runplan._prepared_launch_tiles(
        _plan(forecast_tiles="auto"), _exp(None), _bundle(tmp_path))
    flags = stage_cli.streaming_flags("single", tiles=options)
    assert flags[0] == "--tiles"
    assert json.loads(flags[1])["mode"] == "auto"


def test_the_run_plan_refuses_an_unknown_mode():
    assert runplan._run_option("forecast_tiles", "auto", Path(".")) == "auto"
    with pytest.raises(runplan.PlanError, match="'off', 'auto' or 'on'"):
        runplan._run_option("forecast_tiles", "sometimes", Path("."))


def test_a_prepared_launch_leaves_the_config_alone(tmp_path):
    config = tmp_path / "experiment.toml"
    config.write_text("[experiment]\n", encoding="utf-8")
    args = SimpleNamespace(tiles="auto", prepared_root=tmp_path, restart=None)
    assert go_cli._at_flag_tiles(args, config, {"experiment": {}}) == (
        config, {"experiment": {}})


def test_a_case_data_launch_is_told_where_tiles_goes(tmp_path):
    config = tmp_path / "case.toml"
    args = SimpleNamespace(tiles="auto", prepared_root=None, restart=None,
                           outdir=None)
    with pytest.raises(go_cli.GoRefusal, match=r"\[case_data\]"):
        go_cli._at_flag_tiles(args, config, {"case_data": {}})


def test_a_fetching_launch_runs_a_restated_copy(tmp_path):
    """The shipped pressure-route demo, re-stated with --tiles auto."""

    import shutil
    import tomllib

    root = Path(__file__).parents[1] / "configs"
    for path in root.glob("hrrr_prs_3km_demo.*"):
        shutil.copy(path, tmp_path / path.name)
    config = tmp_path / "hrrr_prs_3km_demo.toml"
    payload = tomllib.loads(config.read_text(encoding="utf-8"))
    args = SimpleNamespace(tiles="auto", prepared_root=None, restart=None,
                           outdir=tmp_path / "out", config=config)
    out, restated = go_cli._at_flag_tiles(args, config, payload)
    assert out == tmp_path / "out" / "tiles" / "auto" / config.name
    assert restated["tiles"] == {"mode": "auto"}
    assert args.config == out
    # Everything else is the original's; [fetch].out is made absolute,
    # keeping its meaning in the other folder, as --cycle's copy does.
    without = {key: value for key, value in restated.items() if key != "tiles"}
    original = {key: value for key, value in payload.items() if key != "tiles"}
    assert Path(without["fetch"].pop("out")).is_absolute()
    original["fetch"].pop("out")
    assert without == original
    # Asked again for the mode it already states: the same file runs.
    assert go_cli._at_flag_tiles(args, out, restated) == (out, restated)


def _demo(tmp_path):
    import shutil
    import tomllib

    root = Path(__file__).parents[1] / "configs"
    for path in root.glob("hrrr_prs_3km_demo.*"):
        shutil.copy(path, tmp_path / path.name)
    config = tmp_path / "hrrr_prs_3km_demo.toml"
    return config, tomllib.loads(config.read_text(encoding="utf-8"))


def _resident_refusal(**extra):
    """The gate dict of X12: a resident 53.69 GiB envelope on a 30.96 GiB card."""

    phases = SimpleNamespace(streamed_forecast=False,
                             peak_envelope_bytes=int(53.69 * 2**30))
    return {"refuse": True, "card_refuse": True, "free_bytes": int(30.96 * 2**30),
            "preparation_refusal": None, "phases": phases, **extra}


def test_a_bare_config_the_card_cannot_hold_streams_instead(tmp_path, capsys):
    """X12: the 5 km CONUS run was refused although it streams in 27 GiB."""

    config, payload = _demo(tmp_path)
    args = SimpleNamespace(tiles=None, prepared_root=None, restart=None,
                           devices=None, outdir=tmp_path / "out", config=config)
    out, restated = go_cli._streams_instead(args, _resident_refusal(), config, payload)
    assert restated["tiles"] == {"mode": "auto"} and args.tiles == "auto"
    assert out == tmp_path / "out" / "tiles" / "auto" / config.name
    said = capsys.readouterr().out
    assert "53.69 GiB against 30.96 GiB free" in said and "streams" in said


@pytest.mark.parametrize("change", [
    "fits", "flag", "table", "domain-table", "streamed", "host", "no-card", "devices"])
def test_the_refusal_stands_wherever_streaming_is_not_the_answer(tmp_path, change):
    config, payload = _demo(tmp_path)
    args = SimpleNamespace(tiles=None, prepared_root=None, restart=None,
                           devices=None, outdir=tmp_path / "out", config=config)
    gate = _resident_refusal()
    if change == "fits":
        gate["refuse"] = False
    elif change == "flag":
        args.tiles = "off"
    elif change == "table":
        payload["tiles"] = {"mode": "off"}
    elif change == "domain-table":
        payload["domain"][0]["tiles"] = {"mode": "off"}
    elif change == "streamed":
        gate["phases"].streamed_forecast = True
    elif change == "host":
        gate["preparation_refusal"] = "the preparation needs more RAM"
    elif change == "no-card":
        gate["free_bytes"] = None
    elif change == "devices":
        args.devices = 2
    assert go_cli._streams_instead(args, gate, config, payload) is None
    assert args.tiles in (None, "off")
