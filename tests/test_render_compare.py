"""``gpuwm render --compare``: what the door decides, and nothing it draws.

Every pixel and every number of a comparison sheet is the Rust engine's
(``rw_compare``), and its own tests hold those.  What this door decides is
which files are one run's frames, which earlier frame an hourly
accumulation needs beside it, and what the engine is told -- so that is
what is held here, without an engine.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from gpuwm import cli, render_compare, rustwx_lanes


def _touch(folder: Path, *names: str) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in names:
        path = folder / name
        path.write_bytes(b"")
        paths.append(path)
    return paths


def _args(**overrides) -> argparse.Namespace:
    words = ["render", "--compare", "hrrr", "frame"]
    args = cli.build_parser(render_only=True).parse_args(words)
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_a_frame_name_gives_its_domain_and_valid_time_in_both_spellings():
    colon = render_compare.frame_identity(
        Path("wrfout_d01_2026-10-03_06:00:00"))
    underscore = render_compare.frame_identity(
        Path("wrfout_d01_2026-10-03_06_00_00"))
    assert colon == underscore
    assert colon[0] == "d01"
    assert colon[1].isoformat() == "2026-10-03T06:00:00"
    assert render_compare.frame_identity(Path("model_output.nc")) is None
    assert render_compare.frame_identity(
        Path("wrfout_d01_2026-13-40_06_00_00")) is None


def test_a_run_folder_becomes_its_history_frames(tmp_path):
    run = tmp_path / "run"
    frames = _touch(run / "out" / "wrfout",
                    "wrfout_d01_2026-10-03_05_00_00",
                    "wrfout_d01_2026-10-03_06_00_00")
    _touch(run / "out", "report.json")
    assert render_compare.expand_inputs([run]) == sorted(frames)
    # The frames' own folder names the same frames.
    assert render_compare.expand_inputs([run / "out" / "wrfout"]) == sorted(frames)
    # A file is taken as given, whatever it is called.
    assert render_compare.expand_inputs([frames[0]]) == [frames[0]]


def test_a_folder_with_no_frames_is_refused_with_where_it_looked(tmp_path):
    _touch(tmp_path / "empty", "notes.txt")
    with pytest.raises(ValueError) as refusal:
        render_compare.expand_inputs([tmp_path / "empty"])
    text = str(refusal.value)
    assert "out/wrfout" in text and "wrfout_dNN" in text


def test_the_frame_an_hour_earlier_rides_along_as_context(tmp_path):
    folder = tmp_path / "wrfout"
    earlier, kept = _touch(folder, "wrfout_d01_2026-10-03_05_00_00",
                           "wrfout_d01_2026-10-03_06_00_00")
    groups = render_compare.group_frames([kept])
    assert groups == [([kept], [earlier])]
    # Asked for itself, the earlier frame is compared, not context; its own
    # earlier frame does not exist, so nothing rides along.
    groups = render_compare.group_frames([kept, earlier])
    assert groups == [([earlier, kept], [])]


def test_each_nest_of_a_run_goes_to_the_engine_on_its_own(tmp_path):
    folder = tmp_path / "wrfout"
    outer, inner = _touch(folder, "wrfout_d01_2026-10-03_06_00_00",
                          "wrfout_d02_2026-10-03_06_00_00")
    groups = render_compare.group_frames([inner, outer])
    assert [members for members, _ in groups] == [[outer], [inner]]


def test_the_engine_is_told_everything_the_door_was(tmp_path):
    frame = tmp_path / "wrfout_d01_2026-10-03_21_00_00"
    context = tmp_path / "wrfout_d01_2026-10-03_20_00_00"
    args = _args(out=tmp_path / "render", compare="hrrr",
                 products="t2m,refc", compare_difference="off",
                 compare_cycle="2026100300", compare_offline=True,
                 compare_gallery=tmp_path / "gallery",
                 compare_reference_dir=tmp_path / "hrrr",
                 compare_cache=tmp_path / "cache",
                 source_label="WOOF test", compare_label=None,
                 layout="flat")
    command = render_compare.engine_command(
        Path("rw_compare"), args, store=tmp_path / "store", frames=[frame],
        context=[context], size=(1200, 900))

    def after(flag):
        return command[command.index(flag) + 1]

    assert after("--reference") == "hrrr"
    assert after("--products") == "t2m,refc"
    assert after("--difference") == "off"
    assert after("--layout") == "flat"
    assert after("--cycle") == "2026100300"
    assert (after("--width"), after("--height")) == ("1200", "900")
    assert after("--source-label") == "WOOF test"
    assert after("--run-label") == render_compare.RUN_LABEL
    assert after("--flat-dir") == str(tmp_path / "gallery")
    assert after("--reference-dir") == str(tmp_path / "hrrr")
    assert after("--reference-cache") == str(tmp_path / "cache")
    assert "--offline" in command
    assert after("--context") == str(context)
    # The frames to draw are the positionals, last.
    assert command[-1] == str(frame)


def test_reference_lists_follow_the_native_catalogue_and_keep_order():
    catalog = render_compare.reference_catalog(
        "REFERENCE\thrrr\tHRRR\nREFERENCE\trrfs\tRRFS\n"
        "REFERENCE\tmrms\tMRMS\nCAPABILITY\tmrms\trefc\tobservation\n"
        "CAPABILITY\tmrms\tqpf1h\tobservation\n")
    assert catalog["mrms"]["products"] == {"refc": "observation", "qpf1h": "observation"}
    assert render_compare.selected_references(" hrrr, mrms,rrfs,hrrr ", catalog) == ["hrrr", "mrms", "rrfs"]
    with pytest.raises(ValueError, match="native build knows"):
        render_compare.selected_references("hrrr,unknown", catalog)
    with pytest.raises(ValueError):
        render_compare.selected_references("hrrr,", catalog)


def test_multi_reference_command_preserves_the_requested_native_order(tmp_path):
    args = _args(out=tmp_path / "render", compare="rrfs,hrrr,mrms", products="refc,qpf1h")
    command = render_compare.engine_command(Path("rw_compare"), args, store=tmp_path / "store",
        frames=[tmp_path / "frame"], context=[], size=(1200, 900))
    assert command[command.index("--reference") + 1] == "rrfs,hrrr,mrms"
    assert command[command.index("--difference") + 1] == "auto"


def test_comparison_forwards_an_explicit_theme_and_keeps_default_implicit(tmp_path):
    args = _args(out=tmp_path / "render", theme=None)
    command = render_compare.engine_command(Path("rw_compare"), args, store=tmp_path / "store",
        frames=[tmp_path / "frame"], context=[], size=(1200, 900))
    assert "--theme" not in command
    args.theme = "woof-dark"
    command = render_compare.engine_command(Path("rw_compare"), args, store=tmp_path / "store",
        frames=[tmp_path / "frame"], context=[], size=(1200, 900))
    assert command[command.index("--theme") + 1] == "woof-dark"


def test_observation_source_lines_keep_actual_seconds(capsys):
    render_compare._relay("SOURCE\tmrms\tobserved\t2026-10-03T13:00:37+00:00\trefc\tfile.grib2.gz\n", {})
    assert "observed 2026-10-03T13:00:37+00:00 product refc" in capsys.readouterr().out


def test_the_cache_is_kept_between_renders_unless_told_otherwise(tmp_path):
    args = _args(out=tmp_path / "render")
    command = render_compare.engine_command(
        Path("rw_compare"), args, store=tmp_path / "store",
        frames=[tmp_path / "frame"], context=[], size=(1200, 900))
    cache = command[command.index("--reference-cache") + 1]
    assert cache == str(render_compare.default_reference_cache())
    assert "--offline" not in command and "--cycle" not in command


def test_the_fast_and_the_full_parser_agree_on_the_comparison_flags():
    words = ["render", "--compare", "hrrr", "--compare-cycle", "2026100300",
             "--compare-difference", "on", "--compare-gallery", "flat",
             "--compare-offline", "run-folder"]
    fast = vars(cli.build_parser(render_only=True).parse_args(words))
    full = vars(cli.build_parser().parse_args(words))
    full.pop("ingest_preflight_handler", None)
    assert fast == full
    assert fast["compare"] == "hrrr"


def test_an_unbuilt_engine_is_refused_with_the_line_that_builds_it(
        monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(rustwx_lanes, "find_compare_bin", lambda: None)
    frame = _touch(tmp_path, "wrfout_d01_2026-10-03_06_00_00")[0]
    args = _args(wrfout=[frame], out=tmp_path / "render")
    assert render_compare.compare_main(args) == 2
    said = capsys.readouterr().err
    assert "rw_compare" in said and "cargo build" in said
    assert not (tmp_path / "render").exists(), "a refusal writes nothing"


def test_a_comparison_does_not_combine_with_pair_sheets(tmp_path, capsys):
    from gpuwm import render

    args = _args(pair=[tmp_path / "a", tmp_path / "b"], wrfout=[])
    assert render.render_main(args) == 2
    assert "--pair" in capsys.readouterr().err


def test_engine_lines_are_relayed_in_the_door_s_own_words(capsys):
    tally = {"rendered": [], "skipped": 0, "failed": 0}
    render_compare._relay("RENDERED\tt2m\tf006\tout/sheet.png\n", tally)
    render_compare._relay(
        "SKIPPED\tqpf1h\t/run/wrfout_d01_2026-10-03_06_00_00\tno earlier "
        "frame\n", tally)
    render_compare._relay(
        "FAILED\t*\t/run/wrfout_d01_2026-10-03_06_00_00\tfetch failed\n",
        tally)
    render_compare._relay("FINISHED rendered=1 skipped=1 failed=1\n", tally)
    assert tally == {"rendered": ["out/sheet.png"], "skipped": 1, "failed": 1}
    said = capsys.readouterr()
    assert "render: out/sheet.png" in said.out
    assert "skipped qpf1h for wrfout_d01_2026-10-03_06_00_00" in said.err
    assert "FAILED every product" in said.err
