"""Frame pairing for ``gpuwm render --diff``: two runs matched by valid time.

The renderer checks the valid time and the grid again from the files'
contents; these tests hold the door's half: which frames it hands over as
pairs, which it reports as unpaired, and which it refuses by name.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from gpuwm import cli, render, render_difference, rustwx


def _touch(folder: Path, name: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_bytes(b"")
    return path


def test_frames_pair_by_valid_time_whatever_each_run_started_from(tmp_path):
    # Run A started at 12Z, run B at 18Z: they share 18Z and 19Z only.
    a = [_touch(tmp_path / "a", f"wrfout_d01_2025-03-15_{h:02d}_00_00")
         for h in range(12, 20)]
    b = [_touch(tmp_path / "b", f"wrfout_d01_2025-03-15_{h:02d}_00_00")
         for h in (18, 19, 20)]
    pairing = render_difference.pair_frames_by_valid_time(a, b)
    assert [(domain, valid.hour) for domain, valid, _, _ in pairing.pairs] == [
        ("d01", 18), ("d01", 19)]
    for _, valid, a_file, b_file in pairing.pairs:
        assert render_difference.wrfout_valid_time(a_file)[1] == valid
        assert render_difference.wrfout_valid_time(b_file)[1] == valid
    assert len(pairing.only_a) == 6
    assert [p.name for p in pairing.only_b] == ["wrfout_d01_2025-03-15_20_00_00"]
    notice = render_difference.unpaired_notice(pairing)
    assert "only run B has d01 03/15 20:00Z" in notice


def test_two_nests_pair_within_their_own_domain(tmp_path):
    a = [_touch(tmp_path / "a", "wrfout_d01_2025-03-16_00_00_00"),
         _touch(tmp_path / "a", "wrfout_d02_2025-03-16_00_00_00")]
    b = [_touch(tmp_path / "b", "wrfout_d02_2025-03-16_00_00_00")]
    pairing = render_difference.pair_frames_by_valid_time(a, b)
    assert [(d, a_file.name) for d, _, a_file, _ in pairing.pairs] == [
        ("d02", "wrfout_d02_2025-03-16_00_00_00")]
    assert [p.name for p in pairing.only_a] == ["wrfout_d01_2025-03-16_00_00_00"]
    assert render_difference.unpaired_notice(
        render_difference.pair_frames_by_valid_time(b, b)) is None


def test_a_frame_with_no_time_in_its_name_is_refused_by_name(tmp_path):
    a = [_touch(tmp_path, "case6-ifs-3km-scoring.nc")]
    b = [_touch(tmp_path, "wrfout_d01_2025-03-16_00_00_00")]
    with pytest.raises(ValueError, match="case6-ifs-3km-scoring.nc does not carry a valid time"):
        render_difference.pair_frames_by_valid_time(a, b)


def test_one_run_with_two_frames_at_one_valid_time_is_refused(tmp_path):
    a = [_touch(tmp_path / "x", "wrfout_d01_2025-03-16_00_00_00"),
         _touch(tmp_path / "y", "wrfout_d01_2025-03-16_00_00_00")]
    with pytest.raises(ValueError, match="one frame per domain and valid time"):
        render_difference.pair_frames_by_valid_time(a, a[:1])


def test_a_run_folder_lists_its_history_frames(tmp_path):
    for h in (1, 0):
        _touch(tmp_path, f"wrfout_d01_2025-03-16_{h:02d}_00_00")
    _touch(tmp_path, "wrfout_d01_2025-03-16_02_00_00.tmp1")
    _touch(tmp_path, "namelist.input")
    frames = render_difference.run_frames(tmp_path)
    assert [p.name for p in frames] == ["wrfout_d01_2025-03-16_00_00_00",
                                        "wrfout_d01_2025-03-16_01_00_00"]
    assert render_difference.wrfout_valid_time(frames[0]) == (
        "d01", dt.datetime(2025, 3, 16, 0, 0, 0))
    # The colon spelling a Linux history writer uses names the same time.
    assert render_difference.wrfout_valid_time(
        Path("wrfout_d01_2025-03-16_00:00:00")) == (
        "d01", dt.datetime(2025, 3, 16, 0, 0, 0))


def test_the_renderer_difference_line_parses_and_the_contract_names_it():
    row = rustwx.parse_difference_line(
        "DIFFERENCE d01-3km_2m_temperature units=°F half_range=10 step=2 "
        "rule=table defined_cells=1835008 max_abs=14.2")
    assert row == {"key": "d01-3km_2m_temperature", "units": "°F",
                   "half_range": "10", "step": "2", "rule": "table",
                   "defined_cells": "1835008", "max_abs": "14.2"}
    # The run-level line names no product.
    assert rustwx.parse_difference_line(
        'DIFFERENCE valid=2025-03-16 00:00Z a_frame=0 b_frame=0') is None
    assert "\t--diff-against\t" in rustwx.RENDERER_ABI_MARKER
    assert "\tDIFFERENCE" in rustwx.RENDERER_ABI_MARKER


def test_default_canvas_and_difference_flags_agree_at_both_cli_doors():
    words = ["render", "--diff", "run-a", "run-b", "--diff-sheet",
             "--diff-labels", "forecast-a", "forecast-b"]
    fast = vars(cli.build_parser(render_only=True).parse_args(words))
    full = vars(cli.build_parser().parse_args(words))
    full.pop("ingest_preflight_handler", None)
    assert fast == full
    assert fast["size"] == "auto"
    assert render.parse_size("auto") is None
    assert render.parse_size("640x480") == (640, 480)
    assert fast["diff"] == [Path("run-a"), Path("run-b")]


@pytest.mark.parametrize("other", ["pair", "compare"])
def test_a_run_difference_refuses_other_comparison_inputs(other, capsys):
    words = ["render", "--diff", "run-a", "run-b"]
    words += ["--pair", "png-a", "png-b"] if other == "pair" else ["--compare", "hrrr"]
    args = cli.build_parser(render_only=True).parse_args(words)
    assert render.render_main(args) == 2
    reason = capsys.readouterr().err
    assert "--diff" in reason and "--" + other in reason
    assert "choose one comparison" in reason


def _recorded_pairing(a, b, stamps, identities=None, **options):
    def metadata(path):
        return ((identities or {}).get(path, (path.parent, path.name.split("_")[1])),
                tuple(dt.datetime(2025, 3, 16, hour) for hour in stamps[path]))
    return render_difference.pair_frames_by_valid_time(
        a, b, reader=metadata, **options)


def test_inner_records_pair_by_actual_time_and_keep_both_full_timelines():
    a = Path("a/wrfout_d01_2025-03-16_00_00_00")
    b = Path("b/wrfout_d01_2025-03-16_01_00_00")
    pairing = _recorded_pairing([a], [b], {a: [0, 1, 2], b: [1, 2, 3]}, timeidx=2)
    assert [(domain, valid.hour) for domain, valid, *_ in pairing.pairs] == [("d01", 2)]
    assert pairing.contexts[("d01", dt.datetime(2025, 3, 16, 2))] == ([a], [b], 2)
    assert "only run B has d01 03/16 03:00Z" in render_difference.unpaired_notice(pairing)


def test_timeidx_is_per_file_unless_the_run_is_a_series():
    a = [Path(f"a/wrfout_d01_2025-03-16_{hour:02d}_00_00") for hour in (0, 2)]
    b = [Path(f"b/{path.name}") for path in a]
    stamps = {a[0]: [0, 1], a[1]: [2, 3], b[0]: [0, 1], b[1]: [2, 3]}
    individual = _recorded_pairing(a, b, stamps, timeidx=1)
    series = _recorded_pairing(a, b, stamps, timeidx=1, series=True)
    assert [valid.hour for _, valid, *_ in individual.pairs] == [1, 3]
    assert [valid.hour for _, valid, *_ in series.pairs] == [1]
    assert all(paths_a == a and paths_b == b
               for paths_a, paths_b, _ordinal in individual.contexts.values())


def test_timeidx_selects_each_domain_and_keeps_its_window_context():
    a = [Path(f"a/wrfout_{domain}_2025-03-16_{hour:02d}_00_00")
         for hour in (0, 1) for domain in ("d01", "d02")]
    b = [Path(f"b/{path.name}") for path in a]
    stamps = {path: [int(path.name[-8:-6])] for path in [*a, *b]}
    pairing = _recorded_pairing(a, b, stamps, timeidx=1, series=True)
    assert [(domain, valid.hour) for domain, valid, *_ in pairing.pairs] == [
        ("d01", 1), ("d02", 1)]
    for (domain, _valid), (paths_a, paths_b, ordinal) in pairing.contexts.items():
        assert len(paths_a) == len(paths_b) == 2 and ordinal == 1
        assert all(f"wrfout_{domain}_" in path.name for path in [*paths_a, *paths_b])


def test_missing_selected_a_time_is_named_and_never_shifted_to_a_shared_time():
    a = Path("a/wrfout_d01_2025-03-16_00_00_00")
    b = Path("b/wrfout_d01_2025-03-16_02_00_00")
    pairing = _recorded_pairing([a], [b], {a: [0, 1, 2], b: [2]}, timeidx=1, series=True)
    assert pairing.pairs == []
    assert "only run A has d01 03/16 01:00Z" in render_difference.unpaired_notice(pairing)


def test_out_of_range_difference_timeidx_is_refused_before_rendering():
    a = Path("a/wrfout_d01_2025-03-16_00_00_00")
    b = Path("b/wrfout_d01_2025-03-16_00_00_00")
    with pytest.raises(ValueError, match="--timeidx 2 out of range; file has 2 frame"):
        _recorded_pairing([a], [b], {a: [0, 1], b: [0, 1]}, timeidx=2)


def test_existing_moving_nest_context_reaches_the_difference_store():
    a = [Path(f"a/wrfout_d02_2025-03-16_{hour:02d}_00_00") for hour in (0, 1)]
    b = [Path(f"b/{path.name}") for path in a]
    stamps = {a[0]: [0], a[1]: [1], b[0]: [0], b[1]: [1]}
    identities = {path: (path.parent, index) for side in (a, b)
                  for index, path in enumerate(side)}

    def groups(paths):
        return [(paths[:1], []), (paths, paths[:1])]

    pairing = _recorded_pairing(a, b, stamps, identities=identities,
                               series_groups=groups, series=True)
    assert [(domain, valid.hour) for domain, valid, *_ in pairing.pairs] == [
        ("d02", 0), ("d02", 1)]
    assert pairing.contexts[("d02", dt.datetime(2025, 3, 16, 1))] == (a, b, 1)
