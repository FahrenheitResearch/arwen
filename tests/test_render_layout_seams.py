"""The layout's own seams: the lead grammar, the frame clock, the delivery.

These are the parts of :mod:`gpuwm.render_layout` that are pure path and
filename work, so they run with no renderer binary, no netCDF and no
``wrf`` package -- which is what separates this file from
``tests/test_render_layout.py``, whose fixtures build real wrfout frames
and which therefore skips wherever ``wrf`` is not installed.  A seam
nothing can exercise on a plain CPU box is a seam that breaks unnoticed.
"""
from __future__ import annotations

import datetime
import os
from pathlib import Path

import pytest

from gpuwm import render_layout


# ------------------------------------------------- the lead is a width

@pytest.mark.parametrize("lead,day", [
    ("f999", "2026-09-30"), ("f1000", "2026-09-30"), ("f1001", "2026-09-30")])
def test_a_lead_past_999_hours_parses_like_any_other(lead, day):
    """Three digits is the zero-padding width, never a ceiling."""
    name = f"arwen_wrf_20260820_0z_{lead}_d01-12km_2m_temperature.png"
    parsed = render_layout.parse_engine_output(name)
    assert parsed is not None, name
    assert parsed[0] == "d01-12km"
    assert parsed[1] == "2m_temperature"
    assert parsed[2] == day


def test_a_lead_past_999_hours_counts_every_hour_of_itself():
    """``f1000`` is 1000 hours, not 100 with a digit left over."""
    hundred = render_layout.parse_engine_output(
        "arwen_wrf_20260820_0z_f100_d01-12km_2m_temperature.png")
    thousand = render_layout.parse_engine_output(
        "arwen_wrf_20260820_0z_f1000_d01-12km_2m_temperature.png")
    assert hundred[2] == "2026-08-24"
    assert thousand[2] == "2026-09-30"


@pytest.mark.parametrize("lead", ["f999", "f1000", "f1001"])
def test_a_lead_past_999_hours_round_trips_through_its_two_folders(lead):
    name = f"arwen_wrf_20260820_0z_{lead}_d01-12km_2m_temperature.png"
    delivered = render_layout.delivered_name(
        name, domain="d01-12km", product="2m_temperature")
    assert "d01-12km" not in delivered
    assert delivered == f"arwen_wrf_20260820_0z_{lead}.png"
    assert render_layout.engine_name(
        delivered, domain="d01-12km", product="2m_temperature") == name


# ---------------------------------------------------- the frame clock

def test_engine_output_time_reads_the_exact_time_suffix_first():
    stamp = render_layout.engine_output_time(
        "arwen_wrf_20260820_0z_f006_d01-12km_2m_temperature"
        "_valid_20260820_061500z_lead_006h15m00s.png")
    assert stamp == datetime.datetime(2026, 8, 20, 6, 15, 0)


def test_engine_output_time_reads_cycle_plus_lead_without_a_suffix():
    assert render_layout.engine_output_time(
        "arwen_wrf_20260820_0z_f006_d01-12km_2m_temperature.png"
    ) == datetime.datetime(2026, 8, 20, 6)
    assert render_layout.engine_output_time(
        "arwen_wrf_20260820_0z_f1000_d01-12km_2m_temperature.png"
    ) == datetime.datetime(2026, 9, 30, 16)


@pytest.mark.parametrize("name", [
    "panel.png",
    "arwen_wrf_20260820_0z_f06_d01-12km_2m_temperature.png",
    "2m_temperature.png",
])
def test_engine_output_time_answers_none_rather_than_raising(name):
    """A total function: the caller gets a fact, never a dead render."""
    assert render_layout.engine_output_time(name) is None


def test_engine_output_time_agrees_with_the_render_door():
    """Two readers of one clock, for as long as both exist."""
    from gpuwm import render

    readable = [
        "arwen_wrf_20260820_0z_f000_d01-12km_2m_temperature.png",
        "arwen_wrf_20260820_18z_f006_d02-3km_composite_reflectivity.png",
        "arwen_wrf_20260820_0z_f1000_d01-12km_2m_temperature.png",
        "arwen_wrf_20260820_0z_f006_d01-12km_2m_temperature"
        "_valid_20260820_061500z_lead_006h15m00s.png",
    ]
    for name in readable:
        assert render_layout.engine_output_time(name) == \
            render._engine_output_time(name), name
    with pytest.raises(ValueError):
        render._engine_output_time("panel.png")
    assert render_layout.engine_output_time("panel.png") is None


# ------------------------------------------------ history frame discovery

def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def test_history_frames_finds_every_episode(tmp_path):
    root = tmp_path / "wrfout"
    flat = _touch(root / "wrfout_d01_1974-04-03_18_00_00")
    first = _touch(root / "d05" / "episode-001" / "wrfout_d05_1974-04-03_18_00_00")
    second = _touch(root / "d05" / "episode-002" / "wrfout_d05_1974-04-03_18_00_00")
    frames = render_layout.history_frames(root)
    assert [frame.path for frame in frames] == [flat, first, second]
    assert [frame.grid for frame in frames] == ["d01", "d05", "d05"]
    assert [frame.episode for frame in frames] == [None, 1, 2]


def test_history_frames_order_is_the_name_then_the_episode(tmp_path):
    root = tmp_path / "wrfout"
    parent = _touch(root / "wrfout_d01_1974-04-03_18_00_00")
    first = _touch(
        root / "d05" / "episode-001" / "wrfout_d05_1974-04-03_18_00_00")
    second = _touch(
        root / "d05" / "episode-002" / "wrfout_d05_1974-04-03_18_00_00")
    frames = [frame.path for frame in render_layout.history_frames(root)]
    assert frames == [parent, first, second]
    # The trap a naive recursive fix reintroduces: a path-string sort
    # puts every nested episode ahead of the top-level frame, because
    # ``d05/`` sorts before ``wrfout_d01_...``.
    assert sorted(root.rglob("wrfout_d*")) != frames
    assert sorted(root.rglob("wrfout_d*"))[0] == first


def test_history_frames_skips_in_flight_temporaries(tmp_path):
    root = tmp_path / "wrfout"
    kept = _touch(root / "wrfout_d01_1974-04-03_18_00_00")
    _touch(root / "wrfout_d01_1974-04-03_19_00_00.tmp1234")
    (root / "wrfout_d09_decoy").mkdir(parents=True, exist_ok=True)
    assert [frame.path for frame in render_layout.history_frames(root)] == [kept]


@pytest.mark.parametrize("episode", [1, 2, 1234])
def test_history_frames_is_the_inverse_of_the_writers_spelling(tmp_path, episode):
    segment = render_layout.episode_segment(episode)
    frame = _touch(tmp_path / "d05" / segment / "wrfout_d05_1974-04-03_18_00_00")
    found = render_layout.history_frames(tmp_path)
    assert [row.path for row in found] == [frame]
    assert found[0].episode == episode


def test_history_frames_of_a_missing_directory_is_empty(tmp_path):
    assert render_layout.history_frames(tmp_path / "never-written") == []


# ------------------------------------------------------- the deliver seam

_NAME = "arwen_scratch_note.png"


def test_deliver_files_an_unreadable_frame_instead_of_leaving_it_flat(tmp_path):
    root = tmp_path / "png"
    source = _touch(root / _NAME)
    path, note = render_layout.deliver(
        root, source, domain="d02-3km", product=None, day=None,
        layout=render_layout.NESTED)
    assert path == root / "d02-3km" / "unclassified" / "undated" / _NAME
    assert path.is_file()
    assert not source.exists()
    assert note is None
    assert render_layout.iter_rendered(root) == [path]


def test_deliver_falls_back_to_copy_when_the_move_cannot_be_done(tmp_path, monkeypatch):
    root = tmp_path / "png"
    source = _touch(root / _NAME)
    source.write_bytes(b"picture")
    calls = []
    real = os.replace

    def one_refusal(src, dst):
        calls.append((src, dst))
        if len(calls) == 1:
            raise PermissionError(13, "held by another process")
        return real(src, dst)

    monkeypatch.setattr(os, "replace", one_refusal)
    path, note = render_layout.deliver(
        root, source, domain="d02-3km", product="2m_temperature",
        day="1974-04-03", layout=render_layout.NESTED)
    assert path == root / "d02-3km" / "2m_temperature" / "1974-04-03" / _NAME
    assert path.read_bytes() == b"picture"
    assert not source.exists()
    assert note is not None and "copy" in note


def test_deliver_that_cannot_move_at_all_names_the_reason_and_keeps_the_file(
        tmp_path, monkeypatch):
    root = tmp_path / "png"
    source = _touch(root / _NAME)
    source.write_bytes(b"picture")

    def refuse(*args, **kwargs):
        raise PermissionError(13, "held by another process")

    monkeypatch.setattr(os, "replace", refuse)
    monkeypatch.setattr(render_layout.shutil, "copyfileobj", refuse)
    path, note = render_layout.deliver(
        root, source, domain="d02-3km", product="2m_temperature",
        day="1974-04-03", layout=render_layout.NESTED)
    assert path == source
    assert path.is_file()
    assert note is not None and "held by another process" in note


def test_deliver_leaves_a_flat_layout_exactly_where_it_is(tmp_path):
    root = tmp_path / "png"
    source = _touch(root / _NAME)
    path, note = render_layout.deliver(
        root, source, domain="d02-3km", product="2m_temperature",
        day="1974-04-03", layout=render_layout.FLAT)
    assert path == source and note is None and path.is_file()


def test_probe_delivery_root_answers_none_for_a_root_that_works(tmp_path):
    assert render_layout.probe_delivery_root(tmp_path / "made-on-demand") is None
    assert not list((tmp_path / "made-on-demand").iterdir())


def test_probe_delivery_root_names_the_breakage_and_the_way_out(tmp_path):
    blocked = tmp_path / "blocked"
    blocked.write_bytes(b"a file where a directory was asked for")
    sentence = render_layout.probe_delivery_root(blocked)
    assert sentence is not None
    assert "--out" in sentence and str(blocked) in sentence
