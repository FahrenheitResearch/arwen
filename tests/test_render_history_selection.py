"""The render front door against a wrfout whose run trimmed its history.

A user who set ``[output] history_drop`` and then asked for a product
that needed one of the dropped variables must be TOLD SO, by name, with
the remedy -- not handed a traceback, and not handed a bare "not in
file" that reads like a corrupt artifact rather than their own choice
from three hours earlier.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gpuwm import render
from gpuwm.io import history_selection as hs
from gpuwm.io.wrfout import WrfoutWriter


NX, NY, NZ = 8, 6, 4


def _trimmed_wrfout(path, dropped=("REFL_10CM",)):
    frame = {
        "T": np.zeros((NZ, NY, NX), np.float32),
        "T2": np.zeros((NY, NX), np.float32),
        "U10": np.zeros((NY, NX), np.float32),
        "V10": np.zeros((NY, NX), np.float32),
        "XLAT": np.zeros((NY, NX), np.float32),
        "XLONG": np.zeros((NY, NX), np.float32),
    }
    selection = hs.HistorySelection.from_mapping(
        {"history_drop": list(dropped)})
    produced = tuple(frame) + tuple(dropped)
    with WrfoutWriter(path, nx=NX, ny=NY, nz=NZ, dx=3000.0, dy=3000.0,
                      title="gpuwm test",
                      global_attrs=selection.wrfout_attrs(produced)) as w:
        w.write_frame("1974-04-03_12:00:00", frame)
    return Path(path)


def _plain_wrfout(path):
    frame = {
        "T2": np.zeros((NY, NX), np.float32),
        "XLAT": np.zeros((NY, NX), np.float32),
        "XLONG": np.zeros((NY, NX), np.float32),
    }
    with WrfoutWriter(path, nx=NX, ny=NY, nz=NZ, dx=3000.0, dy=3000.0,
                      title="gpuwm test") as w:
        w.write_frame("1974-04-03_12:00:00", frame)
    return Path(path)


def test_a_wrfout_reports_the_variables_its_run_dropped(tmp_path):
    path = _trimmed_wrfout(tmp_path / "trim.nc", ("REFL_10CM", "QCLOUD"))
    assert render.history_dropped_variables(path) == frozenset(
        {"REFL_10CM", "QCLOUD"})


def test_a_wrfout_from_a_default_run_reports_no_dropped_variables(tmp_path):
    assert render.history_dropped_variables(
        _plain_wrfout(tmp_path / "full.nc")) == frozenset()


def test_an_unreadable_path_abstains_rather_than_raising(tmp_path):
    """The pre-check must never be the reason a render dies."""
    assert render.history_dropped_variables(
        tmp_path / "nothing-here.nc") == frozenset()


def test_the_precheck_names_the_product_the_variable_and_the_remedy(tmp_path):
    path = _trimmed_wrfout(tmp_path / "trim.nc")
    killed = render.history_killed_products([path], ("refl", "t2"))
    assert set(killed) == {"refl"}
    detail = killed["refl"]
    assert "REFL_10CM" in detail
    assert "[output]" in detail


def test_a_product_whose_inputs_survived_is_not_reported(tmp_path):
    path = _trimmed_wrfout(tmp_path / "trim.nc")
    assert render.history_killed_products([path], ("t2",)) == {}


def test_the_rust_catalog_slug_is_checked_too(tmp_path):
    path = _trimmed_wrfout(tmp_path / "trim.nc")
    killed = render.history_killed_products(
        [path], ("composite_reflectivity",))
    assert "composite_reflectivity" in killed


def test_all_is_not_a_request_for_a_specific_dead_product(tmp_path):
    """``--products all`` asks for what the file supports, so it stands."""
    path = _trimmed_wrfout(tmp_path / "trim.nc")
    assert render.history_killed_products([path], ("all",)) == {}


def test_an_unknown_slug_is_not_claimed_by_the_table(tmp_path):
    """The renderer owns 151 products; this table claims only what it knows."""
    path = _trimmed_wrfout(tmp_path / "trim.nc")
    assert render.history_killed_products([path], ("srh_0_1km",)) == {}


def test_every_requested_product_dead_is_a_refusal_not_a_traceback(tmp_path):
    path = _trimmed_wrfout(tmp_path / "trim.nc")
    refusal = render.history_selection_refusal([path], ("refl",))
    assert refusal is not None
    assert "refl" in refusal
    assert "REFL_10CM" in refusal
    assert "history_drop" in refusal
    # The remedy: how to get the picture back, and how to see what is left.
    assert "--list-products" in refusal


def test_some_products_dead_is_a_note_and_the_render_goes_on(tmp_path):
    path = _trimmed_wrfout(tmp_path / "trim.nc")
    assert render.history_selection_refusal([path], ("refl", "t2")) is None
    note = render.history_selection_notice([path], ("refl", "t2"))
    assert note is not None and "refl" in note and "REFL_10CM" in note


def test_a_full_default_file_produces_neither_refusal_nor_note(tmp_path):
    path = _plain_wrfout(tmp_path / "full.nc")
    assert render.history_selection_refusal([path], ("refl", "t2")) is None
    assert render.history_selection_notice([path], ("refl", "t2")) is None


def _render_argv(argv):
    """One `gpuwm render ...` invocation, through the real CLI parser."""
    import argparse

    parser = argparse.ArgumentParser(prog="gpuwm")
    render.register_cli(parser.add_subparsers(dest="command"))
    return parser.parse_args(["render", *argv])


def test_the_front_door_refuses_by_name_and_leaves_no_run_directory(
        tmp_path, capsys):
    """The whole point, through the real CLI dispatch."""
    path = _trimmed_wrfout(tmp_path / "trim.nc")
    out = tmp_path / "renders"
    code = render.render_main(_render_argv([
        "--engine", "matplotlib", "--products", "refl",
        "--out", str(out), str(path)]))
    captured = capsys.readouterr()
    assert code != 0
    message = captured.err + captured.out
    assert "REFL_10CM" in message
    assert "history_drop" in message
    assert "Traceback" not in message
    assert not out.exists()


def test_the_front_door_draws_the_surviving_product_and_says_what_died(
        tmp_path, capsys):
    """A live product is still drawn; the dead one is named, not silent."""
    path = _trimmed_wrfout(tmp_path / "trim.nc")
    out = tmp_path / "renders"
    render.render_main(_render_argv([
        "--engine", "matplotlib", "--products", "refl,t2",
        "--out", str(out), str(path)]))
    message = "".join(capsys.readouterr())
    assert "REFL_10CM" in message
    assert "history_drop" in message
    assert "Traceback" not in message
