"""Complete difference contexts and labels survive native transport."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile

import pytest

from gpuwm import rustwx


def _timeline(root, count=1441):
    return [root / f"day-{index // 144:02d}"
            / f"wrfout_d01_2026-10-04_{index:04d}, history 🌦.nc"
            for index in range(count)]


def _response(command, *, failed=False):
    return subprocess.CompletedProcess(
        command, int(failed),
        "RENDERED 2m_temperature pictures/difference.png\n"
        "DIFFERENCE 2m_temperature half_range=10 step=2 rule=table\n",
        "FAILED 2m_temperature field missing\n" if failed else "")


def test_short_difference_keeps_complete_context_selected_frame_and_labels(
        tmp_path, monkeypatch):
    a = list(reversed(_timeline(tmp_path / "run a", 3)))
    b = _timeline(tmp_path / "run b", 4)
    labels = ("left, forecast", "right, analysis")
    calls = []

    def native(command, **kwargs):
        calls.append((command, kwargs))
        return _response(command)

    monkeypatch.setattr(rustwx.subprocess, "run", native)
    monkeypatch.setattr(rustwx, "renderer_env", lambda: {"KEEP": "value"})
    written, failures, _, differences = rustwx.run_renderer_difference(
        Path("rw_wrfbatch"), a, b, store_root=tmp_path / "store",
        out_dir=tmp_path / "pictures", products="2m_temperature",
        labels=labels, timeidx=2, overlays=Path("overlays.json"),
        annotate=Path("annotations.json"), streamlines=False,
        width=1200, height=900, sheet=True)
    command, kwargs = calls[0]
    assert command[-len(a):] == list(map(str, a))
    against = [command[index + 1] for index, part in enumerate(command)
               if part == "--diff-against"]
    assert against == list(map(str, b))
    assert command[command.index("--frames") + 1] == "2"
    assert command[command.index("--diff-label-a") + 1] == labels[0]
    assert command[command.index("--diff-label-b") + 1] == labels[1]
    assert "--diff-labels" not in command
    assert "--inputs-json" not in command and "--diff-inputs-json" not in command
    assert command[command.index("--overlays") + 1] == "overlays.json"
    assert command[command.index("--annotate") + 1] == "annotations.json"
    assert "--barbs" in command and "--diff-sheet" in command
    assert command[command.index("--layout") + 1] == "fixed"
    assert kwargs["env"] == {"KEEP": "value"}
    assert written == [Path("pictures/difference.png")] and failures == []
    assert differences[0]["key"] == "2m_temperature"


def test_single_file_difference_defaults_to_first_native_ordinal(
        tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(rustwx.subprocess, "run",
                        lambda command, **_: calls.append(command) or _response(command))
    monkeypatch.setattr(rustwx, "renderer_env", lambda: {})
    rustwx.run_renderer_difference(
        Path("rw_wrfbatch"), Path("a, frame.nc"), Path("b, frame.nc"),
        store_root=tmp_path / "store", out_dir=tmp_path / "pictures",
        products="2m_temperature", labels=("A", "B"))
    command = calls[0]
    assert command[-1] == "a, frame.nc"
    assert command[command.index("--diff-against") + 1] == "b, frame.nc"
    assert command[command.index("--frames") + 1] == "0"


@pytest.mark.parametrize("large_side", ["a", "b", "both"])
@pytest.mark.parametrize("outcome", ["passed", "failed", "cannot start"])
def test_long_difference_transports_both_complete_lists_and_removes_files(
        tmp_path, monkeypatch, large_side, outcome):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    a = list(reversed(_timeline(tmp_path / "run a", 1441 if large_side in {"a", "both"} else 2)))
    b = _timeline(tmp_path / "run b", 1441 if large_side in {"b", "both"} else 2)
    handed = []

    def native(command, **_kwargs):
        assert rustwx._command_units(command) <= rustwx.COMMAND_LINE_BUDGET
        assert "--diff-against" not in command
        assert command[command.index("--frames") + 1] == "1"
        for flag, expected in (("--inputs-json", a), ("--diff-inputs-json", b)):
            inventory = Path(command[command.index(flag) + 1])
            handed.append(inventory)
            assert json.loads(inventory.read_text(encoding="utf-8")) == [
                os.path.abspath(path) for path in expected]
        if outcome == "cannot start":
            raise OSError("process creation failed")
        return _response(command, failed=outcome == "failed")

    monkeypatch.setattr(rustwx.subprocess, "run", native)
    monkeypatch.setattr(rustwx, "renderer_env", lambda: {})
    written, failures, _, _ = rustwx.run_renderer_difference(
        Path("rw_wrfbatch"), a, b, store_root=tmp_path / "store",
        out_dir=tmp_path / "pictures", products="2m_temperature",
        labels=("left, forecast", "right, analysis"), timeidx=1)
    assert bool(failures) == (outcome != "passed")
    assert len(handed) == 2 and handed[0].parent == handed[1].parent
    assert all(not path.exists() for path in handed)
    assert not handed[0].parent.exists()
    if outcome == "passed":
        assert written == [Path("pictures/difference.png")]


def test_long_difference_options_are_refused_without_leaving_inventory_files(
        tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(rustwx, "renderer_env", lambda: {})
    monkeypatch.setattr(rustwx.subprocess, "run",
                        lambda *_, **__: pytest.fail("an overlong command must not launch"))
    _, failures, _, _ = rustwx.run_renderer_difference(
        Path("rw_wrfbatch"), Path("a.nc"), Path("b.nc"),
        store_root=tmp_path / "store", out_dir=tmp_path / "pictures",
        products="x" * 40000, labels=("A", "B"))
    assert "difference options and inventory paths" in failures[0]
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("bad_index", [-1, True, "1", None])
def test_invalid_difference_frame_index_is_refused_before_launch(
        tmp_path, monkeypatch, bad_index):
    monkeypatch.setattr(rustwx.subprocess, "run",
                        lambda *_, **__: pytest.fail("an invalid selection must not launch"))
    with pytest.raises(ValueError, match="nonnegative integer"):
        rustwx.run_renderer_difference(
            Path("rw_wrfbatch"), Path("a.nc"), Path("b.nc"),
            store_root=tmp_path / "store", out_dir=tmp_path / "pictures",
            products="2m_temperature", labels=("A", "B"), timeidx=bad_index)


def test_inventory_absolute_path_errors_keep_the_original_frame_attribution():
    a = [Path("run/earlier.nc"), Path("run/later.nc")]
    assert rustwx._frame_of(f"{os.path.abspath(a[0])}: field missing", a) == (
        str(a[0]), "field missing")


def test_difference_inventory_and_literal_label_abi_agree_with_native():
    root = Path(__file__).resolve().parents[1]
    source = (root / "tools/rustwx/crates/rw-wrfbatch/src/main.rs").read_text(
        encoding="utf-8")
    for flag in ("--diff-inputs-json", "--diff-label-a", "--diff-label-b"):
        assert flag in rustwx.RENDERER_ABI_MARKER
        assert f'"{flag}"' in source
    assert '"--diff-inputs-json" =>' in source
    assert '"--diff-label-a" | "--diff-label-b" =>' in source
    assert "diff_against.extend(input_list::read" in source
