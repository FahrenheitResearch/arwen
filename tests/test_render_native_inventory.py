"""Complete native series survive Windows limits after path shortening."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile

import pytest

from gpuwm import rustwx


def _frames(root):
    return [root / f"day-{index // 144:02d}"
            / f"wrfout_d01_2026-10-{4 + index // 144:02d}_{index % 144:03d}.nc"
            for index in range(1441)]


def _command(root, frames):
    return [str(root / "rw_wrfbatch"), "--store-root", str(root / "store"),
            "--out-dir", str(root / "pictures"), "--products", "t2",
            "--frames", "all", *map(str, frames)]


def test_a_short_native_command_keeps_its_bytes_and_environment(tmp_path):
    frames = _frames(tmp_path)[:2]
    command = _command(tmp_path, frames)
    environment = {"RUSTWX_BASEMAP_DIR": "maps"}
    with rustwx._series_command(command, len(frames), environment) as fitted:
        assert fitted == (command, None, environment)
        assert fitted[0] is command and fitted[2] is environment


def test_shortening_that_still_exceeds_budget_uses_all_ordered_paths(
        tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    frames = list(reversed(_frames(tmp_path / "forecast 🌦")))
    frames.append(frames[0])
    command = _command(tmp_path, frames)
    shortened = [*command[:-len(frames)],
                 *(os.path.relpath(frame, tmp_path / "forecast 🌦")
                   for frame in frames)]
    assert rustwx._command_units(shortened) > rustwx.COMMAND_LINE_BUDGET
    with rustwx._series_command(command, len(frames), {}) as fitted:
        native, cwd, environment = fitted
        assert cwd is None and environment == {}
        assert rustwx._command_units(native) <= rustwx.COMMAND_LINE_BUDGET
        inventory = Path(native[native.index("--inputs-json") + 1])
        assert json.loads(inventory.read_text(encoding="utf-8")) == [
            str(frame.resolve()) for frame in frames]
        assert native[:-2] == command[:-len(frames)]
    assert not inventory.exists()
    assert not inventory.parent.exists()


def test_frames_without_a_shared_drive_use_the_native_inventory(
        tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    frames = _frames(tmp_path)
    command = _command(tmp_path, frames)

    def separate_drives(_paths):
        raise ValueError("Paths don't have the same drive")

    monkeypatch.setattr(os.path, "commonpath", separate_drives)
    with rustwx._series_command(command, len(frames), {}) as fitted:
        inventory = Path(fitted[0][-1])
        assert fitted[0][-2] == "--inputs-json"
        assert json.loads(inventory.read_text(encoding="utf-8")) == [
            str(frame.resolve()) for frame in frames]
        assert fitted[1] is None
    assert not inventory.exists()


def test_utf16_surrogate_pairs_are_counted_after_quoting():
    command = ["rw_wrfbatch", "frame 🌦 with spaces.nc"]
    quoted = subprocess.list2cmdline(command)
    assert rustwx._command_units(command) == len(quoted) + 1


def test_unicode_names_trigger_shortening_before_character_budget(
        tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    frames = [str(tmp_path / f"frame_{index}_{'🌦' * 30}.nc")
              for index in range(360)]
    command = ["rw_wrfbatch", "--store-root", "store", *frames]
    monkey_budget = len(subprocess.list2cmdline(command))
    assert monkey_budget < rustwx._command_units(command)
    # Make the difference between character and UTF-16 counting the
    # deciding factor without depending on the temporary folder length.
    monkeypatch.setattr(rustwx, "COMMAND_LINE_BUDGET", monkey_budget)
    with rustwx._series_command(command, len(frames), {}) as fitted:
        assert fitted[1] == str(tmp_path) or "--inputs-json" in fitted[0]
        assert rustwx._command_units(fitted[0]) <= monkey_budget


@pytest.mark.parametrize("catalog", [False, True])
@pytest.mark.parametrize("outcome", ["passed", "failed", "cannot start"])
def test_native_inventory_lives_until_process_returns_and_is_removed(
        tmp_path, monkeypatch, catalog, outcome):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    frames = _frames(tmp_path)
    handed = []

    def native(command, **_kwargs):
        inventory = Path(command[command.index("--inputs-json") + 1])
        handed.append(inventory)
        assert json.loads(inventory.read_text(encoding="utf-8")) == [
            str(frame.resolve()) for frame in frames]
        if outcome == "cannot start":
            raise OSError("process creation failed")
        return subprocess.CompletedProcess(
            command, 0 if outcome == "passed" else 1,
            "PRODUCT\tt2\tdirect\trenderable\tavailable\tok\nCATALOG ready\n"
            if catalog else "RENDERED t2 pictures/t2.png\n",
            "FAILED t2 input refused\n" if outcome == "failed" else "")

    monkeypatch.setattr(rustwx.subprocess, "run", native)
    monkeypatch.setattr(rustwx, "renderer_env", lambda: {})
    if catalog:
        if outcome == "passed":
            rows, _ = rustwx._catalog_listing(
                Path("rw_wrfbatch"), frames, store_root=tmp_path / "store")
            assert rows[0][0] == "t2"
        else:
            with pytest.raises(RuntimeError):
                rustwx._catalog_listing(
                    Path("rw_wrfbatch"), frames, store_root=tmp_path / "store")
    else:
        written, failures, _ = rustwx.run_renderer_series(
            Path("rw_wrfbatch"), frames, store_root=tmp_path / "store",
            out_dir=tmp_path / "pictures", products="t2", frames="all")
        assert bool(failures) == (outcome != "passed")
        if outcome == "passed":
            assert written == [Path("pictures/t2.png")]
    assert len(handed) == 1
    assert not handed[0].exists() and not handed[0].parent.exists()


def test_long_options_are_refused_before_writing_an_inventory(tmp_path):
    command = ["rw_wrfbatch", "--products", "x" * 40000, "frame.nc"]
    inventory = tmp_path / "inputs.json"
    with pytest.raises(ValueError, match="renderer options"):
        rustwx.fit_series_command(command, 1, {}, inventory=inventory)
    assert not inventory.exists()


def test_python_and_native_renderer_require_the_input_list_abi():
    root = Path(__file__).resolve().parents[1]
    source = (root / "tools/rustwx/crates/rw-wrfbatch/src/main.rs").read_text(
        encoding="utf-8")
    marker = "gpuwm-rw-wrfbatch-inputs-json-v1"
    assert marker in rustwx.RENDERER_ABI_MARKER and marker in source
    assert '"--inputs-json" =>' in source
    assert "inputs.extend(input_list::read" in source
