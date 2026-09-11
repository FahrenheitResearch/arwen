"""The desktop launcher template hands every child a non-writing Python.

THE BREAKAGE THIS PREVENTS: the packaged runtime is verified by hash and
must stay byte-identical after use, and one ``doctor`` run from it wrote
554 ``__pycache__`` files (134 MB) into ``runtime/Lib/site-packages``
because the launcher's own ``-B`` covered the launcher process only.  The
flag has to travel in the environment to every engine, terminal and GUI
subprocess, and the same environment carries the CDS credentials path and
the GUI's chosen cache folder that the shipped package had to patch in
by hand.  ``tools/release/launch_arwen.py`` is the template that package
copies, so it is what is tested here, by path: it is not a package module.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import re
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "tools" / "release" / "launch_arwen.py"


@pytest.fixture(scope="module")
def launcher():
    if not LAUNCHER.is_file():
        pytest.skip("the launcher template needs the source tree")
    spec = importlib.util.spec_from_file_location("_arwen_launcher", LAUNCHER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parent_env() -> dict[str, str]:
    return {"PATH": "C:\\Windows\\System32", "PYTHONPATH": "C:\\elsewhere",
            "GPUWM_DOCTOR_STATE": "x", "ARWEN_CACHE_DIR": "y",
        "NO_COLOR": "1", "HOME": "C:\\ArWen\\profile"}


def test_children_never_write_bytecode_into_the_package(launcher, tmp_path):
    env = launcher.child_environment(tmp_path / "state", tmp_path / "rt" / "python.exe",
                                     parent=_parent_env())
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"
    # The rest of the sealed-runtime posture is unchanged.
    assert env["PYTHONNOUSERSITE"] == "1"
    assert env["PYTHONSAFEPATH"] == "1"
    assert env["PYTHONUTF8"] == "1"
    # Host Python/gpuwm/ArWen settings never leak into the package's children.
    assert "PYTHONPATH" not in env and "GPUWM_DOCTOR_STATE" not in env
    assert "NO_COLOR" not in env
    assert env["HOME"] == "C:\\ArWen\\profile"
    assert env["PATH"].startswith(str(tmp_path / "rt") + os.pathsep)


def test_the_cache_folder_follows_the_gui_preference(launcher, tmp_path):
    state = tmp_path / "state"
    assert launcher.cache_directory(state) == state / "cache"
    pref = state / "appdata" / "ArWenCompanion" / "preferences.json"
    pref.parent.mkdir(parents=True)
    chosen = tmp_path / "chosen"
    pref.write_text(json.dumps({"data_folder": str(chosen)}), encoding="utf-8")
    assert launcher.cache_directory(state) == chosen / "cache"
    env = launcher.child_environment(state, tmp_path / "python.exe", parent={})
    assert env["ARWEN_CACHE_DIR"] == str(chosen / "cache")
    # A relative or malformed preference falls back rather than escaping.
    pref.write_text(json.dumps({"data_folder": "relative/folder"}), encoding="utf-8")
    assert launcher.cache_directory(state) == state / "cache"
    pref.write_text("not json", encoding="utf-8")
    assert launcher.cache_directory(state) == state / "cache"


def test_cds_credentials_reach_the_children_or_refuse_by_name(launcher, tmp_path):
    rc = tmp_path / "cdsapirc"
    rc.write_text("url: https://example.invalid\nkey: 0\n", encoding="utf-8")
    env = launcher.child_environment(tmp_path / "state", tmp_path / "python.exe", rc, parent={})
    assert env["CDSAPI_RC"] == str(rc)
    env = launcher.child_environment(tmp_path / "state", tmp_path / "python.exe", parent={})
    assert "CDSAPI_RC" not in env
    with pytest.raises(ValueError, match="CDS credentials file is unavailable"):
        launcher.child_environment(tmp_path / "state", tmp_path / "python.exe",
                                   tmp_path / "missing", parent={})


def test_the_launcher_parses_the_packaging_flags(launcher):
    """The flags the shipped package added by hand are in the template."""

    source = LAUNCHER.read_text(encoding="utf-8")
    for flag in ("--verify-launcher", "--tui-only", "--state-dir", "--cds-credentials"):
        assert f"'{flag}'" in source, flag
    assert "'cds_credentials_configured'" in source
    assert "'cache_directory'" in source


def test_the_restore_bytes_are_the_controllers_own_sequence(launcher):
    """SGR/urxvt/any-event/button/normal mouse off, paste off, main screen, cursor."""

    assert launcher.TERMINAL_RESTORE == (
        "\x1b[?1006l\x1b[?1015l\x1b[?1003l\x1b[?1002l\x1b[?1000l"
        "\x1b[?2004l\x1b[?1049l\x1b[?25h")
    # Once the controller lane carrying TERMINAL_RESTORE is in, the two
    # sequences have to stay the same one: a launcher that disabled a
    # different set would leave exactly the modes it missed turned on.
    controller = REPO_ROOT / "tools" / "arwen-tui" / "src" / "main.rs"
    source = controller.read_text(encoding="utf-8") if controller.is_file() else ""
    match = re.search(r"const TERMINAL_RESTORE: &\[u8\] =\s*b\"([^\"]*)\";", source)
    if match is None:
        pytest.skip("this tree's controller does not define TERMINAL_RESTORE yet")
    assert match.group(1).replace("\\x1b", "\x1b") == launcher.TERMINAL_RESTORE


def test_a_redirected_launcher_leaves_the_stream_alone(launcher, tmp_path, monkeypatch):
    """THE BREAKAGE THIS PREVENTS: escape bytes in a log file or a pipeline."""

    sink = tmp_path / "captured.txt"
    with sink.open("w", encoding="utf-8") as stream:
        monkeypatch.setattr(sys, "stdout", stream)
        assert launcher.terminal_mode() is None
        launcher.restore_terminal(None)
    assert sink.read_text(encoding="utf-8") == ""


@pytest.mark.skipif(os.name == "nt", reason="the restore is POSIX-only by design")
def test_a_killed_controller_hands_back_a_usable_terminal(launcher, monkeypatch):
    import termios
    import tty

    controller, terminal = os.openpty()
    try:
        with os.fdopen(terminal, "w", closefd=False) as stream:
            monkeypatch.setattr(sys, "stdout", stream)
            before = termios.tcgetattr(terminal)
            saved = launcher.terminal_mode()
            assert saved is not None
            # What the controller does, and what a SIGKILL leaves behind.
            tty.setraw(terminal)
            stream.write("\x1b[?1049h\x1b[?1003h\x1b[?1006h\x1b[?25l")
            stream.flush()
            os.read(controller, 65536)
            launcher.restore_terminal(saved)
        assert os.read(controller, 65536).decode() == launcher.TERMINAL_RESTORE
        assert termios.tcgetattr(terminal) == before
    finally:
        os.close(controller)
        os.close(terminal)


def test_the_signal_line_names_the_signal_and_the_controller_log(launcher, tmp_path):
    state = tmp_path / "state"
    runs = state / "runs"
    command = ["/opt/arwen/arwen-tui", "--python", "/venv/bin/python", "--output", str(runs)]
    line = launcher.signal_report(-9, command, state)
    assert "\n" not in line
    assert "arwen-tui" in line
    # The line is only ever printed on the platform that has the signal.
    if os.name != "nt":
        assert "SIGKILL" in line
    assert str(runs / ".arwen-tui" / "controller.log") in line
    # A forwarded --output moves the log with it; an unknown signal still reports.
    chosen = tmp_path / "elsewhere"
    command[-1] = str(chosen)
    assert str(chosen / ".arwen-tui" / "controller.log") in launcher.signal_report(-9, command, state)
    assert "signal 99" in launcher.signal_report(-99, command, state)
    assert launcher.controller_log(["/opt/arwen/arwen-tui"], state) == runs / ".arwen-tui" / "controller.log"
