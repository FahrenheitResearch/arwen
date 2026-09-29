"""CuPy's false CUDA path warning stays off the engine's stderr on Windows.

The breakage: on the Windows managed runtime (the desktop's install, CuPy
14.2 with the CUDA 12 toolkit wheels and no ``CUDA_*`` variables), every
engine command wrote ``cupy/_environment.py:286: UserWarning: CUDA path
could not be detected. Set CUDA_PATH environment variable if CuPy fails to
load.`` to stderr, and the desktop shows a failed command's stderr as its
error text, so every failure a desktop user saw opened on it.  CuPy loaded
fine every time; see ``gpuwm/cupy_windows_warning.py`` for why the warning
is false there and why ``CUDA_PATH`` cannot silence it.

The step that warns runs only on Windows, so these tests stand a stand-in
``cupy`` package up in front of the real one.  It warns with CuPy 14.2's own
words from CuPy's own module name, which is all the engine's filter matches
on, and the platform and site-packages folders are handed in, so the Windows
decision is exercised on any machine.  The last test imports the real CLI.
"""

from __future__ import annotations

import os
import subprocess
import sys
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: Where the ``nvidia-cuda-nvrtc-cu12`` wheel puts its DLL under
#: site-packages, spelled here rather than imported so the two subprocess
#: tests run against a tree that predates the module and say what it does.
WHEEL_NVRTC = ("nvidia", "cuda_nvrtc", "bin", "nvrtc64_120_0.dll")

#: CuPy 14.2's ``cupy/_environment.py`` lines 286-289, verbatim.
CUPY_142_WARNING = (
    "warnings.warn(\n"
    "    'CUDA path could not be detected.'\n"
    "    ' Set CUDA_PATH environment variable if CuPy '\n"
    "    'fails to load.')\n")

TEXT = "CUDA path could not be detected"


def _stand_in_cupy(root: Path) -> Path:
    """A ``cupy`` whose import warns the way CuPy 14.2's does on Windows."""
    package = root / "cupy"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        "from cupy import _environment\n"
        "_environment._setup_win32_dll_directory()\n", encoding="utf-8")
    body = "".join("    " + line + "\n" for line in CUPY_142_WARNING.splitlines())
    (package / "_environment.py").write_text(
        "import warnings\n\n\ndef _setup_win32_dll_directory():\n" + body,
        encoding="utf-8")
    return root


def _site_packages(root: Path, *, wheel: bool) -> Path:
    """A site-packages folder, holding the CUDA 12 NVRTC wheel's DLL or not."""
    root.mkdir(parents=True)
    if wheel:
        dll = root.joinpath(*WHEEL_NVRTC)
        dll.parent.mkdir(parents=True)
        dll.write_bytes(b"")
    return root


def _without_cuda_variables() -> dict[str, str]:
    """This environment with every ``CUDA_*`` variable taken out, as the
    desktop bootstrap does, except ``CUDA_VISIBLE_DEVICES``: that one picks
    devices, CuPy's path detection never reads it, and it keeps these
    imports off a card the suite was told to leave alone."""
    env = {key: value for key, value in os.environ.items()
           if not key.upper().startswith("CUDA_")
           or key.upper() == "CUDA_VISIBLE_DEVICES"}
    env["PYTHONPATH"] = str(REPO)
    env.pop("PYTHONSAFEPATH", None)
    env.pop("PYTHONWARNINGS", None)
    return env


def _import_gpuwm_then_cupy(tmp_path: Path, *, wheel: bool) -> str:
    """stderr of ``import gpuwm; import cupy`` in a Windows-shaped process.

    ``sys.platform`` and ``site.getsitepackages`` are set before ``gpuwm`` is
    imported, so the package's own import is what decides; nothing here
    calls the filter directly.
    """
    shadow = _stand_in_cupy(tmp_path / "shadow")
    site_dir = _site_packages(tmp_path / "site-packages", wheel=wheel)
    script = (
        # Everything the package's first lines import is loaded before the
        # platform is swapped: the standard library itself branches on it.
        "import collections.abc, importlib.metadata, os, site, sys, warnings\n"
        f"sys.path.insert(0, {str(shadow)!r})\n"
        "sys.platform = 'win32'\n"
        f"site.getsitepackages = lambda: [{str(site_dir)!r}]\n"
        "import gpuwm\n"
        "import cupy\n")
    done = subprocess.run([sys.executable, "-s", "-c", script],
                          capture_output=True, text=True,
                          env=_without_cuda_variables(), timeout=120)
    assert done.returncode == 0, done.stderr
    return done.stderr


def test_importing_gpuwm_keeps_the_false_warning_off_stderr(tmp_path):
    stderr = _import_gpuwm_then_cupy(tmp_path, wheel=True)
    assert TEXT not in stderr, stderr
    assert "UserWarning" not in stderr, stderr


def test_without_the_toolkit_wheel_the_warning_still_reaches_stderr(tmp_path):
    """No wheel: CuPy may really be unable to find CUDA, so it may say so."""
    stderr = _import_gpuwm_then_cupy(tmp_path, wheel=False)
    assert TEXT in stderr, stderr
    assert "cupy" in stderr and "UserWarning" in stderr, stderr


def test_the_filter_is_placed_only_on_windows_and_only_with_the_wheel(tmp_path):
    from gpuwm import cupy_windows_warning as quiet
    assert quiet.WHEEL_NVRTC == WHEEL_NVRTC
    with_wheel = _site_packages(tmp_path / "with", wheel=True)
    without = _site_packages(tmp_path / "without", wheel=False)
    with warnings.catch_warnings():
        before = list(warnings.filters)
        assert quiet.quiet_false_cuda_path_warning(
            platform="linux", folders=[str(with_wheel)]) is None
        assert quiet.quiet_false_cuda_path_warning(
            platform="win32", folders=[str(without)]) is None
        assert warnings.filters == before
        found = quiet.quiet_false_cuda_path_warning(
            platform="win32", folders=[str(without), str(with_wheel)])
        assert found == str(with_wheel.joinpath(*WHEEL_NVRTC))
        assert len(warnings.filters) == len(before) + 1


def test_only_that_warning_from_that_module_is_ignored(tmp_path):
    from gpuwm import cupy_windows_warning as quiet
    site_dir = _site_packages(tmp_path / "site-packages", wheel=True)
    message = ("CUDA path could not be detected. Set CUDA_PATH environment "
               "variable if CuPy fails to load.")
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        assert quiet.quiet_false_cuda_path_warning(
            platform="win32", folders=[str(site_dir)])
        warnings.warn_explicit(message, UserWarning, "_environment.py", 286,
                               module="cupy._environment")
        # CuPy's other import-time warnings are true when they fire.
        warnings.warn_explicit("CuPy may not function correctly because "
                               "multiple CuPy packages are installed",
                               UserWarning, "_environment.py", 677,
                               module="cupy._environment")
        # The same words from anything else are not CuPy's step.
        warnings.warn_explicit(message, UserWarning, "other.py", 1,
                               module="gpuwm.other")
        warnings.warn_explicit(message, UserWarning, "_environment.py", 1,
                               module="cupy._environment_extra")
    assert [(str(w.message)[:28], w.filename) for w in seen] == [
        ("CuPy may not function correc", "_environment.py"),
        ("CUDA path could not be detec", "other.py"),
        ("CUDA path could not be detec", "_environment.py"),
    ]


def test_pathfinder_folders_are_site_packages_then_the_user_site(monkeypatch):
    from gpuwm import cupy_windows_warning as quiet
    monkeypatch.setattr(quiet.site, "getsitepackages", lambda: ["a", "b"])
    monkeypatch.setattr(quiet.site, "ENABLE_USER_SITE", True)
    monkeypatch.setattr(quiet.site, "getusersitepackages", lambda: "u")
    assert quiet.pathfinder_site_packages() == ["a", "b", "u"]
    monkeypatch.setattr(quiet.site, "ENABLE_USER_SITE", False)
    assert quiet.pathfinder_site_packages() == ["a", "b"]


def test_the_cli_imports_without_a_user_warning():
    """``import gpuwm.cli`` with the installed CuPy and no CUDA variables.

    On Windows with the CUDA 12 toolkit wheels this is the defect itself;
    elsewhere it keeps any other import-time warning off the stderr the
    desktop shows.
    """
    done = subprocess.run([sys.executable, "-s", "-c", "import gpuwm.cli"],
                          capture_output=True, text=True, cwd=str(REPO),
                          env=_without_cuda_variables(), timeout=600)
    assert done.returncode == 0, done.stderr
    assert "Warning:" not in done.stderr, done.stderr
