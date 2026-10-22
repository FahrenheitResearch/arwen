"""The Linux wheel's platform tag is read from the staged bytes, not assumed.

The breakage this prevents (2.9.4 cut, 2026-10-22): the battery's measurement
wheel was built from bridges compiled on an Ubuntu glibc 2.43 host and still
tagged ``manylinux_2_28_x86_64``, because ``setup.py`` took the tag from
``PLATFORM_TAGS`` alone.  pip installed it on the gpuwm-hex CI runner (Ubuntu
24.04, glibc 2.39) and 398 tests failed with ``version `GLIBC_2.43' not
found`` (rw_mpas_hostprep, rw_mpas_static, rw_wrfbatch).  Bridges built in the
pinned manylinux_2_28 image keep the 2.28 tag; host-built ones are tagged with
the glibc they need, and a release build of them is refused.

The probe runs in a child interpreter because ``setup.py`` calls ``setup()``
at import (see tests/test_wheel_bridge_staging.py for the measured reason).
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

_ROOT = Path(__file__).resolve().parent.parent

_PROBE = r"""
import functools, importlib.util, json, os, pathlib, sys

import setuptools
setuptools.setup = lambda *args, **kwargs: None
spec = importlib.util.spec_from_file_location("_gpuwm_setup_probe", sys.argv[1])
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)

staged = pathlib.Path(sys.argv[2])
floor = setup._staged_glibc_floor(staged)
setup._staged_glibc_floor = functools.partial(setup._staged_glibc_floor, staged)
result = {"floor": floor,
          "linux": setup._platform_tag("linux-x86_64"),
          "win": setup._platform_tag("win-x86_64")}
for label, value in (("release", None), ("dev", "1")):
    os.environ.pop(setup._UNPINNED_OK_ENV, None)
    if value is not None:
        os.environ[setup._UNPINNED_OK_ENV] = value
    try:
        setup._refuse_off_policy_wheel("linux-x86_64")
        result[label] = "built"
    except SystemExit as stop:
        result[label] = str(stop)
print(json.dumps(result))
"""


def _elf(path: Path, *versions: str) -> None:
    body = b"\x7fELF\x02\x01\x01" + b"\x00" * 9 + b"".join(
        b"\x00" + v.encode() for v in versions) + b"\x00"
    path.write_bytes(body)


def _probe(staged: Path) -> dict:
    setup_py = _ROOT / "setup.py"
    if not setup_py.exists():
        pytest.skip("setup.py is not present outside a source tree")
    completed = subprocess.run([sys.executable, "-c", _PROBE, str(setup_py), str(staged)],
                               capture_output=True, text=True, timeout=600)
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_manylinux_built_bridges_keep_the_policy_tag(tmp_path):
    staged = tmp_path / "bridges"
    staged.mkdir()
    _elf(staged / "rw_mpas_hostprep", "GLIBC_2.2.5", "GLIBC_2.17", "GLIBC_2.28")
    _elf(staged / "libstatic_fields.so", "GLIBC_2.14", "GLIBC_2.18")
    (staged / "BUNDLE.json").write_text('{"platform": "linux-x86_64"}')
    report = _probe(staged)
    assert report["floor"] == [[2, 28], "rw_mpas_hostprep"]
    assert report["linux"] == "manylinux_2_28_x86_64"
    assert report["win"] == "win_amd64"
    assert report["release"] == report["dev"] == "built"


def test_host_built_bridges_are_tagged_with_the_glibc_they_need(tmp_path):
    staged = tmp_path / "bridges"
    (staged / "assets").mkdir(parents=True)
    _elf(staged / "rw_mpas_hostprep", "GLIBC_2.28")
    _elf(staged / "librw_superob.so", "GLIBC_2.34", "GLIBC_2.43", "GLIBC_2.29")
    # Text that names a newer glibc is not a requirement: only ELF members count.
    (staged / "assets" / "NOTES.txt").write_text("needs GLIBC_2.99 somewhere")
    report = _probe(staged)
    assert report["floor"] == [[2, 43], "librw_superob.so"]
    assert report["linux"] == "manylinux_2_43_x86_64"
    assert report["win"] == "win_amd64"
    assert report["dev"] == "built"
    assert "REFUSED" in report["release"] and "librw_superob.so needs glibc 2.43" in report["release"]
    assert "manylinux_2_28" in report["release"]


def test_a_tree_with_no_elf_keeps_the_policy_tag(tmp_path):
    staged = tmp_path / "bridges"
    staged.mkdir()
    (staged / "rw_mpas_hostprep.exe").write_bytes(b"MZ\x90\x00GLIBC_2.43")
    report = _probe(staged)
    assert report["floor"] is None
    assert report["linux"] == "manylinux_2_28_x86_64"
    assert report["release"] == "built"
