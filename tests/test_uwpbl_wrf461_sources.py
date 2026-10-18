"""The UW PBL oracle's WRF v4.7.1 fixtures stand for WRF v4.6.1 too.

tools/uwpbl_wrf461_verify/SOURCES-v4.6.1.sha256 pins the same paths as
tools/uwpbl_wrf471_oracle/SOURCES.sha256, hashed at WRF tag v4.6.1.  Every
file the oracle COMPILES (build.sh's CAM list) must carry the same hash in
both, which is what lets one port be graded against either release; the
driver-side files that are only cited may differ.  If a future pin update
breaks this, the 4.6.1 claim in the UW PBL verification report is void.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V471 = ROOT / "tools/uwpbl_wrf471_oracle/SOURCES.sha256"
V461 = ROOT / "tools/uwpbl_wrf461_verify/SOURCES-v4.6.1.sha256"
BUILD = ROOT / "tools/uwpbl_wrf471_oracle/build.sh"


def _pins(path):
    out = {}
    for line in path.read_text(encoding="ascii").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        digest, name = line.split()
        out[name] = digest
    return out


def _compiled():
    text = BUILD.read_text(encoding="ascii")
    block = re.search(r"^CAM=\(\n(.*?)^\)", text, re.S | re.M).group(1)
    return [line.strip() for line in block.splitlines() if line.strip()]


def test_both_pin_files_name_the_same_paths():
    assert set(_pins(V461)) == set(_pins(V471))


def test_every_compiled_file_is_byte_identical_in_461_and_471():
    compiled = _compiled()
    assert len(compiled) == 15
    a, b = _pins(V461), _pins(V471)
    for name in compiled:
        assert a[name] == b[name], name


def test_the_record_families_are_unchanged():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "make_cases", ROOT / "tools/uwpbl_wrf471_oracle/make_cases.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert m.FAMILIES == ("convective", "stable", "stratocu", "coldpool",
                          "icecloud", "shallowcu")
    assert not set(m.EXTRA_FAMILIES) & set(m.FAMILIES)
