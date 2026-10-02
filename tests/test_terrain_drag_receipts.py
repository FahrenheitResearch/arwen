"""CPU checks for WRF terrain-drag inputs, activation and frame receipts.

The device kernels and their mutation controls run separately in
test_terrain_drag_wrf471_parity.py on the GPU shard.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

ORACLE = (Path(__file__).resolve().parents[1] / "gpuwm" / "data"
          / "terrain_drag" / "oracle")
TOOL = (Path(__file__).resolve().parents[1] / "tools"
        / "terrain_drag_wrf471_oracle")


def _load(case: str) -> dict:
    from gpuwm.verify.urban_oracle import load_case
    return load_case(ORACLE / case)


def test_fixture_is_the_pinned_wrf_and_its_own_receipts():
    """Every published word is the one build.sh wrote, from pinned sources."""
    pins = [line.split() for line in
            (TOOL / "SOURCES.sha256").read_text().splitlines()
            if line and not line.startswith("#")]
    recorded = [line.split() for line in
                (ORACLE / "SOURCES.sha256").read_text().splitlines() if line]
    assert recorded == pins
    for line in (ORACLE / "FIXTURES.sha256").read_text().splitlines():
        digest, rel = line.split()
        got = hashlib.sha256((ORACLE / rel).read_bytes()).hexdigest()
        assert got == digest, rel
    harness = dict(reversed(line.split()) for line in
                   (ORACLE / "HARNESS.sha256").read_text().splitlines())
    for rel, digest in harness.items():
        got = hashlib.sha256((TOOL / rel).read_bytes()).hexdigest()
        assert got == digest, rel
    report = (ORACLE / "libmvec-report.txt").read_text()
    # The reference is scalar libm: no -O0 object took a vector form. WRF's
    # own -O2 flags vectorize the GSL form-drag loop (expf/powf), which
    # the report records rather than hides.
    head = report.split("at WRF's -O2")[0]
    assert "_ZGV" not in head.split("# positive control")[0]
    assert "_ZGVbN4v_expf" in report


def test_fixture_reaches_the_branches_that_matter():
    inputs = _load("topo_static/inputs")
    one = _load("topo_static/topo_wind_1")
    lap = one["lap_hgt"]
    assert (lap > -10).any() and ((lap <= -10) & (lap >= -20)).any()
    assert ((lap < -20) & (lap >= -30)).any() and (lap < -30).any()
    assert (one["ctopo2"] < 1).any() and (one["ctopo"] == 0).any()
    assert (one["ctopo"] > 1).any()
    assert (inputs["xland"] == 2).any()
    ysu = _load("ysu_topo/columns")
    moved = ysu["u10"] != ysu["u10_in"]
    assert moved.any() and (~moved).any()
    gi = _load("gwdo/inputs")
    for dx in (3000, 12000, 30000):
        out = _load(f"gwdo/dx{dx}")
        assert (out["rublten"] != gi["rublten0"]).any()
    seen = {}
    for dx in (1000, 3000, 5000, 9000, 15000):
        out = _load(f"gwdo_gsl/dx{dx}")
        seen[dx] = {c: bool(np.abs(out[f"dtaux3d_{c}"]).max() > 0)
                    for c in ("ls", "bl", "ss", "fd")}
    assert seen[1000] == dict(ls=False, bl=False, ss=False, fd=False)
    assert seen[3000] == dict(ls=False, bl=False, ss=True, fd=True)
    assert all(seen[15000].values())


def test_drag_frame_readings_describe_this_production_unit():
    from gpuwm.core import kernel_frame_recordings as recordings
    from gpuwm.core import terrain_drag
    from gpuwm.core.preflight import CHAINED_TRANSLATION_UNIT_FRAMES

    assert hashlib.sha256(terrain_drag.module_source().encode()).hexdigest() == (
        recordings.TERRAIN_DRAG_COMPOSED_SOURCE_SHA256)
    assert terrain_drag.MODULE_OPTIONS == recordings.TERRAIN_DRAG_COMPOSED_OPTIONS
    ceiling = max(max(row.values()) for row in
                  recordings.TERRAIN_DRAG_COMPOSED_FRAME_READINGS.values())
    assert CHAINED_TRANSLATION_UNIT_FRAMES["terrain_drag_composed"].max_local_size_bytes == ceiling
