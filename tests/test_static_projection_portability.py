"""Windows-prepared fine-grid statics must survive a Linux nest move."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from gpuwm.ingest.relocation_init import OVERLAP_STATIC_EQUALITY_FIELDS
from gpuwm.static import rust_bridge
from gpuwm.static.build import build_static
from gpuwm.static.lambert import LambertGrid
from test_statics_corridor import _synthetic_wps_geog


def geog_30arcsecond(root, origin=(33.5, -98.5)):
    """Exercise interpolated terrain at the production source spacing."""
    root = _synthetic_wps_geog(root, regional_size=1000)
    for name in ("topo_gmted2010_30s", "modis_landuse_20class_30s_with_lakes"):
        path = root / name / "index"
        text = path.read_text().replace("dx = 0.004", "dx = 0.008333333333333333")
        text = text.replace("dy = 0.004", "dy = 0.008333333333333333")
        text = text.replace("known_lat = 33.5", f"known_lat = {origin[0]}")
        text = text.replace("known_lon = -98.5", f"known_lon = {origin[1]}")
        path.write_bytes(text.encode())
    return root


WITNESS_SPECS = json.loads((Path(__file__).parent / "fixtures" /
                           "portable_static_grids.json").read_text())


def witness_grid(key, size=None):
    row = WITNESS_SPECS[str(key)]
    grid = LambertGrid(**row["spec"])
    if row["nest"] is not None:
        nest = list(row["nest"])
        if size is not None:
            nest[-2:] = [size + 1, size + 1]
        grid = grid.nest(*nest)
    return grid


def field_hashes(fields, j, i):
    return {name: hashlib.sha256(np.ascontiguousarray(
        fields[name][..., j, i]).tobytes()).hexdigest()
        for name in OVERLAP_STATIC_EQUALITY_FIELDS}


@pytest.fixture(scope="module")
def portable_geog(tmp_path_factory):
    if rust_bridge.unavailable_reason() is not None:
        pytest.fail("the Rust statics builder is required: cd tools/rustwx && "
                    "cargo build --release --offline -p static-fields")
    roots = {}
    for key, row in WITNESS_SPECS.items():
        origin = tuple(row["origin"])
        if origin not in roots:
            roots[origin] = geog_30arcsecond(tmp_path_factory.mktemp("portable-geog"), origin)
    return roots


@pytest.mark.parametrize("spacing", list(WITNESS_SPECS))
def test_windows_prepared_statics_survive_fine_grid_move(portable_geog, spacing):
    manifest = json.loads((Path(__file__).parent / "fixtures" /
                           "portable_static_witnesses.json").read_text())
    row = manifest["grids"][str(spacing)]
    grid = witness_grid(spacing)
    root = portable_geog[tuple(WITNESS_SPECS[str(spacing)]["origin"])]
    rebuilt = build_static(grid.translated(3, 3), root)
    assert field_hashes(rebuilt, slice(None, -3), slice(None, -3)) == row["overlap_sha256"]


def test_missing_bridge_is_a_failure_with_the_build_command(monkeypatch):
    monkeypatch.setattr(rust_bridge, "unavailable_reason", lambda: "missing")
    with pytest.raises(pytest.fail.Exception, match="cargo build"):
        portable_geog.__wrapped__(None)
