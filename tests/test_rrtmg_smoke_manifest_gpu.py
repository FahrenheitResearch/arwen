"""Congruent native/cut/halo windows and CUDA profile interpolation."""
from datetime import datetime, timezone
import hashlib
import json

import numpy as np
import pytest

pytestmark = pytest.mark.gpu

from gpuwm.core.rrtmg_smoke_manifest import BoundSmokeManifest, SCHEMA

START = datetime(2026, 10, 2, 21, tzinfo=timezone.utc)


def _source(tmp_path):
    """Small numerical grid fixture, not an observed or forecast smoke field."""
    nz, ny, nx = 2, 5, 6
    yy, xx = np.indices((ny, nx))
    lat = 38 + yy * .02 + xx * .0005
    lon = -98 + xx * .03 + yy * .002
    first = np.arange(nz * ny * nx, dtype="<f8").reshape(nz, ny, nx)
    second = first + 20
    def member(name, value, units):
        path = tmp_path / (name + ".f64le")
        value.astype("<f8").tofile(path)
        return {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "dtype": "<f8", "shape": list(value.shape), "units": units}
    manifest = {"schema": SCHEMA, "quantity": "dry_mass_mixing_ratio", "units": "ug/kg-dryair",
        "shape": [nz, ny, nx], "vertical_order": "bottom_to_top", "time_interpolation": "linear",
        "start_time": "2026-10-02T21:00:00Z", "geometry": {
            "latitude": member("lat", lat, "degrees_north"), "longitude": member("lon", lon, "degrees_east")},
        "provenance": {"vertical": {"eta_levels": [1.0, .5, 0.0], "hybrid_opt": 2, "etac": .2, "p_top": 1500.0}},
        "frames": [{"valid_time": "2026-10-02T21:00:00Z", "value": member("a", first, "ug/kg-dryair")},
                   {"valid_time": "2026-10-02T23:00:00Z", "value": member("b", second, "ug/kg-dryair")}]}
    path = tmp_path / "smoke.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path, lat, lon, first, second


@pytest.mark.parametrize("window", ((0, 0, 5, 6), (1, 2, 3, 4), (0, 0, 3, 6)))
def test_full_cut_and_rank_window_keep_all_supplied_cells(tmp_path, window):
    import cupy as cp

    path, lat, lon, first, second = _source(tmp_path)
    j, i, ny, nx = window
    bound = BoundSmokeManifest(path, START, lat[j:j + ny, i:i + nx], lon[j:j + ny, i:i + nx], 2)
    bound.require_vertical((1.0, .5, 0.0), 2, .2, 1500.0)
    bound.require_coverage(7200)
    assert bound.binding_receipt()["offset_ji"] == [j, i]
    np.testing.assert_array_equal(cp.asnumpy(bound.at(0)["value"]), first[:, j:j + ny, i:i + nx].astype(np.float32))
    np.testing.assert_array_equal(cp.asnumpy(bound.at(7200)["value"]), second[:, j:j + ny, i:i + nx].astype(np.float32))
    expected = (first[:, j:j + ny, i:i + nx] + 10).astype(np.float32)
    np.testing.assert_array_equal(cp.asnumpy(bound.at(3600)["value"]), expected)
    assert len(bound._cache) == 2


def test_rotated_or_shifted_grid_is_not_silently_remapped(tmp_path):
    path, lat, lon, _, _ = _source(tmp_path)
    with pytest.raises(ValueError, match="not congruent|outside"):
        BoundSmokeManifest(path, START, lat[::-1], lon[::-1], 2)
    altered = lat.copy()
    altered[2, 3] += .001
    with pytest.raises(ValueError, match="not congruent"):
        BoundSmokeManifest(path, START, altered, lon, 2)


def test_target_vertical_must_be_checked_before_using_a_profile(tmp_path):
    path, lat, lon, _, _ = _source(tmp_path)
    bound = BoundSmokeManifest(path, START, lat, lon, 2)
    with pytest.raises(ValueError, match="vertical provenance must be verified"):
        bound.at(0)
    with pytest.raises(ValueError, match="vertical provenance differs"):
        bound.require_vertical((1.0, .4, 0.0), 2, .2, 1500.0)


def test_rank_rebind_has_an_independent_local_cache_and_source_binding(tmp_path):
    import cupy as cp

    path, lat, lon, first, _ = _source(tmp_path)
    parent = BoundSmokeManifest(path, START, lat, lon, 2)
    parent.require_vertical((1.0, .5, 0.0), 2, .2, 1500.0)
    parent.require_coverage(7200)
    parent.at(3600)
    child = parent.rebind(lat[1:4, 2:6], lon[1:4, 2:6])
    assert child._cache == {}
    assert parent.identity() == child.identity()
    after = child.binding_receipt()
    assert after["local_shape"] == [2, 3, 4]
    assert after["offset_ji"] == [1, 2]
    np.testing.assert_array_equal(cp.asnumpy(child.at(0)["value"]), first[:, 1:4, 2:6].astype(np.float32))
    assert child._cache is not parent._cache


def test_neutral_rank_geography_waits_for_gather_and_tile_reuse_rebinds(tmp_path):
    import cupy as cp

    path, lat, lon, first, _ = _source(tmp_path)
    parent = BoundSmokeManifest(path, START, lat, lon, 2)
    parent.require_vertical((1.0, .5, 0.0), 2, .2, 1500.0)
    parent.require_coverage(7200)
    neutral_lat, neutral_lon = np.zeros((3, 4)), np.zeros((3, 4))
    child = parent.deferred_rebind(neutral_lat, neutral_lon)
    assert child.identity() == parent.identity()
    with pytest.raises(ValueError, match="not congruent|outside"):
        child.at(0)
    neutral_lat[...] = lat[1:4, 2:6]
    neutral_lon[...] = lon[1:4, 2:6]
    np.testing.assert_array_equal(cp.asnumpy(child.at(0)["value"]), first[:, 1:4, 2:6].astype(np.float32))
    assert child.binding_receipt()["offset_ji"] == [1, 2]
    neutral_lat[...] = lat[:3, :4]
    neutral_lon[...] = lon[:3, :4]
    np.testing.assert_array_equal(cp.asnumpy(child.at(0)["value"]), first[:, :3, :4].astype(np.float32))
    assert child.binding_receipt()["offset_ji"] == [0, 0]
