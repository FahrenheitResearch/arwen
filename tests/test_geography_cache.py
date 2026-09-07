"""Exact host geography cache lifetime, independent of source or GPU."""
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.core.geography_cache import geography_cache, geography_cache_snapshots


def owner(latitude):
    return SimpleNamespace(latitude_deg=np.asarray(latitude), cache=None,
                           geography_cache_dependencies={
                               "cache": (("latitude_deg", "latitude_snapshot"),)})


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_cache_follows_in_place_and_replaced_geography(dtype):
    from gpuwm.ingest import wrf_ozone
    climo = wrf_ozone.load_ozone_climatology()
    obj = owner(np.array([[-60, -10], [25, 70]], dtype=dtype))
    calls = []
    def compute():
        calls.append(1)
        return wrf_ozone.interp_ozone_to_latitudes(
            obj.latitude_deg.reshape(-1), climo)
    original = geography_cache(obj, "cache", compute)
    assert geography_cache(obj, "cache", compute) is original
    assert len(calls) == 1
    snapshot = geography_cache_snapshots(obj)["latitude_snapshot"]
    assert snapshot.dtype == dtype
    assert snapshot.nbytes == obj.latitude_deg.nbytes
    assert not np.shares_memory(snapshot, obj.latitude_deg)
    obj.latitude_deg[:] *= -1
    changed = geography_cache(obj, "cache", compute)
    assert not np.array_equal(changed, original)
    np.testing.assert_array_equal(changed, wrf_ozone.interp_ozone_to_latitudes(
        obj.latitude_deg.reshape(-1), climo))
    obj.latitude_deg = obj.latitude_deg[:, :1].copy()
    resized = geography_cache(obj, "cache", compute)
    assert resized.shape[0] == 2
    assert len(calls) == 3
    assert geography_cache(obj, "cache", compute) is resized


def test_dtype_change_rebinds_snapshot_without_coercion():
    obj = owner(np.array([[35]], dtype=np.float32))
    first = geography_cache(obj, "cache", lambda: object())
    obj.latitude_deg = obj.latitude_deg.astype(np.float64)
    assert geography_cache(obj, "cache", lambda: object()) is not first
    assert obj.latitude_snapshot.dtype == np.float64
    assert obj.latitude_snapshot.nbytes == 8


def test_failed_rebuild_keeps_previous_binding_for_retry():
    obj = owner(np.array([[35]], dtype=np.float32))
    first = geography_cache(obj, "cache", lambda: object())
    obj.latitude_deg[:] = -35
    def fail():
        raise RuntimeError("interpolation failed")
    with pytest.raises(RuntimeError, match="interpolation failed"):
        geography_cache(obj, "cache", fail)
    assert obj.cache is first
    assert obj.latitude_snapshot.item() == 35
    assert geography_cache(obj, "cache", lambda: object()) is not first


def test_every_declared_dependency_participates():
    obj = owner(np.array([[35]], dtype=np.float32))
    obj.longitude_deg = np.array([[10]], dtype=np.float64)
    obj.geography_cache_dependencies["cache"] += (
        ("longitude_deg", "longitude_snapshot"),)
    first = geography_cache(obj, "cache", lambda: object())
    obj.longitude_deg[:] = 20
    assert geography_cache(obj, "cache", lambda: object()) is not first
    assert sum(a.nbytes for a in geography_cache_snapshots(obj).values()) == 12


def test_cache_without_inputs_is_refused():
    obj = owner(np.array([[35]], dtype=np.float32))
    obj.geography_cache_dependencies["cache"] = ()
    with pytest.raises(ValueError, match="no declared inputs"):
        geography_cache(obj, "cache", lambda: object())
