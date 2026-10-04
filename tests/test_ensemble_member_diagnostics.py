from datetime import datetime, timedelta
import json

import numpy as np
import pytest

from gpuwm.ensemble.member_diagnostics import MemberDiagnosticArchive


def test_native_member_store_retains_sparse_id_seed_units_and_float_words(tmp_path):
    from netCDF4 import Dataset
    start = datetime(2024, 1, 1)
    archive = MemberDiagnosticArchive(tmp_path, member_order=(19,),
        member_metadata=({"member_id": 19, "seed": 2**64-1, "recipe_sha256": "a"*64,
                          "source_manifests": [{"manifest_sha256": "b"*64}]},), start_time=start)
    words = np.array([0, 0x80000000, 0x3f800001, 0x00000001], np.uint32).view(np.float32).reshape(2, 2)
    fields = {name: words for name in ("T2", "U10", "V10", "RAIN_TOTAL")}
    coordinates = np.ones((2, 2), np.float32)
    assert archive.submit(fields, member_id=19, valid_time=start, grid_id=1, episode=0,
                          latitude=coordinates, longitude=coordinates)
    record = json.loads(archive.path.read_text())
    path = tmp_path / record["files"][0]["path"]
    with Dataset(path) as dataset:
        assert int(dataset.variables["member_id"][:][0]) == 19
        assert int(dataset.variables["member_seed"][:][0]) == 2**64-1
        for name in fields:
            assert dataset.variables[name][:].tobytes() == words.tobytes()
        assert dataset.variables["T2"].getncattr("units") == "K"
        assert dataset.variables["RAIN_TOTAL"].getncattr("units") == "mm"
    assert record["member_order"] == [19]
    assert record["precipitation_accumulation_start"] == start.isoformat()
    with pytest.raises(ValueError, match="submitted twice"):
        archive.submit(fields, member_id=19, valid_time=start, grid_id=1, episode=0,
                       latitude=coordinates, longitude=coordinates)
    assert not archive.submit(fields, member_id=19, valid_time=start+timedelta(seconds=30),
        grid_id=1, episode=0, latitude=coordinates, longitude=coordinates)


def _forecast(start, *, run_seconds=7200, child_start=1800, dynamic=False):
    from types import SimpleNamespace
    domains = tuple(SimpleNamespace(grid_id=grid, history_interval_s=3600.,
        history_begin_s=0., history_end_s=None,
        spawn=object() if dynamic and grid == 2 else None, retire=None, rearm=None)
        for grid in (1, 2))
    return SimpleNamespace(start_time=start, run_seconds=run_seconds, domains=domains,
        domain_start_time=lambda grid: start + timedelta(seconds=0 if grid == 1 else child_start))


def test_coverage_lists_missing_member_hours_and_delayed_child_initial(tmp_path):
    start = datetime(2024, 1, 1)
    archive = MemberDiagnosticArchive(tmp_path, member_order=(17, 23),
        member_metadata=({"member_id": member, "seed": member} for member in (17, 23)), start_time=start)
    exp = _forecast(start)
    for member in archive.order:
        archive.expect_member_forecast(member, exp)
    assert archive.receipt()["coverage"]["status"] == "running"
    words = np.ones((2, 2), np.float32)
    fields = {name: words for name in ("T2", "U10", "V10", "RAIN_TOTAL")}
    # A delayed child's genuine initial plane is retained even off the hour.
    assert archive.submit(fields, member_id=17, valid_time=start+timedelta(seconds=1800),
        grid_id=2, episode=0, latitude=words, longitude=words)
    archive.submit(fields, member_id=17, valid_time=start,
        grid_id=1, episode=0, latitude=words, longitude=words)
    record = archive.finish()
    assert record["coverage"]["status"] == "incomplete"
    assert record["coverage"]["expected_frames"] == 12
    missing = {(row["member_id"], row["grid_id"], row["valid_time"])
               for row in record["coverage"]["missing_frames"]}
    assert len(missing) == 10
    assert (23, 1, start.isoformat()) in missing
    assert (17, 1, (start+timedelta(hours=1)).isoformat()) in missing
    assert (17, 2, (start+timedelta(seconds=1800)).isoformat()) not in missing
    assert record["files"][0]["valid_time"] == (start+timedelta(seconds=1800)).isoformat()


def test_missing_fields_cannot_make_hourly_coverage_complete(tmp_path):
    start = datetime(2024, 1, 1)
    archive = MemberDiagnosticArchive(tmp_path, member_order=(17,),
        member_metadata=({"member_id": 17, "seed": 17},), start_time=start)
    exp = _forecast(start, run_seconds=0, child_start=1800)
    archive.expect_member_forecast(17, exp)
    words = np.ones((2, 2), np.float32)
    assert not archive.submit({"T2": words}, member_id=17, valid_time=start,
        grid_id=1, episode=0, latitude=words, longitude=words)
    record = archive.finish()
    assert record["coverage"]["status"] == "incomplete"
    assert record["coverage"]["missing_frames"][0]["member_id"] == 17
    assert record["unavailable"][0]["missing_fields"] == ["RAIN_TOTAL", "U10", "V10"]


def test_dynamic_child_is_not_declared_complete_from_its_placeholder_calendar(tmp_path):
    start = datetime(2024, 1, 1)
    archive = MemberDiagnosticArchive(tmp_path, member_order=(17,),
        member_metadata=({"member_id": 17, "seed": 17},), start_time=start)
    archive.expect_member_forecast(17, _forecast(start, run_seconds=0, dynamic=True))
    words = np.ones((2, 2), np.float32)
    fields = {name: words for name in ("T2", "U10", "V10", "RAIN_TOTAL")}
    archive.submit(fields, member_id=17, valid_time=start,
        grid_id=1, episode=0, latitude=words, longitude=words)
    record = archive.finish()
    assert record["coverage"]["status"] == "dynamic_lifecycle"
    assert record["coverage"]["missing_frames"] == []
    assert record["coverage"]["dynamic_domains"][0]["grid_id"] == 2
