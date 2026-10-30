"""A nested run prepares as its leads post: nothing at the head waits for a later lead.

WOOF canary run_f530d559497471b8 (2026-10-29, nyc-500m on a cycle still
posting): the tree head held "Validate hierarchy source metadata" until f048
posted, the inland water's daily mean held every start state until f023, and
the forecast then failed at its head with "mapped input manifest does not
exist".  Each test below is one of those three, on the code that read them.
"""
import copy
from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm import source_hierarchy
from gpuwm.ingest import horiz
from gpuwm.ingest.source_metadata import SourceSnapshotMetadata
from test_source_hierarchy import _Grid, _inputs


class _PostingSeries(tuple):
    """Forcing times of which only the first ``ready`` are decoded; asking for a
    later one is what waited for its lead."""

    ready = 1
    asked = ()

    def decoded_positions(self):
        return self.ready

    def posted_positions(self):
        return self.ready

    def snapshot_metadata(self, index):
        if index >= self.ready:
            raise AssertionError(f"the head waited for forcing time {index}")
        self.asked = (*self.asked, index)
        item = self[index]
        return SourceSnapshotMetadata(type(item), item.latitude, item.longitude)


def _head(tmp_path, monkeypatch, snapshots):
    exp, _boundaries, initial, _ = _inputs(tmp_path, 2)
    monkeypatch.setattr(source_hierarchy, "verified_static_catalog",
                        lambda *a, **k: (SimpleNamespace(files=()), {}))
    monkeypatch.setattr(source_hierarchy, "initialize_native_hierarchy_children",
                        lambda **_kwargs: ((), 0.0))
    grids = tuple(_Grid(f"d{grid_id:02d}-grid", 17 + grid_id)
                  for grid_id in (1, 2))
    return source_hierarchy.prepare_regular_source_hierarchy_head(
        exp=exp, grids=grids, snapshots=snapshots, forcing_hours=(0, 1, 2),
        wps_namelist=tmp_path / "namelist.wps", geog_root=tmp_path / "geog",
        source_name="RW-WPS-MAPPED", root_initial_result=initial,
        source_manifest_sha256="b" * 64), exp, initial


def test_the_tree_head_reads_only_decoded_times_and_its_seal_the_rest(
        tmp_path, monkeypatch):
    _exp, _b, _i, plain = _inputs(tmp_path, 2)
    series = _PostingSeries(plain)
    head, exp, initial = _head(tmp_path, monkeypatch, series)
    assert series.asked and set(series.asked) == {0}
    assert head.metadata_positions == 1
    monkeypatch.setattr(source_hierarchy, "export_native_hierarchy",
                        lambda **_kwargs: "native-result")

    def seal(snapshots):
        return source_hierarchy.seal_regular_source_hierarchy(
            head, artifact_output=tmp_path / "artifacts",
            wrf_output=tmp_path / "wrf", root_initial_result=initial,
            root_met="met", root_soil="soil", root_static_fields={},
            root_boundaries=initial.state.lateral_boundaries,
            bridge_manifest_sha256="a" * 64, source_manifest_sha256="a" * 64,
            namelist_sha256="c" * 64, source_identity={"adapter": "mapped"},
            snapshots=snapshots)

    with pytest.raises(ValueError, match="needs the whole forcing series"):
        seal(None)
    series.ready = 3
    assert seal(series).hierarchy == "native-result"
    assert set(series.asked) == {0, 1, 2}

    # A time that posted on another grid is refused at the seal.
    moved = list(plain)
    moved[2] = copy.copy(plain[2])
    moved[2].latitude = np.asarray(plain[2].latitude) + 1.0
    moved = _PostingSeries(moved)
    moved.ready = 3
    with pytest.raises(ValueError, match="source grid changes"):
        seal(moved)


def test_a_series_read_whole_is_still_checked_whole_at_its_head(
        tmp_path, monkeypatch):
    _exp, _b, _i, plain = _inputs(tmp_path, 2)
    head, _exp, _initial = _head(tmp_path, monkeypatch, plain)
    assert head.metadata_positions is None


class _AirSeries(tuple):
    def __new__(cls, count, ready):
        start = datetime(2026, 10, 29, 12)
        self = super().__new__(cls, range(count))
        self.valid_times = tuple(start + timedelta(hours=h) for h in range(count))
        self.ready = ready
        self.read = []
        return self

    def posted_positions(self):
        return self.ready

    def prefetch_through(self, positions):
        assert max(positions) < self.ready, "the mean waited for a later lead"

    def surface_field_values(self, position, name):
        assert position < self.ready, "the mean read a lead not posted"
        self.read.append(position)
        return np.full((2, 2), 280.0 + position)


def test_the_inland_daily_mean_averages_the_leads_posted_at_the_start():
    snapshot = SimpleNamespace(fields={"SKINTEMP": np.zeros((2, 2))})
    series = _AirSeries(49, ready=3)
    field, receipt = horiz.DailyMeanAirTemperature(series)(snapshot)
    assert sorted(series.read) == [0, 1, 2]
    np.testing.assert_allclose(field, 281.0)
    assert "the 3 of the day's 24 posted" in receipt["window"]
    whole = _AirSeries(49, ready=49)
    field, receipt = horiz.DailyMeanAirTemperature(whole)(snapshot)
    np.testing.assert_allclose(field, 280.0 + 11.5)
    assert "posted" not in receipt["window"]


def test_a_child_takes_the_inland_window_its_root_chose():
    from dataclasses import make_dataclass
    Statics = make_dataclass("Statics", [("inland_air_temperature", object)])
    series = _AirSeries(25, ready=25)
    root = horiz.with_inland_air_temperature(Statics(None), series, hours=0.0)
    child = horiz.with_inland_air_temperature(Statics(None), series)
    assert root.inland_air_temperature is None
    assert child.inland_air_temperature is None
    other = _AirSeries(25, ready=25)
    root = horiz.with_inland_air_temperature(Statics(None), other)
    child = horiz.with_inland_air_temperature(Statics(None), other)
    assert child.inland_air_temperature is root.inland_air_temperature


def test_a_sealed_mapped_tree_is_held_to_its_head_by_the_manifest_its_proof_names(
        tmp_path, monkeypatch):
    """The mapped proof names its manifest in its composition receipt; the sealed
    preflight refused every sealed mapped as-posted tree ("'None' is not one")."""

    from gpuwm import prepared_domain_tree_forecast as tree
    from gpuwm.ingest import boundary_stream

    head = {"basis": {"as_posted": {
        "proof_manifest_key": "source_composition.input_manifest.sha256"}}}
    seen = {}

    def children(root, *, head, manifest_sha256):
        seen["manifest"] = manifest_sha256
        return {"d02": {"head_content_sha256": "h" * 64,
                        "sealed_content_sha256": "s" * 64}}

    monkeypatch.setattr(boundary_stream, "read_head", lambda root, **k: head)
    monkeypatch.setattr(boundary_stream, "verify_as_posted_tree_children",
                        children)
    proof = {"posting": {"as_posted": True},
             "boundary_stream": {"head_sha256": "1" * 64},
             "source_composition": {"input_manifest": {"sha256": "m" * 64}}}
    bundles = [SimpleNamespace(parent_id=0, grid_id=1),
               SimpleNamespace(parent_id=1, grid_id=2, cache_reader=SimpleNamespace(
                   content_sha256="s" * 64))]
    assert dict(tree._as_posted_head_children(tmp_path, proof, bundles)) == {
        "d02": "h" * 64}
    assert seen["manifest"] == "m" * 64
