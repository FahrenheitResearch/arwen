"""Geographic placement of cycle children: resolution, refusal, ranking.

A placement REQUEST is geographic -- a latitude, a longitude and a
footprint.  A parent index is a resolution RESULT.  Everything in this
file exists to hold that line, because the moment a request is expressed
in parent indices the mechanism stops being general and starts being a
transcription of one domain's geometry.

Pure tests: no GPU, no network.  Synthetic geometry only -- no case
names, no station names, no CONUS-shaped constants (standing owner rule).
"""
from __future__ import annotations

import numpy as np
import pytest

from gpuwm.cycle.contracts import CHILD_STATES, CycleRefusal
from gpuwm.cycle.placement import (ObsPlacementProvider, PlacementRequest,
                                   ResolvedPlacement,
                                   SchedulePlacementProvider,
                                   TrackerPlacementProvider,
                                   parent_geometry_from_fields,
                                   rank_and_assign, resolve_placement)


def geometry(*, ny: int = 60, nx: int = 80, lat0: float = 30.0,
             lon0: float = -100.0, step: float = 0.05, dx_m: float = 3000.0):
    """A synthetic regular parent: lat rises with j, lon rises with i."""

    lat = lat0 + step * np.arange(ny, dtype=np.float64)[:, None]
    lon = lon0 + step * np.arange(nx, dtype=np.float64)[None, :]
    return parent_geometry_from_fields(
        lat_field=np.broadcast_to(lat, (ny, nx)).copy(),
        lon_field=np.broadcast_to(lon, (ny, nx)).copy(),
        dx_m=dx_m)


def request(lat: float, lon: float, *, nx: int = 30, ny: int = 30,
            source: str = "tracker", strength: float = 55.0,
            dx_m: float = 1000.0) -> PlacementRequest:
    return PlacementRequest(lat=lat, lon=lon, dx_m=dx_m, nx=nx, ny=ny,
                            source=source, strength=strength,
                            evidence={"cells": 12, "max_value": strength})


# ---------------------------------------------------------------------------
# lat/lon in, parent index out
# ---------------------------------------------------------------------------
class TestResolution:
    def test_latlon_resolves_to_expected_parent_index(self):
        parent = geometry()
        # Cell (j=30, i=40) sits at 31.50N / -98.00E by construction.
        got = resolve_placement(request(31.50, -98.00), parent_geometry=parent,
                                grid_id=2)
        assert got.state == "PLANNED"
        assert got.refusal is None
        assert got.clamped is False
        assert got.parent_grid_ratio == 3
        # 30x30 child at ratio 3 spans 10 parent cells; centred on (30,40)
        # in 1-based namelist semantics.
        assert (got.j_parent_start, got.i_parent_start) == (27, 37)

    def test_the_resolved_centre_is_a_result_not_a_request(self):
        """A request carries no indices; only the resolution does."""

        assert not hasattr(request(31.5, -98.0), "i_parent_start")
        assert isinstance(resolve_placement(
            request(31.5, -98.0), parent_geometry=geometry(), grid_id=2),
            ResolvedPlacement)

    def test_a_non_dividing_footprint_refuses_by_name(self):
        with pytest.raises(CycleRefusal) as caught:
            resolve_placement(request(31.5, -98.0, nx=31),
                              parent_geometry=geometry(), grid_id=2)
        assert "31" in str(caught.value)


# ---------------------------------------------------------------------------
# the keepout: refusal, never a silent clamp
# ---------------------------------------------------------------------------
class TestKeepout:
    #: Cell j=51 of a 60-row parent: a 10-parent-cell footprint centred
    #: there starts at j=48 and ends at j=57, keeping 3 parent rows on
    #: the north side of the required 10.
    NORTH_EDGE_LAT = 30.0 + 0.05 * 51

    def test_placement_refuses_outside_parent(self):
        got = resolve_placement(
            request(self.NORTH_EDGE_LAT, -98.00), parent_geometry=geometry(),
            grid_id=2)
        assert got.state == "REFUSED"
        assert got.clamped is False
        assert got.refusal is not None
        text = got.refusal
        # The refusal names WHAT WAS OBSERVED and where it came from: the
        # geographic request, the parent index it mapped to, the rows kept,
        # the rows required, and the two widths that add up to the
        # requirement.  Every one of those numbers is load-bearing for the
        # person reading it at 3am.
        assert "35.20N" not in text          # it must quote the REQUEST
        assert f"{self.NORTH_EDGE_LAT:.2f}N" in text
        assert "-98.00" in text
        assert "north edge" in text
        assert "keeps 3 parent rows of the required 10" in text
        assert "spec_bdy_width 5 + blend_width 5" in text
        assert text.rstrip().endswith("placement REFUSED")

    def test_clamp_is_opt_in_and_flagged(self):
        got = resolve_placement(
            request(self.NORTH_EDGE_LAT, -98.00), parent_geometry=geometry(),
            grid_id=2, allow_clamp=True)
        assert got.state == "PLANNED"
        assert got.clamped is True
        assert got.refusal is None
        # Clamped INTO the keepout, which is the honest cost: the nest
        # integrates but has stopped following its storm.
        assert got.j_parent_start == 60 - (30 // 3) + 1 - 10

    def test_the_default_is_refusal(self):
        """A capability behind a flag is a workaround; this one is not."""

        assert resolve_placement(
            request(self.NORTH_EDGE_LAT, -98.00),
            parent_geometry=geometry(), grid_id=2).state == "REFUSED"


# ---------------------------------------------------------------------------
# providers
# ---------------------------------------------------------------------------
class TestProviders:
    def test_two_storms_one_box_get_two_nests(self):
        parent = geometry()
        signal = np.zeros((60, 80), dtype=np.float64)
        signal[20, 20] = 62.0
        signal[21, 20] = 58.0
        signal[40, 60] = 55.0
        signal[40, 61] = 51.0
        provider = TrackerPlacementProvider(
            field="reflectivity", threshold=45.0, min_separation_km=20.0,
            max_children=4, child_nx=30, child_ny=30, child_dx_m=1000.0)
        got = provider(cycle_index=0, valid_time=None,
                       parent_geometry=parent, signal=signal)
        assert len(got) == 2
        # Two distinct storms, NOT one centroid between them: the mean of
        # the two would land near j=30/i=40, which is neither storm.
        lats = sorted(round(item.lat, 2) for item in got)
        assert lats == [31.00, 32.00]
        assert got[0].strength >= got[1].strength     # ranked by strength

    def test_a_quiet_parent_produces_no_requests(self):
        provider = TrackerPlacementProvider(
            field="uh", threshold=45.0, min_separation_km=20.0,
            max_children=4, child_nx=30, child_ny=30, child_dx_m=1000.0)
        assert provider(cycle_index=0, valid_time=None,
                        parent_geometry=geometry(),
                        signal=np.zeros((60, 80))) == []

    def test_schedule_provider_returns_the_entry_for_its_time(self):
        provider = SchedulePlacementProvider(entries=[
            (10, 31.0, -98.0, 1000.0, 30, 30),
            (20, 32.0, -97.0, 1000.0, 30, 30)])
        got = provider(cycle_index=0, valid_time=20,
                       parent_geometry=geometry(), signal=None)
        assert [(item.lat, item.lon) for item in got] == [(32.0, -97.0)]

    def test_obs_provider_places_on_radar_echo_without_a_model_field(
            self, tmp_path):
        """The capability the model-field triggers cannot have.

        The parent's own plane is DEAD FLAT here -- the model has no storm
        at all.  The radar does, and the child goes where the radar says.
        """

        parent = geometry()
        obs = np.zeros((60, 80), dtype=np.float64)
        obs[35, 55] = 58.0
        provider = ObsPlacementProvider(
            radar_grid_path=tmp_path / "obs.nc", field="z_max",
            threshold_dbz=40.0, min_separation_km=20.0, max_children=4,
            child_nx=30, child_ny=30, child_dx_m=1000.0,
            reader=lambda path, field: obs)
        got = provider(cycle_index=0, valid_time=None,
                       parent_geometry=parent,
                       signal=np.zeros((60, 80)))
        assert len(got) == 1
        assert got[0].source == "obs"
        assert (round(got[0].lat, 2), round(got[0].lon, 2)) == (31.75, -97.25)
        # And the model-field provider on the same parent finds nothing,
        # which is the whole point of the arm.
        tracker = TrackerPlacementProvider(
            field="reflectivity", threshold=40.0, min_separation_km=20.0,
            max_children=4, child_nx=30, child_ny=30, child_dx_m=1000.0)
        assert tracker(cycle_index=0, valid_time=None,
                       parent_geometry=parent,
                       signal=np.zeros((60, 80))) == []


# ---------------------------------------------------------------------------
# ranking and the slot pool's scarcity
# ---------------------------------------------------------------------------
class TestRankAndAssign:
    def test_slot_exhaustion_emits_an_explicit_refusal(self):
        parent = geometry()
        requests = [request(31.0, -99.0, strength=50.0),
                    request(32.0, -98.0, strength=70.0),
                    request(31.5, -97.0, strength=60.0)]
        got = rank_and_assign(requests, free_slots=(2, 3),
                              min_separation_km=5.0,
                              parent_geometry=parent)
        assert len(got) == 3, "a request must never vanish silently"
        assert [item.request.strength for item in got] == [70.0, 60.0, 50.0]
        assert [item.grid_id for item in got][:2] == [2, 3]
        weakest = got[-1]
        assert weakest.state == "REFUSED"
        assert weakest.refusal == "slot pool exhausted"
        assert weakest.grid_id == -1

    def test_near_duplicates_are_dropped_with_a_named_refusal(self):
        parent = geometry()
        requests = [request(31.0, -98.0, strength=70.0),
                    request(31.001, -98.001, strength=40.0)]
        got = rank_and_assign(requests, free_slots=(2, 3),
                              min_separation_km=25.0,
                              parent_geometry=parent)
        assert len(got) == 2
        assert got[0].state == "PLANNED"
        assert got[1].state == "REFUSED"
        assert "min_separation_km" in got[1].refusal

    def test_every_state_is_from_the_shared_vocabulary(self):
        parent = geometry()
        got = rank_and_assign([request(31.0, -98.0)], free_slots=(2,),
                              min_separation_km=5.0, parent_geometry=parent)
        assert all(item.state in CHILD_STATES for item in got)


def test_obs_provider_reads_a_real_radar_grid_files_variables(tmp_path):
    """The canonical reader path must find fields where the writer puts them.

    ``read_radar_grid`` returns the observation planes under a
    ``"variables"`` mapping, not at the document's top level.  A provider
    that looks only at the top level refuses every real file it is handed
    while passing every injected-reader test -- which is exactly how this
    seam shipped.  Real-file shaped, so it stays red on revert.
    """
    import numpy as np
    from gpuwm.cycle.placement import (ObsPlacementProvider,
                                       parent_geometry_from_fields)

    ny = nx = 12
    lat = np.linspace(35.0, 35.4, ny)[:, None] * np.ones((1, nx))
    lon = np.ones((ny, 1)) * np.linspace(-97.5, -97.1, nx)[None, :]
    geometry = parent_geometry_from_fields(lat_field=lat, lon_field=lon,
                                           dx_m=3000.0)
    plane = np.zeros((2, ny, nx))
    plane[1, 6, 6] = 55.0

    # Exactly the shape read_radar_grid returns for a shipped file.
    document = {"schema": "gpuwm-obs.radar-grid.v1",
                "variables": {"z_obs": plane}}

    provider = ObsPlacementProvider(
        radar_grid_path="<document>", field="z_obs", threshold_dbz=40.0,
        min_separation_km=20.0, max_children=1, child_nx=6, child_ny=6,
        child_dx_m=1000.0,
        reader=lambda _path, field: _lookup(document, field))
    requests = provider(cycle_index=0, valid_time=None,
                        parent_geometry=geometry, signal=None)
    assert len(requests) == 1
    assert requests[0].evidence["max_value"] == 55.0


def _lookup(document, field):
    """The lookup the provider itself must perform."""
    from gpuwm.cycle.placement import _document_plane

    return _document_plane(document, field, "<document>")
