"""Site discovery: what it selects, and everything it refuses to guess.

No site id is written into :mod:`gpuwm.obs.coverage`; ids are results.
The fixture table here is synthetic and its ids are invented so that a
test cannot become the place a real site name leaks into the tree.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from gpuwm.obs import coverage as cov


@dataclass
class _Grid:
    """The two attributes discovery reads, and nothing else."""

    lat: np.ndarray
    lon: np.ndarray


#: Half-width chosen so the domain's CORNERS sit ~207 km from its centre,
#: comfortably inside a 250 km range authority.  At 1.8 deg the corners are
#: 250.6 km out and a site directly overhead covers 0.9988 of the domain --
#: correct arithmetic, and a fixture that makes "covers all of it" mean
#: something else.
def _grid(center_lat=42.0, center_lon=-94.0, half_deg=1.5, n=41):
    lat = np.linspace(center_lat - half_deg, center_lat + half_deg, n)
    lon = np.linspace(center_lon - half_deg, center_lon + half_deg, n)
    lon2d, lat2d = np.meshgrid(lon, lat)
    return _Grid(lat=lat2d, lon=lon2d)


def _table(rows):
    return {row[0]: cov.SiteFix(id=row[0], name=row[1], lat_deg=row[2],
                                lon_deg=row[3]) for row in rows}


OVERHEAD = ("AAAA", "over the domain", 42.0, -94.0)
NEARBY = ("BBBB", "one domain away", 42.0, -91.5)
EDGE = ("CCCC", "clipping a corner", 40.0, -90.5)
FAR = ("DDDD", "nowhere near", 30.0, -80.0)


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------
def test_a_site_over_the_domain_outranks_one_clipping_its_corner():
    found = cov.sites_covering(_grid(), _table([EDGE, OVERHEAD, NEARBY]),
                               max_range_km=250.0)
    assert [entry.site.id for entry in found][0] == "AAAA"
    fractions = [entry.coverage_fraction for entry in found]
    assert fractions == sorted(fractions, reverse=True)
    assert found[0].coverage_fraction == pytest.approx(1.0, abs=1e-9)


def test_a_site_out_of_range_is_simply_absent():
    found = cov.sites_covering(_grid(), _table([OVERHEAD, FAR]),
                               max_range_km=250.0)
    assert [entry.site.id for entry in found] == ["AAAA"]


def test_coverage_is_a_measured_fraction_of_the_domains_own_points():
    grid = _grid()
    found = cov.sites_covering(grid, _table([OVERHEAD]), max_range_km=250.0)
    entry = found[0]
    assert entry.points_total == grid.lat.size
    assert entry.points_covered == entry.points_total
    assert entry.nearest_km < entry.farthest_km


def test_the_floor_excludes_a_marginal_site_and_the_receipt_says_so():
    table = _table([OVERHEAD, EDGE])
    generous = cov.sites_covering(_grid(), table, max_range_km=250.0,
                                  min_coverage_fraction=0.0)
    strict = cov.sites_covering(_grid(), table, max_range_km=250.0,
                                min_coverage_fraction=0.99)
    assert len(generous) > len(strict)
    assert [entry.site.id for entry in strict] == ["AAAA"]
    receipt = cov.discovery_receipt(
        strict, max_range_km=250.0, min_coverage_fraction=0.99, limit=None,
        table_size=len(table))
    # A selection is meaningless without the parameters that produced it.
    assert receipt["min_coverage_fraction"] == 0.99
    assert receipt["max_range_km"] == 250.0
    assert [row["id"] for row in receipt["selected"]] == ["AAAA"]


def test_the_limit_keeps_the_best_covered_sites():
    found = cov.sites_covering(_grid(), _table([EDGE, OVERHEAD, NEARBY]),
                               max_range_km=250.0, limit=2)
    assert len(found) == 2
    assert found[0].site.id == "AAAA"


def test_selection_order_is_total_and_reproducible():
    """Two runs of one domain against one table select the same sequence."""

    table = _table([OVERHEAD, NEARBY, EDGE])
    first = cov.sites_covering(_grid(), table, max_range_km=250.0)
    second = cov.sites_covering(_grid(), dict(reversed(list(table.items()))),
                                max_range_km=250.0)
    assert ([e.site.id for e in first] == [e.site.id for e in second])


def test_ties_break_on_the_id_so_the_order_never_depends_on_dict_order():
    # Two sites placed symmetrically about the domain cover it identically.
    table = _table([("ZZZZ", "west", 42.0, -96.0),
                    ("YYYY", "east", 42.0, -92.0)])
    found = cov.sites_covering(_grid(), table, max_range_km=400.0)
    assert [entry.site.id for entry in found] == ["YYYY", "ZZZZ"]


def test_a_domain_no_radar_reaches_returns_empty_rather_than_raising():
    # "Nothing covers this" is a fact about the domain; the caller decides
    # whether it is fatal.
    assert cov.sites_covering(_grid(), _table([FAR]),
                              max_range_km=250.0) == []


# --------------------------------------------------------------------------
# refusals
# --------------------------------------------------------------------------
def test_the_screening_range_must_be_the_real_one():
    for bad in (0.0, -10.0, float("nan")):
        with pytest.raises(cov.SiteCoverageError, match="max_range_km"):
            cov.sites_covering(_grid(), _table([OVERHEAD]), max_range_km=bad)


def test_a_coverage_floor_outside_zero_to_one_is_not_a_fraction():
    with pytest.raises(cov.SiteCoverageError, match=r"\[0, 1\]"):
        cov.sites_covering(_grid(), _table([OVERHEAD]), max_range_km=250.0,
                           min_coverage_fraction=1.5)


def test_asking_for_fewer_than_one_radar_is_a_refusal():
    with pytest.raises(cov.SiteCoverageError, match="empty one"):
        cov.sites_covering(_grid(), _table([OVERHEAD]), max_range_km=250.0,
                           limit=0)


def test_a_table_from_the_wrong_schema_is_refused(monkeypatch):
    monkeypatch.setattr("gpuwm.obs.nexrad.run_sites",
                        lambda binary, site=None: {"schema": "something.v9",
                                                   "sites": []})
    with pytest.raises(cov.SiteCoverageError, match="disagree about the"):
        cov.read_site_table("rw_nexrad")


def test_an_empty_table_is_refused_rather_than_defaulted(monkeypatch):
    monkeypatch.setattr(
        "gpuwm.obs.nexrad.run_sites",
        lambda binary, site=None: {"schema": cov.SITES_SCHEMA, "sites": []})
    with pytest.raises(cov.SiteCoverageError, match="hardcoded site"):
        cov.read_site_table("rw_nexrad")


def test_a_row_with_an_impossible_latitude_is_refused(monkeypatch):
    monkeypatch.setattr(
        "gpuwm.obs.nexrad.run_sites",
        lambda binary, site=None: {
            "schema": cov.SITES_SCHEMA,
            "sites": [{"id": "AAAA", "name": "bad", "lat_deg": 120.0,
                       "lon_deg": -94.0}]})
    with pytest.raises(cov.SiteCoverageError, match="not a latitude"):
        cov.read_site_table("rw_nexrad")


def test_a_front_door_that_cannot_answer_is_a_refusal(monkeypatch):
    def _boom(binary, site=None):
        raise RuntimeError("binary missing")

    monkeypatch.setattr("gpuwm.obs.nexrad.run_sites", _boom)
    with pytest.raises(cov.SiteCoverageError, match="could not produce"):
        cov.read_site_table("rw_nexrad")


# --------------------------------------------------------------------------
# the geometry itself
# --------------------------------------------------------------------------
def test_great_circle_is_zero_at_the_point_and_symmetric():
    assert cov.great_circle_km(42.0, -94.0, np.array([42.0]),
                               np.array([-94.0]))[0] == pytest.approx(0.0)
    there = cov.great_circle_km(42.0, -94.0, np.array([43.0]),
                                np.array([-93.0]))[0]
    back = cov.great_circle_km(43.0, -93.0, np.array([42.0]),
                               np.array([-94.0]))[0]
    assert there == pytest.approx(back, rel=1e-12)


def test_one_degree_of_latitude_is_about_111_km():
    d = cov.great_circle_km(42.0, -94.0, np.array([43.0]), np.array([-94.0]))
    assert d[0] == pytest.approx(111.19, abs=0.1)
