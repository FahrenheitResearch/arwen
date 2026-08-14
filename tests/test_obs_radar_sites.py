"""The radar site inventory, and the refusal that keeps it honest.

The whole point of this module is one behaviour: a site with no antenna
elevation is refused rather than assimilated. Most of this file is that
behaviour, approached from both directions -- it must fire on the shipped
inventory, and it must stop firing the moment an elevation is supplied.
"""

from __future__ import annotations

import json

import pytest

from gpuwm.obs.radar_sites import (TABLE_PATH, TABLE_SCHEMA, RadarSite,
                                   RadarSiteError, RadarSiteTable,
                                   SiteNotAssimilableError, coverage_summary,
                                   load_table, require_assimilable,
                                   sites_for_bbox)


def _site(site_id="0-000-0-aaa", *, elevation=None, velocity=True,
          volume=True, lat=50.0, lon=9.0):
    return RadarSite(
        id=site_id, name=site_id, wmo_block="000", latitude=lat, longitude=lon,
        elevation_m=elevation, moments=("DBZH", "VRADH") if velocity
        else ("DBZH",), methods=("scan",) if volume else ("comp",),
        has_velocity=velocity, has_volume_scan=volume)


def _table(sites):
    return RadarSiteTable(schema=TABLE_SCHEMA, source_url="", source_sha256="",
                          frozen_at="", elevation_basis="",
                          sites=tuple(sites))


# ------------------------------------------------------ the shipped table

def test_the_shipped_inventory_declares_its_schema_and_provenance():
    table = load_table()
    assert table.schema == TABLE_SCHEMA
    assert table.source_url.startswith("https://")
    assert len(table.source_sha256) == 64
    assert table.frozen_at
    # Where the elevations came from is documented in the artifact, not only
    # in a docstring: the feed does not publish them and they are read from
    # the volumes, which is a different provenance from the rest of the row.
    assert "elevation" in table.elevation_basis
    assert "/where/height" in table.elevation_basis


def test_the_inventory_holds_the_network_and_not_the_composite():
    """The composite is a product, not an antenna.

    Leaving it in would put a pseudo-site with no scan strategy in front of
    a caller asking which radars cover a domain.
    """

    table = load_table()
    assert len(table) > 100
    assert not any(site.id.endswith("-OPERA") for site in table.sites)


def test_the_coverage_summary_reports_the_elevations_as_a_number():
    """This used to assert ``with_elevation == 0`` and that was the headline.

    The feed still publishes no antenna elevation; the shipped table now
    carries one for every site because ``tools/harvest_radar_heights.py``
    reads ``/where/height`` out of a volume per site and
    ``tools/freeze_radar_sites.py --heights`` merges it. So the assertion
    flips, and it is an equality rather than a floor: a partial harvest
    silently leaving some sites null is exactly the state that makes a
    domain's coverage look better than it is.
    """

    summary = coverage_summary()
    assert summary["sites"] > 100
    # Nearly every radar serves velocity: the data is there.
    assert summary["with_velocity"] >= summary["sites"] - 5
    assert summary["with_volume_scan"] == summary["sites"]
    assert summary["with_elevation"] == summary["sites"]
    assert summary["wmo_blocks"] > 10


def test_the_inventory_ships_inside_the_package():
    assert TABLE_PATH.is_file()
    assert TABLE_PATH.parent.name == "data"


# ------------------------------------------------------------- the lookup

def test_a_domain_selects_the_radars_inside_it():
    table = _table([_site("a", lat=50.0, lon=9.0),
                    _site("b", lat=60.0, lon=25.0)])
    found = sites_for_bbox(5.0, 47.0, 15.0, 55.0, table=table)
    assert [site.id for site in found] == ["a"]


def test_velocity_filtering_changes_which_sites_not_whether_they_are_usable():
    table = _table([_site("withvel", velocity=True),
                    _site("novel", velocity=False)])
    everything = sites_for_bbox(0.0, 40.0, 20.0, 60.0, table=table)
    assert len(everything) == 2
    only_velocity = sites_for_bbox(0.0, 40.0, 20.0, 60.0, table=table,
                                   require_velocity=True)
    assert [site.id for site in only_velocity] == ["withvel"]
    # Neither call has said anything about assimilability.
    with pytest.raises(SiteNotAssimilableError):
        require_assimilable(only_velocity)


def test_a_box_that_is_not_a_box_refuses():
    with pytest.raises(RadarSiteError):
        sites_for_bbox(15.0, 47.0, 5.0, 55.0)


# ------------------------------------------------------------ the refusal

def test_the_shipped_inventory_is_now_assimilable():
    """The gate, on the real artifact, from the other side.

    This test used to assert the refusal, with a docstring warning that if it
    ever passed without the inventory gaining elevations, the gate had been
    removed. The inventory gained them, so the assertion is inverted -- and
    the guard the old docstring wanted lives in
    :func:`test_the_gate_still_refuses_a_site_whose_elevation_is_missing`
    below, which proves the gate is still armed rather than merely quiet.
    """

    sites = sites_for_bbox(5.0, 47.0, 16.0, 55.5)
    assert sites, "the sample domain should hold radars"
    assert all(site.has_elevation for site in sites)
    assert require_assimilable(sites) == tuple(sites)


def test_the_gate_still_refuses_a_site_whose_elevation_is_missing():
    """The gate is armed, not merely satisfied.

    A harvest that could not reach a site leaves it null, and such a site
    must still be refused by name. Built from a real row so the refusal is
    exercised against the shape the table actually holds.
    """

    from dataclasses import replace

    real = sites_for_bbox(5.0, 47.0, 16.0, 55.5)[0]
    with pytest.raises(SiteNotAssimilableError) as caught:
        require_assimilable((real, replace(real, id="unreachable",
                                           elevation_m=None)))
    message = str(caught.value)
    assert "no antenna elevation" in message
    assert "unreachable" in message
    assert "wrong altitude" in message
    assert "look usable" in message


def test_supplying_the_elevation_is_what_lifts_the_refusal():
    """The other direction, so the gate is not merely always-on.

    An instrument that refuses everything is indistinguishable from one that
    is broken, and it would go unnoticed for exactly as long.
    """

    usable = [_site("a", elevation=140.0), _site("b", elevation=1010.0)]
    assert require_assimilable(usable) == tuple(usable)


def test_a_zero_elevation_is_a_measurement_and_a_null_is_not():
    """Sea level is a real antenna height. Unknown is not zero.

    Collapsing the two is exactly how a placeholder gets in.
    """

    assert require_assimilable([_site("atsealevel", elevation=0.0)])
    with pytest.raises(SiteNotAssimilableError):
        require_assimilable([_site("unknown", elevation=None)])


def test_a_partial_set_is_refused_rather_than_quietly_shortened():
    """Eleven asked for, three usable, is not a successful run of three."""

    sites = [_site("a", elevation=100.0), _site("b", elevation=None),
             _site("c", elevation=200.0)]
    with pytest.raises(SiteNotAssimilableError) as caught:
        require_assimilable(sites)
    assert "1 of 3" in str(caught.value)


def test_an_empty_set_is_refused_rather_than_assimilating_nothing():
    with pytest.raises(RadarSiteError, match="assimilates nothing"):
        require_assimilable([])


def test_a_site_without_velocity_is_refused_only_when_velocity_was_asked_for():
    site = [_site("refl-only", elevation=100.0, velocity=False)]
    assert require_assimilable(site, need_velocity=False) == tuple(site)
    with pytest.raises(SiteNotAssimilableError, match="radial-velocity"):
        require_assimilable(site, need_velocity=True)


# ------------------------------------------------------- table integrity

def test_an_inventory_listing_a_radar_twice_refuses(tmp_path):
    path = tmp_path / "sites.json"
    row = {"id": "x", "name": "x", "wmo_block": "1", "latitude": 1.0,
           "longitude": 1.0, "elevation_m": None, "moments": [],
           "methods": [], "has_velocity": False, "has_volume_scan": False}
    path.write_text(json.dumps({"schema": TABLE_SCHEMA, "sites": [row, row]}))
    with pytest.raises(RadarSiteError, match="twice"):
        load_table(path)


def test_an_empty_inventory_refuses(tmp_path):
    path = tmp_path / "sites.json"
    path.write_text(json.dumps({"schema": TABLE_SCHEMA, "sites": []}))
    with pytest.raises(RadarSiteError):
        load_table(path)


def test_an_inventory_under_another_schema_refuses(tmp_path):
    path = tmp_path / "sites.json"
    path.write_text(json.dumps({"schema": "other.v1", "sites": [
        {"id": "x", "name": "x", "wmo_block": None, "latitude": 1.0,
         "longitude": 1.0, "elevation_m": None, "moments": [], "methods": [],
         "has_velocity": False, "has_volume_scan": False}]}))
    with pytest.raises(RadarSiteError, match=TABLE_SCHEMA):
        load_table(path)
