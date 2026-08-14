"""The frozen surface-network table, and the domain-to-networks resolution.

Nothing here reaches the archive. The table is a repository artifact and the
resolution is arithmetic over it; both are testable offline, which is the
point of freezing the table in the first place.
"""

from __future__ import annotations

import json

import pytest

from gpuwm.obs.surface_networks import (TABLE_PATH, TABLE_SCHEMA,
                                        SurfaceNetwork, SurfaceNetworkError,
                                        SurfaceNetworkTable, describe,
                                        load_table, networks_for_bbox)


def _table(rows):
    return SurfaceNetworkTable(
        schema=TABLE_SCHEMA, source_url="", source_sha256="", frozen_at="",
        networks=tuple(SurfaceNetwork(**row) for row in rows))


def test_the_shipped_table_declares_its_schema_and_its_provenance():
    table = load_table()
    assert table.schema == TABLE_SCHEMA
    # A table nobody can trace back to a listing is a table nobody can
    # refresh with confidence.
    assert table.source_url.startswith("https://")
    assert len(table.source_sha256) == 64
    assert table.frozen_at


def test_the_shipped_table_reaches_well_past_the_united_states():
    """The generalization this module exists for, asserted as a count.

    The surface route began as a US instrument. The claim being made now is
    that it is worldwide, and the honest form of that claim is a number: how
    many of the networks are not US ones.
    """

    table = load_table()
    # The archive spells a non-US network with a doubled separator
    # (``DE__ASOS``) and a US one with a single (``IA_ASOS``).
    international = [n for n in table.networks if "__" in n.id]
    assert len(table) > 200, len(table)
    assert len(international) > 150, len(international)


def test_every_row_is_a_box_that_could_hold_a_station():
    for network in load_table().networks:
        # Latitudes are on the globe. The archive pads every edge, and a
        # polar network's padded southern edge would otherwise land at
        # -90.1: a number no station can have and every range check would
        # then be asserting something false.
        assert -90.0 <= network.south <= network.north <= 90.0, network
        assert -180.0 <= network.west, network
        # East may run past +180 for a network that straddles the
        # antimeridian; it may never wrap more than once around.
        assert network.west <= network.east <= 540.0, network
        assert network.east - network.west <= 360.0, network


def test_a_dateline_network_is_stored_as_the_short_way_round():
    """The defect this representation exists for, on a real network.

    New Zealand reports stations near +178 and near -176. Under ``min``/
    ``max`` its extent is ``[-176, 178]`` -- 354 of the 360 degrees -- and it
    is then offered for domains in the South Atlantic and the Indian Ocean.
    Stored as the shortest containing interval it runs east past +180
    instead, and stays a Pacific network.
    """

    by_id = {network.id: network for network in load_table().networks}
    pacific = by_id["NF__ASOS"]
    assert pacific.crosses_antimeridian
    assert pacific.east > 180.0
    assert pacific.east - pacific.west < 180.0, pacific
    # Both sides of the dateline resolve to it.
    assert "NF__ASOS" in networks_for_bbox(172.0, -42.0, 176.0, -38.0)
    assert "NF__ASOS" in networks_for_bbox(-177.0, -30.0, -173.0, -26.0)
    # And a South Atlantic domain does not.
    assert not pacific.intersects(-25.0, -35.0, -20.0, -30.0)


def test_a_single_and_a_doubled_separator_are_different_networks():
    """``AL_ASOS`` is Alabama and ``AL__ASOS`` is Albania.

    They are 8000 km apart and one underscore apart. A resolver that
    normalized the separator would answer a Balkan domain with Gulf Coast
    stations and never say so.
    """

    table = load_table()
    by_id = {network.id: network for network in table.networks}
    alabama = by_id["AL_ASOS"]
    albania = by_id["AL__ASOS"]
    assert alabama.east < -80.0, alabama
    assert albania.west > 19.0, albania
    assert not alabama.intersects(albania.west, albania.south,
                                  albania.east, albania.north)


def test_a_central_european_domain_resolves_to_its_neighbours_not_to_alabama():
    resolved = networks_for_bbox(11.0, 47.0, 16.0, 51.0)
    assert "DE__ASOS" in resolved
    assert "CZ__ASOS" in resolved
    assert "AL_ASOS" not in resolved
    assert "IA_ASOS" not in resolved
    # Sorted, so a receipt written on two hosts compares equal.
    assert list(resolved) == sorted(resolved)


def test_a_midwest_domain_still_resolves_the_way_it_always_did():
    """The generalization must not have moved the case that already worked."""

    resolved = networks_for_bbox(-96.0, 40.5, -90.5, 43.4)
    assert "IA_ASOS" in resolved
    assert "DE__ASOS" not in resolved


def test_an_empty_ocean_domain_refuses_instead_of_returning_nothing():
    """The failure mode this refusal exists for.

    An empty tuple flows through a fetch, a decode and a score without
    anything raising, and the case ends having assimilated no surface
    observation while looking exactly like one that assimilated every
    available one.
    """

    with pytest.raises(SurfaceNetworkError) as caught:
        # Mid South Pacific, far from any land the archive covers.
        networks_for_bbox(-140.0, -40.0, -135.0, -35.0)
    assert "no surface observations" in str(caught.value)


def test_a_touching_edge_is_offered_rather_than_screened_out():
    table = _table([
        {"id": "X__ASOS", "name": "X", "west": 0.0, "south": 0.0,
         "east": 10.0, "north": 10.0},
    ])
    # The domain's western edge is exactly the network's eastern edge.
    assert networks_for_bbox(10.0, 0.0, 20.0, 10.0, table=table) == ("X__ASOS",)
    # One degree further out, and it is genuinely disjoint.
    with pytest.raises(SurfaceNetworkError):
        networks_for_bbox(11.0, 0.0, 20.0, 10.0, table=table)


@pytest.mark.parametrize("box", [
    (10.0, 50.0, 5.0, 55.0),      # west is not west of east
    (10.0, 55.0, 15.0, 50.0),     # south is not below north
    (10.0, -95.0, 15.0, 55.0),    # latitude off the globe
    (-190.0, 50.0, 15.0, 55.0),   # longitude off the globe
])
def test_a_box_that_is_not_a_box_refuses(box):
    with pytest.raises(SurfaceNetworkError):
        networks_for_bbox(*box)


def test_a_table_that_lists_a_network_twice_refuses(tmp_path):
    path = tmp_path / "surface_networks.json"
    path.write_text(json.dumps({
        "schema": TABLE_SCHEMA,
        "networks": [
            {"id": "X__ASOS", "name": "X", "west": 0.0, "south": 0.0,
             "east": 1.0, "north": 1.0},
            {"id": "X__ASOS", "name": "X again", "west": 50.0, "south": 50.0,
             "east": 51.0, "north": 51.0},
        ],
    }))
    with pytest.raises(SurfaceNetworkError) as caught:
        load_table(path)
    assert "twice" in str(caught.value)


def test_an_empty_table_refuses_rather_than_resolving_everything_to_nothing(
        tmp_path):
    path = tmp_path / "surface_networks.json"
    path.write_text(json.dumps({"schema": TABLE_SCHEMA, "networks": []}))
    with pytest.raises(SurfaceNetworkError):
        load_table(path)


def test_a_table_under_another_schema_refuses(tmp_path):
    path = tmp_path / "surface_networks.json"
    path.write_text(json.dumps({
        "schema": "something-else.v9",
        "networks": [{"id": "X__ASOS", "name": "X", "west": 0.0, "south": 0.0,
                      "east": 1.0, "north": 1.0}],
    }))
    with pytest.raises(SurfaceNetworkError) as caught:
        load_table(path)
    assert TABLE_SCHEMA in str(caught.value)


def test_describe_names_the_networks_for_a_receipt():
    described = describe(("DE__ASOS",))
    assert described[0].startswith("DE__ASOS (")
    assert "Germany" in described[0]


def test_the_table_ships_inside_the_package():
    """A wheel without the table can only fetch what a caller typed by hand."""

    assert TABLE_PATH.is_file()
    assert TABLE_PATH.parent.name == "data"
    assert TABLE_PATH.parent.parent.name == "obs"
