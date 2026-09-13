"""The frozen surface-network table, and the domain-to-networks resolution.

Nothing here reaches the archive. The table is a repository artifact and the
resolution is arithmetic over it; both are testable offline, which is the
point of freezing the table in the first place.
"""

from __future__ import annotations

import json

import pytest

from gpuwm.fetch import Area, parse_area
from gpuwm.obs.surface_networks import (TABLE_PATH, TABLE_SCHEMA,
                                        SurfaceNetwork, SurfaceNetworkError,
                                        SurfaceNetworkTable, describe,
                                        load_table, networks_for_bbox)

# ``gpuwm.fetch`` is imported read-only: it is what decides what an ordered
# longitude pair means, for an ``--area`` and now for a domain box alike, so
# it is the expectation these tests are written against and never a thing
# they edit. Which corner is west is decided before that, and only
# ``--area`` decides it; that difference is asserted rather than assumed in
# ``test_the_ordering_rule_the_area_door_applies_is_not_applied_here``.


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
    that it is worldwide, and the checkable form of that claim is a number:
    how many of the networks are not US ones.
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


def test_an_edge_touching_at_the_antimeridian_touches_like_any_other_edge():
    """The branch the extents are cut on is not a gap on the globe.

    A box whose east edge is +180 and an extent whose west edge is -180 meet
    at one meridian, and a station can sit on it. Compared as four numbers on
    the branch they look the width of the world apart, and the network is
    dropped from the screen with nothing said, which is the one direction
    this screen is built never to err in.
    """

    beginning_there = _table([
        {"id": "X__ASOS", "name": "X", "west": -180.0, "south": 0.0,
         "east": -170.0, "north": 10.0},
    ])
    assert networks_for_bbox(0.0, 0.0, 180.0, 10.0,
                             table=beginning_there) == ("X__ASOS",)
    # The same contact from the other side.
    ending_there = _table([
        {"id": "Y__ASOS", "name": "Y", "west": 170.0, "south": 0.0,
         "east": 180.0, "north": 10.0},
    ])
    assert networks_for_bbox(-180.0, 0.0, -170.0, 10.0,
                             table=ending_there) == ("Y__ASOS",)
    # A gap is still a gap: half a degree short of the meridian is disjoint,
    # so this is a shared boundary and not a wrap that swallows one.
    half_a_degree_short = _table([
        {"id": "Z__ASOS", "name": "Z", "west": 170.0, "south": 0.0,
         "east": 179.5, "north": 10.0},
    ])
    with pytest.raises(SurfaceNetworkError):
        networks_for_bbox(-180.0, 0.0, -170.0, 10.0, table=half_a_degree_short)


@pytest.mark.parametrize("box", [
    (10.0, 55.0, 15.0, 50.0),     # south is not below north
    (10.0, -95.0, 15.0, 55.0),    # latitude off the globe
    (-190.0, 50.0, 15.0, 55.0),   # longitude in neither convention
    (370.0, 50.0, 10.0, 55.0),    # a west no convention spells
])
def test_a_box_that_is_not_a_box_refuses(box):
    with pytest.raises(SurfaceNetworkError):
        networks_for_bbox(*box)


@pytest.mark.parametrize("corner", [float("nan"), float("inf"),
                                    float("-inf")])
def test_a_corner_that_is_not_a_finite_number_refuses(corner):
    with pytest.raises(SurfaceNetworkError) as caught:
        networks_for_bbox(10.0, 50.0, corner, 55.0)
    assert "four finite numbers" in str(caught.value)


def test_a_latitude_off_the_globe_is_named_in_its_refusal():
    """A refusal a caller can act on names the number it objected to.

    The latitude clause survives untouched as the box's own integrity
    check; what it says is the part that changed, because "latitudes must
    lie inside [-90, 90]" leaves a caller with four numbers and no way to
    tell which of the two was the wrong one.
    """

    with pytest.raises(SurfaceNetworkError) as caught:
        networks_for_bbox(10.0, -95.0, 15.0, 55.0)
    message = str(caught.value)
    assert "-95" in message
    assert "[-90, 90]" in message
    with pytest.raises(SurfaceNetworkError) as caught:
        networks_for_bbox(10.0, 50.0, 15.0, 95.0)
    assert "95" in str(caught.value)


def test_a_longitude_in_neither_convention_names_both_of_them():
    """The scope guard on normalizing the ``[0, 360]`` convention.

    -190 is not the ``[0, 360]`` spelling of anything: a general wrap would
    turn it into 170 and hand back a legal 205-degree crossing box, which
    would accept a longitude off the globe as a convention difference. The
    normalization recognizes ``[0, 360]`` and nothing else, so this stays a
    refusal, and the refusal names both conventions and the crossing form.
    """

    with pytest.raises(SurfaceNetworkError) as caught:
        networks_for_bbox(-190.0, 50.0, 15.0, 55.0)
    message = str(caught.value)
    assert "-190" in message
    assert "[-180, 180]" in message
    assert "[0, 360]" in message
    assert "west greater than east" in message


def test_a_domain_that_crosses_the_dateline_resolves_rather_than_refusing():
    """A Pacific domain gets its surface observations.

    A wrapped domain footprint produces exactly this box: west 172, east
    -176. The module used to refuse it, on the grounds that wrapping it
    would resolve the wrong hemisphere. It does not have to guess: the
    product already documents west greater than east as the box that
    crosses the antimeridian (``docs/public/CASE_CATALOG.md``), so the box
    is decomposed the same way a straddling network extent already is.
    """

    crossing = networks_for_bbox(172.0, -42.0, -176.0, -38.0)
    assert "NF__ASOS" in crossing
    # Sorted, so a receipt written on two hosts compares equal.
    assert list(crossing) == sorted(crossing)
    # It is exactly the union of its own two halves, which is what
    # decomposing the box rather than refusing it has to mean.
    halves = set(networks_for_bbox(172.0, -42.0, 180.0, -38.0)) | set(
        networks_for_bbox(-180.0, -42.0, -176.0, -38.0))
    assert set(crossing) == halves
    # And not the wide complement. Those 348 degrees are the failure the old
    # refusal was guarding against, and they are still not this box.
    elsewhere = ("AR__ASOS", "CL__ASOS", "ZA__ASOS")
    assert set(elsewhere) <= set(networks_for_bbox(-176.0, -42.0, 172.0,
                                                   -38.0))
    assert not set(elsewhere) & set(crossing)


def test_a_dateline_box_offers_the_networks_on_both_sides_of_the_seam():
    """``FJ__ASOS`` answers at the latitudes it actually spans.

    Fiji's extent stops at 19.2S, so it is not in the 42S to 38S box above;
    at its own latitudes the crossing box offers it, its neighbour across
    the seam, and New Zealand together, which no half-box does alone.
    """

    crossing = networks_for_bbox(172.0, -20.0, -176.0, -15.0)
    assert "FJ__ASOS" in crossing
    assert "NF__ASOS" in crossing
    assert "WS__ASOS" in crossing
    assert "WS__ASOS" not in networks_for_bbox(172.0, -20.0, 180.0, -15.0)


def test_a_dateline_box_written_the_way_the_extents_are_resolves_the_same():
    """East past +180 is the table's own spelling, so the box may use it."""

    assert networks_for_bbox(172.0, -42.0, 184.0, -38.0) == (
        networks_for_bbox(172.0, -42.0, -176.0, -38.0))


def test_the_box_that_used_to_be_backwards_is_a_crossing_box_now():
    """Retires the ``(10, 50, 5, 55)`` refusal parameter.

    It was listed as "west is not west of east". Under the convention the
    product documents it is a legal 355-degree box that crosses the
    antimeridian, so it resolves instead of refusing, and it reaches both
    sides of the seam that a 5-degree box between the same edges cannot.
    """

    crossing = networks_for_bbox(10.0, 50.0, 5.0, 55.0)
    assert Area(lat_south=50.0, lon_west=10.0, lat_north=55.0,
                lon_east=5.0).crosses_antimeridian
    assert "AK_ASOS" in crossing
    assert "CN__ASOS" in crossing
    narrow = networks_for_bbox(5.0, 50.0, 10.0, 55.0)
    assert set(narrow) < set(crossing)


def test_a_0_360_longitude_box_is_the_same_box_as_its_signed_twin():
    """Only the convention differs, so normalize rather than refuse."""

    assert networks_for_bbox(250.0, 30.0, 260.0, 40.0) == (
        networks_for_bbox(-110.0, 30.0, -100.0, 40.0))


def test_a_0_360_box_that_also_crosses_the_branch_normalizes_and_resolves():
    """350 to 370 is -10 to 10, and resolves to that box exactly."""

    assert networks_for_bbox(350.0, 45.0, 370.0, 55.0) == (
        networks_for_bbox(-10.0, 45.0, 10.0, 55.0))


def test_a_0_360_box_written_west_greater_than_east_crosses_zero_not_180():
    """``(350, 10)`` is the 20-degree box on Greenwich.

    350 is a longitude only the ``[0, 360]`` convention spells, so the
    eastward walk from it to 10 passes 0E and not 180E. Read as a crossing
    box it would have screened the 340-degree complement instead, which is
    a different list and reaches the far side of the globe.
    """

    resolved = networks_for_bbox(350.0, 45.0, 10.0, 55.0)
    assert resolved == networks_for_bbox(-10.0, 45.0, 10.0, 55.0)
    assert not Area(lat_south=45.0, lon_west=350.0, lat_north=55.0,
                    lon_east=10.0).crosses_antimeridian
    complement = networks_for_bbox(10.0, 45.0, 350.0, 55.0)
    assert set(complement) != set(resolved)
    assert "AK_ASOS" in complement
    assert "AK_ASOS" not in resolved


@pytest.mark.parametrize("west,east,signed_west,signed_east", [
    (350.0, 10.0, -10.0, 10.0),
    (190.0, 10.0, -170.0, 10.0),
    (250.0, 100.0, -110.0, 100.0),
    (180.5, 10.0, -179.5, 10.0),
    (180.0, 179.0, -180.0, 179.0),
    (360.0, 10.0, 0.0, 10.0),
])
def test_a_pair_only_the_0_360_convention_spells_is_its_signed_twin(
        west, east, signed_west, signed_east):
    """The families a list of hand-picked agreeing pairs cannot see.

    Every one of these has ``west`` greater than ``east`` and none of them
    crosses the antimeridian, because ``west`` is not on the signed branch
    to begin with. A rule that read ``west`` greater than ``east`` as a
    crossing before putting the edges on one branch screened the complement
    of the box that was asked for.
    """

    assert networks_for_bbox(west, 45.0, east, 55.0) == networks_for_bbox(
        signed_west, 45.0, signed_east, 55.0)


#: Probe networks ringing the globe at 10-degree spacing, 9.5 degrees wide,
#: so a box read on the wrong branch or with the wrong crossing names a
#: visibly different list rather than the same one.
_PROBE_TABLE = _table([
    {"id": f"P{index:02d}__ASOS", "name": f"probe {index}",
     "west": -180.0 + 10.0 * index, "south": -10.0,
     "east": -180.0 + 10.0 * index + 9.5, "north": 10.0}
    for index in range(36)
])

#: Every longitude either convention spells, coarsely: the branch ends, both
#: sides of the seam, the ``[0, 360]`` half nothing on the signed branch can
#: write, and the values that sit exactly on a boundary. Swept against
#: itself, so the pairs written west greater than east are included in both
#: conventions rather than hand-picked.
_LONGITUDE_SWEEP = [-180.0, -176.0, -100.0, -10.0, 0.0, 10.0, 100.0, 172.0,
                    180.0, 180.5, 190.0, 250.0, 350.0, 360.0]

#: The pair the refusal names as the whole band, and the only pair of two
#: spellings of one meridian that the box resolves.
_WHOLE_BAND = (-180.0, 180.0)


def _probes_under(signed_west, span_degrees):
    """The probes a box covers, walking ``span_degrees`` east of its west edge.

    Stated from the two things :class:`gpuwm.fetch.Area` says about a pair:
    where the west edge sits on the signed branch, and how wide the eastward
    walk is. Measured around the circle with one modulus, because the module
    under test answers the same question by cutting both intervals at +180
    and comparing the pieces, and an expectation written that second way is
    a copy of the implementation rather than a second opinion on it.
    """

    covered = set()
    for probe in _PROBE_TABLE.networks:
        start = (probe.west - signed_west) % 360.0
        # Either the probe begins inside the walk, or it began before the
        # walk did and runs past the end of the circle into it. Touching
        # counts on both sides, the way it does for a real extent.
        if (start <= span_degrees
                or start + (probe.east - probe.west) >= 360.0):
            covered.add(probe.id)
    return covered


@pytest.mark.parametrize("west", _LONGITUDE_SWEEP)
def test_every_pair_this_box_takes_covers_what_an_area_puts_under_it(west):
    """Two doors, one reading of an ordered pair, swept not hand-picked.

    ``gpuwm.fetch.Area`` decides what an ordered longitude pair means for an
    ``--area`` and this module calls that same object for a domain box, so
    every pair the box accepts has to name exactly the probes lying under
    the box ``Area`` names. Sixteen pairs chosen by hand can all agree while
    whole families disagree; pairing every listed edge with every other one
    is what catches a family.

    The sweep holds three answers apart rather than two: the probes under
    the box, a refusal because nothing lies under it, and a refusal because
    the pair is not a box at all. Folding a refusal into the empty set is
    what let a pair naming one meridian at +/-180 resolve the whole globe
    with the sweep green.
    """

    for east in _LONGITUDE_SWEEP:
        area = Area(lat_south=-5.0, lon_west=west, lat_north=5.0,
                    lon_east=east)
        span = area.longitude_span_degrees
        if span in (0.0, 360.0) and (west, east) != _WHOLE_BAND:
            # Two spellings of one meridian: zero degrees wide or the whole
            # band, and the numbers do not say which. A list of any length
            # is the screen picking one of the two on the caller's behalf.
            with pytest.raises(SurfaceNetworkError) as caught:
                networks_for_bbox(west, -5.0, east, 5.0, table=_PROBE_TABLE)
            assert "same meridian" in str(caught.value), (west, east)
            continue
        expected = _probes_under(area.as_cds()[1], span)
        try:
            resolved = set(networks_for_bbox(west, -5.0, east, 5.0,
                                             table=_PROBE_TABLE))
        except SurfaceNetworkError as refusal:
            # A refusal is a right answer only where the box is empty. It is
            # not the empty set, so it is asserted against the expectation
            # rather than silently becoming it.
            assert not expected, (west, east, str(refusal))
            resolved = set()
        assert resolved == expected, (west, east)


def test_a_box_whose_two_edges_name_one_meridian_refuses():
    """``(0, 360)`` is zero degrees wide or the whole band.

    The numbers do not say which, and silently picking one hands a caller
    who wrote the whole band a point at Greenwich. The whole band has a
    spelling that says so, and the refusal names it.
    """

    with pytest.raises(SurfaceNetworkError) as caught:
        networks_for_bbox(0.0, 45.0, 360.0, 55.0)
    message = str(caught.value)
    assert "same meridian" in message
    assert "(-180, 180)" in message
    assert networks_for_bbox(-180.0, 45.0, 180.0, 55.0)


@pytest.mark.parametrize("west,east", [
    (180.0, 180.0),      # one meridian, spelled +180 twice
    (-180.0, -180.0),    # and spelled -180 twice
    (180.0, -180.0),     # and once under each of its two spellings
    (180.0, 540.0),      # and as a walk that starts there and comes back
])
def test_one_meridian_is_still_one_meridian_at_the_end_of_the_branch(
        west, east):
    """The zero-width check has to hold where the two branches meet.

    ``west`` is put on ``[-180, 180)`` and ``east`` on ``(-180, 180]``, so
    the antimeridian is the one meridian with two branch values and a pair
    naming it twice comes back as the pair ``(-180, 180)``: two different
    numbers, which is how a degenerate box resolved to all 54 networks while
    ``(10, 45, 10, 55)`` was refused. The width the pair encloses is what
    the check reads now, so the same box is refused at 180 as anywhere else.
    """

    with pytest.raises(SurfaceNetworkError) as caught:
        networks_for_bbox(west, 45.0, east, 55.0)
    message = str(caught.value)
    assert "same meridian" in message
    assert "(-180, 180)" in message
    # And the spelling the refusal names is not refused with it: the whole
    # band is a box a caller can still ask for, by saying so.
    assert len(networks_for_bbox(-180.0, 45.0, 180.0, 55.0)) > 1


def test_the_ordering_rule_the_area_door_applies_is_not_applied_here():
    """What the two doors share is the reading of an ordered pair.

    ``parse_area`` is handed two corners with no west among them: it sorts
    them and then reads a span wider than 180 degrees as the complementary
    crossing box, so both orders of one pair of corners are the narrow
    Pacific box. A domain box arrives ordered and is read as written, so
    the two orders are two different boxes. Both statements are true and
    the module docstring may claim only the first one.
    """

    for corners in ("45,-170,55,170", "45,170,55,-170"):
        area = parse_area(corners)
        assert (area.lon_west, area.lon_east) == (170.0, -170.0), corners
    # The box door, handed the pair that door produced, agrees with it.
    pacific = networks_for_bbox(170.0, 45.0, -170.0, 55.0)
    assert set(pacific) == set(
        networks_for_bbox(area.lon_west, 45.0, area.lon_east, 55.0))
    # And the other order is the other box: the 340 degrees that exclude the
    # Pacific, which is what an unordered pair cannot tell you.
    complement = networks_for_bbox(-170.0, 45.0, 170.0, 55.0)
    assert set(complement) != set(pacific)
    assert len(complement) > len(pacific)


def test_the_box_a_refusal_names_is_the_box_the_screen_compared():
    """A refusal has to be reproducible from its own numbers.

    ``(350, -60, 10, -50)`` is screened as ``(-10, -60, 10, -50)``. Naming
    the pair before the edges reached one branch sent the reader hunting
    for a hemisphere the screen never used.
    """

    with pytest.raises(SurfaceNetworkError) as caught:
        networks_for_bbox(350.0, -60.0, 10.0, -50.0)
    message = str(caught.value)
    assert "(-10, -60, 10, -50)" in message
    assert "370" not in message
    # The numbers as passed are still there to recognize the call by.
    assert "(350, 10)" in message
    # And the box the message names is a box that really screens to nothing,
    # which is what makes the message reproducible.
    with pytest.raises(SurfaceNetworkError):
        networks_for_bbox(-10.0, -60.0, 10.0, -50.0)


def test_a_crossing_refusal_names_the_carried_pair_the_screen_used():
    """East past +180 is the canonical spelling, so that is what is named."""

    table = _table([
        {"id": "X__ASOS", "name": "X", "west": 0.0, "south": 0.0,
         "east": 10.0, "north": 10.0},
    ])
    with pytest.raises(SurfaceNetworkError) as caught:
        networks_for_bbox(172.0, 20.0, -176.0, 25.0, table=table)
    message = str(caught.value)
    assert "(172, 20, 184, 25)" in message
    assert "(172, -176)" in message


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
