"""Live smoke for the packaged acquisition routes.

The rule this suite is written under: a fetch test must not download
gigabytes.  Route resolution is proved with a ranged read of the first
few bytes of each route's real first object -- which checks the host,
the key grammar, the cycle stamp, the lead stamp AND that the payload is
the product the row claims -- and exactly one small end-to-end runs the
whole front door on real bytes.

Gated twice: the ``network`` marker and ``GPUWM_NETWORK_TESTS=1``.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import os
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from gpuwm import fetch_routes


live = pytest.mark.skipif(
    os.environ.get("GPUWM_NETWORK_TESTS") != "1",
    reason="live network smoke; set GPUWM_NETWORK_TESTS=1")


def _recent_cycle(route: fetch_routes.Route, *, lag_hours: int) -> datetime:
    """The newest cycle this producer runs that is at least ``lag`` old.

    Publication lag differs by hours between these producers, so the lag
    is the caller's, measured: GDAS is about +7 h, ICON-EU about +2 h.
    """

    when = (datetime.now(timezone.utc).replace(tzinfo=None)
            - timedelta(hours=lag_hours))
    for back in range(0, 48):
        candidate = (when - timedelta(hours=back)).replace(
            minute=0, second=0, microsecond=0)
        if candidate.hour in route.cycle_hours:
            return candidate
    raise AssertionError(f"no cycle hour for {route.source_id}")


#: ``source -> publication lag (hours)`` to look back by, measured on
#: 2026-08-17.  Deliberately generous: this suite proves that a resolved
#: URL is real, not that a producer is punctual.
_LAG_HOURS = {
    "hrrr-prs": 3, "rap": 3, "rrfs": 5, "gefs": 8, "aigfs": 6,
    "aigefs": 6, "ecmwf-open-data": 12, "aifs": 8, "icon-eu": 5,
    "gem-gdps": 8,
}


def _head_bytes(url: str, count: int = 64) -> bytes:
    request = Request(url, headers={
        "User-Agent": "gpuwm-fetch-test", "Range": f"bytes=0-{count - 1}"})
    with urlopen(request, timeout=60) as response:
        assert response.status in (200, 206), (url, response.status)
        return response.read(count)


@live
@pytest.mark.network
@pytest.mark.parametrize("source_id", fetch_routes.route_ids())
def test_live_every_route_resolves_to_a_real_object_of_its_own_product(
        source_id):
    """One ranged read per route: 64 bytes, and they have to be the product.

    The breakage this prevents is the whole reason the lane exists: a
    key grammar that was right when it was written and has since moved
    (RRFS's prototype bucket froze mid-day and the live feed is a
    different bucket entirely), or a host whose second copy is a
    DIFFERENT product under identical key names (AIGFS on S3).
    """

    route = fetch_routes.route_for(source_id)
    cycle = _recent_cycle(route, lag_hours=_LAG_HOURS[source_id])
    plan = fetch_routes.resolve_request(source_id, cycle=cycle, hours=0)
    first = plan.objects[0]
    magic = fetch_routes._magic_for(plan, first.role)
    try:
        head = _head_bytes(first.url)
    except HTTPError as error:  # pragma: no cover - live service
        pytest.fail(
            f"{source_id}: {first.url} answered HTTP {error.code}; the "
            "route table's key grammar or its cycle/lead ladder has moved")
    assert head.startswith(magic.encode("ascii")), (source_id, first.url,
                                                    head[:8])


@live
@pytest.mark.network
def test_live_rap_fetch_is_a_front_door_end_to_end(tmp_path):
    """The one small real end-to-end: two 19 MB objects, whole route.

    RAP awip32 is the smallest complete valid time any of these routes
    publishes, so it is the arm that pays for the real proof: real
    bytes, the real pool, the real receipts, and a handoff whose bound
    half is what `gpuwm prep --source rap` consumes.
    """

    route = fetch_routes.route_for("rap")
    cycle = _recent_cycle(route, lag_hours=3)
    plan = fetch_routes.resolve_request("rap", cycle=cycle, hours=1)
    payload = fetch_routes.run_plan(plan, out=tmp_path)
    inputs, command = fetch_routes.write_handoff(plan, tmp_path)

    assert len(payload["files"]) == 2
    # Sizes recorded, not asserted to a constant: the product's own
    # volume drifts, an empty or truncated answer does not.
    assert all(entry["bytes"] > 5_000_000 for entry in payload["files"])
    for entry in payload["files"]:
        body = (tmp_path / entry["relpath"]).read_bytes()
        assert body[:4] == b"GRIB" and body[-4:] == b"7777"
    listed = inputs.read_text().splitlines()
    assert len(listed) == 2
    assert "rap_awip32_in_band_surface=" in command.read_text()
    assert (tmp_path / fetch_routes.SHA256SUMS_NAME).is_file()


@live
@pytest.mark.network
def test_live_gefs_pairs_two_real_disjoint_halves(tmp_path):
    """The compose stage, on real bytes, at the smallest window there is.

    One member, one lead: the ``pgrb2a``+``pgrb2b`` pair whose isobaric
    level sets are exactly disjoint, concatenated into the multi-record
    form the packaged profile decodes.  Roughly 110 MB.
    """

    route = fetch_routes.route_for("gefs")
    cycle = _recent_cycle(route, lag_hours=8)
    plan = fetch_routes.resolve_request("gefs", cycle=cycle, hours=0)
    payload = fetch_routes.run_plan(plan, out=tmp_path)

    assert len(payload["files"]) == 2
    assert len(payload["composed"]) == 1
    pair = tmp_path / payload["composed"][0]["name"]
    body = pair.read_bytes()
    assert body[:4] == b"GRIB" and body[-4:] == b"7777"
    assert pair.stat().st_size == sum(
        entry["bytes"] for entry in payload["files"])
    # Member identity survived as a path component.
    assert all(entry["relpath"].startswith("upstream/gefs.")
               for entry in payload["files"])
