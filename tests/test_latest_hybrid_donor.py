"""``--cycle latest`` for a hybrid route resolves a cycle it can fetch whole.

AIGFS and AIGEFS take their land surface from the same-cycle GDAS
analysis, which publishes after the AI atmosphere.  Latest used to ask
only about the atmosphere, picked the newest cycle, and the fetch then
refused it because its GDAS analysis was not out yet, telling the reader
to pass --cycle latest, which they had.  The cycle before had
everything.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from gpuwm import fetch, fetch_endpoints, fetch_routes
from test_audit_acquisition_contract import args


_NOW = datetime(2026, 9, 27, 20)


def _publication(source, *, donor_cycles):
    """A probe for: the primary published at its two newest cycles, the
    donor only at ``donor_cycles`` (indices into those two)."""

    grid = fetch.require_cycle_grid(source)
    cycles = tuple(grid.candidates(_NOW))[:2]
    route = fetch_routes.route_for(source)
    last = route.default_cadence
    published = set()
    for cycle in cycles:
        for endpoint in fetch_endpoints.serving_ladder(source, cycle=cycle, now=_NOW):
            published.update(fetch.cycle_probe_urls(
                source, cycle, last, transport=endpoint.name))
    for index in donor_cycles:
        for donor in route.donors:
            for endpoint in fetch_endpoints.serving_ladder(
                    donor.source, cycle=cycles[index], now=_NOW):
                published.update(fetch.cycle_probe_urls(
                    donor.source, cycles[index], max(donor.leads),
                    transport=endpoint.name))
    asked = []

    def probe(url):
        asked.append(url)
        return url in published

    return cycles, last, probe, asked


@pytest.mark.parametrize("source", ["aigfs", "aigefs"])
def test_latest_passes_over_a_cycle_whose_donor_is_not_published(source):
    (newest, older), last, probe, asked = _publication(source, donor_cycles=[1])
    assert fetch_routes.route_for(source).donors
    selected = fetch.resolve_latest_cycle(source, last, now=_NOW, probe=probe)
    assert selected == older, (
        f"latest chose {selected:%Y-%m-%dT%H}Z, whose GDAS analysis is not published")
    assert any("gdas" in url for url in asked)


@pytest.mark.parametrize("source", ["aigfs", "aigefs"])
def test_latest_keeps_the_newest_cycle_when_its_donor_is_out(source):
    (newest, _older), last, probe, _asked = _publication(source, donor_cycles=[0, 1])
    assert fetch.resolve_latest_cycle(source, last, now=_NOW, probe=probe) == newest


def test_latest_refuses_by_name_when_no_cycle_has_its_donor():
    _cycles, last, probe, _asked = _publication("aigfs", donor_cycles=[])
    with pytest.raises(RuntimeError, match=r"no complete AIGFS cycle covering "
                       r"f\d{3} \(with its same-cycle GDAS f000\)"):
        fetch.resolve_latest_cycle("aigfs", last, now=_NOW, probe=probe)


def test_a_route_without_a_donor_never_asks_about_one():
    source = "gefs"
    assert not fetch_routes.route_for(source).donors
    asked = []
    expected = fetch.require_cycle_grid(source).newest(_NOW)
    selected = fetch.resolve_latest_cycle(
        source, 6, now=_NOW, probe=lambda url: asked.append(url) or True)
    assert selected == expected
    assert asked and not any("gdas" in url for url in asked)


def test_the_fetch_command_downloads_the_cycle_latest_chose(tmp_path, monkeypatch):
    source = "aigfs"
    (_newest, older), last, probe, _asked = _publication(source, donor_cycles=[1])
    resolve, published = fetch.resolve_latest_cycle, fetch.require_published_cycle

    # The clock and the hosts of the fixture, for every caller: a
    # refusal's remedy resolves latest again with its own clock and probe.
    def latest(name, end, **options):
        options.setdefault("now", _NOW)
        options.setdefault("probe", probe)
        return resolve(name, end, **options)

    def check(name, cycle, end, **options):
        options.setdefault("now", _NOW)
        options.setdefault("probe", probe)
        return published(name, cycle, end, **options)

    downloaded = []
    monkeypatch.setattr(fetch, "resolve_latest_cycle", latest)
    monkeypatch.setattr(fetch, "require_published_cycle", check)
    monkeypatch.setattr(fetch_routes, "run_plan",
                        lambda plan, **kw: downloaded.append(plan.cycle))
    monkeypatch.setattr(fetch, "_fetch_route_donors", lambda *a: {})
    monkeypatch.setattr(fetch_routes, "write_handoff", lambda *a, **kw: None)
    monkeypatch.setattr(fetch_routes, "handoff_lines", lambda *a: ())
    # The whole-cycle rule (A136 L2: as-posted is the default, so this
    # names --whole-cycle and keeps every assertion).
    parsed = args("--whole-cycle", "--source", source, "--cycle", "latest", "--hours", str(last),
                  "--out", str(tmp_path / "fetch"))
    assert fetch.fetch_main(parsed) == 0
    assert downloaded == [older]
