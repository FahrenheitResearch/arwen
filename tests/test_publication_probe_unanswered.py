"""A publication probe that was not answered is not a "not published".

The breakage these guard (GS-05): a GEM-GDPS start about a day old was
refused after 284 s as "cycle ... is not published through f006 yet",
because three of its 174 HEADs hit the Windows connect timeout and the
fetch's publication check read every failed HEAD as "not there".  The
same URLs answered 200 in a quarter of a second a little later.  Only
the network is replaced here: the probe, the governor's urlopen wrapper
and the endpoint ladder are the production ones.
"""

from __future__ import annotations

from datetime import datetime, timezone
import functools
from urllib.error import HTTPError, URLError

import pytest

import gpuwm.cli as cli
import gpuwm.fetch as fetch
from gpuwm import fetch_endpoints, nomads_governor, rustwx_fetch


CYCLE = datetime(2026, 9, 25, 12)
NOW = datetime(2026, 9, 26, 13)


class _Answer:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Network:
    """A stand-in for the wire: per URL, a list of what each ask gets."""

    def __init__(self, plan=None, default=200):
        self.plan = {url: list(steps) for url, steps in (plan or {}).items()}
        self.default = default
        self.asked: list[str] = []

    def __call__(self, request, timeout=None):
        url = request.full_url
        self.asked.append(url)
        steps = self.plan.get(url)
        step = steps.pop(0) if steps else self.default
        if step == "timeout":
            # What urllib raises for a connect that the OS gave up on.
            raise URLError(TimeoutError(10060, "timed out"))
        if step != 200:
            raise HTTPError(url, step, "status", {}, None)
        return _Answer()


@pytest.fixture
def wire(monkeypatch):
    """Route every probe through the real governed urlopen onto a fake network, with no real pauses."""

    monkeypatch.setattr(fetch_endpoints, "SETTLE_BACKOFF_S", (0.0, 0.0), raising=False)
    real = nomads_governor.paced_urlopen

    def install(network):
        monkeypatch.setattr(nomads_governor, "paced_urlopen",
                            functools.partial(real, opener=network))
        return network

    return install


def _gdps_urls(cycle=CYCLE):
    return fetch.cycle_probe_urls("gem-gdps", cycle, 6, transport="msc")


def test_three_timeouts_among_174_heads_do_not_refuse_a_published_start(wire):
    urls = _gdps_urls()
    assert len(urls) == 174
    flaky = (urls[5], urls[80], urls[170])
    network = wire(_Network({url: ["timeout", 200] for url in flaky}))

    refusal = fetch.cycle_publication_refusal("gem-gdps", CYCLE, 6, now=NOW)

    assert refusal is None
    # Each flaky object was asked again, and nothing else was.
    assert all(network.asked.count(url) == 2 for url in flaky)
    assert len(network.asked) == 174 + 3

    network = wire(_Network({url: ["timeout", 200] for url in flaky}))
    assert fetch.cycle_publication_check("gem-gdps", CYCLE, 6, now=NOW).state == "published"


def test_a_host_that_never_answers_is_said_unreached_not_unpublished(wire):
    silent = _gdps_urls()[40]

    network = wire(_Network({silent: ["timeout"] * 3}))
    refusal = fetch.cycle_publication_refusal("gem-gdps", CYCLE, 6, now=NOW)
    assert refusal is None
    # Three asks of the silent object (one, then one after each pause).
    assert network.asked.count(silent) == 3

    said: list[str] = []
    wire(_Network({silent: ["timeout"] * 3}))
    fetch.require_published_cycle("gem-gdps", CYCLE, 6, now=NOW, progress=said.append)
    assert len(said) == 1
    assert said[0].startswith("fetch gem-gdps: could not reach dd.weather.gc.ca to check whether "
                              "GEM-GDPS cycle 2026-09-25T12Z is published through f006")
    assert "not published" not in said[0]

    wire(_Network({silent: ["timeout"] * 3}))
    check = fetch.cycle_publication_check("gem-gdps", CYCLE, 6, now=NOW)
    assert check.state == "unchecked"
    assert said == [f"fetch gem-gdps: {check.why}"]


def test_an_object_the_host_says_is_missing_is_still_not_published(wire):
    missing = _gdps_urls()[40]
    wire(_Network({missing: [404] * 10}))

    refusal = fetch.cycle_publication_refusal("gem-gdps", CYCLE, 6, now=NOW)

    assert refusal is not None
    # Older than the newest complete cycle, so it is not still publishing:
    # the refusal says waiting will not bring it rather than "not published yet".
    assert ("GEM-GDPS cycle 2026-09-25T12Z is not on dd.weather.gc.ca through f006, "
            "and waiting will not bring it") in refusal
    assert "the newest complete GEM-GDPS cycle covering f006 is 2026-09-26T00Z" in refusal
    with pytest.raises(RuntimeError, match="is not on dd.weather.gc.ca through f006"):
        fetch.require_published_cycle("gem-gdps", CYCLE, 6, now=NOW, progress=lambda line: None)


def test_a_rung_not_heard_keeps_the_answer_open_when_another_rung_says_no():
    """The operational server timed out and the archive has not caught up: either may be the truth."""

    def probe(url):
        return None if "nomads.ncep.noaa.gov" in url else False

    check = fetch.cycle_publication_check("gfs", CYCLE, 6, now=NOW, probe=probe)

    assert check.state == "unchecked"
    assert "could not reach nomads.ncep.noaa.gov to check" in check.why
    assert fetch.cycle_publication_refusal("gfs", CYCLE, 6, now=NOW, probe=probe) is None


def test_a_probe_that_answers_only_yes_or_no_is_read_as_before():
    check = fetch.cycle_publication_check("gfs", CYCLE, 6, now=NOW, probe=lambda url: False)

    assert check.state == "not-published"
    assert "GFS cycle 2026-09-25T12Z is not published through f006 yet" in check.why


def test_settled_answer_asks_again_only_while_unanswered(wire):
    url = _gdps_urls()[0]
    pauses: list[float] = []

    network = wire(_Network({url: [404]}))
    assert fetch_endpoints.settled_object_answer(url, backoff_s=(1.0, 2.0), sleep=pauses.append) is False
    assert network.asked == [url] and pauses == []

    network = wire(_Network({url: ["timeout", 200]}))
    assert fetch_endpoints.settled_object_answer(url, backoff_s=(1.0, 2.0), sleep=pauses.append) is True
    assert network.asked == [url, url] and pauses == [1.0]

    pauses.clear()
    network = wire(_Network({url: [503, "timeout", "timeout"]}))
    assert fetch_endpoints.settled_object_answer(url, backoff_s=(1.0, 2.0), sleep=pauses.append) is None
    assert network.asked == [url] * 3 and pauses == [1.0, 2.0]


def test_a_named_fetch_goes_ahead_when_the_host_cannot_be_heard(
        tmp_path, monkeypatch, capsys, wire):
    """At the command line: the transfer starts, and the line says why nothing was confirmed."""

    class Started(Exception):
        pass

    def transfer(**kwargs):
        raise Started

    monkeypatch.setattr(rustwx_fetch, "find_fetch_bin", lambda: None)
    monkeypatch.setattr(fetch, "fetch_hrrr", transfer)
    wire(_Network(default="timeout"))

    with pytest.raises(Started):
        cli.main(["fetch", "--source", "hrrr", "--cycle", "2026-01-31T06",
                  "--hours", "1", "--transport", "s3", "--out", str(tmp_path / "hrrr")])

    out = capsys.readouterr().out
    assert ("fetch hrrr: could not reach noaa-hrrr-bdp-pds.s3.amazonaws.com to check whether "
            "HRRR cycle 2026-01-31T06Z is published through f001") in out
    assert "not published" not in out


def test_the_background_review_falls_back_to_declared_timing_for_a_silent_host():
    """The local-DA review: a start whose host timed out is not replaced by an older one as if missing."""

    from gpuwm.background_contract import plan

    when = datetime(2026, 9, 13, tzinfo=timezone.utc)
    selected = plan("gfs", init=when, cycle=when, now=when, run_seconds=900., probe=lambda url: None)

    assert selected.cycle == when.isoformat()
    assert "could not reach its host" in selected.publication_basis
    window = fetch.probe_cycle_window("gfs", when.replace(tzinfo=None), (0, 1),
                                      now=when.replace(tzinfo=None), probe=lambda url: None)
    assert window["available"] is None
