"""The bounded-concurrency transfer pool behind every multi-file fetch.

Pure CPU: jobs are callables, no network.  What is pinned here is the
contract the fetch routes rely on:

* admission is in submission order, on the caller's thread, so manifest
  publication keeps its exact serial semantics;
* concurrency is bounded by the worker count AND by the per-host
  politeness caps (NOMADS is capped harder than everyone else);
* one failed file fails the whole request closed, re-raising the
  original refusal so its message still names the file;
* the receipt records files, bytes, workers, wall and the effective
  speedup against the serial model (the sum of per-file seconds).
"""
from __future__ import annotations

import threading
import time

import pytest

from gpuwm import fetch_pool


def _job(name, action, url=None):
    return fetch_pool.TransferJob(name=name, url=url, action=action)


def _entry(name, size=10):
    return {"name": name, "bytes": size}


# ---------------------------------------------------------------------------
# Worker-count resolution
# ---------------------------------------------------------------------------

def test_default_worker_count_is_bounded_and_documented():
    assert fetch_pool.DEFAULT_FILE_WORKERS == 6
    assert fetch_pool.resolve_file_workers(None) == 6
    assert fetch_pool.resolve_file_workers(1) == 1
    assert fetch_pool.resolve_file_workers(8) == 8


@pytest.mark.parametrize("bad", (0, -1, -6))
def test_worker_count_must_be_positive(bad):
    with pytest.raises(ValueError, match="fetch-workers"):
        fetch_pool.resolve_file_workers(bad)


def test_host_key_is_the_netloc_and_tolerates_absence():
    assert fetch_pool.host_key(
        "https://nomads.ncep.noaa.gov/cgi-bin/x?f=1") == "nomads.ncep.noaa.gov"
    assert fetch_pool.host_key("https://Example.COM:443/a") == "example.com:443"
    assert fetch_pool.host_key(None) == ""


def test_nomads_host_cap_is_two_and_other_hosts_are_uncapped():
    assert fetch_pool.host_worker_cap("nomads.ncep.noaa.gov", 6) == 2
    assert fetch_pool.host_worker_cap("nomads.ncep.noaa.gov", 1) == 1
    assert fetch_pool.host_worker_cap("noaa-gfs-bdp-pds.s3.amazonaws.com",
                                      6) == 6
    assert fetch_pool.host_worker_cap("", 6) == 6


# ---------------------------------------------------------------------------
# Ordering and admission
# ---------------------------------------------------------------------------

def test_serial_runs_in_order_on_the_callers_thread():
    order = []
    caller = threading.get_ident()
    threads = []

    def make(name):
        def action():
            order.append(name)
            threads.append(threading.get_ident())
            return _entry(name)
        return action

    admitted = []
    entries, receipt = fetch_pool.run_transfers(
        [_job(name, make(name)) for name in ("a", "b", "c")],
        workers=1, on_admitted=lambda index, entry: admitted.append(
            (index, entry["name"])))
    assert order == ["a", "b", "c"]
    assert set(threads) == {caller}
    assert admitted == [(0, "a"), (1, "b"), (2, "c")]
    assert [entry["name"] for entry in entries] == ["a", "b", "c"]
    assert receipt["workers_effective"] == 1


def test_admission_is_in_submission_order_even_when_completion_is_not():
    release_first = threading.Event()
    admitted = []

    def slow_first():
        release_first.wait(timeout=10.0)
        return _entry("first")

    def fast_second():
        release_first.set()
        return _entry("second")

    entries, _receipt = fetch_pool.run_transfers(
        [_job("first", slow_first), _job("second", fast_second)],
        workers=2,
        on_admitted=lambda index, entry: admitted.append(entry["name"]))
    assert admitted == ["first", "second"]
    assert [entry["name"] for entry in entries] == ["first", "second"]


def test_admission_runs_on_the_callers_thread():
    caller = threading.get_ident()
    admitting_threads = []
    entries, _receipt = fetch_pool.run_transfers(
        [_job(name, lambda name=name: _entry(name)) for name in "abcd"],
        workers=3,
        on_admitted=lambda index, entry: admitting_threads.append(
            threading.get_ident()))
    assert set(admitting_threads) == {caller}
    assert len(entries) == 4


# ---------------------------------------------------------------------------
# Bounded concurrency and per-host politeness
# ---------------------------------------------------------------------------

class _Gauge:
    """Records the maximum simultaneous occupancy of the actions."""

    def __init__(self):
        self.lock = threading.Lock()
        self.current = 0
        self.peak = 0

    def action(self, name, hold_s=0.05):
        def run():
            with self.lock:
                self.current += 1
                self.peak = max(self.peak, self.current)
            time.sleep(hold_s)
            with self.lock:
                self.current -= 1
            return _entry(name)
        return run


def test_concurrency_is_bounded_by_the_worker_count():
    gauge = _Gauge()
    jobs = [_job(f"j{i}", gauge.action(f"j{i}")) for i in range(8)]
    _entries, receipt = fetch_pool.run_transfers(jobs, workers=3)
    assert gauge.peak <= 3
    assert receipt["workers_requested"] == 3
    assert receipt["workers_effective"] == 3


def test_nomads_jobs_never_exceed_the_politeness_cap():
    gauge = _Gauge()
    url = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?f=x"
    jobs = [_job(f"n{i}", gauge.action(f"n{i}"), url=url) for i in range(6)]
    _entries, receipt = fetch_pool.run_transfers(jobs, workers=6)
    assert gauge.peak <= 2
    assert receipt["host_caps"] == {"nomads.ncep.noaa.gov": 2}


def test_other_hosts_keep_the_requested_worker_count():
    gauge = _Gauge()
    url = "https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.t00z.f000"
    jobs = [_job(f"s{i}", gauge.action(f"s{i}"), url=url) for i in range(6)]
    barrier_gauge = gauge  # same gauge; peak proves >2 ran together
    _entries, receipt = fetch_pool.run_transfers(jobs, workers=4)
    assert barrier_gauge.peak >= 3   # genuinely parallel beyond the NOMADS cap
    assert receipt["host_caps"] == {}


# ---------------------------------------------------------------------------
# Fail-closed
# ---------------------------------------------------------------------------

def test_a_failed_file_fails_the_request_and_keeps_the_refusal_text():
    admitted = []

    def good(name):
        return lambda: _entry(name)

    def bad():
        raise ValueError("downloaded f006 carries 3 GRIB2 messages, "
                         "expected 124")

    with pytest.raises(ValueError, match="f006 carries 3"):
        fetch_pool.run_transfers(
            [_job("f000", good("f000")), _job("f003", good("f003")),
             _job("f006", bad), _job("f009", good("f009"))],
            workers=2,
            on_admitted=lambda index, entry: admitted.append(entry["name"]))
    # The verified prefix was admitted before the refusal propagated.
    assert admitted == ["f000", "f003"]


def test_jobs_after_a_serial_failure_never_run():
    ran = []

    def track(name):
        def action():
            ran.append(name)
            return _entry(name)
        return action

    def bad():
        ran.append("bad")
        raise RuntimeError("NOMADS did not serve f003")

    with pytest.raises(RuntimeError, match="f003"):
        fetch_pool.run_transfers(
            [_job("f000", track("f000")), _job("f003", bad),
             _job("f006", track("f006"))],
            workers=1)
    assert ran == ["f000", "bad"]


def test_keyboard_interrupt_propagates_for_the_callers_own_handling():
    def interrupt():
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        fetch_pool.run_transfers(
            [_job("f000", lambda: _entry("f000")), _job("f003", interrupt)],
            workers=1)


# ---------------------------------------------------------------------------
# The receipt
# ---------------------------------------------------------------------------

def test_receipt_records_files_bytes_workers_wall_and_speedup():
    jobs = [_job(f"j{i}", (lambda i=i: (time.sleep(0.05),
                                        _entry(f"j{i}", size=100 + i))[1]))
            for i in range(4)]
    _entries, receipt = fetch_pool.run_transfers(jobs, workers=4)
    assert receipt["schema"] == fetch_pool.POOL_RECEIPT_SCHEMA
    assert receipt["files"] == 4
    assert receipt["bytes"] == 100 + 101 + 102 + 103
    assert receipt["workers_requested"] == 4
    assert receipt["workers_effective"] == 4
    assert receipt["wall_seconds"] > 0.0
    # The serial model is the sum of per-file seconds; four 50 ms jobs on
    # four workers must beat it.
    assert receipt["modeled_serial_seconds"] >= 4 * 0.05
    assert receipt["effective_speedup"] > 1.0


def test_receipt_speedup_is_honest_about_serial_runs():
    jobs = [_job(f"j{i}", (lambda i=i: (time.sleep(0.02),
                                        _entry(f"j{i}"))[1]))
            for i in range(3)]
    _entries, receipt = fetch_pool.run_transfers(jobs, workers=1)
    assert receipt["workers_effective"] == 1
    assert 0.8 <= receipt["effective_speedup"] <= 1.1


def test_effective_workers_never_exceed_the_job_count():
    _entries, receipt = fetch_pool.run_transfers(
        [_job("only", lambda: _entry("only"))], workers=6)
    assert receipt["workers_requested"] == 6
    assert receipt["workers_effective"] == 1
