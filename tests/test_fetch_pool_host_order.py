"""Host admission keeps early files ahead of later files under the same cap."""

from __future__ import annotations

import threading
import time

import pytest

from gpuwm import fetch_pool


HOST = "transfer.example"


def _job(index, action, *, host=HOST):
    return fetch_pool.TransferJob(
        name=f"object-{index}", url=f"https://{host}/{index}", action=action)


def _entry(index):
    return {"name": f"object-{index}", "bytes": 10}


def test_a_reused_worker_cannot_take_a_slot_ahead_of_an_older_submission(
        monkeypatch):
    monkeypatch.setitem(fetch_pool.HOST_FILE_WORKER_CAPS, HOST, 1)
    allow_older = threading.Event()
    later_waiting = threading.Event()
    first_finished = threading.Event()
    later_started = threading.Event()
    started = []
    original_executor = fetch_pool.ThreadPoolExecutor

    class ScheduledPool(original_executor):
        def submit(self, function, index, job):
            def scheduled():
                # The later job reaches admission first, as can happen
                # when a worker finishes while older workers await it.
                if index == 1:
                    assert allow_older.wait(5.0)
                elif index == 2:
                    later_waiting.set()
                return function(index, job)
            return super().submit(scheduled)

    monkeypatch.setattr(fetch_pool, "ThreadPoolExecutor", ScheduledPool)

    def action(index):
        def transfer():
            started.append(index)
            if index == 0:
                assert later_waiting.wait(5.0)
                first_finished.set()
            elif index >= 2:
                later_started.set()
            return _entry(index)
        return transfer

    outcome = {}

    def run():
        try:
            outcome["result"] = fetch_pool.run_transfers(
                [_job(index, action(index)) for index in range(8)], workers=6)
        except BaseException as error:
            outcome["error"] = error

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    try:
        assert first_finished.wait(5.0)
        # The slot is free, but ticket 1 still owns the next admission.
        assert not later_started.wait(0.05)
    finally:
        allow_older.set()
        worker.join(5.0)
    assert not worker.is_alive()
    if "error" in outcome:
        raise outcome["error"]
    entries, receipt = outcome["result"]
    assert started == list(range(8))
    assert [entry["name"] for entry in entries] == [
        f"object-{index}" for index in range(8)]
    assert receipt["host_caps"] == {HOST: 1}


def test_fifo_admission_still_uses_every_permitted_host_slot(monkeypatch):
    monkeypatch.setitem(fetch_pool.HOST_FILE_WORKER_CAPS, HOST, 2)
    lock = threading.Lock()
    both_running = threading.Event()
    active = 0
    peak = 0

    def action(index):
        def transfer():
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
                if active == 2:
                    both_running.set()
            try:
                assert both_running.wait(5.0)
                time.sleep(0.01)
                return _entry(index)
            finally:
                with lock:
                    active -= 1
        return transfer

    entries, receipt = fetch_pool.run_transfers(
        [_job(index, action(index)) for index in range(8)], workers=6)
    assert peak == 2
    assert active == 0
    assert len(entries) == 8
    assert receipt["workers_effective"] == 6
    assert receipt["host_caps"] == {HOST: 2}


def test_a_host_wait_does_not_block_another_hosts_available_slot(monkeypatch):
    other_host = "another.transfer.example"
    for host in (HOST, other_host):
        monkeypatch.setitem(fetch_pool.HOST_FILE_WORKER_CAPS, host, 1)
    other_finished = threading.Event()

    def first():
        assert other_finished.wait(5.0)
        return _entry(0)

    def other():
        other_finished.set()
        return _entry(1)

    entries, receipt = fetch_pool.run_transfers(
        [_job(0, first), _job(1, other, host=other_host),
         _job(2, lambda: _entry(2))], workers=3)
    assert [entry["name"] for entry in entries] == [
        "object-0", "object-1", "object-2"]
    assert receipt["host_caps"] == {other_host: 1, HOST: 1}


@pytest.mark.parametrize("refusal", [
    ValueError("object-0 integrity check failed"), KeyboardInterrupt()])
def test_failure_wakes_host_waiters_and_preserves_the_original_refusal(
        monkeypatch, refusal):
    monkeypatch.setitem(fetch_pool.HOST_FILE_WORKER_CAPS, HOST, 1)
    waiter_reached = threading.Event()
    original_executor = fetch_pool.ThreadPoolExecutor
    started = []

    class ObservedPool(original_executor):
        def submit(self, function, index, job):
            def observed():
                if index > 0:
                    waiter_reached.set()
                return function(index, job)
            return super().submit(observed)

    monkeypatch.setattr(fetch_pool, "ThreadPoolExecutor", ObservedPool)

    def first():
        started.append(0)
        assert waiter_reached.wait(5.0)
        raise refusal

    def later(index):
        def transfer():
            started.append(index)
            return _entry(index)
        return transfer

    begin = time.monotonic()
    with pytest.raises(type(refusal)) as raised:
        fetch_pool.run_transfers(
            [_job(0, first), *[
                _job(index, later(index)) for index in range(1, 8)]], workers=6)
    assert raised.value is refusal
    assert started == [0]
    assert time.monotonic() - begin < 2.0
