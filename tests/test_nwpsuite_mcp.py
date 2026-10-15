from __future__ import annotations

import os

import pytest

from gpuwm import proc_identity
from gpuwm.mcp.gpulock import GpuLock, CardHeld


def test_disjoint_card_admission_and_overlap(tmp_path):
    lock = GpuLock(tmp_path)
    identity = proc_identity.identify(os.getpid())
    lock.claim(lambda: ("first", os.getpid(), identity), cards=[0, 1])
    lock.claim(lambda: ("second", os.getpid(), identity), cards=[2, 3])
    with pytest.raises(CardHeld):
        lock.claim(lambda: ("overlap", os.getpid(), identity), cards=[1, 2])
    assert {h["job_id"] for h in lock.holders()} == {"first", "second"}
    lock.release("first")
    assert lock.held_by([0]) is None
    assert lock.held_by([2])["job_id"] == "second"
    lock.release("second")
    assert lock.holders() == []


def test_stale_card_holder_reclaims(tmp_path):
    lock = GpuLock(tmp_path)
    lock.acquire("gone", 2147483646, {"pid": 2147483646, "start": "never"})
    identity = proc_identity.identify(os.getpid())
    lock.claim(lambda: ("now", os.getpid(), identity), cards=[0])
    assert lock.held_by([0])["job_id"] == "now"


def test_suite_tools_registered():
    from gpuwm.mcp.suite_tools import register
    class Server:
        def tool(self, *, name):
            return lambda fn: fn
    assert set(register(Server(), None)) == {"suite_plan", "suite_status", "suite_halt", "suite_resume", "suite_rehearse", "suite_run", "suite_events", "suite_apply"}
